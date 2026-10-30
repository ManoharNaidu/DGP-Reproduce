"""DGP graph-to-prompt pipeline: the data flow of the paper's Figure 3.

    target raw text ───────────────────────────────────────────────┐ fine-grained
    graph ─> metapaths up to K hops ─> MDK Top-M trimming           │
          ─> node summaries s_u (B_node) ─> metapath summary S_P(v) (B_meta)   ├─> prompt(v)
          ─> neighbour mean a_P(v)                                  │ coarse-grained
                                                                    ┘

Provenance: PAPER_RECONSTRUCTION. Every stage is cached (datasets/cache/<stage>/<key>/),
keyed by the settings that determine it, so summaries are never regenerated needlessly.

Component switches (configs `components:`), mapping to the paper's Figure 4 ablations:
    text_summary     False -> "w/o TextSumm": no neighbour text in the prompt
    mdk              False -> "w/o MDK": M neighbours chosen at random instead of by diffusion distance
    path_summary     False -> "w/o PathSumm": node summaries are concatenated, not summarized again
    numeric_summary  False -> "w/o NumSumm": no neighbour means
    node_summary     False -> neighbours enter path summarization as raw text (not a paper ablation)

Leakage guards: labels are read only to pick train/val/test nodes (done before this module);
MDK, summaries and prompts see node text, numeric features and structure only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from dgp_repro.data import HeteroGraph, Split
from dgp_repro.embeddings import build_mdk_features
from dgp_repro.mdk import enumerate_metapaths, metapath_name, trim_metapath
from dgp_repro.prompts import MetapathBlock, build_prompt
from dgp_repro.prompts.builder import load_final_template
from dgp_repro.summarization import load_template, metapath_summaries, node_summaries, prompt_version, template_names
from dgp_repro.summarization.numeric import numeric_summary
from dgp_repro.summarization.pipeline import generation_shard, join_neighbor_texts
from dgp_repro.utils.cache import ArtifactCache, TextStore

TokenCounter = Callable[[str], int]
PATH_SUMM_OFF_SEPARATOR = " | "
STAGES = ("mdk", "fa_text", "target_text", "node_summaries", "metapath_summaries", "prompts")


@dataclass
class PromptSet:
    prompts: dict[int, str]
    token_rows: list[dict]
    trim_stats: list[dict]
    metadata: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)      # node -> {"target_disc": ..., "target_residual": ...}
    aux: dict = field(default_factory=dict)         # FLAG-architecture study: branch-only prompts (see flag_arch)


def target_text_specs(cfg: dict) -> dict:
    """Optional LLM-generated target text (extension; empty for the baseline): kind -> {template, budget}."""
    return {k: v for k, v in (cfg.get("target_text") or {}).items() if v}


def _components(cfg: dict) -> dict:
    defaults = {"node_summary": True, "text_summary": True, "mdk": True, "path_summary": True, "numeric_summary": True}
    return {**defaults, **cfg.get("components", {})}


def _node_token_counts(graph: HeteroGraph, counter: TokenCounter, counter_name: str, cache_root) -> np.ndarray:
    cache = ArtifactCache("token_counts", {"dataset": graph.name, "nodes": graph.num_nodes, "counter": counter_name},
                          cache_root)
    path = cache.path("counts.npy")
    if path.exists():
        return np.load(path)
    counts = np.array([counter(t) for t in graph.texts], dtype=np.int64)
    np.save(path, counts)
    return counts


def text_embedding_cache_path(graph: HeteroGraph, cfg: dict, cache_root=None):
    """Where the DeBERTaV3 embeddings for this graph and embedder config are cached (shared by MDK and the MLP)."""
    emb_cfg = cfg.get("embedder", {})
    if emb_cfg.get("backend", "deberta") == "deberta":
        name = emb_cfg.get("model_id", "microsoft/deberta-v3-base")  # == DebertaEmbedder.name
    else:
        name = "hashing-debug"  # == HashingEmbedder.name
    cache = ArtifactCache("embeddings", {"dataset": graph.name, "nodes": graph.num_nodes, "embedder": name,
                                         "embedder_cfg": emb_cfg}, cache_root)
    return cache.path("text_embeddings.npy")


def _mdk_features(graph: HeteroGraph, cfg: dict, embedder, cache_root, log) -> np.ndarray:
    mdk_cfg = cfg.get("mdk", {})
    mode = mdk_cfg.get("features", "deberta_concat")
    text_emb = None
    if mode == "deberta_concat":
        path = text_embedding_cache_path(graph, cfg, cache_root)
        if path.exists():
            text_emb = np.load(path)
        else:
            log(f"embedding {graph.num_nodes} node texts with {embedder.name}")
            text_emb = embedder.embed(graph.texts)
            np.save(path, text_emb)
    return build_mdk_features(text_emb, graph.numeric, mode, mdk_cfg.get("scaling", "none"))


def _save_trim(path, result) -> None:
    targets = np.array(sorted(result.neighbors), dtype=np.int64)
    lists = [result.neighbors[v] for v in targets]
    np.savez(path, targets=targets, offsets=np.cumsum([0] + [len(x) for x in lists]),
             neighbors=np.concatenate(lists) if lists else np.zeros(0, dtype=np.int64),
             sizes=np.array([result.neighborhood_sizes[v] for v in targets]),
             raw_tokens=np.array([result.neighborhood_raw_tokens.get(v, 0) for v in targets]),
             stats=json.dumps(result.stats))


def _load_trim(path) -> tuple[dict, dict, dict, dict]:
    # Read every array exactly once: NpzFile re-reads the whole array on each z[...] access, so
    # indexing inside the loop would give each slice its own full copy (GBs on real graphs).
    with np.load(path, allow_pickle=False) as z:
        targets, offsets, flat = z["targets"], z["offsets"], z["neighbors"]
        sizes = dict(zip(targets.tolist(), z["sizes"].tolist()))
        raw = dict(zip(targets.tolist(), z["raw_tokens"].tolist()))
        stats = json.loads(str(z["stats"]))
    neighbors = {int(v): flat[offsets[i]:offsets[i + 1]] for i, v in enumerate(targets)}
    return neighbors, sizes, raw, stats


def run_dgp_prompt_pipeline(graph: HeteroGraph, split: Split, cfg: dict, *, summarizer, embedder,
                            token_counter: TokenCounter, token_counter_name: str,
                            truncate_target: Callable[[str], tuple[str, bool]] | None = None,
                            cache_root=None, log: Callable[[str], None] = print,
                            until: str = "prompts") -> PromptSet:
    """until: 'mdk' | 'node_summaries' | 'prompts'. Earlier stops fill the caches and return no prompts."""
    if until not in STAGES:
        raise ValueError(f"until must be one of {STAGES}")
    if generation_shard() and until not in ("target_text", "node_summaries", "metapath_summaries"):
        raise ValueError("DGP_GEN_SHARD is set: a shard may only fill text caches (target_text, node_summaries, "
                         "metapath_summaries); build MDK and prompts in one unsharded process")
    dgp = cfg["dgp"]
    K, M, B_node, B_meta = dgp["K"], dgp["M"], dgp["B_node"], dgp["B_meta"]
    comp = _components(cfg)
    mdk_cfg = cfg.get("mdk", {})
    task_aware = cfg.get("task_aware", {})
    targets = split.all_targets()
    relation_names = cfg["dataset"]["graph"].get("relations") if "graph" in cfg["dataset"] else None
    metapaths = enumerate_metapaths(relation_names or list(graph.relations), K)
    uses_neighbors = comp["text_summary"] or comp["numeric_summary"]

    token_counts = _node_token_counts(graph, token_counter, token_counter_name, cache_root)

    # ---- 1. diffusion-based metapath trimming (Eq. 6-9) ----------------------------------------
    trims: dict[str, dict] = {}
    trim_stats: list[dict] = []
    if uses_neighbors:
        features = _mdk_features(graph, cfg, embedder, cache_root, log) if comp["mdk"] else None
        targets_hash = hashlib.sha1(np.sort(targets).astype(np.int64).tobytes()).hexdigest()[:16]
        trim_meta = {"dataset": graph.name, "split_seed": split.seed, "targets": int(len(targets)),
                     "targets_hash": targets_hash, "K": K, "M": M,
                     "use_mdk": comp["mdk"], "operator": mdk_cfg.get("operator", "paper_eq6"),
                     "adjacency": mdk_cfg.get("adjacency", "weighted"), "features": mdk_cfg.get("features", "deberta_concat"),
                     "scaling": mdk_cfg.get("scaling", "none"),
                     "embedder": embedder.name if (comp["mdk"] and embedder is not None) else None,
                     "random_seed": mdk_cfg.get("random_seed", 0), "counter": token_counter_name}
        trim_cache = ArtifactCache("mdk", trim_meta, cache_root)
        for path in metapaths:
            name = metapath_name(path)
            file = trim_cache.path(f"{name}.npz")
            if not file.exists():
                if generation_shard():
                    raise RuntimeError(f"MDK cache {trim_cache.key}/{name}.npz missing: build it unsharded "
                                       "(--until mdk) before starting shard processes")
                log(f"trimming metapath {name} (K={K}, M={M}, mdk={comp['mdk']})")
                result = trim_metapath(graph.relations, path, features, targets, K, M, use_mdk=comp["mdk"],
                                       operator=trim_meta["operator"], adjacency=trim_meta["adjacency"],
                                       seed=trim_meta["random_seed"], node_token_counts=token_counts)
                _save_trim(file, result)
            neighbors, sizes, raw, stats = _load_trim(file)
            trims[name] = {"neighbors": neighbors, "sizes": sizes, "raw_tokens": raw}
            trim_stats.append(stats)

    if until == "mdk":
        return PromptSet(prompts={}, token_rows=[], trim_stats=trim_stats, metadata={"stopped_after": "mdk"})

    # ---- 1b. FLAG-architecture study (extension): D / R text generated per trimmed subgraph -----------
    fa_cfg = cfg.get("flag_arch") or {}
    fa: dict = {}
    if fa_cfg.get("enabled"):
        fa = _flag_arch_text(graph, targets, trims, fa_cfg, summarizer, trim_cache.key, targets_hash, cache_root, log)
    if until == "fa_text":                        # aux carries the subgraph texts and trims for the pilot script
        return PromptSet(prompts={}, token_rows=[], trim_stats=trim_stats,
                         metadata={"stopped_after": "fa_text", "fa_coverage": fa.get("coverage")},
                         aux={"fa": fa, "trims": trims, "mdk_key": trim_cache.key if uses_neighbors else None})
    if until == "target_text":                    # target D/R lines only; they read raw text, not the summaries
        _generate_target_texts(graph, targets, cfg, summarizer, cache_root, log)
        return PromptSet(prompts={}, token_rows=[], trim_stats=trim_stats, metadata={"stopped_after": "target_text"})

    # ---- 2. node-level summaries s_u (Eq. 5) --------------------------------------------------
    node_template_name, meta_template_name = template_names(task_aware.get("enabled", False),
                                                            task_aware.get("apply_to_metapath", False))
    node_template_name = (cfg.get("templates") or {}).get("node", node_template_name)   # extension override
    per_node_text: dict[int, str] = {}
    if comp["text_summary"] and not fa:          # FLAG architecture: the subgraph lines replace node summaries
        used = sorted({int(u) for t in trims.values() for nbrs in t["neighbors"].values() for u in nbrs})
        if comp["node_summary"]:
            store_meta = {"dataset": graph.name, "summarizer": summarizer.name, "prompt_version": prompt_version(),
                          "template": node_template_name, "B_node": B_node}
            store = TextStore(ArtifactCache("node_summaries", store_meta, cache_root).path("summaries.jsonl"))
            log(f"node summaries: {len(used)} nodes needed, {sum(str(u) in store for u in used)} cached")
            per_node_text = node_summaries(graph.texts, used, summarizer, load_template(node_template_name), B_node, store)
        else:
            per_node_text = {u: graph.texts[u] for u in used}

    if until == "node_summaries":
        return PromptSet(prompts={}, token_rows=[], trim_stats=trim_stats, metadata={"stopped_after": "node_summaries"})

    # ---- 3. metapath-level summaries S_P(v) (Eq. 10) ------------------------------------------
    path_text: dict[tuple[int, str], str] = {}
    fa_path_text: dict[str, dict] = {}
    if comp["text_summary"] and fa:
        requests = [(int(v), name, [int(u) for u in t["neighbors"][int(v)]]) for name, t in trims.items() for v in targets]
        for kind in fa["kinds"]:
            fa_path_text[kind] = _flag_arch_path_text(graph, fa, kind, requests, summarizer, meta_template_name,
                                                      B_meta, trim_cache.key, cache_root, log)
        path_text = fa_path_text["discriminative"]
    elif comp["text_summary"]:
        requests = [(int(v), name, [int(u) for u in t["neighbors"][int(v)]]) for name, t in trims.items() for v in targets]
        if comp["path_summary"]:
            store_meta = {"dataset": graph.name, "summarizer": summarizer.name, "prompt_version": prompt_version(),
                          "node_template": node_template_name, "template": meta_template_name,
                          "node_summary": comp["node_summary"], "B_node": B_node, "B_meta": B_meta,
                          "mdk_cache": trim_cache.key}
            store = TextStore(ArtifactCache("metapath_summaries", store_meta, cache_root).path("summaries.jsonl"))
            log(f"metapath summaries: {sum(1 for r in requests if r[2])} requests, {len(store)} cached")
            path_text = metapath_summaries(requests, per_node_text, summarizer, load_template(meta_template_name),
                                           B_meta, store)
        else:
            path_text = {(v, name): PATH_SUMM_OFF_SEPARATOR.join(per_node_text[u] for u in nbrs)
                         for v, name, nbrs in requests if nbrs}

    if until == "metapath_summaries":
        return PromptSet(prompts={}, token_rows=[], trim_stats=trim_stats, metadata={"stopped_after": "metapath_summaries"})

    # ---- 3b. optional generated target text (discriminative / residual), cached per template and budget
    target_extra = _generate_target_texts(graph, targets, cfg, summarizer, cache_root, log)
    for kind in fa.get("kinds", []):              # the target's own line from its subgraph prompt (raw text if failed)
        key = "disc" if kind == "discriminative" else "residual"
        target_extra[key] = {int(v): fa[kind][int(v)].text_for(int(v), graph.texts) for v in targets}

    # ---- 4. prompts (Eq. 11-12) and token accounting -------------------------------------------
    question = cfg["dataset"].get("task", {}).get("question", "Is this fraud?")
    precision = dgp.get("numeric_precision", 2)
    split_of = {int(v): s for s, ids in (("train", split.train), ("val", split.val), ("test", split.test)) for v in ids}
    prompts: dict[int, str] = {}
    rows: list[dict] = []
    extras_out: dict[int, dict] = {}
    final_name = (cfg.get("templates") or {}).get("final")
    final_template = load_final_template(final_name) if final_name else None
    for v in targets.tolist():
        target_text, truncated = truncate_target(graph.texts[v]) if truncate_target else (graph.texts[v], False)
        blocks, node_sum_tokens, trimmed_raw, n_full, n_trim, raw_neighbor = [], 0, 0, 0, 0, 0
        for name, t in trims.items():
            nbrs = t["neighbors"][v]
            n_full += t["sizes"][v]
            n_trim += len(nbrs)
            raw_neighbor += t["raw_tokens"][v]
            trimmed_raw += int(token_counts[nbrs].sum()) if len(nbrs) else 0
            if comp["text_summary"] and len(nbrs) and fa:
                node_sum_tokens += token_counter("\n".join(fa["discriminative"][v].text_for(int(u), graph.texts)
                                                           for u in nbrs))
            elif comp["text_summary"] and len(nbrs):
                node_sum_tokens += token_counter(join_neighbor_texts([int(u) for u in nbrs], per_node_text))
            blocks.append(MetapathBlock(
                name=name, num_neighbors=len(nbrs),
                text=path_text.get((v, name)) if comp["text_summary"] else None,
                numeric=numeric_summary(graph.numeric, nbrs) if comp["numeric_summary"] else None))
        extras = {f"target_{k}": t[v] for k, t in target_extra.items()}
        if extras:
            extras_out[v] = extras
        prompt = build_prompt(target_text, blocks, graph.numeric_names, question, template=final_template,
                              precision=precision, extras=extras)
        prompts[v] = prompt
        rows.append({
            "node": v, "split": split_of[v], "raw_target_tokens": int(token_counts[v]),
            "raw_neighbor_tokens": raw_neighbor, "trimmed_neighbor_raw_tokens": trimmed_raw,
            "node_summary_tokens": node_sum_tokens,
            "metapath_summary_tokens": sum(token_counter(path_text[(v, n)]) for n in trims if (v, n) in path_text),
            "final_prompt_tokens": token_counter(prompt), "num_neighbors_full": n_full,
            "num_neighbors_trimmed": n_trim, "num_metapaths": len(trims), "K": K, "M": M,
            "B_node": B_node, "B_meta": B_meta, "target_truncated": truncated,
        })

    metadata = {"metapaths": [metapath_name(p) for p in metapaths], "components": comp,
                "prompt_version": prompt_version(), "token_counter": token_counter_name,
                "summarizer": summarizer.name if summarizer else None,
                "embedder": embedder.name if embedder is not None else None}
    aux = {}
    if fa:
        metadata["fa_coverage"] = fa["coverage"]
        aux = _flag_arch_branch_prompts(graph, targets, trims, fa, fa_path_text, question, precision)
    return PromptSet(prompts=prompts, token_rows=rows, trim_stats=trim_stats, metadata=metadata, extras=extras_out,
                     aux=aux)


def _generate_target_texts(graph: HeteroGraph, targets, cfg: dict, summarizer, cache_root, log) -> dict[str, dict[int, str]]:
    """Optional generated target text (discriminative / residual), cached per template and budget."""
    target_extra: dict[str, dict[int, str]] = {}
    for kind, spec in target_text_specs(cfg).items():
        store_meta = {"dataset": graph.name, "summarizer": summarizer.name, "prompt_version": prompt_version(),
                      "template": spec["template"], "B_node": spec["budget"], "role": "target"}
        cut = spec.get("truncate_chars")      # FLAG feeds the LLM raw_text[:1200]; key omitted when unset
        if cut:
            store_meta["truncate_chars"] = cut
        source = [t[:cut] for t in graph.texts] if cut else graph.texts
        store = TextStore(ArtifactCache("node_summaries", store_meta, cache_root).path("summaries.jsonl"))
        tl = [int(t) for t in targets]
        log(f"target {kind}: {len(tl)} targets, {sum(str(u) in store for u in tl)} cached")
        generated = node_summaries(source, tl, summarizer, load_template(spec["template"]), spec["budget"], store)
        if spec.get("fallback_raw"):         # FLAG: a node whose LLM text is unusable keeps its raw text
            empty = [u for u, s in generated.items() if not s.strip()]
            generated.update({u: source[u] for u in empty})
            log(f"target {kind}: {len(empty)} empty outputs replaced by raw text")
        target_extra[kind] = generated
    return target_extra


# ---------------------------------------------------------------- FLAG-architecture study (extension)
def _flag_arch_text(graph, targets, trims, fa_cfg, summarizer, mdk_key, targets_hash, cache_root, log) -> dict:
    """D (and R) lines for every target subgraph, FLAG style (summarization/flag_subgraph.py)."""
    from dgp_repro.summarization.flag_subgraph import FlagPrompts, coverage, store_metadata, subgraph_texts

    prompts = FlagPrompts.load(fa_cfg.get("prompt_set", "fa"))
    kinds = list(fa_cfg.get("kinds", ["discriminative"]))
    out = {"kinds": kinds, "coverage": {}, "cache_keys": {}}
    # `only_targets` (pilot only): generate for a subset, in the same cache as the full run so answers are reused
    gen_targets = [int(v) for v in fa_cfg.get("only_targets") or targets]
    for kind in kinds:
        meta = store_metadata(graph.name, summarizer.name, prompts, kind, fa_cfg.get("max_new_tokens", 550),
                              fa_cfg.get("truncate_chars", 1200), mdk_key, targets_hash, fa_cfg.get("prompt_set", "fa"))
        cache = ArtifactCache("fa_subgraph_text", meta, cache_root)
        store = TextStore(cache.path("answers.jsonl"))
        out[kind] = subgraph_texts(graph.texts, gen_targets, trims, summarizer, prompts, kind, store,
                                   max_new_tokens=fa_cfg.get("max_new_tokens", 550),
                                   truncate=fa_cfg.get("truncate_chars", 1200),
                                   batch_size=fa_cfg.get("batch_size", 8),
                                   max_input_tokens=fa_cfg.get("max_input_tokens", 32768), log=log)
        out["coverage"][kind] = coverage(out[kind])
        out["cache_keys"][kind] = cache.key
        log(f"FA {kind}: {out['coverage'][kind]}")
    return out


def _flag_arch_path_text(graph, fa, kind, requests, summarizer, meta_template_name, B_meta, mdk_key, cache_root,
                         log) -> dict:
    """Metapath summaries S_P(v) of the neighbours' lines from target v's own subgraph prompt."""
    from dgp_repro.summarization.pipeline import NEIGHBOR_SEPARATOR, _batched_fill, metapath_key

    store_meta = {"dataset": graph.name, "summarizer": summarizer.name, "prompt_version": prompt_version(),
                  "node_text": f"fa_subgraph_text/{fa['cache_keys'][kind]}", "template": meta_template_name,
                  "B_meta": B_meta, "mdk_cache": mdk_key}
    store = TextStore(ArtifactCache("metapath_summaries", store_meta, cache_root).path("summaries.jsonl"))
    items = [(metapath_key(v, name, nbrs), NEIGHBOR_SEPARATOR.join(fa[kind][v].text_for(u, graph.texts) for u in nbrs))
             for v, name, nbrs in requests if nbrs]
    log(f"FA {kind} metapath summaries: {len(items)} requests, {len(store)} cached")
    _batched_fill(store, items, summarizer, load_template(meta_template_name), B_meta, 32, None)
    return {(v, name): store[metapath_key(v, name, nbrs)] for v, name, nbrs in requests if nbrs}


def _flag_arch_branch_prompts(graph, targets, trims, fa, fa_path_text, question, precision) -> dict:
    """Branch-only prompts for FLAG's +FLAG* phase: the shared model reads ONLY one kind of text, as FLAG's backbone
    reads only x_disc or x_common there (FLAG-Reproduce/src/flagbench/training/trainer.py:finetune_extra).
    A target is usable only if both its D and R subgraphs parsed (FLAG drops failed subgraphs from that phase)."""
    out = {"usable": [], "disc_only": {}, "resid_only": {}}
    if "residual" not in fa["kinds"]:
        return out
    template = load_final_template("final_prompt")
    for v in targets.tolist():
        if not (fa["discriminative"][v].ok and fa["residual"][v].ok):
            continue
        out["usable"].append(int(v))
        for kind, key in (("discriminative", "disc_only"), ("residual", "resid_only")):
            blocks = [MetapathBlock(name=name, num_neighbors=len(t["neighbors"][v]),
                                    text=fa_path_text[kind].get((v, name)), numeric=None) for name, t in trims.items()]
            out[key][int(v)] = build_prompt(fa[kind][v].text_for(int(v), graph.texts), blocks, graph.numeric_names,
                                            question, template=template, precision=precision)
    return out
