"""Pilot for the FLAG-architecture study (stage 2): measure before the full run is approved.

    python scripts/fa_pilot.py --config configs/experiments/flagarch/F2_disc_resid_finetune.yaml --n 200 --device cuda:0

1. FLAG-style D and R generation for n random targets (all splits, labels never read): subgraph format success,
   node coverage, prompt / answer tokens, context overflows, answers hitting the 550-token cap, seconds per subgraph.
   Answers go into the same caches the full run uses, so they are not regenerated later.
2. Metapath summaries of the new lines for those targets, timed.
3. Classifier check on a few targets with both texts: the three +FLAG* loss terms at initialisation (to see whether
   FLAG's raw squared-dot orthogonality is usable on 4096-d hidden states) and forward/backward times.
4. Cost estimate for the full run. Writes results/flag_arch/pilot/{summary.json,examples.jsonl}.
"""

import json
import time

import numpy as np
from _common import ROOT, base_parser, device_for, experiment_config

from dgp_repro import pipeline as P
from dgp_repro.data import load_graph, load_split
from dgp_repro.embeddings import make_embedder
from dgp_repro.experiment import make_classifier, make_token_counter, release_accelerator_memory
from dgp_repro.models.readout import first_token_loss, residual_uniform_loss
from dgp_repro.prompts import MetapathBlock, build_prompt
from dgp_repro.prompts.builder import load_final_template
from dgp_repro.summarization import make_summarizer
from dgp_repro.summarization.numeric import numeric_summary
from dgp_repro.training.flag_finetune import orthogonal_loss
from dgp_repro.training.trainer import _set_mode
from dgp_repro.utils.cache import TextStore

OUT = ROOT / "results" / "flag_arch" / "pilot"


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else None


def size_bucket(n):
    return "01-10" if n <= 10 else "11-25" if n <= 25 else "26-40" if n <= 40 else "41+"


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--check", type=int, default=16, help="targets used in the classifier check")
    args = parser.parse_args()
    cfg = experiment_config(args)
    device = device_for(args, cfg)
    OUT.mkdir(parents=True, exist_ok=True)

    graph = load_graph(cfg["dataset"])
    split = load_split(cfg["dataset"], graph)
    all_targets = split.all_targets()
    sample = sorted(int(v) for v in np.random.default_rng(0).choice(all_targets, args.n, replace=False))
    counter, counter_name, _ = make_token_counter(cfg)
    summarizer = make_summarizer(cfg["summarizer"], device)
    embedder = make_embedder(cfg["embedder"], device)
    summary = {"n": args.n, "sample_seed": 0, "config": args.config, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "kinds": {}}

    # 1. generation, one kind at a time so each is timed on its own
    ps = None
    for kinds in (["discriminative"], ["discriminative", "residual"]):
        c = json.loads(json.dumps(cfg))
        c["flag_arch"].update(kinds=kinds, only_targets=sample)
        t0 = time.time()
        ps = P.run_dgp_prompt_pipeline(graph, split, c, summarizer=summarizer, embedder=embedder, token_counter=counter,
                                       token_counter_name=counter_name, until="fa_text")
        summary["kinds"][kinds[-1]] = {"generation_seconds": round(time.time() - t0, 1)}
    fa, trims, mdk_key = ps.aux["fa"], ps.aux["trims"], ps.aux["mdk_key"]
    stage_dir = ROOT / "datasets" / "cache" / "fa_subgraph_text"
    for kind in fa["kinds"]:
        store = TextStore(stage_dir / fa["cache_keys"][kind] / "answers.jsonl")
        ex = [store.extra[str(v)] for v in sample]
        s = summary["kinds"][kind]
        s.update({k: v for k, v in fa["coverage"][kind].items()})
        sizes, ptoks = [len(e["nodes"]) for e in ex], [e["prompt_tokens"] for e in ex]
        s.update({"cache_key": fa["cache_keys"][kind],
                  "nodes_per_subgraph": {"mean": float(np.mean(sizes)), "p95": pct(sizes, 95), "max": max(sizes)},
                  "prompt_tokens": {"mean": float(np.mean(ptoks)), "p95": pct(ptoks, 95), "max": max(ptoks)},
                  "generated_tokens_mean": float(np.mean([e["generated_tokens"] or 0 for e in ex])),
                  "hit_cap": sum(bool(e.get("hit_cap")) for e in ex), "overflow": sum(bool(e.get("overflow")) for e in ex),
                  "seconds_per_subgraph": s["generation_seconds"] / args.n})
        by_size = {}
        for e in ex:
            by_size.setdefault(size_bucket(len(e["nodes"])), []).append(bool(e["ok"]))
        s["success_by_subgraph_size"] = {b: {"n": len(v), "ok": sum(v)} for b, v in sorted(by_size.items())}
    with open(OUT / "examples.jsonl", "w", encoding="utf-8") as f:
        for kind in fa["kinds"]:
            for v in sample[:3]:
                r = fa[kind][v]
                f.write(json.dumps({"kind": kind, "target": v, "nodes": r.nodes, "ok": r.ok, "lines": r.lines}) + "\n")

    # 2. metapath summaries of the new lines, for the sample only
    requests = [(v, name, [int(u) for u in t["neighbors"][v]]) for name, t in trims.items() for v in sample]
    n_req = sum(1 for r in requests if r[2])
    path_text = {}
    for kind in fa["kinds"]:
        t0 = time.time()
        path_text[kind] = P._flag_arch_path_text(graph, fa, kind, requests, summarizer, "metapath_summary",
                                                 cfg["dgp"]["B_meta"], mdk_key, None, print)
        took = time.time() - t0
        summary["kinds"][kind]["metapath_summaries"] = {"requests": n_req, "seconds": round(took, 1),
                                                        "seconds_per_summary": took / max(n_req, 1)}
    question = cfg["dataset"].get("task", {}).get("question", "Is this fraud?")
    precision = cfg["dgp"].get("numeric_precision", 2)
    branch = P._flag_arch_branch_prompts(graph, np.array(sample), trims, fa, path_text, question, precision)
    main_tpl = load_final_template(cfg["templates"]["final"])
    main = {}
    for v in sample:
        blocks = [MetapathBlock(name, len(t["neighbors"][v]), path_text["discriminative"].get((v, name)),
                                numeric_summary(graph.numeric, t["neighbors"][v])) for name, t in trims.items()]
        main[v] = build_prompt(graph.texts[v], blocks, graph.numeric_names, question, template=main_tpl,
                               precision=precision, extras={"target_disc": fa["discriminative"][v].text_for(v, graph.texts)})
    summary["prompt_tokens"] = {k: {"mean": float(np.mean([counter(p) for p in d.values()])), "max": max(counter(p) for p in d.values())}
                                for k, d in (("main_F1", main), ("disc_only", branch["disc_only"]),
                                             ("resid_only", branch["resid_only"])) if d}
    summary["usable_both_kinds"] = len(branch["usable"])
    with open(OUT / "examples.jsonl", "a", encoding="utf-8") as f:
        for v in branch["usable"][:2]:
            f.write(json.dumps({"kind": "prompts", "target": v, "main_F1": main[v], "disc_only": branch["disc_only"][v],
                                "resid_only": branch["resid_only"][v]}) + "\n")
    del summarizer, embedder
    release_accelerator_memory()

    # 3. classifier check: the +FLAG* loss terms at initialisation, and timing
    import torch
    model, lt = make_classifier(cfg, device, seed=0)
    use = branch["usable"][:args.check]
    rows, fb_times, fwd_times = [], [], []
    for i in range(0, len(use), 2):
        ids = use[i:i + 2]
        _set_mode(model, True)
        t0 = time.time()
        d_logits, h_d = model.last_token_logits_hidden([branch["disc_only"][v] for v in ids])
        r_logits, h_r = model.last_token_logits_hidden([branch["resid_only"][v] for v in ids])
        y = torch.as_tensor(graph.labels[ids].astype(int), device=d_logits.device)
        ce = first_token_loss(d_logits, y, lt.yes_id, lt.no_id)
        kl = residual_uniform_loss(r_logits, lt.yes_id, lt.no_id)
        dot2 = orthogonal_loss(h_d, h_r, "squared_dot") / len(ids)
        cos2 = orthogonal_loss(h_d, h_r, "squared_cosine") / len(ids)
        (ce + 0.1 * kl + 0.1 * dot2).backward()
        torch.cuda.synchronize()
        fb_times.append((time.time() - t0) / len(ids))
        for p in model.trainable_parameters():
            p.grad = None
        rows.append({"ce": float(ce), "kl": float(kl), "squared_dot": float(dot2), "squared_cosine": float(cos2),
                     "h_norm_d": float(h_d.float().norm(dim=-1).mean()), "h_norm_r": float(h_r.float().norm(dim=-1).mean())})
    _set_mode(model, False)
    with torch.no_grad():
        for i in range(0, min(len(sample), 32), 8):
            t0 = time.time()
            model.last_token_logits([main[v] for v in sample[i:i + 8]])
            torch.cuda.synchronize()
            fwd_times.append((time.time() - t0) / 8)
    summary["classifier_check"] = {
        "examples": len(use),
        "per_example_mean": {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else None,
        "fwd_bwd_seconds_per_example_D_plus_R": float(np.mean(fb_times)) if fb_times else None,
        "fwd_seconds_per_example_main": float(np.mean(fwd_times)) if fwd_times else None}

    # 4. full-run estimate (hours), from the measured rates
    n_t, n_train, n_val = len(all_targets), len(split.train), len(split.val)
    k = summary["kinds"]
    est = {"generation_h": {kind: k[kind]["seconds_per_subgraph"] * n_t / 3600 for kind in k},
           "metapath_summaries_h": {kind: k[kind]["metapath_summaries"]["seconds_per_summary"] * n_req / args.n * n_t / 3600
                                    for kind in k}}
    cc = summary["classifier_check"]
    if cc["fwd_bwd_seconds_per_example_D_plus_R"]:
        usable_rate = summary["usable_both_kinds"] / args.n
        epoch = cc["fwd_bwd_seconds_per_example_D_plus_R"] * usable_rate * n_train + cc["fwd_seconds_per_example_main"] * n_val
        est["F2_plus_flag_star_phase_h_per_seed"] = 30 * epoch / 3600
    summary["estimate"] = est
    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
