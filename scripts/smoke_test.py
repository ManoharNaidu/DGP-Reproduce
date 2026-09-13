"""CPU_DEBUG smoke test of the whole DGP pipeline. DEBUG_ONLY — results are meaningless.

    python scripts/smoke_test.py
    python scripts/smoke_test.py --real-tokenizer   # also validate Yes/No against the real Qwen3 tokenizer (downloads ~11 MB)

Checks, each through the same code used for real runs:
    heterogeneous graph loading, metapath adjacency, MDK diffusion, Top-M selection,
    node and metapath summarization interfaces, numeric aggregation, prompt construction,
    every Figure 4 ablation variant, Yes/No token resolution, Eq. 13 training and Eq. 14 readout,
    metrics, and token accounting.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dgp_repro.config import load_experiment  # noqa: E402
from dgp_repro.data import build_graph, make_split  # noqa: E402
from dgp_repro.embeddings import make_embedder  # noqa: E402
from dgp_repro.evaluation.complexity import dgp_final_tokens, full_neighbor_tokens  # noqa: E402
from dgp_repro.evaluation.token_usage import summarize_token_usage  # noqa: E402
from dgp_repro.mdk import build_metapath_adjacency, enumerate_metapaths, trim_metapath  # noqa: E402
from dgp_repro.metrics import aggregate_seeds, compute_metrics  # noqa: E402
from dgp_repro.models import fraud_probability, resolve_label_token_ids  # noqa: E402
from dgp_repro.models.mock_llm import MockLLM  # noqa: E402
from dgp_repro.pipeline import run_dgp_prompt_pipeline  # noqa: E402
from dgp_repro.summarization import make_summarizer  # noqa: E402
from dgp_repro.training import predict, train  # noqa: E402
from dgp_repro.utils.seed import set_seed  # noqa: E402

ABLATIONS = {
    "DGP": {},
    "w/o MDK": {"mdk": False},
    "w/o PathSumm": {"path_summary": False},
    "w/o TextSumm": {"text_summary": False},
    "w/o NumSumm": {"numeric_summary": False},
}

results: list[tuple[str, bool, str]] = []


def check(name: str):
    def wrap(fn):
        def run(*args, **kwargs):
            try:
                detail = fn(*args, **kwargs) or ""
                results.append((name, True, detail))
                print(f"  PASS  {name}  {detail}")
            except Exception as exc:  # report and continue so every stage is covered
                results.append((name, False, repr(exc)))
                print(f"  FAIL  {name}: {exc!r}")
                traceback.print_exc()
        return run
    return wrap


def word_count(text: str) -> int:
    """DEBUG_ONLY token counter: whitespace words."""
    return len(text.split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/smoke.yaml")
    parser.add_argument("--real-tokenizer", action="store_true")
    args = parser.parse_args()

    started = time.time()
    set_seed(0)
    cfg = load_experiment(args.config)
    cache_root = Path(tempfile.mkdtemp(prefix="dgp_smoke_"))
    print(f"CPU_DEBUG smoke test  (DEBUG_ONLY; cache: {cache_root})\n")
    state: dict = {}

    @check("heterogeneous graph loading")
    def _graph():
        g = build_graph(cfg["dataset"])
        assert len(g.relations) == 3 and g.num_frauds > 0 and g.num_labeled < g.num_nodes
        state["graph"] = g
        state["split"] = make_split(g.labels, cfg["dataset"]["split"]["sizes"], seed=0)
        return f"N={g.num_nodes}, relations={list(g.relations)}, edges={g.num_edges}"
    _graph()
    g, split = state["graph"], state["split"]

    @check("metapath adjacency (Eq. 3)")
    def _adj():
        paths = enumerate_metapaths(list(g.relations), cfg["dgp"]["K"])
        assert len(paths) == 12  # (3^3 - 3)/2
        A = build_metapath_adjacency(g.relations, ("RUR", "RSR")).toarray()
        dense = g.relations["RUR"].toarray() @ g.relations["RSR"].toarray()
        assert np.allclose(A, dense)
        return f"{len(paths)} metapaths for K={cfg['dgp']['K']}"
    _adj()

    @check("MDK diffusion + Top-M (Eq. 6-9)")
    def _mdk():
        X = make_embedder(cfg["embedder"]).embed(g.texts)
        result = trim_metapath(g.relations, ("RSR",), X, split.all_targets(), K=2, M=3)
        sizes = [len(v) for v in result.neighbors.values()]
        assert max(sizes) <= 3
        return f"mean trimmed={np.mean(sizes):.2f}, empty={result.stats['empty_neighborhoods']}"
    _mdk()

    summarizer = make_summarizer(cfg["summarizer"])
    embedder = make_embedder(cfg["embedder"])
    prompt_sets = {}

    for label, switches in ABLATIONS.items():
        @check(f"prompt pipeline: {label}")
        def _variant(label=label, switches=switches):
            variant = {**cfg, "components": {**cfg["components"], **switches}}
            ps = run_dgp_prompt_pipeline(g, split, variant, summarizer=summarizer, embedder=embedder,
                                         token_counter=word_count, token_counter_name="words-debug",
                                         cache_root=cache_root, log=lambda *_: None)
            assert len(ps.prompts) == len(split.all_targets())
            sample = next(iter(ps.prompts.values()))
            assert sample.startswith("Target: ") and sample.endswith("Question: Is this fraud?")
            if switches.get("text_summary") is False:
                assert "neighbor mean" in sample and all("no neighbors" in line or "neighbor mean" in line
                                                         for line in sample.splitlines() if line.startswith("- "))
            if switches.get("numeric_summary") is False:
                assert "neighbor mean" not in sample
            prompt_sets[label] = ps
            return f"avg prompt={summarize_token_usage(ps.token_rows)['average_prompt_length']:.1f} words"
        _variant()

    if "DGP" in prompt_sets:
        example_node = int(split.train[0])
        print("\n  --- example DGP prompt (DEBUG_ONLY, mock summaries) ---")
        print("  " + prompt_sets["DGP"].prompts[example_node].replace("\n", "\n  "))
        print("  ------------------------------------------------------\n")

    @check("numeric aggregation (Eq. 11) matches manual mean")
    def _numeric():
        ps = prompt_sets["DGP"]
        cache_dir = next((cache_root / "mdk").iterdir())
        from dgp_repro.pipeline import _load_trim
        neighbors, *_ = _load_trim(cache_dir / "RUR.npz")
        v = next(n for n, nb in neighbors.items() if len(nb))
        expected = g.numeric[neighbors[v]].mean(axis=0)[0]
        assert f"RUR: " in ps.prompts[v] and f"rating={expected:.2f}" in ps.prompts[v]
        return f"node {v}: rating mean {expected:.2f} present in prompt"
    _numeric()

    @check("Yes/No label tokens (mock tokenizer)")
    def _labels_mock():
        model = MockLLM()
        tokens = resolve_label_token_ids(model.tokenizer)
        state["label_tokens"] = tokens
        state["model"] = model
        return f"yes_id={tokens.yes_id}, no_id={tokens.no_id}"
    _labels_mock()

    if args.real_tokenizer:
        @check("Yes/No label tokens (real Qwen/Qwen3-8B tokenizer, chat boundary)")
        def _labels_real():
            from transformers import AutoTokenizer
            from dgp_repro.models.qwen_classifier import render_chat_prompt
            tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
            rendered = render_chat_prompt(tok, prompt_sets["DGP"].prompts[int(split.train[0])])
            assert rendered.endswith("<think>\n\n</think>\n\n"), "thinking was not disabled"
            t = resolve_label_token_ids(tok, rendered_prompt=rendered)
            assert (t.yes_id, t.no_id) == (9454, 2753), (t.yes_id, t.no_id)
            return f"yes_id={t.yes_id}, no_id={t.no_id}, boundary verified"
        _labels_real()

    @check("Eq. 14 readout is a two-way softmax")
    def _readout():
        logits = torch.zeros(2, 32)
        logits[0, 2], logits[0, 3] = 2.0, 0.0
        p = fraud_probability(logits, 2, 3)
        assert torch.allclose(p[0], torch.sigmoid(torch.tensor(2.0))) and torch.allclose(p[1], torch.tensor(0.5))
        return "p = sigmoid(logit_Yes - logit_No)"
    _readout()

    @check("training (Eq. 13) + evaluation across 2 seeds")
    def _train():
        ps = prompt_sets["DGP"]
        tokens = state["label_tokens"]
        per_seed = []
        for seed in (0, 1):
            set_seed(seed)
            model = MockLLM(seed=seed)
            tr = [ps.prompts[int(v)] for v in split.train]
            va = [ps.prompts[int(v)] for v in split.val]
            te = [ps.prompts[int(v)] for v in split.test]
            result = train(model, tr, g.labels[split.train], va, g.labels[split.val], tokens, cfg["training"], seed)
            assert result.best_epoch >= 1 and result.history[0]["train_loss"] > 0
            per_seed.append(compute_metrics(g.labels[split.test], predict(model, te, tokens)))
        agg = aggregate_seeds(per_seed)
        assert all(np.isfinite(agg[f"{m}_mean"]) for m in ("macro_f1", "auroc", "auprc"))
        return f"test AUROC {agg['auroc_mean']:.1f} +/- {agg['auroc_std']:.1f} (DEBUG_ONLY)"
    _train()

    @check("token accounting + complexity formulas")
    def _tokens():
        s = summarize_token_usage(prompt_sets["DGP"].token_rows)
        assert s["token_reduction"] < 1 and s["num_metapaths"] == 12
        assert full_neighbor_tokens(170, 133, 2) > dgp_final_tokens(170, 3, 2, 10)
        return f"compression_ratio={s['compression_ratio']:.2f}, token_reduction={100 * s['token_reduction']:.1f}%"
    _tokens()

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed in {time.time() - started:.1f}s (CPU_DEBUG, DEBUG_ONLY)")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
