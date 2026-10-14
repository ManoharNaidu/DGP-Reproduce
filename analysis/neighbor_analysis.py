"""Task 3a: what happens to the neighbours at K = 1, 2, 3 (LLM-free).

Run:  /venv/main/bin/python analysis/neighbor_analysis.py
Needs the MDK caches for K=1,2,3 (scripts/build_mdk.py --set dgp.K=<k>) and the cached
DeBERTa embeddings. Reads labels ONLY for the homophily diagnostic and ONLY for train+val
targets; nothing here feeds the training pipeline.

Coupled sweep   : K sets metapath length (<=K) AND diffusion steps K (the repo's behaviour).
Decoupled check : metapath length <=2 with diffusion steps 1,2,3 (task 3 item 5, analysis only).
Outputs analysis/out/neighbor_analysis.{json,csv}
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dgp_repro.config import load_experiment  # noqa: E402
from dgp_repro.data import load_graph, load_split  # noqa: E402
from dgp_repro.embeddings import build_mdk_features  # noqa: E402
from dgp_repro.mdk import enumerate_metapaths, metapath_name, trim_metapath  # noqa: E402
from dgp_repro.pipeline import _load_trim, text_embedding_cache_path  # noqa: E402

CACHE = ROOT / "datasets" / "cache"
M = 4


def find_mdk_cache(K: int, split_seed: int, n_targets: int) -> Path:
    hits = []
    for meta in (CACHE / "mdk").glob("*/metadata.json"):
        m = json.loads(meta.read_text())
        if (m.get("K") == K and m.get("M") == M and m.get("use_mdk") and m.get("dataset") == "amazonvideo"
                and m.get("targets") == n_targets and m.get("features") == "deberta_concat"
                and "Qwen3" in str(m.get("counter"))):
            hits.append(meta.parent)
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one MDK cache for K={K}, found {hits}")
    return hits[0]


def qwen_token_counts(n_nodes: int) -> np.ndarray:
    for meta in (CACHE / "token_counts").glob("*/metadata.json"):
        m = json.loads(meta.read_text())
        if "Qwen3" in str(m.get("counter")) and m.get("nodes") == n_nodes:
            return np.load(meta.parent / "counts.npy")
    raise SystemExit("no Qwen token-count cache found")


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a, b = set(a.tolist()), set(b.tolist())
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def main() -> None:
    cfg = load_experiment("configs/experiments/dgp_amazon.yaml")
    graph = load_graph(cfg["dataset"])
    split = load_split(cfg["dataset"], graph)
    targets = split.all_targets()
    tok = qwen_token_counts(graph.num_nodes)
    y = graph.labels.astype(int)
    probe = np.concatenate([split.train, split.val])          # diagnostic never touches test labels
    rows, per_k_trims = [], {}

    for K in (1, 2, 3):
        d = find_mdk_cache(K, split.seed, len(targets))
        trims = {}
        for path in enumerate_metapaths(list(graph.relations), K):
            name = metapath_name(path)
            nb, sizes, raw, _ = _load_trim(d / f"{name}.npz")
            trims[name] = (nb, sizes)
        per_k_trims[K] = trims
        # ---- per-metapath statistics
        for name, (nb, sizes) in trims.items():
            s = np.array([sizes[int(v)] for v in targets])
            kept = np.array([len(nb[int(v)]) for v in targets])
            rows.append({"K": K, "metapath": name, "hops": name.count("-") + 1,
                         "mean_full_nbrs": round(float(s.mean()), 1), "median_full_nbrs": float(np.median(s)),
                         "p95_full_nbrs": float(np.percentile(s, 95)), "empty_pct": round(100 * float((s == 0).mean()), 2),
                         "lt_M_pct": round(100 * float((kept < M).mean()), 2),
                         "mean_kept": round(float(kept.mean()), 2),
                         "mean_kept_raw_tokens": round(float(np.mean([tok[nb[int(v)]].sum() for v in targets])), 1)})
        # ---- per-target aggregates over all metapaths of this K
        total_kept = np.zeros(len(targets)); uniq = np.zeros(len(targets)); raw_tok = np.zeros(len(targets))
        for i, v in enumerate(targets):
            allb = np.concatenate([trims[n][0][int(v)] for n in trims]) if trims else np.zeros(0, int)
            total_kept[i], uniq[i] = len(allb), len(np.unique(allb))
            raw_tok[i] = tok[allb].sum() if len(allb) else 0
        # ---- label-signal diagnostic (train+val targets only)
        pos = {int(v): i for i, v in enumerate(targets)}
        fr_pos, fr_neg = [], []
        for v in probe:
            allb = np.concatenate([trims[n][0][int(v)] for n in trims])
            if len(allb):
                (fr_pos if y[v] == 1 else fr_neg).append(y[allb].mean())
        rows.append({"K": K, "metapath": "ALL", "hops": "-", "mean_kept": round(float(total_kept.mean()), 2),
                     "unique_nbrs_mean": round(float(uniq.mean()), 2),
                     "dup_ratio": round(float(1 - uniq.sum() / max(total_kept.sum(), 1)), 3),
                     "mean_kept_raw_tokens": round(float(raw_tok.mean()), 1),
                     "n_metapaths": len(trims),
                     "nbr_fraud_rate_if_target_fraud": round(float(np.mean(fr_pos)), 4),
                     "nbr_fraud_rate_if_target_benign": round(float(np.mean(fr_neg)), 4),
                     "nbr_fraud_gap": round(float(np.mean(fr_pos) - np.mean(fr_neg)), 4)})
        print(f"K={K}: {len(trims)} metapaths, mean kept {total_kept.mean():.1f}, unique {uniq.mean():.1f}")

    # ---- top-M stability across K for metapaths present in several K (coupled: diffusion differs)
    stab = []
    for name in per_k_trims[2]:
        for k1, k2 in ((1, 2), (2, 3), (1, 3)):
            if name in per_k_trims[k1] and name in per_k_trims[k2]:
                a, b = per_k_trims[k1][name][0], per_k_trims[k2][name][0]
                j = np.mean([jaccard(a[int(v)], b[int(v)]) for v in targets])
                stab.append({"metapath": name, "K_a": k1, "K_b": k2, "mean_jaccard": round(float(j), 3)})
    # ---- decoupled: path length <=2, diffusion steps 1,2,3 (analysis only)
    emb = np.load(text_embedding_cache_path(graph, cfg))
    X = build_mdk_features(emb, graph.numeric, "deberta_concat", "none")
    by_steps = {s: {} for s in (1, 2, 3)}
    for steps in by_steps:
        for path in enumerate_metapaths(list(graph.relations), 2):
            r = trim_metapath(graph.relations, path, X, targets, steps, M)
            by_steps[steps][metapath_name(path)] = r.neighbors
        print(f"decoupled diffusion={steps} done")
    for steps in (1, 3):
        for name, nbrs in by_steps[steps].items():
            j = np.mean([jaccard(nbrs[int(v)], by_steps[2][name][int(v)]) for v in targets])
            stab.append({"metapath": name, "K_a": f"diff{steps}", "K_b": "diff2",
                         "mean_jaccard": round(float(j), 3), "decoupled": True})

    out = ROOT / "analysis" / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "neighbor_analysis.json").write_text(json.dumps({"stats": rows, "stability": stab}, indent=2))
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "neighbor_analysis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
