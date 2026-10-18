"""Shared loading for the K = 1, 2, 3 neighbour inspection (CPU only, read-only on every existing cache).

Run the scripts with the project's python and scipy/pandas/matplotlib on PYTHONPATH, from the repo root:
    python analysis/neighbors_k123/<script>.py

Labels are only ever read for train + val targets (same rule as analysis/neighbor_analysis.py).
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "out" / "neighbors_k123"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))

from dgp_repro.config import load_experiment  # noqa: E402
from dgp_repro.data import load_graph, load_split  # noqa: E402
from dgp_repro.embeddings import build_mdk_features  # noqa: E402
from dgp_repro.mdk import enumerate_metapaths, metapath_name  # noqa: E402
from dgp_repro.pipeline import _load_trim, text_embedding_cache_path  # noqa: E402
from neighbor_analysis import find_mdk_cache, qwen_token_counts  # noqa: E402

M = 4
KS = (1, 2, 3)
EXPERIMENT = "configs/experiments/dgp_amazon.yaml"


class Ctx:
    pass


@lru_cache(maxsize=1)
def context() -> Ctx:
    c = Ctx()
    c.cfg = load_experiment(str(ROOT / EXPERIMENT))
    c.graph = load_graph(c.cfg["dataset"])
    c.split = load_split(c.cfg["dataset"], c.graph)
    c.targets = c.split.all_targets()
    c.probe = np.concatenate([c.split.train, c.split.val])      # the only nodes whose labels may be read
    c.emb = np.load(text_embedding_cache_path(c.graph, c.cfg))
    c.X = build_mdk_features(c.emb, c.graph.numeric, "deberta_concat", "none")
    c.rating = c.graph.numeric[:, 0]
    c.labels = c.graph.labels
    c.tok = qwen_token_counts(c.graph.num_nodes)
    c.cache_dirs = {K: find_mdk_cache(K, c.split.seed, len(c.targets)) for K in KS}
    c.paths = {K: enumerate_metapaths(list(c.graph.relations), K) for K in KS}
    return c


@lru_cache(maxsize=None)
def cached_trims(K: int) -> dict:
    """metapath name -> (neighbours {target: array, closest first}, sizes {target: |N_P(v)|})."""
    c = context()
    out = {}
    for path in c.paths[K]:
        name = metapath_name(path)
        nb, sizes, _raw, _stats = _load_trim(c.cache_dirs[K] / f"{name}.npz")
        out[name] = (nb, sizes)
    return out
