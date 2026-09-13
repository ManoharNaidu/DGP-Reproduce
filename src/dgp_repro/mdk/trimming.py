"""Diffusion-based metapath trimming (paper Eq. 9).

Provenance: PAPER_RECONSTRUCTION.

    N~_P(v) = TopM_{u in N_P(v)} ( -delta_K^(P)(u, v) )

i.e. the M neighbours inside the metapath neighbourhood with the smallest diffusion
distance to v. Ties are broken by node id so results are deterministic.

`random_m_neighbors` is used ONLY for the "w/o MDK" ablation. The paper does not say what
replaces MDK when it is removed; keeping M neighbours but choosing them uniformly at random
holds the prompt budget fixed, so only the selection rule changes. This choice is a
PAPER_RECONSTRUCTION and is recorded in research/reproduction_matrix.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from dgp_repro.mdk.adjacency import Metapath, Relations, metapath_name, metapath_neighbors
from dgp_repro.mdk.diffusion import compute_diffusion_embeddings
from dgp_repro.mdk.distance import compute_diffusion_distance


def select_top_m_neighbors(candidates: np.ndarray, distances: np.ndarray, M: int) -> np.ndarray:
    """The M candidates with smallest distance, closest first. Keeps all if fewer than M."""
    candidates = np.asarray(candidates, dtype=np.int64)
    if M < 1:
        raise ValueError(f"M must be >= 1, got {M}")
    order = np.lexsort((candidates, distances))  # primary key: distance, tie-break: node id
    return candidates[order[:M]]


def random_m_neighbors(candidates: np.ndarray, M: int, rng: np.random.Generator) -> np.ndarray:
    """ABLATION ONLY (w/o MDK): M neighbours chosen uniformly at random."""
    candidates = np.asarray(candidates, dtype=np.int64)
    if len(candidates) <= M:
        return candidates
    return np.sort(rng.choice(candidates, size=M, replace=False))


@dataclass
class TrimResult:
    """Trimmed neighbours of every target for one metapath, plus bookkeeping for token accounting."""
    path: Metapath
    neighbors: dict[int, np.ndarray]           # target -> selected neighbour ids (closest first)
    neighborhood_sizes: dict[int, int]         # target -> |N_P(v)| before trimming
    neighborhood_raw_tokens: dict[int, int] = field(default_factory=dict)  # sum of raw text tokens over N_P(v)
    stats: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return metapath_name(self.path)


def trim_metapath(relations: Relations, path: Metapath, features: np.ndarray, targets: np.ndarray,
                  K: int, M: int, use_mdk: bool = True, operator: str = "paper_eq6",
                  adjacency: str = "weighted", seed: int = 0,
                  neighborhoods: Mapping[int, np.ndarray] | None = None,
                  node_token_counts: np.ndarray | None = None) -> TrimResult:
    """Select Top-M neighbours of each target along one metapath.

    If use_mdk is False, diffusion is skipped and neighbours are sampled at random
    (the w/o MDK ablation). If node_token_counts is given, the raw text tokens of the full
    untrimmed neighbourhood are recorded for token accounting.
    """
    targets = np.asarray(targets, dtype=np.int64)
    if neighborhoods is None:
        lists = metapath_neighbors(relations, path, targets)
        neighborhoods = dict(zip(targets.tolist(), lists))

    embeddings = None
    if use_mdk:
        embeddings = compute_diffusion_embeddings(relations, path, features, K, operator, adjacency)
    rng = np.random.default_rng(seed)

    selected: dict[int, np.ndarray] = {}
    sizes: dict[int, int] = {}
    raw_tokens: dict[int, int] = {}
    for v in targets.tolist():
        candidates = neighborhoods[v]
        sizes[v] = int(len(candidates))
        if node_token_counts is not None:
            raw_tokens[v] = int(node_token_counts[candidates].sum())
        if len(candidates) == 0:
            selected[v] = candidates
        elif use_mdk:
            distances = compute_diffusion_distance(embeddings, v, candidates)
            selected[v] = select_top_m_neighbors(candidates, distances, M)
        else:
            selected[v] = random_m_neighbors(candidates, M, rng)

    size_values = np.array(list(sizes.values()), dtype=np.float64)
    stats = {
        "metapath": metapath_name(path),
        "K": K, "M": M, "use_mdk": use_mdk, "operator": operator, "adjacency": adjacency,
        "targets": int(len(targets)),
        "empty_neighborhoods": int((size_values == 0).sum()),
        "mean_neighborhood_size": float(size_values.mean()) if size_values.size else 0.0,
        "max_neighborhood_size": int(size_values.max()) if size_values.size else 0,
    }
    return TrimResult(path=path, neighbors=selected, neighborhood_sizes=sizes,
                      neighborhood_raw_tokens=raw_tokens, stats=stats)
