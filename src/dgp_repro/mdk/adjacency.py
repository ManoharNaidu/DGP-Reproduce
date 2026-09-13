"""Metapath adjacency (paper Eq. 2-4).

Provenance: PAPER_RECONSTRUCTION (no official DGP code exists).

A metapath is a sequence of relation names, e.g. ("RUR", "RSR"). Its adjacency is the
matrix product A_P = A_r1 A_r2 ... A_rL (Eq. 3) and its neighbourhood is
N_P(v) = {u : (A_P)_vu > 0} (Eq. 4).

The metapath set P_K is every relation sequence of length 1..K. This follows from the
paper's own count of summaries per prompt, (R^{K+1} - R) / (R - 1) = R + R^2 + ... + R^K
(see research/evidence_matrix.md, section B).
"""

from __future__ import annotations

import itertools
from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp

Metapath = tuple[str, ...]
Relations = Mapping[str, sp.csr_matrix]


def enumerate_metapaths(relation_names: Sequence[str], max_hops: int) -> list[Metapath]:
    """All relation sequences of length 1..max_hops, shortest first."""
    if max_hops < 1:
        raise ValueError(f"max_hops must be >= 1, got {max_hops}")
    names = list(relation_names)
    paths: list[Metapath] = []
    for length in range(1, max_hops + 1):
        paths.extend(itertools.product(names, repeat=length))
    return paths


def metapath_name(path: Metapath) -> str:
    """Human-readable name used in prompts and cache keys, e.g. 'RTR-RSR'."""
    return "-".join(path)


def check_relations(relations: Relations) -> int:
    """Validate that all relations are square CSR matrices of the same size. Returns N."""
    sizes = {name: mat.shape for name, mat in relations.items()}
    shapes = set(sizes.values())
    if len(shapes) != 1:
        raise ValueError(f"relations have different shapes: {sizes}")
    (n_rows, n_cols), = shapes
    if n_rows != n_cols:
        raise ValueError(f"relation matrices must be square, got {n_rows}x{n_cols}")
    return n_rows


def build_metapath_adjacency(
    relations: Relations,
    path: Metapath,
    rows: np.ndarray | None = None,
    binary: bool = False,
) -> sp.csr_matrix:
    """Materialise A_P = A_r1 ... A_rL as a sparse matrix (Eq. 3).

    Entries are path counts unless binary=True. Pass `rows` to build only those rows
    (A_P[rows]), which is how neighbourhoods are computed for target nodes without
    ever forming the full N x N product.
    """
    check_relations(relations)
    first = relations[path[0]]
    result = first if rows is None else first[rows]
    for rel in path[1:]:
        result = result @ relations[rel]
    result = sp.csr_matrix(result)
    if binary:
        result.data = np.ones_like(result.data)
    return result


def metapath_neighbors(
    relations: Relations,
    path: Metapath,
    targets: np.ndarray,
    exclude_self: bool = True,
    chunk_size: int = 2048,
) -> list[np.ndarray]:
    """N_P(v) for each target v (Eq. 4), as sorted arrays of node ids.

    exclude_self=True removes v from its own neighbourhood. Composite paths such as
    RUR return to v; the paper's Figure 3 averages neighbour ratings without the target
    (research/mdk_research.md, ambiguity M6).
    """
    targets = np.asarray(targets, dtype=np.int64)
    neighbors: list[np.ndarray] = []
    for start in range(0, len(targets), chunk_size):
        chunk = targets[start:start + chunk_size]
        rows = build_metapath_adjacency(relations, path, rows=chunk)
        for i, v in enumerate(chunk):
            cols = rows.indices[rows.indptr[i]:rows.indptr[i + 1]]
            cols = np.sort(cols[rows.data[rows.indptr[i]:rows.indptr[i + 1]] > 0])
            if exclude_self:
                cols = cols[cols != v]
            neighbors.append(cols.astype(np.int64))
    return neighbors
