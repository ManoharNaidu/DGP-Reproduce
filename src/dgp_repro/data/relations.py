"""Build review-review relations from shared attributes ("same user", "same product", ...).

Two reviews are connected in a relation when they share the relation's group key.
Each review has exactly one key per relation, so A = B B^T (B = node-by-group incidence)
has 0/1 off-diagonal entries; the diagonal is removed (no self-loops).

Edge-count convention, verified exactly on AmazonVideo (research/evidence_matrix.md, D):
paper Table 1 "# Edges" = sum over relations of directed nnz = 2 x undirected pairs.
"""

from __future__ import annotations

from typing import Hashable, Sequence

import numpy as np
import scipy.sparse as sp


def group_incidence(keys: Sequence[Hashable]) -> sp.csr_matrix:
    """N x G incidence matrix; rows with key None belong to no group."""
    index: dict[Hashable, int] = {}
    rows, cols = [], []
    for node, key in enumerate(keys):
        if key is None:
            continue
        rows.append(node)
        cols.append(index.setdefault(key, len(index)))
    data = np.ones(len(rows), dtype=np.float64)
    return sp.csr_matrix((data, (rows, cols)), shape=(len(keys), len(index)))


def relation_from_keys(keys: Sequence[Hashable]) -> sp.csr_matrix:
    """Symmetric binary adjacency connecting every pair of nodes that share a key."""
    B = group_incidence(keys)
    A = (B @ B.T).tocsr()
    A = (A - sp.diags(A.diagonal())).tocsr()
    A.eliminate_zeros()
    A.data = np.ones_like(A.data)
    return A


def count_pairs(keys: Sequence[Hashable]) -> int:
    """Undirected pairs a key would create, without building a matrix (for probing definitions)."""
    counts: dict[Hashable, int] = {}
    for key in keys:
        if key is not None:
            counts[key] = counts.get(key, 0) + 1
    return sum(c * (c - 1) // 2 for c in counts.values())
