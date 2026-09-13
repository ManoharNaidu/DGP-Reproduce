"""Markov diffusion operator and diffused embeddings (paper Eq. 6-7).

Provenance: PAPER_RECONSTRUCTION. Derivation and evidence: research/mdk_research.md.

    T_P    = D_P^{-1} A_P,   D_P = diag(A_P 1)
    Z_P(K) = (1/K) * sum_{k=0}^{K} T_P^k        <- Eq. 6, reproduced literally
    H      = Z_P(K) X                            <- Eq. 7

Operator variants (config key `mdk.operator`):
    paper_eq6  : (1/K)     * sum_{k=0..K} T^k   literal Eq. 6 (default)
    mean_k0    : (1/(K+1)) * sum_{k=0..K} T^k   same terms, true average
    walk_only  : (1/K)     * sum_{k=1..K} T^k   classical form without identity (ABLATION ONLY)

paper_eq6 and mean_k0 differ by a positive constant, so they select identical Top-M
neighbours; tests/unit/test_mdk.py checks this. walk_only changes the ranking.

Adjacency variants (config key `mdk.adjacency`):
    weighted : A_P holds path counts, exactly as Eq. 3 defines it (default).
               T_P x is applied as a chain of sparse products, never forming A_P.
    binary   : A_P is re-binarised. This needs A_P materialised, so it is only
               practical for small graphs.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from dgp_repro.mdk.adjacency import Metapath, Relations, build_metapath_adjacency, check_relations

OPERATORS = ("paper_eq6", "mean_k0", "walk_only")
ADJACENCY_MODES = ("weighted", "binary")


def _operator_terms(operator: str, K: int) -> tuple[int, float]:
    """Return (first power in the sum, scale factor) for an operator variant."""
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if operator == "paper_eq6":
        return 0, 1.0 / K
    if operator == "mean_k0":
        return 0, 1.0 / (K + 1)
    if operator == "walk_only":
        return 1, 1.0 / K
    raise ValueError(f"unknown operator {operator!r}; choose from {OPERATORS}")


def build_transition_matrix(adjacency: sp.spmatrix) -> sp.csr_matrix:
    """Row-stochastic T = D^{-1} A. Rows with zero degree stay all-zero (ambiguity M4)."""
    adjacency = sp.csr_matrix(adjacency, dtype=np.float64)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_degree = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    return sp.diags(inv_degree) @ adjacency


def compute_diffusion_operator(transition: sp.spmatrix | np.ndarray, K: int,
                               operator: str = "paper_eq6") -> np.ndarray:
    """Dense Z_P(K). Only for small graphs and tests: it is N x N."""
    first, scale = _operator_terms(operator, K)
    T = transition.toarray() if sp.issparse(transition) else np.asarray(transition, dtype=np.float64)
    power = np.eye(T.shape[0])
    total = np.zeros_like(T)
    for k in range(K + 1):
        if k >= first:
            total += power
        power = power @ T
    return scale * total


class MetapathTransition:
    """Applies T_P to a dense matrix without materialising the metapath adjacency."""

    def __init__(self, relations: Relations, path: Metapath, adjacency: str = "weighted"):
        if adjacency not in ADJACENCY_MODES:
            raise ValueError(f"unknown adjacency mode {adjacency!r}; choose from {ADJACENCY_MODES}")
        self.n = check_relations(relations)
        self.path = path
        self.adjacency = adjacency
        self._relations = relations
        self._transition: sp.csr_matrix | None = None
        if adjacency == "binary":
            self._transition = build_transition_matrix(build_metapath_adjacency(relations, path, binary=True))
            self._inv_degree = None
        else:
            degree = self._chain(np.ones((self.n, 1))).ravel()
            self._inv_degree = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)

    def _chain(self, x: np.ndarray) -> np.ndarray:
        """A_r1 (A_r2 ( ... (A_rL x))) — the product applied right to left."""
        for rel in reversed(self.path):
            x = self._relations[rel] @ x
        return x

    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        if self._transition is not None:
            return self._transition @ x
        return self._inv_degree[:, None] * self._chain(x)

    @property
    def zero_degree_nodes(self) -> int:
        if self._transition is not None:
            return int((np.diff(self._transition.indptr) == 0).sum())
        return int((self._inv_degree == 0).sum())


def compute_diffusion_embeddings(relations: Relations, path: Metapath, features: np.ndarray, K: int,
                                 operator: str = "paper_eq6", adjacency: str = "weighted",
                                 dtype=np.float64) -> np.ndarray:
    """H = Z_P(K) X, computed iteratively (Eq. 7).

    Memory is O(N * d) plus the sparse relations; Z_P(K) is never formed.
    """
    first, scale = _operator_terms(operator, K)
    T = MetapathTransition(relations, path, adjacency)
    current = np.asarray(features, dtype=dtype)
    if current.ndim != 2 or current.shape[0] != T.n:
        raise ValueError(f"features must have shape (N={T.n}, d), got {current.shape}")
    total = current.copy() if first == 0 else np.zeros_like(current)
    for _ in range(K):
        current = (T @ current).astype(dtype, copy=False)
        total += current
    return scale * total
