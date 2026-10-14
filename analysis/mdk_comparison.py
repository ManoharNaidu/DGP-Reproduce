"""Task 1: verify and compare the Markov diffusion distance in DGP and in FLAG-MD.

Run:  /venv/main/bin/python analysis/mdk_comparison.py

Checks (all CPU, seconds):
  1. DGP iterative H = Z(K)X equals a dense reference (weighted + binary adjacency).
  2. FLAG-MD H equals the same dense reference.
  3. Cross-feed: on a single-relation graph with identical X, DGP and FLAG give the same H,
     the same distances and the same top-M sets.
  4. Where they differ by design: multi-relation metapath (path counts vs binary), and the
     candidate set (metapath neighbourhood vs 1-hop neighbours).
Writes analysis/out/mdk_comparison.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
FLAG_SRC = ROOT.parent / "FLAG-Reproduce" / "src"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(FLAG_SRC))

from dgp_repro.mdk import (compute_diffusion_embeddings, compute_diffusion_distance,  # noqa: E402
                           metapath_neighbors, select_top_m_neighbors)
from flagbench.sampling.markov_diffusion import (adjacency_to_csr, diffusion_distances,  # noqa: E402
                                                 diffusion_embeddings, transition_matrix)


def dense_reference(A: np.ndarray, X: np.ndarray, K: int, first: int = 0) -> np.ndarray:
    """H = (1/K) sum_{k=first..K} T^k X with T = D^-1 A (zero-degree rows stay zero)."""
    deg = A.sum(1)
    T = np.divide(A, deg[:, None], out=np.zeros_like(A, dtype=float), where=deg[:, None] > 0)
    P, total = np.eye(len(A)), np.zeros_like(A, dtype=float)
    for k in range(K + 1):
        if k >= first:
            total += P
        P = P @ T
    return (total @ X) / K


def random_symmetric(n, density, rng):
    U = np.triu((rng.random((n, n)) < density).astype(float), 1)
    return U + U.T


def main() -> None:
    rng = np.random.default_rng(0)
    n, d = 40, 6
    X = rng.normal(size=(n, d))
    A1 = random_symmetric(n, 0.12, rng)
    A2 = random_symmetric(n, 0.12, rng)
    out: dict = {"checks": {}}

    # 1 + 2 + 3: single relation, binary -> DGP and FLAG must coincide with the dense reference.
    rel = {"R": sp.csr_matrix(A1)}
    adj_list = [np.flatnonzero(A1[i]) for i in range(n)]
    T_flag = transition_matrix(adjacency_to_csr(adj_list, n))
    for K in (1, 2, 3):
        ref = dense_reference(A1, X, K)
        h_dgp = compute_diffusion_embeddings(rel, ("R",), X, K)
        h_flag = diffusion_embeddings(T_flag, X, K)
        out["checks"][f"single_relation_K{K}"] = {
            "dgp_vs_dense_maxabs": float(np.abs(h_dgp - ref).max()),
            "flag_vs_dense_maxabs": float(np.abs(h_flag - ref).max()),
            "dgp_vs_flag_maxabs": float(np.abs(h_dgp - h_flag).max()),
        }
        # same distances and same top-M sets for every node, M=4, candidates = 1-hop neighbours
        same_sets, same_dist = True, True
        for v in range(n):
            cands = np.flatnonzero(A1[v])
            if cands.size == 0:
                continue
            dd = compute_diffusion_distance(h_dgp, v, cands)
            df = diffusion_distances(h_flag, v, cands)
            same_dist &= bool(np.allclose(dd, df))
            a = select_top_m_neighbors(cands, dd, 4)
            b = select_top_m_neighbors(cands, df, 4)
            same_sets &= bool(np.array_equal(np.sort(a), np.sort(b)))
        out["checks"][f"single_relation_K{K}"].update(distances_equal=same_dist, top4_sets_equal=same_sets)

    # 4a: composite metapath R1-R2. DGP weights by path counts, a binarised A_P differs.
    rel2 = {"R1": sp.csr_matrix(A1), "R2": sp.csr_matrix(A2)}
    for K in (2,):
        hw = compute_diffusion_embeddings(rel2, ("R1", "R2"), X, K, adjacency="weighted")
        hb = compute_diffusion_embeddings(rel2, ("R1", "R2"), X, K, adjacency="binary")
        ref_w = dense_reference(A1 @ A2, X, K)
        ref_b = dense_reference((A1 @ A2 > 0).astype(float), X, K)
        out["checks"]["composite_metapath_K2"] = {
            "weighted_vs_dense": float(np.abs(hw - ref_w).max()),
            "binary_vs_dense": float(np.abs(hb - ref_b).max()),
            "weighted_vs_binary_maxabs": float(np.abs(hw - hb).max()),
        }
        # how often does weighted vs binary change the top-4 set?
        nb = metapath_neighbors(rel2, ("R1", "R2"), np.arange(n))
        changed = tot = 0
        for v, c in enumerate(nb):
            if c.size <= 4:
                continue
            tot += 1
            a = np.sort(select_top_m_neighbors(c, compute_diffusion_distance(hw, v, c), 4))
            b = np.sort(select_top_m_neighbors(c, compute_diffusion_distance(hb, v, c), 4))
            changed += int(not np.array_equal(a, b))
        out["checks"]["composite_metapath_K2"]["targets_with_gt4_candidates"] = tot
        out["checks"]["composite_metapath_K2"]["top4_set_changed_by_weighting"] = changed

    # 4b: operator variants keep / change the ranking (identity term).
    T = transition_matrix(adjacency_to_csr(adj_list, n))
    for op in ("paper_eq6", "mean_k0", "walk_only"):
        H = diffusion_embeddings(T, X, 2, op)
        ref = diffusion_embeddings(T, X, 2, "paper_eq6")
        diff = 0
        for v in range(n):
            c = np.flatnonzero(A1[v])
            if c.size > 4:
                a = np.sort(select_top_m_neighbors(c, diffusion_distances(H, v, c), 4))
                b = np.sort(select_top_m_neighbors(c, diffusion_distances(ref, v, c), 4))
                diff += int(not np.array_equal(a, b))
        out["checks"][f"operator_{op}_vs_paper_eq6_top4_sets_changed"] = diff

    ok = all(v.get("dgp_vs_flag_maxabs", 0) < 1e-10 and v.get("top4_sets_equal", True)
             for k, v in out["checks"].items() if k.startswith("single_relation"))
    out["single_relation_equivalence_holds"] = bool(ok)
    dest = ROOT / "analysis" / "out"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "mdk_comparison.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
