"""MDK tests: the sparse, iterative implementation must equal brute-force dense algebra."""

import numpy as np
import pytest
import scipy.sparse as sp

from dgp_repro.mdk import (
    build_metapath_adjacency,
    build_transition_matrix,
    compute_diffusion_distance,
    compute_diffusion_embeddings,
    compute_diffusion_operator,
    enumerate_metapaths,
    metapath_neighbors,
    random_m_neighbors,
    select_top_m_neighbors,
    trim_metapath,
)


def random_relations(n=30, density=0.12, seed=0, names=("RUR", "RSR", "RTR")):
    """Symmetric binary relations without self-loops, like review-review graphs."""
    rng = np.random.default_rng(seed)
    relations = {}
    for name in names:
        upper = np.triu((rng.random((n, n)) < density).astype(float), k=1)
        relations[name] = sp.csr_matrix(upper + upper.T)
    # one isolated node in every relation, to exercise zero-degree handling
    for name in names:
        dense = relations[name].toarray()
        dense[0, :] = 0
        dense[:, 0] = 0
        relations[name] = sp.csr_matrix(dense)
    return relations


def dense_path(relations, path, binary=False):
    out = np.eye(relations[path[0]].shape[0])
    for rel in path:
        out = out @ relations[rel].toarray()
    return (out > 0).astype(float) if binary else out


# ---------------------------------------------------------------- metapaths

@pytest.mark.parametrize("R,K", [(3, 1), (3, 2), (3, 3), (2, 3), (9, 2)])
def test_metapath_count_matches_paper_formula(R, K):
    names = [f"r{i}" for i in range(R)]
    expected = (R ** (K + 1) - R) // (R - 1)  # paper's (R^{K+1} - R) / (R - 1)
    assert len(enumerate_metapaths(names, K)) == expected


def test_metapaths_are_all_sequences_up_to_k():
    paths = enumerate_metapaths(["A", "B"], 2)
    assert paths == [("A",), ("B",), ("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")]


def test_metapath_adjacency_equals_dense_product():
    rel = random_relations()
    for path in [("RUR",), ("RSR", "RTR"), ("RUR", "RSR", "RTR")]:
        np.testing.assert_allclose(build_metapath_adjacency(rel, path).toarray(), dense_path(rel, path))
        rows = np.array([3, 7, 11])
        np.testing.assert_allclose(build_metapath_adjacency(rel, path, rows=rows).toarray(),
                                   dense_path(rel, path)[rows])


def test_neighbors_are_nonzero_entries_without_self():
    rel = random_relations()
    path = ("RUR", "RSR")
    targets = np.arange(30)
    dense = dense_path(rel, path)
    for v, nbrs in zip(targets, metapath_neighbors(rel, path, targets, chunk_size=7)):
        expected = np.flatnonzero(dense[v] > 0)
        expected = expected[expected != v]
        np.testing.assert_array_equal(nbrs, expected)


# ---------------------------------------------------------------- diffusion

def test_transition_is_row_stochastic_and_zero_rows_stay_zero():
    T = build_transition_matrix(random_relations()["RUR"]).toarray()
    sums = T.sum(axis=1)
    assert np.allclose(sums[sums > 0], 1.0)
    assert np.all(T[0] == 0)  # isolated node


def test_operator_matches_eq6_literally():
    """Z(K) = (1/K) * sum_{k=0}^{K} T^k, including the identity term and dividing by K."""
    T = build_transition_matrix(random_relations()["RSR"]).toarray()
    I = np.eye(len(T))
    np.testing.assert_allclose(compute_diffusion_operator(T, 1), (I + T) / 1)
    np.testing.assert_allclose(compute_diffusion_operator(T, 2), (I + T + T @ T) / 2)
    np.testing.assert_allclose(compute_diffusion_operator(T, 2, "mean_k0"), (I + T + T @ T) / 3)
    np.testing.assert_allclose(compute_diffusion_operator(T, 2, "walk_only"), (T + T @ T) / 2)


@pytest.mark.parametrize("operator", ["paper_eq6", "mean_k0", "walk_only"])
@pytest.mark.parametrize("K", [1, 2, 3])
@pytest.mark.parametrize("path", [("RUR",), ("RTR", "RSR"), ("RUR", "RSR", "RTR")])
def test_iterative_weighted_embeddings_equal_dense(operator, K, path):
    rel = random_relations()
    X = np.random.default_rng(1).normal(size=(30, 5))
    T = build_transition_matrix(sp.csr_matrix(dense_path(rel, path)))
    expected = compute_diffusion_operator(T, K, operator) @ X
    got = compute_diffusion_embeddings(rel, path, X, K, operator, adjacency="weighted")
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_binary_adjacency_embeddings_equal_dense():
    rel = random_relations()
    path = ("RTR", "RSR")
    X = np.random.default_rng(2).normal(size=(30, 4))
    T = build_transition_matrix(sp.csr_matrix(dense_path(rel, path, binary=True)))
    expected = compute_diffusion_operator(T, 2) @ X
    got = compute_diffusion_embeddings(rel, path, X, 2, adjacency="binary")
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_weighted_and_binary_differ_on_composite_paths():
    """Guards against the two ambiguity-M1 variants silently collapsing into one."""
    rel = random_relations()
    X = np.random.default_rng(3).normal(size=(30, 4))
    w = compute_diffusion_embeddings(rel, ("RUR", "RSR"), X, 2, adjacency="weighted")
    b = compute_diffusion_embeddings(rel, ("RUR", "RSR"), X, 2, adjacency="binary")
    assert not np.allclose(w, b)


# ---------------------------------------------------------------- distance and Top-M

def test_distance_is_l2_on_diffused_embeddings():
    H = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 0.0]])
    np.testing.assert_allclose(compute_diffusion_distance(H, 0, np.array([1, 2])), [5.0, 1.0])


def test_top_m_orders_by_distance_breaks_ties_by_id_and_keeps_short_lists():
    cands = np.array([9, 4, 7, 2])
    dists = np.array([0.5, 0.1, 0.1, 0.9])
    np.testing.assert_array_equal(select_top_m_neighbors(cands, dists, 3), [4, 7, 9])
    np.testing.assert_array_equal(select_top_m_neighbors(cands, dists, 10), [4, 7, 9, 2])


def test_eq6_scale_does_not_change_selection():
    """1/K versus 1/(K+1) is a uniform rescaling, so Top-M sets must be identical."""
    for seed in range(5):
        rel = random_relations(n=40, seed=seed)
        X = np.random.default_rng(seed).normal(size=(40, 6))
        targets = np.arange(40)
        for path in [("RUR",), ("RSR", "RTR")]:
            a = trim_metapath(rel, path, X, targets, K=2, M=3, operator="paper_eq6")
            b = trim_metapath(rel, path, X, targets, K=2, M=3, operator="mean_k0")
            for v in targets:
                np.testing.assert_array_equal(a.neighbors[v], b.neighbors[v])


def test_mdk_selection_differs_from_raw_feature_distance():
    """MDK must rank by diffused embeddings, not raw features.

    Node 1 is raw-identical to the target but sits in a different community, so after
    diffusion it is farther away than node 2, whose raw features differ.
    """
    edges = [(0, 2), (0, 3), (2, 3), (1, 4), (1, 5), (4, 5)]
    A = np.zeros((6, 6))
    for u, v in edges:
        A[u, v] = A[v, u] = 1
    A[0, 1] = A[1, 0] = 1  # make node 1 a direct neighbour of the target
    rel = {"R": sp.csr_matrix(A)}
    X = np.array([[1.0], [1.0], [0.0], [0.0], [9.0], [9.0]])
    result = trim_metapath(rel, ("R",), X, np.array([0]), K=2, M=1)
    raw_nearest = 1
    assert result.neighbors[0][0] != raw_nearest


def test_trim_reports_neighborhood_sizes_and_handles_isolated_nodes():
    rel = random_relations()
    X = np.random.default_rng(0).normal(size=(30, 3))
    result = trim_metapath(rel, ("RUR",), X, np.array([0, 5]), K=1, M=2)
    assert result.neighbors[0].size == 0 and result.neighborhood_sizes[0] == 0
    assert len(result.neighbors[5]) <= 2
    assert result.stats["empty_neighborhoods"] >= 1


def test_random_ablation_is_seeded_subset():
    cands = np.arange(100)
    a = random_m_neighbors(cands, 5, np.random.default_rng(7))
    b = random_m_neighbors(cands, 5, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)
    assert set(a) <= set(cands) and len(a) == 5
