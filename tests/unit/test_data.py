"""Data layer: relation construction, labelling rules, splits, persistence, config."""

import numpy as np
import pytest

from dgp_repro.config import deep_merge, parse_overrides
from dgp_repro.data.amazon import label_review, probe_relation_definitions, relation_keys, week_index
from dgp_repro.data.graph import UNLABELED, HeteroGraph
from dgp_repro.data.proprietary import NotReproducibleError, build_proprietary
from dgp_repro.data.relations import count_pairs, relation_from_keys
from dgp_repro.data.splits import Split, make_split
from dgp_repro.data.synthetic import make_synthetic_graph


def test_relation_from_keys_connects_shared_keys_without_self_loops():
    A = relation_from_keys(["u1", "u2", "u1", "u1", None]).toarray()
    expected = np.zeros((5, 5))
    for i, j in [(0, 2), (0, 3), (2, 3)]:
        expected[i, j] = expected[j, i] = 1
    np.testing.assert_array_equal(A, expected)


def test_count_pairs_matches_matrix_nnz():
    keys = ["a", "b", "a", "c", "a", "b"]
    assert 2 * count_pairs(keys) == relation_from_keys(keys).nnz == 8


@pytest.mark.parametrize("helpful,label", [([0, 0], 0), ([1, 3], 1), ([2, 4], 0), ([5, 5], 0), ([0, 1], 1)])
def test_amazon_label_rule(helpful, label):
    assert label_review(helpful) == label


def test_amazon_zero_vote_reading_is_switchable():
    assert label_review([0, 0], zero_vote_label="unlabeled") == UNLABELED
    with pytest.raises(ValueError):
        label_review([0, 0], zero_vote_label="spam")


def test_week_index_is_saturday_aligned():
    # 1970-01-03 was a Saturday: it starts a new week under the +5 day offset
    friday, saturday = 1 * 86400, 2 * 86400
    assert week_index(saturday) == week_index(friday) + 1
    assert week_index(saturday) == week_index(saturday + 6 * 86400)


def test_amazon_rsr_scope_switch():
    reviews = [
        {"reviewerID": "u1", "asin": "p1", "overall": 5.0, "unixReviewTime": 3 * 86400},
        {"reviewerID": "u2", "asin": "p2", "overall": 5.0, "unixReviewTime": 3 * 86400},
    ]
    global_keys = relation_keys(reviews, {"rsr_product_scope": False})["RSR"]
    scoped_keys = relation_keys(reviews, {"rsr_product_scope": True})["RSR"]
    assert global_keys[0] == global_keys[1] and scoped_keys[0] != scoped_keys[1]
    counts = probe_relation_definitions(reviews)
    assert counts["RSR_rating_week"] == 2 and counts["RSR_product_rating_week"] == 0


def test_split_sizes_labeled_only_disjoint_and_stratified():
    labels = np.array([UNLABELED] * 50 + [1] * 30 + [0] * 120, dtype=np.int8)
    split = make_split(labels, {"train": 20, "val": 20, "test": 40}, seed=3)
    assert split.sizes() == {"train": 20, "val": 20, "test": 40}
    all_ids = split.all_targets()
    assert len(set(all_ids.tolist())) == len(all_ids)
    assert np.all(labels[all_ids] != UNLABELED)
    assert labels[split.test].sum() == 8  # 30/150 fraud rate of the labeled pool -> 8 of 40


def test_split_is_deterministic_and_rejects_oversized_requests():
    labels = np.array([0, 1] * 50, dtype=np.int8)
    a = make_split(labels, {"train": 10, "val": 10, "test": 20}, seed=1)
    b = make_split(labels, {"train": 10, "val": 10, "test": 20}, seed=1)
    np.testing.assert_array_equal(a.train, b.train)
    with pytest.raises(ValueError):
        make_split(labels, {"train": 60, "val": 30, "test": 30})


def test_graph_save_load_roundtrip(tmp_path):
    g = make_synthetic_graph(num_nodes=25, seed=2)
    g.save(tmp_path / "g")
    h = HeteroGraph.load(tmp_path / "g")
    assert h.texts == g.texts and h.numeric_names == g.numeric_names
    np.testing.assert_array_equal(h.labels, g.labels)
    for rel in g.relations:
        assert (h.relations[rel] != g.relations[rel]).nnz == 0
    split = make_split(g.labels, {"train": 5, "val": 5, "test": 5})
    split.save(tmp_path / "s.json")
    np.testing.assert_array_equal(Split.load(tmp_path / "s.json").test, split.test)


def test_synthetic_graph_is_heterogeneous_and_partly_unlabeled():
    g = make_synthetic_graph()
    assert set(g.relations) == {"RUR", "RSR", "RTR"}
    assert g.num_frauds > 0 and g.num_labeled == g.num_nodes - 1


def test_proprietary_datasets_refuse_to_build():
    with pytest.raises(NotReproducibleError, match="proprietary dataset access was unavailable"):
        build_proprietary("ecommerce")


def test_config_merge_and_overrides():
    base = {"dgp": {"K": 2, "M": 4}, "components": {"mdk": True}}
    merged = deep_merge(base, parse_overrides(["dgp.M=8", "components.mdk=false"]))
    assert merged == {"dgp": {"K": 2, "M": 8}, "components": {"mdk": False}}
