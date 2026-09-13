"""Prompt building, numeric summaries, readout maths, label tokens, metrics, complexity, baselines I/O."""

import numpy as np
import pytest
import scipy.io
import torch

from dgp_repro.baselines.export import export_care_gnn_mat
from dgp_repro.baselines.mlp import standardize
from dgp_repro.data import make_split
from dgp_repro.data.synthetic import make_synthetic_graph
from dgp_repro.evaluation.attention_dilution import fraud_token_fraction, fraud_token_fraction_bound
from dgp_repro.evaluation.complexity import (dgp_final_tokens, dgp_summarization_tokens, full_neighbor_tokens,
                                             vectorized_tokens)
from dgp_repro.evaluation.token_usage import summarize_token_usage
from dgp_repro.metrics import aggregate_seeds, compute_metrics
from dgp_repro.models import LabelTokenizationError, first_token_loss, fraud_probability, gather_last_logits
from dgp_repro.models import resolve_label_token_ids
from dgp_repro.models.mock_llm import MockTokenizer
from dgp_repro.prompts import MetapathBlock, build_prompt
from dgp_repro.summarization import MockSummarizer, fill_summary_template, load_template
from dgp_repro.summarization.numeric import numeric_summary


# ---------------------------------------------------------------- summaries and prompts

def test_task_agnostic_template_is_verbatim_and_has_no_task_words():
    filled = fill_summary_template(load_template("node_summary"), "some review", 10)
    assert filled.startswith("Summarize the text within 10 tokens.")
    for word in ("fraud", "spam", "helpful", "unhelpful", "fake"):
        assert word not in filled.lower()


def test_task_aware_template_matches_paper_sentence():
    filled = fill_summary_template(load_template("node_summary_task_aware"), "x", 10)
    assert filled.startswith("Summarize the text within 10 tokens, focusing on signals indicative of fraudulent behavior.")


def test_mock_summarizer_respects_budget_in_words():
    out = MockSummarizer().summarize([fill_summary_template(load_template("node_summary"), "a b c d e f", 3)], 3)
    assert out == ["a b c"]


def test_numeric_summary_is_plain_mean_over_trimmed_neighbors_matching_figure_3():
    # Figure 3: neighbours v1..v3 with ratings 2, 2, 1 and categories <0,1>, <1,0>, <1,0>
    numeric = np.array([[5.0, 1, 0], [2.0, 0, 1], [2.0, 1, 0], [1.0, 1, 0]])
    mean = numeric_summary(numeric, np.array([1, 2, 3]))
    np.testing.assert_allclose(mean, [5 / 3, 2 / 3, 1 / 3])
    assert numeric_summary(numeric, np.array([], dtype=int)) is None


def test_prompt_layout_and_component_switches():
    blocks = [MetapathBlock("RUR", 2, "short summary", np.array([1.666])), MetapathBlock("RTR-RSR", 0)]
    prompt = build_prompt("the target review", blocks, ["rating"], "Is this fraud?")
    assert prompt.splitlines() == ["Target: the target review", "Metapaths:",
                                   "- RUR: short summary (neighbor mean: rating=1.67)",
                                   "- RTR-RSR: no neighbors", "Question: Is this fraud?"]
    text_off = build_prompt("t", [MetapathBlock("RUR", 2, None, np.array([3.0]))], ["rating"], "Q?")
    assert "- RUR: neighbor mean: rating=3.00" in text_off
    target_only = build_prompt("t", [MetapathBlock("RUR", 2, None, None)], ["rating"], "Q?")
    assert target_only == "Target: t\nQuestion: Q?"


# ---------------------------------------------------------------- readout

def test_fraud_probability_uses_only_the_two_label_logits():
    logits = torch.randn(3, 50)
    p = fraud_probability(logits, yes_id=7, no_id=9)
    expected = torch.softmax(torch.stack([logits[:, 7], logits[:, 9]], -1), -1)[:, 0]
    assert torch.allclose(p, expected)
    logits[:, 20] += 100  # a huge logit elsewhere must not change p (Eq. 14)
    assert torch.allclose(fraud_probability(logits, 7, 9), expected)


def test_first_token_loss_is_full_vocab_cross_entropy():
    logits = torch.randn(2, 50)
    labels = torch.tensor([1, 0])
    loss = first_token_loss(logits, labels, yes_id=7, no_id=9)
    manual = -(torch.log_softmax(logits, -1)[0, 7] + torch.log_softmax(logits, -1)[1, 9]) / 2
    assert torch.allclose(loss, manual)


def test_gather_last_logits_for_both_padding_sides():
    logits = torch.arange(2 * 4 * 3, dtype=torch.float).view(2, 4, 3)
    right = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    left = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    assert torch.equal(gather_last_logits(logits, right), torch.stack([logits[0, 1], logits[1, 3]]))
    assert torch.equal(gather_last_logits(logits, left), torch.stack([logits[0, 3], logits[1, 3]]))


class MultiTokenTokenizer(MockTokenizer):
    def encode(self, text, add_special_tokens=False):
        return [1, 2] if text.strip() == "Yes" else super().encode(text)  # "Yes" split into two in-vocab ids


def test_label_resolution_refuses_multi_token_labels():
    assert resolve_label_token_ids(MockTokenizer()).yes_id == 2
    with pytest.raises(LabelTokenizationError, match="2 tokens"):
        resolve_label_token_ids(MultiTokenTokenizer())


# ---------------------------------------------------------------- metrics and accounting

def test_metrics_use_sklearn_definitions_and_reject_unlabeled():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert m["auroc"] == pytest.approx(75.0)
    from sklearn.metrics import average_precision_score
    assert m["auprc"] == pytest.approx(100 * average_precision_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]))
    with pytest.raises(ValueError):
        compute_metrics([0, -1, 1], [0.1, 0.2, 0.9])
    agg = aggregate_seeds([{"macro_f1": 60, "auroc": 70, "auprc": 30}, {"macro_f1": 62, "auroc": 72, "auprc": 34}])
    assert agg["auroc_mean"] == 71 and agg["auroc_std"] == 1 and agg["n_seeds"] == 2


def test_complexity_formulas_match_paper_closed_forms():
    L, D, R, K, M, B = 170, 133, 3, 2, 4, 10
    assert full_neighbor_tokens(L, D, K) == pytest.approx((D ** 3 - 1) / (D - 1) * L)
    assert vectorized_tokens(L, D, K) == pytest.approx(L + (D ** 3 - D) / (D - 1))
    assert dgp_summarization_tokens(L, R, K, M, B) == pytest.approx(L + (R ** 3 - R) / (R - 1) * M * B)
    assert dgp_final_tokens(L, R, K, B) == pytest.approx(L + 12 * B)


def test_attention_dilution_fraction_and_bound():
    r = fraud_token_fraction(L=170, m=50, D=133, K=2, p=0.12)
    assert 0.12 < r < 1 and r <= fraud_token_fraction_bound(170, 50, 133, 2, 0.12)
    assert fraud_token_fraction(170, 50, 133, 3, 0.12) < r  # more hops dilute further


def test_token_usage_summary():
    rows = [{"raw_target_tokens": 10, "raw_neighbor_tokens": 990, "trimmed_neighbor_raw_tokens": 40,
             "node_summary_tokens": 20, "metapath_summary_tokens": 10, "final_prompt_tokens": 50,
             "num_neighbors_full": 100, "num_neighbors_trimmed": 4, "num_metapaths": 3, "K": 1, "M": 4,
             "B_node": 10, "B_meta": 10, "target_truncated": False}]
    s = summarize_token_usage(rows)
    assert s["compression_ratio"] == 20 and s["token_reduction"] == pytest.approx(0.95)


# ---------------------------------------------------------------- baselines

def test_mlp_standardisation_uses_training_rows_only():
    tr, te = np.array([[0.0], [2.0]]), np.array([[100.0]])
    tr_s, te_s = standardize(tr, te)
    np.testing.assert_allclose(tr_s.ravel(), [-1, 1])
    assert te_s[0, 0] == pytest.approx(99.0)


def test_care_gnn_export_roundtrip(tmp_path):
    g = make_synthetic_graph(num_nodes=30)
    split = make_split(g.labels, {"train": 5, "val": 5, "test": 5})
    path = export_care_gnn_mat(g, np.random.default_rng(0).normal(size=(30, 4)), split, tmp_path)
    mat = scipy.io.loadmat(path)
    assert {"net_rur", "net_rsr", "net_rtr", "homo", "features", "label"} <= set(mat)
    assert mat["net_rur"].shape == (30, 30) and mat["label"].shape == (1, 30)
