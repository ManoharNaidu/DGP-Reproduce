"""Extension (discriminative / residual text, residual aux loss): additive hooks must not change the baseline."""
import numpy as np
import torch

from dgp_repro.models.readout import orthogonality_loss, residual_uniform_loss
from dgp_repro.prompts import MetapathBlock, build_prompt
from dgp_repro.prompts.builder import load_final_template
from dgp_repro.summarization import fill_summary_template, load_template
from dgp_repro.utils.cache import TextStore


def test_baseline_prompt_unchanged_without_extras():
    blocks = [MetapathBlock("RUR", 2, "s", np.array([1.0]))]
    assert build_prompt("t", blocks, ["rating"], "Q?") == build_prompt("t", blocks, ["rating"], "Q?", extras={})


def test_extras_fill_new_fields_on_one_line():
    tpl = load_final_template("final_prompt_disc_resid")
    p = build_prompt("t", [MetapathBlock("RUR", 0)], ["rating"], "Q?", template=tpl,
                     extras={"target_disc": "a\nb", "target_residual": "  c  d "})
    assert "Key signals: a b\n" in p and "Background: c d\n" in p and "{" not in p


def test_new_templates_fill_and_never_contain_label_values():
    for name in ("node_discriminative", "node_residual"):
        filled = fill_summary_template(load_template(name), "some review", 40)
        assert "40 tokens" in filled and "some review" in filled and "{" not in filled


def test_residual_uniform_loss_zero_when_logits_equal_and_positive_otherwise():
    eq = torch.zeros(3, 10)
    assert residual_uniform_loss(eq, 2, 3).item() < 1e-6
    skew = torch.zeros(3, 10); skew[:, 2] = 4.0
    assert residual_uniform_loss(skew, 2, 3).item() > 0.1


def test_orthogonality_loss_is_scale_free_and_zero_for_orthogonal():
    a = torch.tensor([[1.0, 0.0]]); b = torch.tensor([[0.0, 5.0]])
    assert orthogonality_loss(a, b).item() < 1e-9
    assert abs(orthogonality_loss(a, a * 100).item() - 1.0) < 1e-6


def test_textstore_ignores_truncated_trailing_line(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text('{"key": "1", "text": "ok"}\n{"key": "2", "te', encoding="utf-8")
    store = TextStore(path)
    assert "1" in store and "2" not in store
