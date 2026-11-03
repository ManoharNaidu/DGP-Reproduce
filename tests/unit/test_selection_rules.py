"""Selection-rule study (E19): multi-rule training must reproduce the legacy early-stopping run for the paper rule."""
import numpy as np
import pytest
import torch

from dgp_repro.models.label_tokens import resolve_label_token_ids
from dgp_repro.models.mock_llm import MockLLM
from dgp_repro.training import predict, train
from dgp_repro.training.trainer import _Rule, parse_rule


def _data(n=40, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    prompts = [f"review {i} {'bad scam fake' if y[i] else 'good nice fine'} {rng.integers(0, 9)}" for i in range(n)]
    return prompts, y


def _run(cfg, seed=0):
    torch.manual_seed(seed)
    model = MockLLM(seed=seed)
    labels = resolve_label_token_ids(model.tokenizer)
    tp, ty = _data(40, 0)
    vp, vy = _data(24, 1)
    res = train(model, tp, ty, vp, vy, labels, cfg, seed)
    return model, labels, res, vp


def test_parse_rule():
    assert parse_rule("loss_es2") == ("val_loss", 2)
    assert parse_rule("auroc_all") == ("val_auroc", None)
    assert parse_rule("auprc_es3") == ("val_auprc", 3)
    for bad in ("loss", "f1_es2", "auroc_esx", "auroc_last"):
        with pytest.raises(ValueError):
            parse_rule(bad)


def test_rule_freezes_after_patience():
    r = _Rule("loss_es2")
    for e, v in enumerate([1.0, 0.8, 0.9, 0.95, 0.1], 1):   # stops after epoch 4; the late 0.1 is ignored
        r.update(e, {"val_loss": v})
    assert (r.best_epoch, r.stopped) == (2, True)
    a = _Rule("auroc_all")
    for e, v in enumerate([70, 72, 71, 75, 74], 1):
        a.update(e, {"val_auroc": v})
    assert a.best_epoch == 4


def test_multi_rule_reproduces_legacy_paper_rule():
    base = {"learning_rate": 0.05, "batch_size": 4, "max_epochs": 6, "early_stopping_patience": 2, "eval_batch_size": 8}
    m_old, labels, old, vp = _run(base)
    multi = {**base, "selection_rules": ["loss_es2", "auroc_all", "auroc_es2"], "primary_rule": "loss_es2"}
    m_new, _, new, _ = _run(multi)
    assert new.rule_epochs["loss_es2"] == old.best_epoch
    assert len(new.history) == 6                              # multi-rule mode runs every epoch
    assert new.history[:len(old.history)] == old.history      # same trajectory up to the legacy stop
    np.testing.assert_allclose(predict(m_new, vp, labels), predict(m_old, vp, labels), rtol=0, atol=1e-6)
    assert set(new.epoch_states) == set(new.rule_epochs.values())
    best_auroc = max(new.history, key=lambda r: r["val_auroc"])["val_auroc"]
    assert new.history[new.rule_epochs["auroc_all"] - 1]["val_auroc"] == best_auroc
