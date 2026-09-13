"""Macro-F1, AUROC, AUPRC with scikit-learn, and mean/std over seeds.

arXiv v1 §5.1: "All evaluation metrics are computed using the scikit-learn library."
    AUROC  = sklearn.metrics.roc_auc_score
    AUPRC  = sklearn.metrics.average_precision_score (not trapezoidal PR-AUC; the two differ)

Macro-F1 threshold (PAPER_RECONSTRUCTION; the paper does not state it, and the evidence is MIXED):
    Primary  `macro_f1`      threshold chosen on the VALIDATION set, applied to test. Grid and rule mirror the
                             official ConsisGAD evaluation (modules/evaluation.py: 19 thresholds in [0.05, 0.95],
                             predict fraud if p > t, first best macro-F1).
    Secondary `macro_f1_at_0.5`  fixed threshold 0.5, as in the official PMP evaluation
                             (training_procedure/evaluate.py, `thres: 0.5`).
    Both same-lab baselines are hosted with DGP's authors yet disagree, so neither is "DGP's protocol". Every method is
    recomputed from saved probabilities under BOTH, and both columns are reported. The val-tuned variant is primary only
    because it is the fairer of the two for models with uncalibrated scores (LLM Yes/No softmax vs GNN logits).
Test labels are never used to pick the threshold.

std uses ddof=0 (numpy default). Results are reported in percent, like the paper's tables.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

METRICS = ("macro_f1", "auroc", "auprc")
THRESHOLD_GRID = np.linspace(0.05, 0.95, 19)


def _check_labels(y_true) -> np.ndarray:
    y_true = np.asarray(y_true).astype(int)
    if set(np.unique(y_true)) - {0, 1}:
        raise ValueError("y_true must contain only 0/1; unlabeled nodes must not reach evaluation")
    return y_true


def macro_f1_at(y_true, prob_fraud, threshold: float) -> float:
    return 100 * f1_score(_check_labels(y_true), (np.asarray(prob_fraud) > threshold).astype(int),
                          average="macro", zero_division=0)


def select_f1_threshold(y_val, prob_val) -> float:
    """Validation threshold maximising macro-F1 (first maximum), as in the official ConsisGAD evaluation."""
    best_f1, best_t = -1.0, 0.5
    for t in THRESHOLD_GRID:
        f1 = macro_f1_at(y_val, prob_val, t)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def compute_metrics(y_true, prob_fraud, threshold: float = 0.5) -> dict[str, float]:
    """Metrics at a GIVEN threshold (default 0.5). Use evaluate_split for the primary protocol."""
    y_true = _check_labels(y_true)
    prob_fraud = np.asarray(prob_fraud, dtype=np.float64)
    out = {"macro_f1": macro_f1_at(y_true, prob_fraud, threshold)}
    if len(np.unique(y_true)) == 2:
        out["auroc"] = 100 * roc_auc_score(y_true, prob_fraud)
        out["auprc"] = 100 * average_precision_score(y_true, prob_fraud)
    else:
        out["auroc"] = out["auprc"] = float("nan")
    return out


def evaluate_split(y_val, prob_val, y_test, prob_test) -> dict[str, dict[str, float]]:
    """Primary protocol: threshold from validation, applied to both splits; 0.5-threshold F1 kept as secondary."""
    t = select_f1_threshold(y_val, prob_val)
    val = compute_metrics(y_val, prob_val, t)
    test = compute_metrics(y_test, prob_test, t)
    for name, (y, p) in {"val": (y_val, prob_val), "test": (y_test, prob_test)}.items():
        (val if name == "val" else test)["macro_f1_at_0.5"] = macro_f1_at(y, p, 0.5)
    val["threshold"] = test["threshold"] = t
    return {"val": val, "test": test}


def aggregate_seeds(per_seed: list[dict[str, float]]) -> dict[str, float]:
    """{'macro_f1_mean', 'macro_f1_std', ...} across seed runs (also for macro_f1_at_0.5 when present)."""
    out: dict[str, float] = {"n_seeds": len(per_seed)}
    names = list(METRICS) + (["macro_f1_at_0.5"] if per_seed and "macro_f1_at_0.5" in per_seed[0] else [])
    for metric in names:
        values = np.array([run[metric] for run in per_seed], dtype=np.float64)
        out[f"{metric}_mean"] = float(values.mean())
        out[f"{metric}_std"] = float(values.std(ddof=0))
    return out
