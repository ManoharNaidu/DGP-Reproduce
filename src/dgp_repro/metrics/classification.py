"""Macro-F1, AUROC, AUPRC with scikit-learn, and mean/std over seeds.

arXiv v1 §5.1: "All evaluation metrics are computed using the scikit-learn library."
    AUPRC  = sklearn.metrics.average_precision_score (not trapezoidal PR-AUC; the two differ)
Not specified by the paper (PAPER_RECONSTRUCTION):
    Macro-F1 threshold = 0.5 on p_v (the natural cut for a Yes-vs-No softmax)
    std uses ddof=0 (numpy default)
Results are reported in percent, like the paper's tables.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

METRICS = ("macro_f1", "auroc", "auprc")


def compute_metrics(y_true, prob_fraud, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    prob_fraud = np.asarray(prob_fraud, dtype=np.float64)
    if set(np.unique(y_true)) - {0, 1}:
        raise ValueError("y_true must contain only 0/1; unlabeled nodes must not reach evaluation")
    out = {"macro_f1": 100 * f1_score(y_true, (prob_fraud >= threshold).astype(int), average="macro", zero_division=0)}
    if len(np.unique(y_true)) == 2:
        out["auroc"] = 100 * roc_auc_score(y_true, prob_fraud)
        out["auprc"] = 100 * average_precision_score(y_true, prob_fraud)
    else:
        out["auroc"] = out["auprc"] = float("nan")
    return out


def aggregate_seeds(per_seed: list[dict[str, float]]) -> dict[str, float]:
    """{'macro_f1_mean', 'macro_f1_std', ...} across seed runs."""
    out: dict[str, float] = {"n_seeds": len(per_seed)}
    for metric in METRICS:
        values = np.array([run[metric] for run in per_seed], dtype=np.float64)
        out[f"{metric}_mean"] = float(values.mean())
        out[f"{metric}_std"] = float(values.std(ddof=0))
    return out
