"""MLP baseline (paper Table 2, "MLP (Rosenblatt 1958)"). No official repository exists.

Provenance: PAPER_RECONSTRUCTION.
Unspecified by the paper, reconstructed:
    input features   X = DeBERTaV3(text) concat numeric, the same matrix DGP's MDK uses, standardised
    architecture     one or two hidden layers with ReLU and dropout
    selection        grid below, chosen by mean validation AUROC (the paper's criterion for all models)
Trained on the same split and evaluated with the same scikit-learn metrics as DGP.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch
import torch.nn as nn

from dgp_repro.metrics import compute_metrics
from dgp_repro.utils.seed import set_seed

GRID = {"hidden": [128, 256], "layers": [1, 2], "dropout": [0.0, 0.3], "lr": [1e-3, 1e-4]}


def standardize(train: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """Fit mean/std on training rows only (no information from validation or test)."""
    mean, std = train.mean(axis=0), train.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    return [(a - mean) / std for a in (train, *others)]


def build_mlp(in_dim: int, hidden: int, layers: int, dropout: float) -> nn.Module:
    blocks, dim = [], in_dim
    for _ in range(layers):
        blocks += [nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
        dim = hidden
    return nn.Sequential(*blocks, nn.Linear(dim, 1))


def fit_predict(X_tr, y_tr, X_va, y_va, X_te, params: dict, seed: int, max_epochs: int = 200, patience: int = 20):
    set_seed(seed)
    model = build_mlp(X_tr.shape[1], params["hidden"], params["layers"], params["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"])
    loss_fn = nn.BCEWithLogitsLoss()
    tr, va, te = (torch.as_tensor(a, dtype=torch.float32) for a in (X_tr, X_va, X_te))
    y = torch.as_tensor(y_tr, dtype=torch.float32)
    best_auc, best_state, waited = -1.0, None, 0
    for _ in range(max_epochs):
        model.train()
        for idx in torch.randperm(len(tr)).split(256):
            opt.zero_grad()
            loss_fn(model(tr[idx]).squeeze(-1), y[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            auc = compute_metrics(y_va, torch.sigmoid(model(va).squeeze(-1)).numpy())["auroc"]
        if auc > best_auc:
            best_auc, best_state, waited = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best_state)
    with torch.no_grad():
        return (torch.sigmoid(model(va).squeeze(-1)).numpy(), torch.sigmoid(model(te).squeeze(-1)).numpy())


def run_mlp(features: np.ndarray, labels: np.ndarray, split, seeds: list[int], grid: dict = GRID) -> dict:
    X_tr, X_va, X_te = standardize(features[split.train], features[split.val], features[split.test])
    y_tr, y_va, y_te = labels[split.train], labels[split.val], labels[split.test]
    best = None
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values))
        val_auc = np.mean([compute_metrics(y_va, fit_predict(X_tr, y_tr, X_va, y_va, X_te, params, s)[0])["auroc"]
                           for s in seeds])
        if best is None or val_auc > best[0]:
            best = (val_auc, params)
    per_seed = []
    for s in seeds:
        p_va, p_te = fit_predict(X_tr, y_tr, X_va, y_va, X_te, best[1], s)
        per_seed.append({"seed": s, "val": compute_metrics(y_va, p_va), "test": compute_metrics(y_te, p_te),
                         "val_prob": p_va, "test_prob": p_te})
    return {"selected_params": best[1], "selection_val_auroc": best[0], "per_seed": per_seed}
