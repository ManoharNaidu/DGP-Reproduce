"""First-token fine-tuning loop.

Known from the paper (arXiv v1 §5.1):
    optimizer AdamW; batch size 4; up to 10 epochs; early stopping on VALIDATION LOSS;
    hyperparameters chosen by validation AUROC (done one level up, in scripts/run_grid.py).
Reconstructed (PAPER_RECONSTRUCTION, configs/models/dgp.yaml): early-stopping patience,
weight decay, gradient accumulation, gradient clipping.

The model object only needs:
    last_token_logits(prompts) -> Tensor[B, V]
    trainable_parameters(), state_for_checkpoint(), load_checkpoint_state(state)
and optionally train()/eval(). Test labels are never passed to this module.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from dgp_repro.metrics import compute_metrics
from dgp_repro.models.readout import (first_token_loss, fraud_probability, orthogonality_loss,
                                      residual_uniform_loss)


@dataclass
class TrainResult:
    best_epoch: int
    best_val_loss: float
    val_metrics: dict
    history: list[dict] = field(default_factory=list)
    train_seconds: float = 0.0
    # Multi-rule mode only (training.selection_rules set): {rule: kept epoch}, and the adapter state and validation
    # probabilities of every epoch some rule kept, so the caller can score the test set once per distinct epoch.
    rule_epochs: dict = field(default_factory=dict)
    epoch_states: dict = field(default_factory=dict)
    epoch_val_probs: dict = field(default_factory=dict)


def parse_rule(rule: str) -> tuple[str, int | None]:
    """'loss_es2' -> ('val_loss', 2): early stopping on val loss with patience 2, keep the best epoch up to the stop.
    'auroc_all' -> ('val_auroc', None): best epoch over every epoch run. Metrics: loss (lower is better), auroc, auprc."""
    metric, _, stop = rule.partition("_")
    if metric not in ("loss", "auroc", "auprc") or not (stop == "all" or (stop.startswith("es") and stop[2:].isdigit())):
        raise ValueError(f"unknown selection rule {rule!r}; expected e.g. loss_es2, auroc_es3, auroc_all")
    return f"val_{metric}", (None if stop == "all" else int(stop[2:]))


class _Rule:
    def __init__(self, rule: str):
        self.name, (self.key, self.patience) = rule, parse_rule(rule)
        self.lower = self.key == "val_loss"
        self.best_epoch, self.best = -1, (math.inf if self.lower else -math.inf)
        self.wait, self.stopped = 0, False

    def update(self, epoch: int, record: dict) -> None:
        if self.stopped:
            return
        value = record[self.key]
        if (value < self.best) if self.lower else (value > self.best):
            self.best_epoch, self.best, self.wait = epoch, value, 0
        else:
            self.wait += 1
            if self.patience is not None and self.wait >= self.patience:
                self.stopped = True


def _set_mode(model, training: bool) -> None:
    """Both nn.Module and QwenFraudClassifier expose no-argument train() and eval()."""
    method = getattr(model, "train" if training else "eval", None)
    if callable(method):
        method()


@torch.no_grad()
def predict(model, prompts: list[str], label_tokens, batch_size: int = 8) -> np.ndarray:
    """Fraud probability p_v for each prompt (Eq. 14)."""
    _set_mode(model, False)
    probs = []
    for start in range(0, len(prompts), batch_size):
        logits = model.last_token_logits(prompts[start:start + batch_size])
        probs.append(fraud_probability(logits, label_tokens.yes_id, label_tokens.no_id).cpu())
    return torch.cat(probs).numpy()


@torch.no_grad()
def evaluation_loss(model, prompts: list[str], labels: np.ndarray, label_tokens, batch_size: int) -> float:
    _set_mode(model, False)
    total, count = 0.0, 0
    for start in range(0, len(prompts), batch_size):
        batch_labels = torch.as_tensor(labels[start:start + batch_size])
        logits = model.last_token_logits(prompts[start:start + batch_size])
        loss = first_token_loss(logits, batch_labels.to(logits.device), label_tokens.yes_id, label_tokens.no_id)
        total += float(loss) * len(batch_labels)
        count += len(batch_labels)
    return total / max(count, 1)


@torch.no_grad()
def evaluation_loss_and_predict(model, prompts: list[str], labels: np.ndarray, label_tokens,
                                batch_size: int) -> tuple[float, np.ndarray]:
    """evaluation_loss and predict from ONE forward pass over the prompts (2026-11-03 speed-up: the epoch loop used to
    run the validation set through the model twice). Same batches, same logits, so both outputs are identical."""
    _set_mode(model, False)
    total, count, probs = 0.0, 0, []
    for start in range(0, len(prompts), batch_size):
        batch_labels = torch.as_tensor(labels[start:start + batch_size])
        logits = model.last_token_logits(prompts[start:start + batch_size])
        loss = first_token_loss(logits, batch_labels.to(logits.device), label_tokens.yes_id, label_tokens.no_id)
        total += float(loss) * len(batch_labels)
        count += len(batch_labels)
        probs.append(fraud_probability(logits, label_tokens.yes_id, label_tokens.no_id).cpu())
    return total / max(count, 1), torch.cat(probs).numpy()


def train(model, train_prompts: list[str], train_labels: np.ndarray, val_prompts: list[str], val_labels: np.ndarray,
          label_tokens, cfg: dict, seed: int, log_path: str | Path | None = None,
          aux_prompts: list[str] | None = None) -> TrainResult:
    """aux_prompts (optional, aligned with train_prompts): residual-only prompts for the FLAG-style auxiliary loss
    L = CE + alpha*KL(uniform || p_residual) + beta*cos^2(h_main, h_residual). Validation loss stays pure CE."""
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=cfg["learning_rate"],
                                  weight_decay=cfg.get("weight_decay", 0.0))
    batch_size = cfg.get("batch_size", 4)
    accumulation = cfg.get("gradient_accumulation", 1)
    patience = cfg.get("early_stopping_patience", 2)
    # Selection-rule study (E19): train every epoch, track several epoch-selection rules on one trajectory. With
    # selection_rules unset the loop below is the original one (early stopping on val loss), bit for bit.
    rules = [_Rule(r) for r in (cfg.get("selection_rules") or [])]
    eval_batch = cfg.get("eval_batch_size", batch_size)
    train_labels = np.asarray(train_labels)
    val_labels = np.asarray(val_labels)

    best = TrainResult(best_epoch=-1, best_val_loss=math.inf, val_metrics={})
    best_state = None
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(1, cfg.get("max_epochs", 10) + 1):
        _set_mode(model, True)
        order = rng.permutation(len(train_prompts))
        optimizer.zero_grad()
        running, steps = 0.0, 0
        for step, start in enumerate(range(0, len(order), batch_size), 1):
            idx = order[start:start + batch_size]
            if aux_prompts is None:
                logits = model.last_token_logits([train_prompts[i] for i in idx])
            else:
                logits, h_main = model.last_token_logits_hidden([train_prompts[i] for i in idx])
            labels = torch.as_tensor(train_labels[idx], device=logits.device)
            loss = first_token_loss(logits, labels, label_tokens.yes_id, label_tokens.no_id)
            if aux_prompts is not None:
                r_logits, h_res = model.last_token_logits_hidden([aux_prompts[i] for i in idx])
                aux_cfg = cfg.get("residual_aux", {})
                loss = (loss + aux_cfg.get("alpha", 0.1) * residual_uniform_loss(r_logits, label_tokens.yes_id, label_tokens.no_id)
                        + aux_cfg.get("beta", 0.1) * orthogonality_loss(h_main, h_res))
            (loss / accumulation).backward()
            running += float(loss.detach())
            steps += 1
            if step % accumulation == 0 or start + batch_size >= len(order):
                if cfg.get("max_grad_norm"):
                    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad()

        val_loss, val_prob = evaluation_loss_and_predict(model, val_prompts, val_labels, label_tokens, eval_batch)
        val_metrics = compute_metrics(val_labels, val_prob)
        record = {"epoch": epoch, "train_loss": running / max(steps, 1), "val_loss": val_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        best.history.append(record)
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        if rules:
            for rule in rules:
                rule.update(epoch, record)
            kept = {rule.best_epoch for rule in rules}
            if epoch in kept:
                best.epoch_states[epoch] = model.state_for_checkpoint()
                best.epoch_val_probs[epoch] = val_prob
            for old in [e for e in best.epoch_states if e not in kept]:
                del best.epoch_states[old], best.epoch_val_probs[old]
            continue

        if val_loss < best.best_val_loss:
            best.best_epoch, best.best_val_loss, best.val_metrics = epoch, val_loss, val_metrics
            best_state = model.state_for_checkpoint()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if rules:
        best.rule_epochs = {rule.name: rule.best_epoch for rule in rules}
        primary = cfg.get("primary_rule", rules[0].name)
        best.best_epoch = best.rule_epochs[primary]
        record = best.history[best.best_epoch - 1]
        best.best_val_loss = record["val_loss"]
        best.val_metrics = {k[4:]: v for k, v in record.items() if k.startswith("val_") and k != "val_loss"}
        best_state = best.epoch_states[best.best_epoch]
    if best_state is not None:
        model.load_checkpoint_state(best_state)  # restore the early-stopping checkpoint
    best.train_seconds = time.time() - start_time
    return best
