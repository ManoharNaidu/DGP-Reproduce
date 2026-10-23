"""FLAG's +FLAG* phase on DGP's classifier (FLAG-architecture study; extension, NOT from the DGP paper).

Port of `FLAG-Reproduce/src/flagbench/training/trainer.py:finetune_extra` with `FinetuneConfig`'s defaults:

    continue from the model already fit by train(); 3 outer x 10 inner epochs; Adam(lr 1e-4, weight decay 0);
    per example  loss = CE(disc-only input) + alpha * KL(uniform || p(resid-only input)) + beta * orth(h_D, h_R)
    losses summed over 10 examples per optimizer step (accumulation_steps = 10), no gradient clipping;
    only targets whose D and R subgraphs both parsed take part (FLAG drops failed subgraphs from this phase);
    after every epoch, select on validation macro-F1 of the normal (fused) input at the validation-chosen
    threshold, and keep the phase's result only if it beats the starting point.

FLAG runs one shared GNN on x_disc and on x_common; here the one LoRA model reads the D-only and the R-only prompt
(pipeline._flag_arch_branch_prompts). h_D, h_R are the final hidden states at the answer position.
orthogonality 'squared_dot' is FLAG's paper Eq. 9 default (flag_losses.py); 'squared_cosine' is DGP's earlier
scale-free form, available because 4096-d LLM states can make the raw squared dot very large.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from dgp_repro.metrics.classification import macro_f1_at, select_f1_threshold
from dgp_repro.models.readout import label_targets, residual_uniform_loss
from dgp_repro.training.trainer import _set_mode, predict

DEFAULTS = {"outer_epochs": 3, "inner_epochs": 10, "lr": 1e-4, "weight_decay": 0.0, "alpha": 0.1, "beta": 0.1,
            "accumulation_steps": 10, "orthogonality": "squared_dot", "micro_batch": 2, "eval_batch_size": 8}


def orthogonal_loss(h_d: torch.Tensor, h_r: torch.Tensor, mode: str) -> torch.Tensor:
    """Per-example loss summed over the micro-batch (FLAG sums per-subgraph losses)."""
    a, b = h_d.float(), h_r.float()
    if mode == "squared_dot":
        return ((a * b).sum(-1) ** 2).sum()
    if mode == "squared_cosine":
        return ((F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(-1) ** 2).sum()
    raise ValueError(f"unknown orthogonality mode {mode!r}")


def _val_metric(model, val_prompts, val_labels, label_tokens, batch) -> float:
    prob = predict(model, val_prompts, label_tokens, batch)
    return macro_f1_at(val_labels, prob, select_f1_threshold(val_labels, prob))


def flag_finetune(model, disc_prompts: list[str], resid_prompts: list[str], labels: np.ndarray, val_prompts: list[str],
                  val_labels: np.ndarray, label_tokens, cfg: dict, seed: int, log_path=None) -> dict:
    c = {**DEFAULTS, **(cfg or {})}
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    best_metric = _val_metric(model, val_prompts, val_labels, label_tokens, c["eval_batch_size"])
    best_state, best_epoch = model.state_for_checkpoint(), 0
    stats = {"usable_train_targets": len(disc_prompts), "start_val_macro_f1": best_metric, "history": []}
    if not disc_prompts:
        stats.update(best_epoch=0, applied=False, seconds=0.0)
        return stats
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    started, epoch = time.time(), 0
    for _outer in range(c["outer_epochs"]):
        for _inner in range(c["inner_epochs"]):
            epoch += 1
            t0 = time.time()
            _set_mode(model, True)
            order = rng.permutation(len(disc_prompts))
            optimizer.zero_grad()
            seen, total, parts = 0, 0.0, {"ce": 0.0, "kl": 0.0, "orth": 0.0}
            for start in range(0, len(order), c["micro_batch"]):
                idx = order[start:start + c["micro_batch"]]
                d_logits, h_d = model.last_token_logits_hidden([disc_prompts[i] for i in idx])
                r_logits, h_r = model.last_token_logits_hidden([resid_prompts[i] for i in idx])
                y = torch.as_tensor(labels[idx], device=d_logits.device)
                n = len(idx)
                ce = F.cross_entropy(d_logits.float(), label_targets(y, label_tokens.yes_id, label_tokens.no_id),
                                     reduction="sum")
                kl = residual_uniform_loss(r_logits, label_tokens.yes_id, label_tokens.no_id) * n   # batchmean -> sum
                orth = orthogonal_loss(h_d, h_r, c["orthogonality"])
                loss = ce + c["alpha"] * kl + c["beta"] * orth
                loss.backward()                       # gradients accumulate across micro-batches
                total += float(loss.detach())
                for k, v in (("ce", ce), ("kl", kl), ("orth", orth)):
                    parts[k] += float(v.detach())
                before, seen = seen, seen + n
                if seen // c["accumulation_steps"] > before // c["accumulation_steps"] or start + n >= len(order):
                    optimizer.step()
                    optimizer.zero_grad()
            metric = _val_metric(model, val_prompts, val_labels, label_tokens, c["eval_batch_size"])
            record = {"phase": "flag_finetune", "epoch": epoch, "train_loss": total / len(order),
                      **{f"train_{k}": v / len(order) for k, v in parts.items()},
                      "val_macro_f1": metric, "seconds": round(time.time() - t0, 1)}
            stats["history"].append(record)
            if log_path:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
            if metric > best_metric:
                best_metric, best_epoch, best_state = metric, epoch, model.state_for_checkpoint()
    model.load_checkpoint_state(best_state)
    stats.update(best_epoch=best_epoch, best_val_macro_f1=best_metric, applied=best_epoch > 0,
                 seconds=round(time.time() - started, 1), config=c)
    return stats
