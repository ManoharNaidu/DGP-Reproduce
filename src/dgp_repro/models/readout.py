"""First-token loss and fraud probability (paper Eq. 13 and Eq. 14).

Note the two equations normalise differently, exactly as the paper writes them:

    training  (Eq. 13)  L   = -log p_theta(y_v | prompt(v))
                        p_theta is the LLM's next-token probability over the FULL vocabulary,
                        with target token "Yes" (fraud) or "No" (benign).
    inference (Eq. 14)  p_v = exp(logit_Yes) / (exp(logit_Yes) + exp(logit_No))
                        softmax over ONLY the two label logits.

Classification never decodes generated text.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def label_targets(labels: torch.Tensor, yes_id: int, no_id: int) -> torch.Tensor:
    """1 (fraud) -> yes_id, 0 (benign) -> no_id."""
    return torch.where(labels.bool(), torch.full_like(labels, yes_id), torch.full_like(labels, no_id)).long()


def first_token_loss(last_logits: torch.Tensor, labels: torch.Tensor, yes_id: int, no_id: int) -> torch.Tensor:
    """Eq. 13: cross-entropy over the full vocabulary at the first generated position."""
    return F.cross_entropy(last_logits.float(), label_targets(labels, yes_id, no_id))


def fraud_probability(last_logits: torch.Tensor, yes_id: int, no_id: int) -> torch.Tensor:
    """Eq. 14: two-way softmax over the Yes/No logits."""
    pair = torch.stack([last_logits[:, yes_id], last_logits[:, no_id]], dim=-1).float()
    return torch.softmax(pair, dim=-1)[:, 0]


def gather_last_logits(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Logits at each row's last real token, for either padding side. logits: [B, T, V]."""
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    last = (attention_mask * positions).argmax(dim=1)
    return logits[torch.arange(logits.shape[0], device=logits.device), last]
