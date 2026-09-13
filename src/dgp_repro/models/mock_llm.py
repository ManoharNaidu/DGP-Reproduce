"""DEBUG_ONLY stand-in for an LLM, so the full train/evaluate path runs on CPU in seconds.

It hashes prompt words into a bag-of-words vector and maps it linearly to logits over a tiny
vocabulary. It exposes the same two methods as the real Qwen classifier, so the trainer,
the Eq. 13 loss and the Eq. 14 readout are exercised by exactly the code used for Qwen3-8B.
Its numbers mean nothing and must never be reported.
"""

from __future__ import annotations

import zlib

import torch
import torch.nn as nn


class MockTokenizer:
    """Word-level toy tokenizer where 'Yes' and 'No' are single tokens."""

    vocab = ["<pad>", "<unk>", "Yes", "No"]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = []
        for word in text.split(" "):
            if word in self.vocab:
                ids.append(self.vocab.index(word))
            elif word:
                ids.append(1)
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.vocab[i] for i in ids)


class MockLLM(nn.Module):
    name = "mock-bow-llm"

    def __init__(self, vocab_size: int = 32, hash_dim: int = 512, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.hash_dim = hash_dim
        self.head = nn.Linear(hash_dim, vocab_size)
        self.tokenizer = MockTokenizer()

    def _featurize(self, prompts: list[str]) -> torch.Tensor:
        x = torch.zeros(len(prompts), self.hash_dim)
        for i, prompt in enumerate(prompts):
            for word in prompt.lower().split():
                x[i, zlib.crc32(word.encode()) % self.hash_dim] += 1.0
        return x / x.sum(dim=1, keepdim=True).clamp(min=1.0)

    def last_token_logits(self, prompts: list[str]) -> torch.Tensor:
        return self.head(self._featurize(prompts))

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def state_for_checkpoint(self) -> dict:
        return {k: v.detach().clone() for k, v in self.state_dict().items()}

    def load_checkpoint_state(self, state: dict) -> None:
        self.load_state_dict(state)
