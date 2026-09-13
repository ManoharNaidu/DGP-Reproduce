"""Node embeddings for the diffusion kernel: X = emb(X_text) concat X_num (camera-ready Eq. 7 context).

Verified: the paper's reference "He, Gao, and Chen 2021" is DeBERTaV3 (arXiv:2111.09543).
Not specified, reconstructed (research/mdk_research.md, M2/M3):
    model size   microsoft/deberta-v3-base
    pooling      mean over non-padding tokens (DeBERTaV3 has no sentence-level CLS objective)
    max_length   512 tokens
    scaling      none: the literal concatenation. 'standardize' is a sensitivity option.

HashingEmbedder is DEBUG_ONLY and exists so the smoke test needs no model download.
"""

from __future__ import annotations

import zlib

import numpy as np


class HashingEmbedder:
    """DEBUG_ONLY bag-of-words hashing embedder."""

    name = "hashing-debug"

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                out[i, zlib.crc32(word.encode()) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)


class DebertaEmbedder:
    def __init__(self, model_id: str = "microsoft/deberta-v3-base", device: str = "cpu", batch_size: int = 64,
                 max_length: int = 512, revision: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.name = model_id
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        # Always float32. transformers >= 5 otherwise loads the checkpoint's stored dtype, which is float16 for
        # deberta-v3-base: ~10x slower on CPU and prone to overflow in DeBERTa's disentangled attention on GPU,
        # which would silently corrupt the embeddings that drive MDK neighbour selection.
        self.model = AutoModel.from_pretrained(model_id, revision=revision, dtype=torch.float32).to(device).eval()
        self._torch = torch

    def embed(self, texts: list[str], log_every: int = 20) -> np.ndarray:
        """Mean-pooled embeddings in input order. Texts are batched by length to minimise padding;
        padding is masked out of the mean, so the result does not depend on the batching."""
        torch = self._torch
        order = np.argsort([len(t) for t in texts], kind="stable")
        out = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        with torch.no_grad():
            for step, start in enumerate(range(0, len(texts), self.batch_size)):
                idx = order[start:start + self.batch_size]
                enc = self.tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="pt").to(self.device)
                hidden = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                out[idx] = pooled.float().cpu().numpy()
                if log_every and step % log_every == 0:
                    print(f"  embedded {min(start + self.batch_size, len(texts))}/{len(texts)}", flush=True)
        return out


def make_embedder(cfg: dict, device: str = "cpu"):
    backend = cfg.get("backend", "deberta")
    if backend == "hashing":
        return HashingEmbedder(cfg.get("dim", 64))
    if backend == "deberta":
        return DebertaEmbedder(cfg.get("model_id", "microsoft/deberta-v3-base"), device, cfg.get("batch_size", 64),
                               cfg.get("max_length", 512), cfg.get("revision"))
    raise ValueError(f"unknown embedder backend {backend!r}")


def build_mdk_features(text_embeddings: np.ndarray | None, numeric: np.ndarray, mode: str = "deberta_concat",
                       scaling: str = "none") -> np.ndarray:
    """mode 'deberta_concat' (camera-ready) or 'raw' (arXiv v1: raw node features only)."""
    numeric = np.asarray(numeric, dtype=np.float64)
    if mode == "raw":
        blocks = [numeric]
    elif mode == "deberta_concat":
        if text_embeddings is None:
            raise ValueError("mode 'deberta_concat' needs text embeddings")
        blocks = [np.asarray(text_embeddings, dtype=np.float64), numeric]
    else:
        raise ValueError(f"unknown MDK feature mode {mode!r}")
    if scaling == "standardize":
        blocks = [(b - b.mean(axis=0)) / np.where(b.std(axis=0) > 0, b.std(axis=0), 1.0) for b in blocks]
    elif scaling != "none":
        raise ValueError(f"unknown scaling {scaling!r}")
    return np.concatenate(blocks, axis=1)
