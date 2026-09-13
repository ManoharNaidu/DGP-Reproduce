"""Joint diffusion distance (paper Eq. 8).

Provenance: PAPER_RECONSTRUCTION.

    delta_K^(P)(u, v) = || h_u^(P)(K) - h_v^(P)(K) ||_2

This is a Euclidean distance between *diffused* embeddings. It is deliberately not cosine
similarity and not a distance between raw embeddings.
"""

from __future__ import annotations

import numpy as np


def compute_diffusion_distance(embeddings: np.ndarray, target: int, candidates: np.ndarray) -> np.ndarray:
    """L2 distance from the target's diffused embedding to each candidate's."""
    candidates = np.asarray(candidates, dtype=np.int64)
    if candidates.size == 0:
        return np.zeros(0, dtype=np.float64)
    diff = embeddings[candidates] - embeddings[target]
    return np.sqrt(np.einsum("ij,ij->i", diff, diff, dtype=np.float64))
