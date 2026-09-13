"""Numerical summarization (paper Eq. 11).

    a_P(v) = (1 / |N~_P(v)|) * sum_{u in N~_P(v)} x_u^num

Plain mean over the TRIMMED neighbours; the target's own features are not included.
Works unchanged for continuous values and for one-hot / multi-hot category columns
(the mean of one-hot vectors is the category distribution, as in Figure 3: c_mean=<0.7,0.3>).
Values are kept at full float64 precision; rounding happens only when writing the prompt.
"""

from __future__ import annotations

import numpy as np


def numeric_summary(numeric: np.ndarray, neighbors: np.ndarray) -> np.ndarray | None:
    """Mean feature vector of the neighbours, or None when there are no neighbours."""
    neighbors = np.asarray(neighbors, dtype=np.int64)
    if neighbors.size == 0:
        return None
    return numeric[neighbors].astype(np.float64).mean(axis=0)
