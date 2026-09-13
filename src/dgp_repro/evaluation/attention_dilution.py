"""THEORETICAL_ANALYSIS: fraud-related token fraction as the neighbourhood grows (paper Eq. 15).

    n_K = (D^{K+1} - D) / (D - 1)
    r   = (L + p*m*n_K) / (L + m*n_K) = p + L(1-p) / (L + m*n_K)
    upper bound  r <= p + L(1-p) / (m*n_K)

Under the paper's assumption of roughly uniform token similarity, r is also the expected
attention mass on fraud-related tokens. This module only evaluates that formula. It is NOT
evidence about how Qwen3-8B actually distributes attention, and must not be presented as such.
"""

from __future__ import annotations

from dgp_repro.evaluation.complexity import geometric_sum

LABEL = "THEORETICAL_ANALYSIS"


def neighborhood_size(D: float, K: int) -> float:
    return geometric_sum(D, K, 1)


def fraud_token_fraction(L: float, m: float, D: float, K: int, p: float) -> float:
    nK = neighborhood_size(D, K)
    return (L + p * m * nK) / (L + m * nK)


def fraud_token_fraction_bound(L: float, m: float, D: float, K: int, p: float) -> float:
    return p + L * (1 - p) / (m * neighborhood_size(D, K))


def dilution_curves(L: float, p: float, D_values: list[float], m_values: list[float], K_values: list[int]) -> list[dict]:
    return [{"label": LABEL, "L": L, "p": p, "D": D, "m": m, "K": K, "n_K": neighborhood_size(D, K),
             "fraud_token_fraction": fraud_token_fraction(L, m, D, K, p)}
            for D in D_values for m in m_values for K in K_values]
