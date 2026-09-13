"""Theoretical prompt-length formulas from the paper's "Complexity Analysis of DGP".

    L  average token length of a node's text     D  average out-degree
    R  number of relation types                   B  summarization budget
    K  number of hops                              M  metapath neighbour truncation

    full-neighbour prompt        (D^{K+1} - 1) / (D - 1) * L
    fully-vectorized prompt      L + (D^{K+1} - D) / (D - 1)
    DGP summarization prompts    L + (R^{K+1} - R) / (R - 1) * M * B
    DGP final prompt             L + (R^{K+1} - R) / (R - 1) * B

Measured anchors from the paper, used to evaluate the formulas on real settings:
    AmazonVideo  D = 133   (both versions). Our graph gives 4,941,703 undirected pairs / 37,126 nodes = 133.1,
                           so D is computed as undirected pairs per node.
    YelpReviews  L = 170   (arXiv v1 only).
"""

from __future__ import annotations

import csv
from pathlib import Path


def geometric_sum(base: float, K: int, start: int) -> float:
    """sum_{k=start}^{K} base^k, written in the paper's closed forms (base != 1)."""
    if base == 1:
        return float(K - start + 1)
    return (base ** (K + 1) - base ** start) / (base - 1)


def full_neighbor_tokens(L: float, D: float, K: int) -> float:
    return geometric_sum(D, K, 0) * L                      # (D^{K+1} - 1)/(D - 1) * L


def vectorized_tokens(L: float, D: float, K: int) -> float:
    return L + geometric_sum(D, K, 1)                      # L + (D^{K+1} - D)/(D - 1)


def dgp_summarization_tokens(L: float, R: int, K: int, M: int, B: int) -> float:
    return L + geometric_sum(R, K, 1) * M * B              # L + (R^{K+1} - R)/(R - 1) * M * B


def dgp_final_tokens(L: float, R: int, K: int, B: int) -> float:
    return L + geometric_sum(R, K, 1) * B                  # L + (R^{K+1} - R)/(R - 1) * B


def degree_from_graph(num_undirected_pairs: int, num_nodes: int) -> float:
    return num_undirected_pairs / num_nodes


def complexity_table(settings: list[dict]) -> list[dict]:
    rows = []
    for s in settings:
        L, D, R, K, M, B = s["L"], s["D"], s["R"], s["K"], s["M"], s["B"]
        rows.append({**s,
                     "full_neighbor": full_neighbor_tokens(L, D, K),
                     "vectorized": vectorized_tokens(L, D, K),
                     "dgp_summarization": dgp_summarization_tokens(L, R, K, M, B),
                     "dgp_final": dgp_final_tokens(L, R, K, B),
                     "num_metapaths": int(geometric_sum(R, K, 1))})
    return rows


def write_complexity_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
