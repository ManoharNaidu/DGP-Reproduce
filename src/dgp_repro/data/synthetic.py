"""Tiny synthetic review graph for smoke tests. DEBUG_ONLY — never used for reported results.

Structure mirrors the real review graphs: reviews are nodes, relations come from shared
user / product / (rating, month) keys, and fraudulent users write short, low-rated,
templated reviews so that text, numbers and structure all carry signal.
"""

from __future__ import annotations

import numpy as np

from dgp_repro.data.graph import HeteroGraph
from dgp_repro.data.relations import relation_from_keys

BENIGN_PHRASES = [
    "the food was warm and the staff were friendly", "great location and a quiet comfortable room",
    "portions were generous and the service was quick", "we waited a while but the pasta was worth it",
    "clean tables, fair prices and a relaxed atmosphere", "the breakfast menu had plenty of good options",
]
FRAUD_PHRASES = [
    "best place ever amazing amazing must visit now", "terrible terrible avoid this place worst ever",
    "five stars best deal click my profile for coupons", "worst service ever do not go never again",
]


def make_synthetic_graph(num_nodes: int = 80, num_users: int = 16, num_products: int = 6,
                         fraud_user_ratio: float = 0.25, seed: int = 0) -> HeteroGraph:
    rng = np.random.default_rng(seed)
    fraud_users = set(rng.choice(num_users, size=max(1, int(num_users * fraud_user_ratio)), replace=False).tolist())

    users = rng.integers(0, num_users, size=num_nodes)
    products = rng.integers(0, num_products, size=num_nodes)
    months = rng.integers(1, 4, size=num_nodes)
    labels = np.array([1 if u in fraud_users else 0 for u in users], dtype=np.int8)
    ratings = np.where(labels == 1, rng.choice([1.0, 5.0], size=num_nodes), rng.integers(2, 6, size=num_nodes))

    texts = []
    for label in labels:
        phrases = FRAUD_PHRASES if label == 1 else BENIGN_PHRASES
        texts.append(". ".join(rng.choice(phrases, size=rng.integers(1, 4))) + ".")

    relations = {
        "RUR": relation_from_keys(users.tolist()),
        "RSR": relation_from_keys(list(zip(products.tolist(), ratings.tolist()))),
        "RTR": relation_from_keys(list(zip(products.tolist(), months.tolist()))),
    }
    # leave one node unlabeled, as in the real datasets
    labels[-1] = -1
    return HeteroGraph(
        name="synthetic", relations=relations, texts=texts,
        numeric=ratings.reshape(-1, 1).astype(np.float32), numeric_names=["rating"], labels=labels,
        node_ids=[f"syn{i}" for i in range(num_nodes)], metadata={"provenance": "DEBUG_ONLY", "seed": seed},
    )
