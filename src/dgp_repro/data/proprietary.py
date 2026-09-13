"""E-Commerce and LifeService: proprietary ByteDance datasets. NOT_REPRODUCIBLE.

The paper states these are "real-world graphs sampled from our industry partner, ByteDance".
They are not public. Only their Table 1 statistics are known, recorded below for reference.
No data is generated, simulated or approximated in their place.

If access is ever granted, implement a builder that returns a HeteroGraph (see amazon.py)
and register it in dgp_repro/data/__init__.py.
"""

from __future__ import annotations

PAPER_TABLE_1 = {
    "ecommerce": {"node_type": "Shop Profile", "nodes": 182_043, "edges": 27_196_608, "edge_types": 9,
                  "frauds": 3_256, "split": (1_309, 1_309, 3_928)},
    "lifeservice": {"node_type": "Shop Profile", "nodes": 12_868, "edges": 82_912, "edge_types": 5,
                    "frauds": 2_868, "split": (1_287, 1_287, 2_574)},
}

MESSAGE = "Not reproduced because proprietary dataset access was unavailable."


class NotReproducibleError(RuntimeError):
    pass


def build_proprietary(name: str, *args, **kwargs):
    raise NotReproducibleError(f"{name}: {MESSAGE} (ByteDance-internal data; see research/reproduction_matrix.md)")
