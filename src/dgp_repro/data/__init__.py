"""Dataset registry: build, validate against paper Table 1, save and load processed graphs."""

from __future__ import annotations

from pathlib import Path

from dgp_repro.config import REPO_ROOT
from dgp_repro.data.graph import UNLABELED, HeteroGraph
from dgp_repro.data.splits import Split, make_split

PUBLIC = ("amazonvideo", "yelpchi", "synthetic")
PROPRIETARY = ("ecommerce", "lifeservice")


def processed_dir(cfg: dict) -> Path:
    return REPO_ROOT / cfg.get("processed_dir", f"datasets/processed/{cfg['name']}")


def build_graph(cfg: dict) -> HeteroGraph:
    name = cfg["name"]
    if name == "amazonvideo":
        from dgp_repro.data.amazon import build_amazon_video, download
        raw = download(cfg["source"]["url"], REPO_ROOT / cfg["source"]["raw_path"], cfg["source"].get("sha256"))
        return build_amazon_video(raw, cfg["graph"])
    if name == "yelpchi":
        from dgp_repro.data.yelpchi import build_yelpchi
        return build_yelpchi(REPO_ROOT / cfg["source"]["canonical_path"], cfg["graph"])
    if name == "synthetic":
        from dgp_repro.data.synthetic import make_synthetic_graph
        return make_synthetic_graph(**cfg.get("graph", {}))
    if name in PROPRIETARY:
        from dgp_repro.data.proprietary import build_proprietary
        return build_proprietary(name)
    raise ValueError(f"unknown dataset {name!r}")


def validate_against_paper(graph: HeteroGraph, expected: dict | None) -> list[str]:
    """Compare graph statistics with the paper's Table 1. Returns human-readable mismatches."""
    if not expected:
        return []
    actual = {"nodes": graph.num_nodes, "edges": graph.num_edges,
              "edge_types": len(graph.relations), "frauds": graph.num_frauds}
    return [f"{key}: paper={expected[key]:,} ours={actual[key]:,}"
            for key in actual if key in expected and expected[key] != actual[key]]


def load_graph(cfg: dict) -> HeteroGraph:
    directory = processed_dir(cfg)
    if not (directory / "graph.json").exists():
        raise FileNotFoundError(f"{directory} has no processed graph; run scripts/prepare_data.py --dataset {cfg['name']}")
    return HeteroGraph.load(directory)


def load_split(cfg: dict, graph: HeteroGraph) -> Split:
    path = processed_dir(cfg) / "split.json"
    if path.exists():
        return Split.load(path)
    split = make_split(graph.labels, cfg["split"]["sizes"], seed=cfg["split"].get("seed", 0),
                       stratify=cfg["split"].get("stratify", True))
    split.save(path)
    return split


__all__ = ["HeteroGraph", "UNLABELED", "Split", "make_split", "build_graph", "load_graph",
           "load_split", "validate_against_paper", "processed_dir"]
