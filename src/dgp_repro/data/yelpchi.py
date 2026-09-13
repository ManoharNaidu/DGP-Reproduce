"""YelpReviews (original Rayana & Akoglu YelpChi release, 67,395 reviews with raw text).

Provenance: PAPER_RECONSTRUCTION. Evidence: research/evidence_matrix.md section D.

STATUS: BLOCKED ON DATA ACCESS. The release is available only by email request to
srayana@cs.stonybrook.edu. The standard GNN `YelpChi.mat` (45,954 nodes, 32 handcrafted
features, no text) is NOT this dataset and must not be substituted.

The raw release's file layout has not been seen, so no parser for it is written here
(writing one would mean guessing the preprocessing). Instead this module reads a documented
canonical file, `reviews.jsonl`, one JSON object per review:

    {"review_id": str, "user_id": str, "product_id": str, "date": "YYYY-MM-DD",
     "rating": float, "text": str, "label": 1 | 0}

label 1 = filtered/spam review, 0 = recommended. The converter from the raw release to
this format will be written against the actual files once they are obtained, and
`build_yelpchi` validates the result against Table 1 (67,395 nodes, 8,919 frauds).

Relations (arXiv v1 section 5.1, following CARE-GNN):
    RUR : reviews written by the same user
    RSR : reviews on the same product with the same star rating
    RTR : reviews posted in the same month for the same product
Target edge count: 17,486,608. On AmazonVideo the product constraint in the prose did not
match the data, so both scopings are exposed and must be checked against that number.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dgp_repro.data.graph import HeteroGraph
from dgp_repro.data.relations import count_pairs, relation_from_keys

REQUIRED_FIELDS = ("review_id", "user_id", "product_id", "date", "rating", "text", "label")


def read_canonical_reviews(path: str | Path) -> list[dict]:
    path = Path(path)
    mat = path.parent / "YelpChi.mat"
    if not path.exists() and mat.exists():
        raise FileNotFoundError(
            f"{mat} is the CARE-GNN preprocessed YelpChi (45,954 reviews, 32 handcrafted features, NO review text). "
            "DGP's YelpReviews needs the original release with 67,395 reviews and their raw text (paper Table 1), "
            "which DGP summarizes with an LLM; this file cannot be converted into it (no text, no review ids). "
            "Request the original release from srayana@cs.stonybrook.edu. See docs/datasets.md.")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. YelpReviews requires the original YelpChi release, obtainable only by "
            "emailing srayana@cs.stonybrook.edu. See docs/datasets.md.")
    reviews = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [k for k in REQUIRED_FIELDS if k not in row]
            if missing:
                raise ValueError(f"{path}:{line_no} is missing fields {missing}")
            reviews.append(row)
    return reviews


def relation_keys(reviews: list[dict], cfg: dict) -> dict[str, list]:
    rsr_scope = cfg.get("rsr_product_scope", True)
    rtr_scope = cfg.get("rtr_product_scope", True)
    keys = {
        "RUR": [r["user_id"] for r in reviews],
        "RSR": [((r["product_id"],) if rsr_scope else ()) + (float(r["rating"]),) for r in reviews],
        "RTR": [((r["product_id"],) if rtr_scope else ()) + (r["date"][:7],) for r in reviews],
    }
    return {name: keys[name] for name in cfg.get("relations", ["RUR", "RSR", "RTR"])}


def probe_relation_definitions(reviews: list[dict]) -> dict[str, int]:
    """Directed edge counts for each candidate scoping, to compare against 17,486,608."""
    out = {"RUR": 2 * count_pairs([r["user_id"] for r in reviews])}
    for scoped in (True, False):
        tag = "product" if scoped else "global"
        out[f"RSR_{tag}"] = 2 * count_pairs(
            [((r["product_id"],) if scoped else ()) + (float(r["rating"]),) for r in reviews])
        out[f"RTR_{tag}"] = 2 * count_pairs(
            [((r["product_id"],) if scoped else ()) + (r["date"][:7],) for r in reviews])
    return out


def build_yelpchi(canonical_path: str | Path, cfg: dict) -> HeteroGraph:
    reviews = read_canonical_reviews(canonical_path)
    relations = {name: relation_from_keys(keys) for name, keys in relation_keys(reviews, cfg).items()}
    return HeteroGraph(
        name="yelpchi", relations=relations,
        texts=[r["text"] for r in reviews],
        numeric=np.array([[float(r["rating"])] for r in reviews], dtype=np.float32),
        numeric_names=["rating"],
        labels=np.array([int(r["label"]) for r in reviews], dtype=np.int8),
        node_ids=[r["review_id"] for r in reviews],
        metadata={"source_file": Path(canonical_path).name, "provenance": "PAPER_RECONSTRUCTION",
                  "relation_config": cfg},
    )
