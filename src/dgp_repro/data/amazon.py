"""AmazonVideo (paper: "AmazonVideo"; FraudCoT: "InstantVideo").

Provenance: PAPER_RECONSTRUCTION, validated by exact reproduction of every Table 1 number.
Evidence: research/evidence_matrix.md section D, research/dataset_provenance.md.

Source file: reviews_Amazon_Instant_Video_5.json.gz (McAuley 2014, 5-core), 37,126 reviews.

Node = review. Label (verified exactly -> 4,379 frauds):
    total_votes >= 1 and helpful_votes / total_votes <  0.5  -> 1 (unhelpful)
    total_votes >= 1 and helpful_votes / total_votes >= 0.5  -> 0 (helpful)
    total_votes == 0                                         -> 0 (benign)   [default: zero_vote_label=benign]

Why zero-vote reviews are benign (research/evidence_matrix.md, "Loop 2 addendum"): if they were
unlabeled, every split would be 33.3% fraud, and 12 of the 13 methods in paper Table 2 would score an
AUPRC below that of random guessing despite AUROC 70-77. Under a binormal score model the paper's
AUROC/AUPRC pairs are reproduced within 1.6 points on average at 11.8% prevalence (all zero-vote
reviews benign) versus 27.6 points at 33.3%. `zero_vote_label: unlabeled` keeps the other reading.

Relations (verified exactly -> 9,883,406 directed edges):
    RUR : same reviewer
    RPR : same product
    RSR : same star rating AND same week
          The paper's prose says "same-product reviews ... same rating ... same week", but
          adding the product constraint gives 86,314 directed edges instead of the reported
          6,225,010. The data reproduces Table 1 only WITHOUT the product constraint, so that
          is the default. `rsr_product_scope: true` restores the prose reading.
    Weeks are Saturday-aligned: (unixReviewTime + 5 * 86400) // 604800.

Numeric features: the star rating only. The helpful-vote counts are EXCLUDED because the
label is computed from them (direct label leakage).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np

from dgp_repro.data.graph import UNLABELED, HeteroGraph
from dgp_repro.data.relations import count_pairs, relation_from_keys

SECONDS_PER_DAY = 86400
SECONDS_PER_WEEK = 604800


def week_index(unix_time: int, offset_days: int = 5) -> int:
    return (int(unix_time) + offset_days * SECONDS_PER_DAY) // SECONDS_PER_WEEK


def sha256_of(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: str | Path, expected_sha256: str | None = None) -> Path:
    destination = Path(destination)
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_suffix(destination.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(destination)
    if expected_sha256:
        actual = sha256_of(destination)
        if actual != expected_sha256:
            raise ValueError(f"checksum mismatch for {destination}: expected {expected_sha256}, got {actual}")
    return destination


def read_reviews(path: str | Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def label_review(helpful: list[int], min_votes: int = 1, threshold: float = 0.5,
                 zero_vote_label: str = "benign") -> int:
    helpful_votes, total_votes = helpful
    if total_votes < min_votes:
        if zero_vote_label not in ("benign", "unlabeled"):
            raise ValueError(f"zero_vote_label must be 'benign' or 'unlabeled', got {zero_vote_label!r}")
        return 0 if zero_vote_label == "benign" else UNLABELED
    return 1 if helpful_votes / total_votes < threshold else 0


def relation_keys(reviews: list[dict], cfg: dict) -> dict[str, list]:
    offset = cfg.get("week_offset_days", 5)
    rsr_scope = cfg.get("rsr_product_scope", False)
    keys = {
        "RUR": [r["reviewerID"] for r in reviews],
        "RPR": [r["asin"] for r in reviews],
        "RSR": [((r["asin"],) if rsr_scope else ()) + (r["overall"], week_index(r["unixReviewTime"], offset))
                for r in reviews],
    }
    return {name: keys[name] for name in cfg.get("relations", ["RUR", "RPR", "RSR"])}


def probe_relation_definitions(reviews: list[dict]) -> dict[str, int]:
    """Directed edge counts for candidate definitions, to compare against Table 1."""
    return {
        "RUR": 2 * count_pairs([r["reviewerID"] for r in reviews]),
        "RPR": 2 * count_pairs([r["asin"] for r in reviews]),
        "RSR_rating_week": 2 * count_pairs([(r["overall"], week_index(r["unixReviewTime"])) for r in reviews]),
        "RSR_product_rating_week": 2 * count_pairs(
            [(r["asin"], r["overall"], week_index(r["unixReviewTime"])) for r in reviews]),
    }


def build_amazon_video(raw_path: str | Path, cfg: dict) -> HeteroGraph:
    reviews = read_reviews(raw_path)
    label_cfg = cfg.get("label", {})
    labels = np.array([label_review(r["helpful"], label_cfg.get("min_votes", 1), label_cfg.get("threshold", 0.5),
                                    label_cfg.get("zero_vote_label", "benign"))
                       for r in reviews], dtype=np.int8)
    relations = {name: relation_from_keys(keys) for name, keys in relation_keys(reviews, cfg).items()}
    texts = [r.get("reviewText", "") for r in reviews]
    numeric = np.array([[float(r["overall"])] for r in reviews], dtype=np.float32)
    return HeteroGraph(
        name="amazonvideo", relations=relations, texts=texts, numeric=numeric, numeric_names=["rating"],
        labels=labels, node_ids=[f"{r['reviewerID']}|{r['asin']}" for r in reviews],
        metadata={"source_file": Path(raw_path).name, "provenance": "PAPER_RECONSTRUCTION",
                  "relation_config": {k: v for k, v in cfg.items() if k != "label"}},
    )
