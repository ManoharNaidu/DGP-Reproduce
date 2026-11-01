"""Convert the YelpChi release mirror used by FLAG-Reproduce into DGP's canonical reviews.jsonl.

    python scripts/convert_yelpchi_mirror.py [--src ../FLAG-Reproduce/datasets/raw/yelpchi] \
                                             [--out datasets/raw/yelpchi/reviews.jsonl]

Source (2026-11-01, user decision to use it): FLAG-Reproduce/datasets/raw/yelpchi/, an unofficial mirror of the
Rayana & Akoglu YelpChi release, downloaded by FLAG on 2026-09-17 (FLAG research/_evidence/loop1_c_yelp_text.md).
FLAG checked it against the author page (67,395 reviews / 38,063 users / 201 products / 8,919 filtered) and rebuilt
two of CARE-GNN's three YelpChi.mat relations from it exactly; its dates did NOT reproduce CARE-GNN's same-month
relation (412,824 vs 1,147,232 entries), so they may come from a different snapshot.

Files and the row join used here (the same as FLAG's flagbench/fraud_text/yelpchi.py):
    metadata.txt                                 67,395 rows "user_id product_id rating label date", label -1 = filtered
    raw_text.txt                                 5,854 hotel reviews, one per line
    output_review_yelpResData_NRYRcleaned.txt    61,541 restaurant reviews, one per line
    row i of metadata.txt  <->  line i of (hotel lines + restaurant lines)
FLAG's evidence for this join is statistical (within-product text similarity 1.72x a shifted control), not a proof.

Output: one JSON object per review in metadata order, review_id = "r<row>", label 1 = filtered (-1 in the source).
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT.parent / "FLAG-Reproduce" / "datasets" / "raw" / "yelpchi"))
    ap.add_argument("--out", default=str(ROOT / "datasets" / "raw" / "yelpchi" / "reviews.jsonl"))
    args = ap.parse_args()
    src = Path(args.src)
    meta = (src / "metadata.txt").read_text(encoding="utf-8").splitlines()
    hotel = (src / "raw_text.txt").read_text(encoding="utf-8", errors="replace").splitlines()
    rest = (src / "output_review_yelpResData_NRYRcleaned.txt").read_text(encoding="utf-8", errors="replace").splitlines()
    texts = hotel + rest
    if len(meta) != len(texts):
        raise SystemExit(f"row mismatch: {len(meta)} metadata rows vs {len(texts)} text lines")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    labels = {"-1": 1, "1": 0}
    n_fraud = 0
    with open(out, "w", encoding="utf-8") as f:
        for i, (line, text) in enumerate(zip(meta, texts)):
            user, product, rating, label, date = line.split()
            y = labels[label]
            n_fraud += y
            f.write(json.dumps({"review_id": f"r{i}", "user_id": user, "product_id": product, "date": date,
                                "rating": float(rating), "text": text.strip(), "label": y}, ensure_ascii=False) + "\n")
    print(f"wrote {len(meta)} reviews ({n_fraud} filtered = label 1) to {out}")


if __name__ == "__main__":
    main()
