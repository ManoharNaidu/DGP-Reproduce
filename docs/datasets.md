# Datasets

| Paper name | Status | Source | Access |
|---|---|---|---|
| AmazonVideo | **Ready** — rebuilt graph matches paper Table 1 exactly | McAuley 2014, Amazon Instant Video 5-core | Automatic download, no registration |
| YelpReviews | **Blocked on access** | Rayana & Akoglu 2015, original YelpChi release | Email request |
| E-Commerce | Not reproducible | ByteDance internal | None |
| LifeService | Not reproducible | ByteDance internal | None |

Full provenance and evidence: `research/dataset_provenance.md`, `research/evidence_matrix.md` (section D).

## AmazonVideo

```bash
python scripts/prepare_data.py --dataset amazonvideo
```

Downloads `reviews_Amazon_Instant_Video_5.json.gz` (9,517,526 bytes) from SNAP, checks its SHA-256
(`7816bf30c235c5261560e6ebd9ff6875c154ed15fb9827f0eded74b2c48c4f54`), builds the graph and refuses to save
unless every Table 1 statistic matches:

| Statistic | Paper | Ours |
|---|---|---|
| Nodes | 37,126 | 37,126 |
| Edges | 9,883,406 | 9,883,406 |
| Edge types | 3 | 3 |
| Frauds (unhelpful reviews) | 4,379 | 4,379 |
| Train / Val / Test | 1,299 / 1,299 / 7,425 | 1,299 / 1,299 / 7,425 |

Construction:

- **Node** = review. **Text** = `reviewText`. **Numeric feature** = star rating.
- **Label**: unhelpful (fraud) if the review has ≥1 helpfulness vote and fewer than half are helpful;
  all other reviews benign.
- **Relations**: R-U-R (same reviewer), R-P-R (same product), R-S-R (same rating and same Saturday-aligned week).
- The helpfulness vote counts are **never** used as features: they define the label.

Two places where the data contradict the paper's prose, and why we follow the data:

1. The prose defines R-S-R as *same-product* reviews with the same rating in the same week. That gives
   86,314 edges. Dropping "same-product" gives 6,225,010, which is the only way to reach the reported
   9,883,406 total. (`graph.rsr_product_scope`)
2. Treating the 23,993 zero-vote reviews as unlabelled would make every split 33% fraud, which is
   inconsistent with the AUROC/AUPRC pairs of all 13 methods in Table 2; treating them as benign (11.8%)
   fits them within 1.6 AUPRC points on average. (`graph.label.zero_vote_label`)

## YelpReviews — how to obtain it

DGP uses the **original** YelpChi release (67,395 Chicago hotel and restaurant reviews **with raw text**,
13.23% filtered by Yelp). This is **not** the `YelpChi.mat` used by CARE-GNN, PC-GNN, ConsisGAD and DGL
(45,954 nodes, 32 handcrafted features, no text). Do not substitute it.

The dataset page (https://shebuti.com/yelpchi-dataset/) states: *"To get the datasets with ground truth
please email: srayana@cs.stonybrook.edu"*. The ODDS mirror was unreachable (TLS failure) when checked.

Suggested email:

> Subject: Request for the YelpChi dataset (with ground truth) for academic reproduction
>
> Dear Dr. Rayana,
>
> I am a student at [institution] reproducing the AAAI-26 paper "DGP: A Dual-Granularity Prompting
> Framework for Fraud Detection with Graph-Enhanced LLMs", which evaluates on the YelpChi dataset from
> your KDD 2015 work "Collective Opinion Spam Detection: Bridging Review Networks and Metadata".
> Would you be willing to share the YelpChi release with ground-truth labels and review text? It will be
> used for non-commercial academic research only and will not be redistributed.
>
> Thank you very much,
> [name, institution, supervisor]

**When the files arrive**, place them under `datasets/raw/yelpchi/` and open an issue/task to write the
converter: the raw layout has not been seen, so no parser for it was written in advance (that would be
guessing preprocessing). The converter produces `datasets/raw/yelpchi/reviews.jsonl` with one object per review:

```json
{"review_id": "...", "user_id": "...", "product_id": "...", "date": "YYYY-MM-DD", "rating": 4.0, "text": "...", "label": 0}
```

Then:

```bash
python scripts/prepare_data.py --dataset yelpchi --probe   # which R-S-R / R-T-R scoping gives 17,486,608 edges?
python scripts/prepare_data.py --dataset yelpchi           # refuses to save unless Table 1 matches
```

## Splits

Sizes come from paper Table 1. The procedure is reconstructed: stratified by label, drawn from all
labelled nodes (every node on both public datasets), fixed by `split.seed` (default 0); the five seeds
vary only training. Stored in `datasets/processed/<dataset>/split.json`.

## Proprietary datasets

`src/dgp_repro/data/proprietary.py` records their Table 1 statistics and raises
`NotReproducibleError("Not reproduced because proprietary dataset access was unavailable.")`.
Nothing is generated in their place.
