# DGP Reproduction Report

**Version:** interim, 2026-09-13 — before any GPU experiment; includes the first real CPU baseline (MLP).
**Paper:** Li, Hu, Hooi, He, Chen. *DGP: A Dual-Granularity Prompting Framework for Fraud Detection with Graph-Enhanced LLMs.* AAAI 2026.

> This report will be reissued after the GPU runs. At this version **no DGP result from the paper has been
> reproduced or refuted**: every experiment that needs Qwen3-8B is implemented and tested on CPU but has not
> been executed. One baseline row (MLP, AmazonVideo) has been run and is partially reproduced. No numbers below are model results unless explicitly marked MEASURED, and none are paper
> numbers presented as ours.

## 1. Executive summary

- **No official code exists.** The official repository holds a single README; the release is pending
  industry-partner approval, and a July 2026 issue asking for the code has no reply. All DGP code in this project
  is a `PAPER_RECONSTRUCTION`.
- **The two paper versions differ materially** and both were needed. arXiv v1 contains the hyperparameter grid,
  batch size, epochs, selection criterion and all six relation definitions that the camera-ready cut; the
  camera-ready changes the diffusion input from raw features to DeBERTaV3 embeddings and names the summarizer.
- **AmazonVideo is reproduced at the data level, exactly.** Rebuilt from the raw 2014 McAuley file, the graph
  matches every Table 1 statistic. Doing so exposed two errors in the paper's prose (the R-S-R definition and the
  treatment of unvoted reviews), both resolved from the data and the paper's own reported metrics.
- **The full DGP pipeline is implemented** with the paper's equations as code, and verified by 77 unit tests, a
  14-check CPU smoke test (including the real Qwen3 tokenizer) and an end-to-end CPU run on the real AmazonVideo
  graph with mock language models.
- **Blocked:** GPU compute (≈100 GPU-hours, ≈$100–$320 for both public datasets without a hyperparameter
  search), the YelpReviews data (email-gated), and a decision on how much of the paper's 1,728-configuration grid
  to search.
- **Not reproducible:** E-Commerce and LifeService (proprietary ByteDance data).

## 2. What we reproduced

| Item | Outcome |
|---|---|
| AmazonVideo graph: nodes, edges, edge types, frauds, split sizes | **Exact match** with Table 1 |
| Average degree `D = 133` on Amazon (paper complexity section) | **Match**: 133.1 (MEASURED) |
| Paper's complexity formulas | Implemented and evaluated at MEASURED L and D |
| MLP baseline, AmazonVideo, 5 seeds | **PARTIALLY_REPRODUCED**: Macro-F1 59.86±0.51 vs 61.74 (CLOSE), AUROC 68.35±0.49 vs 70.47 (DEVIATES, −2.12), AUPRC 25.14±1.06 vs 26.55 (MATCH) |
| DGP main results, ablations, budget sweep, task-aware study | Not yet run (GPU) |

## 3. What matched the paper

- All four AmazonVideo Table 1 statistics and the three split sizes.
- The claim that multi-hop neighbourhoods explode: at K = 2 the 12 metapath neighbourhoods of an AmazonVideo
  review hold on average ~1.5M raw text tokens (MEASURED), consistent with the paper's "up to 2 million tokens".
- The headline "+6.8% AUPRC" corresponds to Yelp DGP 48.87 vs ConsisGAD 42.11 (+6.76 absolute points).

## 4. What deviated

| # | Deviation from the paper's text | Resolution |
|---|---|---|
| D1 | AmazonVideo R-S-R: prose says same product + rating + week | Data reproduce Table 1 only without "same product" |
| D2 | AmazonVideo unvoted reviews: treating them as unlabelled (a natural reading of "majority of nodes are unlabeled") | Labelled benign; only this fits the paper's AUROC/AUPRC pairs |
| D3 | Qwen3 thinking mode, never mentioned | Disabled; the first-token classifier is impossible otherwise |
| D4 | "w/o MDK" replacement, never described | Random M neighbours at the same budget |
| D5 | MDK input differs between versions | Camera-ready (DeBERTaV3 ⊕ numeric) |

## 5. Why deviations occurred

D1 and D2 are cases where the prose contradicts the data or the paper's own numbers; we follow the evidence and
keep the prose reading as a config switch. D3–D5 fill genuine omissions. Each is recorded with its evidence in
`evidence_matrix.md` and `reproduction_matrix.md`.

## 6. Repository provenance

Official DGP repository: README-only (Git tree API, 1 blob, 557 bytes, not truncated). Baselines: ten official
repositories verified to exist with pinned commits and licences (`methods/README.md`); three have no licence.
The same authors' follow-up paper FraudCoT (arXiv:2601.22949) was used only as corroboration: it uses the
identical AmazonVideo graph and the identical LoRA grid.

## 7. Dataset provenance

AmazonVideo: `reviews_Amazon_Instant_Video_5.json.gz`, SNAP, SHA-256 `7816bf30c235…2b48c4f54`. YelpReviews: the
original Rayana & Akoglu release (67,395 reviews, 13.23% filtered), not the widely used 45,954-node `YelpChi.mat`.
Details: `dataset_provenance.md`, `docs/datasets.md`.

## 8. Model provenance

Qwen3-8B (`Qwen/Qwen3-8B`, 8,190,735,360 parameters, Apache-2.0, not gated). Label tokens verified on the real
tokenizer: `Yes` = 9454, `No` = 2753, each a single token at the chat boundary with thinking disabled. LoRA targets
`q_proj, k_proj, v_proj, o_proj`, verified in the transformers source. DeBERTaV3 identified from the paper's
bibliography (arXiv:2111.09543); the base size and mean pooling are reconstructions.

## 9. Hardware

Development: Windows 11 laptop, 12 threads, 15 GB RAM, **no CUDA GPU**. Paper: 4× A100 80GB; the authors'
follow-up ran the same LoRA recipe on 1× A100 80GB, which is the recommendation (`compute_plan.md`).

## 10. Environment

Python 3.11.9 · torch 2.14.0+cpu · transformers 5.17.0 · numpy 2.4.6 · scipy 1.17.1 · scikit-learn 1.9.1
(development). GPU runs record their own versions in each run manifest.

## 11. Main results

**DGP: not yet available** (GPU). Paper targets: `reported_results.csv`.

| Dataset | Method | Mode | Macro-F1 | AUROC | AUPRC | Macro-F1 @0.5 |
|---|---|---|---|---|---|---|
| AmazonVideo | MLP (ours) | CPU_REPRODUCTION, 5 seeds | 59.86 ± 0.51 | 68.35 ± 0.49 | 25.14 ± 1.06 | 49.65 ± 1.67 |
| AmazonVideo | MLP (paper) | — | 61.74 ± 0.39 | 70.47 ± 0.18 | 26.55 ± 0.48 | — |

Status by the documented tolerance rule: Macro-F1 CLOSE, AUROC DEVIATES (−2.12), AUPRC MATCH. The gap may come from unstated MLP input features, architecture grid or DeBERTa pooling; none can be checked against the paper. The run independently supports two earlier decisions: the benign reading of unvoted reviews (AUPRC lands near the paper's), and a non-0.5 Macro-F1 threshold (0.5 gives 49.65, far from the paper's 61.74).

## 12. Ablation results

**Not yet available.** The paper's Figure 4 prints no numbers, so only its stated ordering can be compared.

## 13. Sensitivity results

**Not yet available** (budget sweep and task-aware study). Figure 5 prints no numbers; Table 3 does.

## 14. Token results

MEASURED on AmazonVideo without any LLM: mean review length 115.7 Qwen3 tokens (median 53); per-metapath
neighbourhood sizes and raw token volumes in `results/complexity/measured_neighborhoods_amazonvideo.csv`.
Realised prompt lengths require real summaries and are pending.

## 15. Baseline results

MLP on AmazonVideo: see §11. ConsisGAD: official code runs unmodified on our graph and split (adapter verified); full 5-seed training deferred to the GPU machine. PMP requires CUDA. Remaining official baselines not yet adapted.

## 16. Unresolved issues

1. Selected hyperparameters (only the grid is published) — needs a search decision.
2. Exact final classification and metapath-summary prompts — reconstructed.
3. Split sampling procedure — reconstructed (stratified, fixed split seed).
4. YelpReviews raw-file format — unseen; converter to be written on arrival; relation scoping to be probed
   against 17,486,608 edges.
5. Node features for the GNN/MLP baselines on these text graphs — unstated.
6. Swapping GraphGPT/HiGPT to a Qwen3-8B backbone — substantial adaptation.

## 17. Proprietary-data limitations

E-Commerce and LifeService: **not reproduced because proprietary dataset access was unavailable.** Half of
Table 2 (26 of 52 method-dataset cells) is therefore outside the reach of any external reproduction. No substitute
data were generated.

## 18. Recommended next actions

1. Request the YelpChi release (email template in `docs/datasets.md`) — lead time of days to weeks.
2. Approve GPU compute and a hyperparameter-search option (`compute_plan.md` §6).
3. On the GPU: run the calibration job, then the AmazonVideo main result with five seeds.
4. Reissue this report with measured results and `compare_with_paper.py` output.
5. Adapt ConsisGAD and PMP first (same lab, native fraud-graph loaders), then the LLM baselines.
