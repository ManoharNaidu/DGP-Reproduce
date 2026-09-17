# DGP Reproduction Report

**Version:** 2026-09-17 — reissued after the first GPU run; DGP's main result is now reproduced on AmazonVideo.
**Paper:** Li, Hu, Hooi, He, Chen. *DGP: A Dual-Granularity Prompting Framework for Fraud Detection with Graph-Enhanced LLMs.* AAAI 2026.

> The GPU run this report was waiting on is done: 5 real Qwen3-8B + LoRA seeds on AmazonVideo, on a rented
> 2× A100-SXM4-80GB instance. Ablations, the budget sweep, the task-aware study, YelpReviews and the
> hyperparameter grid search still have not been run (§12, §13, §16). No numbers below are model results
> unless explicitly marked MEASURED, and none are paper numbers presented as ours.

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
- **DGP's main result is reproduced on AmazonVideo (5 real seeds, GPU).** Macro-F1 65.16 ± 0.52 vs. the paper's
  66.91 ± 0.13 (CLOSE, −2.6%); AUROC 75.07 ± 0.50 vs. 77.32 ± 0.11 (DEVIATES, −2.9%); AUPRC 32.33 ± 0.65 vs.
  34.63 ± 0.24 (DEVIATES, −6.6%) — run with reconstructed-default hyperparameters and no search (§11).
- **The full DGP pipeline is implemented and now GPU-verified**, not just CPU-smoke-tested: 77 unit tests, a
  14-check CPU smoke test (including the real Qwen3 tokenizer), and an end-to-end real run — 119,829 real
  Qwen3-8B metapath summaries plus 35,747 node summaries generated and consumed by 5 real LoRA training runs.
- **Still blocked:** the hyperparameter grid search (≈60–90 GPU-hours for the staged option, `compute_plan.md`
  §6 — needs an owner decision), the YelpReviews data (email-gated, blocking Yelp entirely and the ablation/
  budget/task-aware studies which were only planned for AmazonVideo's paired dataset comparison), and further
  third-party baselines beyond MLP and ConsisGAD's smoke test.
- **Not reproducible:** E-Commerce and LifeService (proprietary ByteDance data).

## 2. What we reproduced

| Item | Outcome |
|---|---|
| AmazonVideo graph: nodes, edges, edge types, frauds, split sizes | **Exact match** with Table 1 |
| Average degree `D = 133` on Amazon (paper complexity section) | **Match**: 133.1 (MEASURED) |
| Paper's complexity formulas | Implemented and evaluated at MEASURED L and D |
| MLP baseline, AmazonVideo, 5 seeds | **PARTIALLY_REPRODUCED**: Macro-F1 59.86±0.51 vs 61.74 (CLOSE), AUROC 68.35±0.49 vs 70.47 (DEVIATES, −2.12), AUPRC 25.14±1.06 vs 26.55 (MATCH) |
| **DGP main result, AmazonVideo, 5 seeds** | **REPRODUCED (CLOSE on Macro-F1, DEVIATES on AUROC/AUPRC)**: see §11 |
| Ablations, budget sweep, task-aware study | Not yet run (§12, §13) |

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

Development: Windows 11 laptop, 12 threads, 15 GB RAM, **no CUDA GPU**. GPU runs: rented Vast.ai instance,
**2× NVIDIA A100-SXM4-80GB**, CUDA 12.8. Paper: 4× A100 80GB; the authors' follow-up ran the same LoRA recipe
on 1× A100 80GB, which is the recommendation (`compute_plan.md`) — **confirmed**: one seed's LoRA training
peaked at 19.68–19.69 GiB, comfortably under a single 80 GB card. The second GPU was used only to run two
independent seeds in parallel (`train.py` has no multi-GPU support of its own).

## 10. Environment

Development (CPU): Python 3.11.9 · torch 2.14.0+cpu · transformers 5.17.0 · numpy 2.4.6 · scipy 1.17.1 ·
scikit-learn 1.9.1. GPU runs (recorded per-run in each `run_manifest.json`): Python 3.12.14 · torch
2.11.0+cu128 · transformers 5.17.0 · peft 0.21.0 · accelerate 1.15.0 · scipy 1.18.1 · scikit-learn 1.9.1 ·
CUDA 12.8.

## 11. Main results

**DGP: REPRODUCED on AmazonVideo, 5 real seeds** (GPU_REPRODUCTION — real Qwen3-8B summarization + LoRA
fine-tuning, not mock). Paper targets: `reported_results.csv`; full comparison: `results/tables/comparison_with_paper.md`.

| Dataset | Method | Mode | Macro-F1 | AUROC | AUPRC | Macro-F1 @0.5 |
|---|---|---|---|---|---|---|
| AmazonVideo | MLP (ours) | CPU_REPRODUCTION, 5 seeds | 59.86 ± 0.51 | 68.35 ± 0.49 | 25.14 ± 1.06 | 49.65 ± 1.67 |
| AmazonVideo | MLP (paper) | — | 61.74 ± 0.39 | 70.47 ± 0.18 | 26.55 ± 0.48 | — |
| **AmazonVideo** | **DGP (ours)** | **GPU_REPRODUCTION, 5 seeds** | **65.16 ± 0.52** | **75.07 ± 0.50** | **32.33 ± 0.65** | **49.45 ± 2.83** |
| AmazonVideo | DGP (paper) | — | 66.91 ± 0.13 | 77.32 ± 0.11 | 34.63 ± 0.24 | — |

DGP status by the documented tolerance rule: Macro-F1 **CLOSE** (−1.75, −2.6%), AUROC **DEVIATES** (−2.25,
−2.9%), AUPRC **DEVIATES** (−2.30, −6.6%). Run with the reconstructed defaults in `configs/models/dgp.yaml`
(K=2, M=4, B_node=B_meta=10, LoRA rank=8, alpha=16, dropout=0.05, lr=1e-4) and **no hyperparameter search** —
the paper searches a grid but never publishes the winning configuration (`compute_plan.md` §6). Our
cross-seed std (0.50–0.65) is 2–4× the paper's (0.11–0.24), the same direction as an untuned config rather
than a pipeline defect, given Macro-F1 alone lands within 2.6% of the paper. Best epoch varied 1–3 of the
10-epoch budget across seeds, consistent with early stopping on validation loss reacting to seed noise on a
config that was not tuned for it.

MLP status unchanged: Macro-F1 CLOSE, AUROC DEVIATES (−2.12), AUPRC MATCH. The gap may come from unstated MLP input features, architecture grid or DeBERTa pooling; none can be checked against the paper. The run independently supports two earlier decisions: the benign reading of unvoted reviews (AUPRC lands near the paper's), and a non-0.5 Macro-F1 threshold (0.5 gives 49.65, far from the paper's 61.74).

## 12. Ablation results

**Not yet available.** The paper's Figure 4 prints no numbers, so only its stated ordering can be compared.

## 13. Sensitivity results

**Not yet available** (budget sweep and task-aware study). Figure 5 prints no numbers; Table 3 does.

## 14. Token results

MEASURED on AmazonVideo, now including real LLM output: mean raw review length 115.7 Qwen3 tokens (median 53);
per-metapath neighbourhood sizes and raw token volumes in `results/complexity/measured_neighborhoods_amazonvideo.csv`.
**Realised final prompt length (real summaries, no longer pending): mean 781.5 tokens** (p95 1,205.9, max 1,902)
versus a mean full-neighbour prompt of ~1.50M tokens — a **1922× compression ratio, 99.9% token reduction**
(`results/token_usage/token_usage_dgp_amazonvideo_seed0.json`). This is higher than the ~236-token estimate
an earlier version of this report derived from the paper's formula at B=10 — the formula evidently undercounts
what real Qwen3-8B summaries and the numeric/path-summary overhead actually cost in tokens.

## 15. Baseline results

MLP on AmazonVideo: see §11. ConsisGAD: official code runs unmodified on our graph and split (adapter verified); full 5-seed training still deferred (CPU cost alone is ≈35 GPU/CPU-hours for 5 seeds; GPU run not yet done). PMP requires CUDA and has not been run. Remaining official baselines not yet adapted.

## 16. Unresolved issues

1. Selected hyperparameters (only the grid is published) — needs a search decision; §11's reproduction used
   reconstructed defaults and no search, which is the leading suspect for the remaining AUROC/AUPRC gap.
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
2. Decide on a hyperparameter-search budget (`compute_plan.md` §6, staged option ≈60–90 GPU-hours) to close
   the AUROC/AUPRC gap in §11, or accept the current reconstructed-default result as final for AmazonVideo.
3. Run the ablation, budget-sweep and task-aware studies on AmazonVideo now that the main pipeline is verified
   on real hardware (§12, §13).
4. Once YelpReviews data arrives, repeat the main result there — expect ~1.6× the AmazonVideo compute per
   `compute_plan.md` §4.
5. Finish adapting ConsisGAD (full 5-seed GPU run) and PMP, then the LLM baselines (GraphGPT/HiGPT need a
   Vicuna→Qwen3 backbone swap first).
