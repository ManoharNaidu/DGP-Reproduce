# Reproduction Matrix

**As of:** 2026-11-03 (previous revision 2026-09-17). Updated whenever a status changes.

**Status labels:** `NOT_STARTED` · `INVESTIGATING` · `IMPLEMENTED` · `SMOKE_TESTED` · `PARTIALLY_REPRODUCED` · `REPRODUCED` · `BLOCKED` · `NOT_REPRODUCIBLE`
**Provenance labels:** `OFFICIAL` · `OFFICIAL_ADAPTED` · `PAPER_RECONSTRUCTION` · `BASELINE_OFFICIAL` · `BASELINE_ADAPTED` · `APPROXIMATION` · `DEBUG_ONLY`

> **No component is `OFFICIAL`.** The official DGP repository contains only a README; code release is pending industry-partner approval (`repository_audit.md`). **DGP's main result is now `REPRODUCED`** on AmazonVideo: 5 real Qwen3-8B + LoRA seeds on 2× A100 80GB, Macro-F1 65.16 ± 0.52 vs. the paper's 66.91 ± 0.13 (CLOSE), AUROC and AUPRC DEVIATE by 2.9% and 6.6% respectively — attributed to running with reconstructed defaults instead of the paper's (unpublished) searched hyperparameters. The MLP baseline remains `PARTIALLY_REPRODUCED` (below).
>
> **Since 2026-11-01:** YelpReviews was obtained from a release mirror and its graph matches Table 1 exactly; DGP `DEVIATES` there (AUROC 78.38 vs 84.28, AUPRC 31.05 vs 48.87) while the MLP matches the paper (README §11). The target-only LLM baseline ran on GPU (README §13), and the w/o MDK ablation ran (no measurable change, README §14). The other three ablations, the budget sweep and the task-aware study have not been run. A reduced hyperparameter search (the epoch-selection rule, E19) is running since 2026-11-03 (README §23).

`SMOKE_TESTED` means the code ran end to end and passed checks on CPU (synthetic graph and/or the real AmazonVideo graph with mock language models). It says nothing about matching the paper's numbers.

## DGP method

| Component | Paper | Official code | Our code | Provenance | Status | Evidence / notes |
|---|---|---|---|---|---|---|
| Heterogeneous graph model `G={V,E,R,X}` | Yes | None | `data/graph.py` | PAPER_RECONSTRUCTION | SMOKE_TESTED | Relations kept separate |
| Metapath enumeration `P_K` | Implicit | None | `mdk/adjacency.py` | PAPER_RECONSTRUCTION | SMOKE_TESTED | All sequences of length ≤ K, derived from the paper's `(R^{K+1}−R)/(R−1)` count; unit-tested |
| Metapath adjacency (Eq. 3) | Yes | None | `mdk/adjacency.py` | PAPER_RECONSTRUCTION | SMOKE_TESTED | Equals dense product in tests |
| MDK operator (Eq. 6) | Yes | None | `mdk/diffusion.py` | PAPER_RECONSTRUCTION | SMOKE_TESTED | Literal `(1/K)Σ_{k=0}^{K}`; iterative sparse form equals dense for all variants; 1/K scale proven ranking-invariant by test |
| Diffusion distance + Top-M (Eq. 8–9) | Yes | None | `mdk/distance.py`, `mdk/trimming.py` | PAPER_RECONSTRUCTION | IMPLEMENTED (real run done) | Real DeBERTaV3 ⊕ rating features, K=2, M=4: all 12 AmazonVideo metapaths trimmed for 10,023 targets and cached (K, M are GRID placeholders) |
| DeBERTaV3 features for MDK | Camera-ready only | None | `embeddings/text.py` | PAPER_RECONSTRUCTION | IMPLEMENTED (real run done) | Variant verified from bibliography; base size + mean pooling reconstructed. All 37,126 AmazonVideo reviews embedded on CPU (float32): finite, 37,111 distinct, 1- vs 5-star separable at 91.5 AUROC |
| Node-level summarization (Eq. 5) | Yes | None | `summarization/` | PAPER_RECONSTRUCTION | IMPLEMENTED (real run done) | Instruction verbatim; decoding reconstructed; 35,747 real Qwen3-8B node summaries generated and cached for AmazonVideo |
| Metapath-level summarization (Eq. 10) | Yes | None | `summarization/pipeline.py` | PAPER_RECONSTRUCTION | IMPLEMENTED (real run done) | Prompt not published → reuses node instruction; 119,829 real Qwen3-8B metapath summaries generated and cached, ~10.5/s on one A100 80GB |
| Numerical summarization (Eq. 11) | Yes | None | `summarization/numeric.py` | PAPER_RECONSTRUCTION | SMOKE_TESTED | Matches Figure 3 arithmetic in tests |
| Final prompt (Eq. 12) | Figure sketch | None | `prompts/builder.py`, `prompts/dgp/` | PAPER_RECONSTRUCTION | SMOKE_TESTED | Field labels from Figure 3; layout reconstructed, versioned |
| Qwen3-8B backbone | Yes | None | `models/qwen_classifier.py` | PAPER_RECONSTRUCTION | REPRODUCED | `Qwen/Qwen3-8B`, not gated; loaded and fine-tuned for real on 2× A100 80GB (5 seeds) |
| Thinking mode disabled | Not mentioned | None | `render_chat_prompt` | PAPER_RECONSTRUCTION | SMOKE_TESTED | Required for first-token readout; verified on real tokenizer |
| Yes/No first-token readout (Eq. 13–14) | Yes | None | `models/readout.py`, `models/label_tokens.py` | PAPER_RECONSTRUCTION | SMOKE_TESTED | Real Qwen3 tokenizer: Yes=9454, No=2753, single tokens at the chat boundary |
| LoRA on attention (q/k/v/o_proj) | Yes | None | `models/qwen_classifier.py` | PAPER_RECONSTRUCTION | IMPLEMENTED | Module names verified in two transformers versions; alpha unstated |
| AdamW, batch 4, ≤10 epochs, early stop on val loss | arXiv v1 | None | `training/trainer.py` | PAPER_RECONSTRUCTION | IMPLEMENTED (real run done) | Patience reconstructed; real runs stopped at epoch 1–3 of 10 across the 5 seeds. On YelpChi validation AUROC was still rising when training stopped (all 5 seeds); validation-AUROC epoch selection raises YelpChi test AUROC by +3.24 and AUPRC by +12.96 (E19, README §23) |
| Metrics via scikit-learn | arXiv v1 | None | `metrics/classification.py` | PAPER_RECONSTRUCTION | SMOKE_TESTED | Macro-F1 threshold unstated; official baselines disagree (ConsisGAD tunes on validation, PMP uses 0.5). Both recomputed for every method; the choice MEASURED to move Macro-F1 by ~13 points |
| Five seeds, mean ± std | Yes | None | `experiment.py` | PAPER_RECONSTRUCTION | REPRODUCED | Seed values 0–4 reconstructed; all 5 run for real on AmazonVideo (Macro-F1 65.16 ± 0.52) |
| Hyperparameter grid search | Grid in arXiv v1 | None | configs `grid:`, `scripts/run_grid.py`, `training.selection_rules` | PAPER_RECONSTRUCTION | PARTIAL | Selected values unpublished. Owner decision 2026-11-03: only the epoch-selection rule is searched (E19; stopped 2026-11-04 with YelpChi 5 seeds, AmazonVideo 1). YelpChi with the best-val-AUROC epoch: 67.30 / 82.00 / 44.58 (paper 69.07 / 84.28 / 48.87); the LR sweep (E20) was dropped; K was swept earlier (README §17). `hparam_search_plan.md`, README §23 |
| Caching / resumability | — | — | `utils/cache.py` | — | SMOKE_TESTED | Keys include target-id hash (Loop 2 fix) |
| Token accounting | Implicit | None | `evaluation/token_usage.py` | PAPER_RECONSTRUCTION | SMOKE_TESTED | — |
| Complexity formulas | Yes | None | `evaluation/complexity.py` | PAPER_RECONSTRUCTION | SMOKE_TESTED | Evaluated at MEASURED Amazon L=115.7, D=133.1 |
| Attention-dilution curves | Yes | None | `evaluation/attention_dilution.py` | PAPER_RECONSTRUCTION (THEORETICAL_ANALYSIS) | SMOKE_TESTED | Formula evaluation only; not evidence of real attention |
| Mock summarizer / hashing embedder / mock LLM | — | — | various | DEBUG_ONLY | SMOKE_TESTED | Never used for results; debug tables are segregated |

## Datasets

| Dataset | Paper | Official code | Our code | Status | Notes |
|---|---|---|---|---|---|
| **AmazonVideo** | Yes | None | `data/amazon.py` | **SMOKE_TESTED** (graph fully matches Table 1) | Nodes 37,126 · edges 9,883,406 · 3 types · frauds 4,379 — all exact. Source file SHA-256 pinned. Relation R-S-R follows the data, not the prose. Zero-vote reviews benign (Loop 2) |
| **YelpReviews** | Yes | None | `data/yelpchi.py`, `scripts/convert_yelpchi_mirror.py` | **REPRODUCED (graph)** (2026-11-01) | The release mirror downloaded for FLAG-Reproduce (67,395 reviews with text) gives Table 1 exactly with RSR = same product + same star, RTR = same product + same month: 67,395 nodes, 17,486,608 edges, 8,919 frauds; split 1,348 / 1,348 / 13,479. The text join is supported statistically, not proven. Earlier (2026-09-14) a `YelpChi.mat` (45,954 nodes, no text, CARE-GNN variant) was rejected (`docs/datasets.md`) |
| E-Commerce | Yes | Proprietary | `data/proprietary.py` (interface only) | **NOT_REPRODUCIBLE** | Not reproduced because proprietary dataset access was unavailable |
| LifeService | Yes | Proprietary | `data/proprietary.py` (interface only) | **NOT_REPRODUCIBLE** | Not reproduced because proprietary dataset access was unavailable |

## Experiments

| Experiment | Paper artefact | AmazonVideo | YelpReviews | E-Commerce / LifeService |
|---|---|---|---|---|
| MLP baseline, 5 seeds | Table 2 | **PARTIALLY_REPRODUCED** (see Baselines) | **REPRODUCED**: 63.64±0.43 / 74.66±0.71 / 33.67±1.01 (paper 62.09 / 75.00 / 32.24) | NOT_REPRODUCIBLE |
| DGP main result, 5 seeds | Table 2 | **REPRODUCED**: Macro-F1 65.16±0.52 (paper 66.91±0.13, CLOSE) · AUROC 75.07±0.50 (paper 77.32±0.11, DEVIATES) · AUPRC 32.33±0.65 (paper 34.63±0.24, DEVIATES) | **DEVIATES**: Macro-F1 61.76±1.86 · AUROC 78.38±0.88 · AUPRC 31.05±1.31 (paper 69.07 / 84.28 / 48.87); beats the MLP on AUROC (+3.72), below it on AUPRC (−2.62). With the epoch chosen by validation AUROC (E19, a recorded deviation): 67.30 / 82.00 / 44.58, CLOSE on AUROC | NOT_REPRODUCIBLE |
| Ablations (w/o MDK, PathSumm, TextSumm, NumSumm) | Figure 4 (no printed numbers) | **w/o MDK run** (2026-11-02, 5 seeds; M = 4 random neighbours): 65.28±0.80 / 74.61±0.54 / 32.51±0.61 vs R0 64.74 / 74.97 / 32.37, no significant difference; the paper reports a decline. PathSumm / TextSumm / NumSumm: SMOKE_TESTED, not run | Not run | NOT_REPRODUCIBLE |
| Budget B ∈ {5,10,20,40,80} | Figure 5 (no printed numbers) | SMOKE_TESTED; not run | Not run | — |
| Task-aware vs task-agnostic | Table 3 | SMOKE_TESTED (mock cannot differentiate); not run | Not run | — |
| Token efficiency | Figure 2 (no printed numbers) | REPRODUCED: mean final prompt 781.5 tokens (real summaries), vs. ~1.5M raw neighbour tokens — 1922× compression, 99.9% reduction (`results/token_usage/`) | Measured: mean final prompt 792 tokens (max 1,845) | — |
| Complexity analysis | Section "Complexity Analysis" | IMPLEMENTED, run on measured statistics | Not run (data now available) | — |

## Baselines (Table 2)

| Method | Official repo @ commit | Provenance | Status | Notes |
|---|---|---|---|---|
| MLP | none exists | PAPER_RECONSTRUCTION | **PARTIALLY_REPRODUCED** (AmazonVideo); **REPRODUCED** (YelpReviews) | AmazonVideo, 5 seeds, CPU: Macro-F1 59.86±0.51 (paper 61.74, CLOSE) · AUROC 68.35±0.49 (paper 70.47, DEVIATES by 2.12) · AUPRC 25.14±1.06 (paper 26.55, MATCH); re-run 2026-11-01: 59.83 / 68.33 / 25.11. YelpReviews: 63.64 / 74.66 / 33.67 (paper 62.09 / 75.00 / 32.24). Input features and MLP grid unstated by the paper |
| LLM (Qwen3-8B target-only) | n/a | PAPER_RECONSTRUCTION | **DEVIATES (above the paper)** (AmazonVideo, 2026-11-01) | Same classifier and recipe as DGP, prompt = target review only, 5 seeds: 63.35±2.84 / 73.29±0.36 / 30.54±0.60 vs paper 59.90 / 71.78 / 27.03. DGP − LLM is +1.69 AUROC / +1.83 AUPRC (paper +5.54 / +7.60). README §13. Not run on YelpReviews |
| GraphSAGE | `williamleif/GraphSAGE` @ a0fdef95 | BASELINE_ADAPTED | NOT_STARTED | Export adapter ready; repo is TF1 |
| HGT | `acbull/pyHGT` @ 85eaccd4 | BASELINE_ADAPTED | NOT_STARTED | OAG-specific loader needs data adapter |
| ConsisGAD | `Xtra-Computing/ConsisGAD` @ 36811c5b | BASELINE_ADAPTED | SMOKE_TESTED | Official code runs unmodified on our AmazonVideo graph + split (`baselines/consisgad_adapter.py`, env torch 1.13.1 / dgl 1.1.0). 1 epoch verified: prediction ids and labels match our split exactly. CPU cost MEASURED uncontended: 509 s for 2 epochs on real features (~4 min/epoch) × 100 upstream epochs ≈ 7 h/seed, ≈35 h for 5 seeds → run on the GPU machine |
| PMP | `Xtra-Computing/PMP` @ 3f7629f6 | BASELINE_ADAPTED | NOT_STARTED | **No licence**. Hard-codes `torch.cuda.current_device()` → GPU machine; same bundle interface as ConsisGAD |
| GAAP | `AtwoodDuan/GAAP` @ 6a7dbb04 | BASELINE_ADAPTED | NOT_STARTED | **No licence**; Amazon config absent upstream |
| TAPE | `XiaoxinHe/TAPE` @ d9881f7e | BASELINE_ADAPTED | NOT_STARTED | Needs LLM explanations (GPU) |
| FLAG (KDD'25) | `BUPT-GAMMA/FLAG` @ cb83944e | BASELINE_ADAPTED | NOT_STARTED | **No licence**, no README; attribution high-confidence, not declared |
| GraphGPT | `HKUDS/GraphGPT` @ db25a66f | BASELINE_ADAPTED | NOT_STARTED | Vicuna backbone hard-coded; Qwen3 swap is substantial work |
| HiGPT | `HKUDS/HiGPT` @ 2b0793e7 | BASELINE_ADAPTED | NOT_STARTED | Same |
| InstructGLM | `agiresearch/InstructGLM` @ dd2dd5ec | BASELINE_ADAPTED | NOT_STARTED | — |

## Recorded deviations from a literal reading of the paper

| # | Deviation | Why | Switch |
|---|---|---|---|
| D1 | AmazonVideo R-S-R without the "same-product" constraint in the prose | Only this reproduces Table 1's 9,883,406 edges exactly | `graph.rsr_product_scope` |
| D2 | AmazonVideo zero-vote reviews labelled benign | Only this is consistent with the AUROC/AUPRC pairs of all 13 Table 2 methods | `graph.label.zero_vote_label` |
| D3 | Qwen3 thinking mode disabled | First-token Yes/No readout is impossible otherwise | — (required) |
| D4 | "w/o MDK" = random M neighbours | Paper does not say what replaces MDK | ablation config |
| D5 | MDK input follows camera-ready (DeBERTa ⊕ numeric), not arXiv v1 (raw features) | Camera-ready is the archival version | `mdk.features` |
| D6 | Macro-F1 reported under two threshold protocols (validation-tuned primary, 0.5 secondary) | Paper silent; official baselines disagree | both columns |
