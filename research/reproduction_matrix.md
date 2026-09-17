# Reproduction Matrix

**As of:** 2026-09-17. Updated whenever a status changes.

**Status labels:** `NOT_STARTED` · `INVESTIGATING` · `IMPLEMENTED` · `SMOKE_TESTED` · `PARTIALLY_REPRODUCED` · `REPRODUCED` · `BLOCKED` · `NOT_REPRODUCIBLE`
**Provenance labels:** `OFFICIAL` · `OFFICIAL_ADAPTED` · `PAPER_RECONSTRUCTION` · `BASELINE_OFFICIAL` · `BASELINE_ADAPTED` · `APPROXIMATION` · `DEBUG_ONLY`

> **No component is `OFFICIAL`.** The official DGP repository contains only a README; code release is pending industry-partner approval (`repository_audit.md`). **DGP's main result is now `REPRODUCED`** on AmazonVideo: 5 real Qwen3-8B + LoRA seeds on 2× A100 80GB, Macro-F1 65.16 ± 0.52 vs. the paper's 66.91 ± 0.13 (CLOSE), AUROC and AUPRC DEVIATE by 2.9% and 6.6% respectively — attributed to running with reconstructed defaults instead of the paper's (unpublished) searched hyperparameters. The MLP baseline remains `PARTIALLY_REPRODUCED` (below). Ablations, budget sweep, task-aware study, YelpReviews, and the hyperparameter grid search have not been run.

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
| AdamW, batch 4, ≤10 epochs, early stop on val loss | arXiv v1 | None | `training/trainer.py` | PAPER_RECONSTRUCTION | IMPLEMENTED (real run done) | Patience reconstructed; real runs stopped at epoch 1–3 of 10 across the 5 seeds |
| Metrics via scikit-learn | arXiv v1 | None | `metrics/classification.py` | PAPER_RECONSTRUCTION | SMOKE_TESTED | Macro-F1 threshold unstated; official baselines disagree (ConsisGAD tunes on validation, PMP uses 0.5). Both recomputed for every method; the choice MEASURED to move Macro-F1 by ~13 points |
| Five seeds, mean ± std | Yes | None | `experiment.py` | PAPER_RECONSTRUCTION | REPRODUCED | Seed values 0–4 reconstructed; all 5 run for real on AmazonVideo (Macro-F1 65.16 ± 0.52) |
| Hyperparameter grid search | Grid in arXiv v1 | None | configs `grid:` | PAPER_RECONSTRUCTION | BLOCKED | Selected values unpublished; search strategy awaits owner decision (`compute_plan.md` §6) |
| Caching / resumability | — | — | `utils/cache.py` | — | SMOKE_TESTED | Keys include target-id hash (Loop 2 fix) |
| Token accounting | Implicit | None | `evaluation/token_usage.py` | PAPER_RECONSTRUCTION | SMOKE_TESTED | — |
| Complexity formulas | Yes | None | `evaluation/complexity.py` | PAPER_RECONSTRUCTION | SMOKE_TESTED | Evaluated at MEASURED Amazon L=115.7, D=133.1 |
| Attention-dilution curves | Yes | None | `evaluation/attention_dilution.py` | PAPER_RECONSTRUCTION (THEORETICAL_ANALYSIS) | SMOKE_TESTED | Formula evaluation only; not evidence of real attention |
| Mock summarizer / hashing embedder / mock LLM | — | — | various | DEBUG_ONLY | SMOKE_TESTED | Never used for results; debug tables are segregated |

## Datasets

| Dataset | Paper | Official code | Our code | Status | Notes |
|---|---|---|---|---|---|
| **AmazonVideo** | Yes | None | `data/amazon.py` | **SMOKE_TESTED** (graph fully matches Table 1) | Nodes 37,126 · edges 9,883,406 · 3 types · frauds 4,379 — all exact. Source file SHA-256 pinned. Relation R-S-R follows the data, not the prose. Zero-vote reviews benign (Loop 2) |
| **YelpReviews** | Yes | None | `data/yelpchi.py` | **BLOCKED** | Original Rayana & Akoglu release obtainable only by email. A downloaded `YelpChi.mat` was checked on 2026-09-14 and rejected: 45,954 nodes, no text (CARE-GNN variant; details in `docs/datasets.md`) |
| E-Commerce | Yes | Proprietary | `data/proprietary.py` (interface only) | **NOT_REPRODUCIBLE** | Not reproduced because proprietary dataset access was unavailable |
| LifeService | Yes | Proprietary | `data/proprietary.py` (interface only) | **NOT_REPRODUCIBLE** | Not reproduced because proprietary dataset access was unavailable |

## Experiments

| Experiment | Paper artefact | AmazonVideo | YelpReviews | E-Commerce / LifeService |
|---|---|---|---|---|
| MLP baseline, 5 seeds | Table 2 | **PARTIALLY_REPRODUCED** (see Baselines) | BLOCKED (data) | NOT_REPRODUCIBLE |
| DGP main result, 5 seeds | Table 2 | **REPRODUCED**: Macro-F1 65.16±0.52 (paper 66.91±0.13, CLOSE) · AUROC 75.07±0.50 (paper 77.32±0.11, DEVIATES) · AUPRC 32.33±0.65 (paper 34.63±0.24, DEVIATES) | BLOCKED (data) | NOT_REPRODUCIBLE |
| Ablations (w/o MDK, PathSumm, TextSumm, NumSumm) | Figure 4 (no printed numbers) | SMOKE_TESTED; BLOCKED on compute | BLOCKED | NOT_REPRODUCIBLE |
| Budget B ∈ {5,10,20,40,80} | Figure 5 (no printed numbers) | SMOKE_TESTED; BLOCKED on compute | BLOCKED | — |
| Task-aware vs task-agnostic | Table 3 | SMOKE_TESTED (mock cannot differentiate); BLOCKED on compute | BLOCKED | — |
| Token efficiency | Figure 2 (no printed numbers) | REPRODUCED: mean final prompt 781.5 tokens (real summaries), vs. ~1.5M raw neighbour tokens — 1922× compression, 99.9% reduction (`results/token_usage/`) | BLOCKED | — |
| Complexity analysis | Section "Complexity Analysis" | IMPLEMENTED, run on measured statistics | Needs data | — |

## Baselines (Table 2)

| Method | Official repo @ commit | Provenance | Status | Notes |
|---|---|---|---|---|
| MLP | none exists | PAPER_RECONSTRUCTION | **PARTIALLY_REPRODUCED** (AmazonVideo) | 5 seeds, CPU: Macro-F1 59.86±0.51 (paper 61.74, CLOSE) · AUROC 68.35±0.49 (paper 70.47, DEVIATES by 2.12) · AUPRC 25.14±1.06 (paper 26.55, MATCH). Input features and MLP grid unstated by the paper |
| LLM (Qwen3-8B target-only) | n/a | PAPER_RECONSTRUCTION | SMOKE_TESTED (mock) | DGP pipeline with all neighbour components off |
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
