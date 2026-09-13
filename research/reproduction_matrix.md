# Reproduction Matrix

**As of:** 2026-09-13. Updated whenever a status changes.

**Status labels:** `NOT_STARTED` · `INVESTIGATING` · `IMPLEMENTED` · `SMOKE_TESTED` · `PARTIALLY_REPRODUCED` · `REPRODUCED` · `BLOCKED` · `NOT_REPRODUCIBLE`
**Provenance labels:** `OFFICIAL` · `OFFICIAL_ADAPTED` · `PAPER_RECONSTRUCTION` · `BASELINE_OFFICIAL` · `BASELINE_ADAPTED` · `APPROXIMATION` · `DEBUG_ONLY`

> **No component is `OFFICIAL`.** The official DGP repository contains only a README; code release is pending industry-partner approval (`repository_audit.md`). **Nothing is `REPRODUCED` yet**: no Qwen3-8B run has been executed, because this machine has no CUDA GPU.

`SMOKE_TESTED` means the code ran end to end and passed checks on CPU (synthetic graph and/or the real AmazonVideo graph with mock language models). It says nothing about matching the paper's numbers.

## DGP method

| Component | Paper | Official code | Our code | Provenance | Status | Evidence / notes |
|---|---|---|---|---|---|---|
| Heterogeneous graph model `G={V,E,R,X}` | Yes | None | `data/graph.py` | PAPER_RECONSTRUCTION | SMOKE_TESTED | Relations kept separate |
| Metapath enumeration `P_K` | Implicit | None | `mdk/adjacency.py` | PAPER_RECONSTRUCTION | SMOKE_TESTED | All sequences of length ≤ K, derived from the paper's `(R^{K+1}−R)/(R−1)` count; unit-tested |
| Metapath adjacency (Eq. 3) | Yes | None | `mdk/adjacency.py` | PAPER_RECONSTRUCTION | SMOKE_TESTED | Equals dense product in tests |
| MDK operator (Eq. 6) | Yes | None | `mdk/diffusion.py` | PAPER_RECONSTRUCTION | SMOKE_TESTED | Literal `(1/K)Σ_{k=0}^{K}`; iterative sparse form equals dense for all variants; 1/K scale proven ranking-invariant by test |
| Diffusion distance + Top-M (Eq. 8–9) | Yes | None | `mdk/distance.py`, `mdk/trimming.py` | PAPER_RECONSTRUCTION | SMOKE_TESTED | Runs on all 12 real AmazonVideo metapaths in ~1 min CPU |
| DeBERTaV3 features for MDK | Camera-ready only | None | `embeddings/text.py` | PAPER_RECONSTRUCTION | IMPLEMENTED | Variant verified from bibliography; base size + mean pooling reconstructed. Real CPU run in progress |
| Node-level summarization (Eq. 5) | Yes | None | `summarization/` | PAPER_RECONSTRUCTION | SMOKE_TESTED (mock) | Instruction verbatim; decoding reconstructed; Qwen path not yet run |
| Metapath-level summarization (Eq. 10) | Yes | None | `summarization/pipeline.py` | PAPER_RECONSTRUCTION | SMOKE_TESTED (mock) | Prompt not published → reuses node instruction |
| Numerical summarization (Eq. 11) | Yes | None | `summarization/numeric.py` | PAPER_RECONSTRUCTION | SMOKE_TESTED | Matches Figure 3 arithmetic in tests |
| Final prompt (Eq. 12) | Figure sketch | None | `prompts/builder.py`, `prompts/dgp/` | PAPER_RECONSTRUCTION | SMOKE_TESTED | Field labels from Figure 3; layout reconstructed, versioned |
| Qwen3-8B backbone | Yes | None | `models/qwen_classifier.py` | PAPER_RECONSTRUCTION | IMPLEMENTED | `Qwen/Qwen3-8B`, not gated; not yet loaded (no GPU) |
| Thinking mode disabled | Not mentioned | None | `render_chat_prompt` | PAPER_RECONSTRUCTION | SMOKE_TESTED | Required for first-token readout; verified on real tokenizer |
| Yes/No first-token readout (Eq. 13–14) | Yes | None | `models/readout.py`, `models/label_tokens.py` | PAPER_RECONSTRUCTION | SMOKE_TESTED | Real Qwen3 tokenizer: Yes=9454, No=2753, single tokens at the chat boundary |
| LoRA on attention (q/k/v/o_proj) | Yes | None | `models/qwen_classifier.py` | PAPER_RECONSTRUCTION | IMPLEMENTED | Module names verified in two transformers versions; alpha unstated |
| AdamW, batch 4, ≤10 epochs, early stop on val loss | arXiv v1 | None | `training/trainer.py` | PAPER_RECONSTRUCTION | SMOKE_TESTED (mock LLM) | Patience reconstructed |
| Metrics via scikit-learn | arXiv v1 | None | `metrics/classification.py` | PAPER_RECONSTRUCTION | SMOKE_TESTED | Macro-F1 threshold unstated; official baselines disagree (ConsisGAD tunes on validation, PMP uses 0.5). Both recomputed for every method; the choice MEASURED to move Macro-F1 by ~13 points |
| Five seeds, mean ± std | Yes | None | `experiment.py` | PAPER_RECONSTRUCTION | SMOKE_TESTED | Seed values 0–4 reconstructed |
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
| **YelpReviews** | Yes | None | `data/yelpchi.py` | **BLOCKED** | Original Rayana & Akoglu release obtainable only by email; relations and validation ready |
| E-Commerce | Yes | Proprietary | `data/proprietary.py` (interface only) | **NOT_REPRODUCIBLE** | Not reproduced because proprietary dataset access was unavailable |
| LifeService | Yes | Proprietary | `data/proprietary.py` (interface only) | **NOT_REPRODUCIBLE** | Not reproduced because proprietary dataset access was unavailable |

## Experiments

| Experiment | Paper artefact | AmazonVideo | YelpReviews | E-Commerce / LifeService |
|---|---|---|---|---|
| DGP main result, 5 seeds | Table 2 | SMOKE_TESTED (mock); GPU run BLOCKED on compute | BLOCKED (data + compute) | NOT_REPRODUCIBLE |
| Ablations (w/o MDK, PathSumm, TextSumm, NumSumm) | Figure 4 (no printed numbers) | SMOKE_TESTED; BLOCKED on compute | BLOCKED | NOT_REPRODUCIBLE |
| Budget B ∈ {5,10,20,40,80} | Figure 5 (no printed numbers) | SMOKE_TESTED; BLOCKED on compute | BLOCKED | — |
| Task-aware vs task-agnostic | Table 3 | SMOKE_TESTED (mock cannot differentiate); BLOCKED on compute | BLOCKED | — |
| Token efficiency | Figure 2 (no printed numbers) | Formula + measured neighbourhoods done; realised prompt lengths need real summaries | BLOCKED | — |
| Complexity analysis | Section "Complexity Analysis" | IMPLEMENTED, run on measured statistics | Needs data | — |

## Baselines (Table 2)

| Method | Official repo @ commit | Provenance | Status | Notes |
|---|---|---|---|---|
| MLP | none exists | PAPER_RECONSTRUCTION | IMPLEMENTED | `baselines/mlp.py`; input features unstated by the paper |
| LLM (Qwen3-8B target-only) | n/a | PAPER_RECONSTRUCTION | SMOKE_TESTED (mock) | DGP pipeline with all neighbour components off |
| GraphSAGE | `williamleif/GraphSAGE` @ a0fdef95 | BASELINE_ADAPTED | NOT_STARTED | Export adapter ready; repo is TF1 |
| HGT | `acbull/pyHGT` @ 85eaccd4 | BASELINE_ADAPTED | NOT_STARTED | OAG-specific loader needs data adapter |
| ConsisGAD | `Xtra-Computing/ConsisGAD` @ 36811c5b | BASELINE_ADAPTED | SMOKE_TESTED | Official code runs unmodified on our AmazonVideo graph + split (`baselines/consisgad_adapter.py`, env torch 1.13.1 / dgl 1.1.0). 1 epoch verified: prediction ids and labels match our split exactly. CPU cost MEASURED ~7.5 min/epoch under contention × 100 epochs → run 5 seeds on the GPU machine |
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
