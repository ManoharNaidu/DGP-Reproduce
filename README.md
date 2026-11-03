# DGP Reproduction

A reproduction of **"DGP: A Dual-Granularity Prompting Framework for Fraud Detection with Graph-Enhanced LLMs"**
(Yuan Li, Jun Hu, Bryan Hooi, Bingsheng He, Cheng Chen — AAAI 2026;
[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/38541) · [arXiv:2507.21653](https://arxiv.org/abs/2507.21653)).

> **Status (2026-10-30): DGP reproduced on AmazonVideo; five follow-up studies finished (§17–§22).**
> The official repository ([Xtra-Computing/DGP](https://github.com/Xtra-Computing/DGP)) contains only a README —
> code release is pending industry-partner approval — so **every DGP component here is a `PAPER_RECONSTRUCTION`**,
> built from both paper versions and traced to evidence in [`research/`](research/).
>
> - **Main result** (5 seeds, real Qwen3-8B + LoRA, 2× A100 80GB): Macro-F1 65.16 ± 0.52 vs. the paper's 66.91 ± 0.13
>   (**CLOSE**, −2.6%), AUROC 75.07 ± 0.50 vs. 77.32 ± 0.11 (**DEVIATES**, −2.9%), AUPRC 32.33 ± 0.65 vs. 34.63 ± 0.24
>   (**DEVIATES**, −6.6%), using reconstructed defaults (K=2, M=4, B=10, LoRA rank=8) with **no hyperparameter search**.
> - **Extension study** (2026-10-12 → 10-13, 46 GPU jobs, all complete): a K sweep (K=1, 2, 3) and six arms that bring
>   FLAG's *discriminative / residual* text and its residual loss into DGP, 5 seeds each (§17). **No arm moves DGP by
>   more than about one point.** Discriminative text for the neighbours (arms B, B+C) gives the only consistent gain,
>   +0.7 to +1.0 AUPRC. FLAG's residual loss (E1, E2) makes Macro-F1 worse and much less stable.
> - **Markov diffusion: DGP vs FLAG-MD** — the two projects' diffusion operators were verified to be the same operator to
>   ~1e-16 (§18).
> - **dr study** (2026-10-15 → 10-16, §19): FLAG's published prompts inside DGP against a same-stack baseline. AUROC up
>   0.21–0.45 on all seeds for every variant; AUPRC flat.
> - **Neighbours at K = 1, 2, 3** (2026-10-17, §20): K = 2 already finds 64% of K = 3's distinct neighbours.
> - **FLAG-architecture study** (2026-10-23 → 10-26, §21): FLAG-style text per sampled subgraph parsed for only 2.8% of
>   subgraphs with Qwen3-8B; stopped after its baseline.
> - **Gemma swap / zero-shot prompt** (2026-10-29 → 10-30, §22): **Gemma-2-9B-it instead of Qwen3-8B raises AUPRC by
>   about 2 points in every arm** (95% CIs exclude 0) — the largest gain found. A structured zero-shot classification
>   prompt does nothing for Qwen and costs Gemma about 2 AUPRC. **The 2 × 2 writer × classifier design** (2026-11-01 →
>   11-03, §22 Exp 4): about three quarters of that gain comes from Gemma as the classifier, whichever role is swapped
>   first; Gemma's text adds 0.2–0.6 points (not significant) and the two roles add up (interaction ≈ 0).
>
> - **Hyperparameter search, running** (2026-11-03 →, §23): the epoch-selection rule (E19, both datasets, 5 seeds); the
>   YelpChi learning-rate sweep (E20) was dropped. The 178 existing training logs show that on YelpChi the epoch with the
>   best validation AUROC was never the one kept (+4.0 val AUROC, +6.7 val AUPRC given up); on AmazonVideo it barely matters.
> - **Target-only LLM control** (2026-11-01, §13): the same fine-tuned Qwen3-8B on the review alone reaches 73.29 AUROC /
>   30.54 AUPRC; **the graph context adds +1.7 AUROC / +1.8 AUPRC** (significant), against +5.5 / +7.6 in the paper.
>
> The MLP baseline remains partially reproduced. See [`research/reproduction_matrix.md`](research/reproduction_matrix.md).

---

## 1. DGP Reproduction

What exists and works today:

- The **AmazonVideo** graph rebuilt from raw data, matching the paper's Table 1 **exactly** (37,126 nodes,
  9,883,406 edges, 4,379 frauds, 1,299/1,299/7,425 split).
- The full DGP pipeline — metapaths, Markov-diffusion trimming, bi-level summarization, numeric summaries,
  prompt construction, Qwen3-8B + LoRA first-token classification — run end to end on GPU, not just smoke-tested.
- **DGP's main result reproduced**: 5 seeds, real Qwen3-8B summarization (119,829 metapath + 35,747 node summaries)
  and LoRA fine-tuning on 2× A100 80GB. See §16.
- **Extension study complete**: K sweep, FLAG-style discriminative/residual text, residual auxiliary loss, a
  software-stack parity check, an LLM-free neighbour analysis and a DGP-vs-FLAG diffusion comparison. See §17–§18.
- **Follow-up studies complete**: the dr study (§19), the K = 1, 2, 3 neighbour analysis (§20), the FLAG-architecture
  study (§21, stopped after its baseline) and the Gemma swap / zero-shot prompt study (§22).
- **Second dataset (YelpChi, 2026-11-02)**: graph matches Table 1 exactly; DGP 61.76 / 78.38 / 31.05 vs the paper's
  69.07 / 84.28 / 48.87, MLP within 0.34 AUROC of the paper. See §11.
- **Ablation w/o MDK (2026-11-02)**: M = 4 random neighbours instead of MDK's choice perform the same as MDK
  (Δ R0 − noMDK: Macro-F1 −0.53, AUROC +0.36, AUPRC −0.14; none significant). The paper reports a decline. See §14.
- Every experiment of the paper wired to a script: main table, ablations, budget sweep, task-aware study,
  token accounting, complexity analysis, comparison with the paper.
- 108 unit tests (all passing on 2026-11-03) and a 14-check CPU smoke test, including a check against the real Qwen3 tokenizer.
- Read-only progress dashboards (`tools/dgp-dashboard/`, v0.4.0; `tools/dr-dashboard/v2/`).

What is outstanding: the paper's other three ablations, the budget / task-aware studies, the rest of the hyperparameter
grid (the owner chose the epoch-selection rule only, running since 2026-11-03, §23), and the
third-party GNN/LLM baselines beyond ConsisGAD and MLP.

## 2. What is DGP?

Graph-enhanced LLMs turn a node's neighbourhood into a prompt. On heterogeneous fraud graphs multi-hop
neighbourhoods explode (on AmazonVideo we measured an average of **~1.5 million raw tokens** of neighbour text
per review at 2 hops), drowning the target node's own signal. DGP keeps two granularities:

- **fine**: the target node's full raw text;
- **coarse**: for each metapath, the neighbours closest under a **Markov diffusion distance** are summarized
  twice by a frozen LLM (node level, then metapath level) to a small token budget `B`, and their numeric
  features are averaged.

The resulting prompt is classified by LoRA-tuned **Qwen3-8B** from the logits of the first generated token
(`Yes` = fraud, `No` = benign). Paper-to-code map: [`docs/architecture.md`](docs/architecture.md).

## 3. Repository structure

```
configs/     datasets/, models/dgp.yaml (every value tagged PAPER / GRID / RECON), experiments/
             experiments/ext/  K1, K3, armB, armC, armD, armE1, armE2, armBC  (extension study, §17)
             experiments/disc_resid/ (§19), flagarch/ (§21), gemma_swap/, zs_qwen/, zs_gemma/ (§22)
src/dgp_repro/
  data/ mdk/ embeddings/ summarization/ prompts/ models/ training/ metrics/ evaluation/ baselines/ utils/
  pipeline.py      graph -> prompts (the paper's Figure 3)
  experiment.py    prompts -> train -> evaluate -> records
prompts/dgp/     versioned prompt templates + PROVENANCE.md; study templates + PROVENANCE_{EXT,DR,FA,ZS}.md
scripts/         prepare_data, build_mdk, generate_summaries, train, evaluate, run_ablation, run_sensitivity,
                 run_grid, run_mlp, reproduce, compare_with_paper, analyze_complexity, fetch_baselines,
                 export_baseline_data, smoke_test, ext_queue (extension scheduler), ext_drain, make_handoff,
                 dr_* (§19), fa_* (§21), gs_queue / gs_checks / gs_analysis (§22)
analysis/        mdk_comparison.py + MDK_COMPARISON.md (DGP vs FLAG-MD), neighbor_analysis.py, neighbors_k123/, out/
methods/         official baselines, fetched at pinned commits (never committed) + README.md provenance
research/        paper specification, version divergence, evidence matrix, provenance, gaps, compute plan,
                 FINAL_REPRODUCTION_REPORT.md
docs/            installation, reproduction, datasets, architecture, troubleshooting,
                 Markov_Diffusion_DGP_vs_FLAG.docx
tools/           dgp-dashboard/, dr-dashboard/v2/ (read-only progress dashboards)
tests/           unit/ (105 tests)
results/         raw/<run_id>/, aggregated/, tables/, figures/, token_usage/, complexity/,
                 ext/, disc_resid/, flag_arch/, gemma_swap/ (queue state, checks, analyses per study)
checkpoints/     LoRA adapters, one per run (adapter_model.safetensors + adapter_config.json)
HANDOFF_4GPU.md  hand-off note from the 2-GPU to the 4-GPU instance (now historical, see §17)
```

## 4. Datasets

| Dataset | Status |
|---|---|
| AmazonVideo | **Ready.** Automatic download; graph matches Table 1 exactly |
| YelpReviews | **Run (2026-11-02, §11).** The official release is email-gated; we use FLAG-Reproduce's release mirror, whose graph matches Table 1 exactly. (A user-supplied `YelpChi.mat` was rejected earlier: it is the CARE-GNN variant and has no text.) |
| E-Commerce, LifeService | **Not reproducible.** Proprietary ByteDance data |

Details, including two places where the data contradict the paper's prose: [`docs/datasets.md`](docs/datasets.md).

## 5. Environment

Python 3.11+, PyTorch ≥ 2.4, transformers ≥ 4.56 (Qwen3 support and the `dtype=` loading argument),
PEFT, scikit-learn (the paper computes all metrics with scikit-learn). DGP's own graph code uses
`scipy.sparse`; DGL is needed only by some baselines. Each baseline gets its own environment.

Stacks actually used (recorded per run in `run_manifest.json`):

| Runs | Python | torch | transformers | peft | CUDA | GPU |
|---|---|---|---|---|---|---|
| Main result (5 seeds) | 3.12.14 | 2.11.0+cu128 | 5.17.0 | 0.21.0 | 12.8 | 2× A100-SXM4-80GB |
| Extension study + parity run | 3.12.14 | 2.14.0+cu130 | 5.19.0 | 0.21.2 | 13.0 | 2×, then 4× A100-SXM4-80GB |

The stack changed between the two, so the baseline was re-run once (seed 0) on the new stack: `parity_dgp` gives
65.43 / 75.15 / 32.54 against 65.70 / 75.13 / 32.78 for the original seed 0 (Δ −0.28 F1, +0.02 AUROC, −0.25 AUPRC).
The new stack is comparable, so extension arms are compared with the original baseline seed by seed.

## 6. CPU installation

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]" "transformers>=4.56" sentencepiece protobuf
```

## 7. GPU installation

```bash
conda env create -f environment/gpu.yml && conda activate dgp-gpu && pip install -e .
```

One A100 80GB is enough (the authors' follow-up paper used exactly one with the same recipe). Measured peak GPU memory
per LoRA training run: **18.3 GB** (K=1), **19.3–19.9 GB** (K=2, every arm), **24.1 GB** (K=3, `max_seq_len` 8192).
Two runs fit on one 80 GB card (the allocator cache then fills ~79 GB; use one slot per GPU if memory is tighter).
`train.py` has no multi-GPU support of its own; extra GPUs only run more seeds or arms in parallel.
Full notes: [`docs/installation.md`](docs/installation.md).

## 8. Model downloads

Not gated; no Hugging Face token needed. Downloaded automatically on first use.

| Model | Id | Size |
|---|---|---|
| Qwen3-8B (summarizer and classifier) | `Qwen/Qwen3-8B` | 16.4 GB |
| DeBERTaV3-base (MDK embeddings) | `microsoft/deberta-v3-base` | ~0.7 GB |

```bash
hf download Qwen/Qwen3-8B
hf download microsoft/deberta-v3-base
```

## 9. Dataset downloads

```bash
python scripts/prepare_data.py --dataset amazonvideo        # downloads, verifies SHA-256, validates Table 1
```

YelpReviews: the official release is email-gated (template in [`docs/datasets.md`](docs/datasets.md)); the mirror route is in §11.

## 10. Quick smoke test

```bash
python scripts/smoke_test.py --real-tokenizer    # CPU, ~30 s, DEBUG_ONLY
python -m pytest tests/unit -q                   # 105 tests
```

Checks graph loading, metapath adjacency, MDK, Top-M, both summarization levels, numeric aggregation,
prompt construction, all four ablation variants, Yes/No token ids at the real Qwen3 chat boundary,
first-token training and readout, metrics and token accounting.

## 11. Yelp reproduction

**Data (2026-11-01).** The official release is email-gated; we use the release mirror that FLAG-Reproduce downloaded
(`../FLAG-Reproduce/datasets/raw/yelpchi/`: metadata, hotel and restaurant review text, 67,395 rows), converted by
`scripts/convert_yelpchi_mirror.py` to `datasets/raw/yelpchi/reviews.jsonl` (row join as in FLAG; the join is supported
statistically, not proven). With RSR = same product + same star and RTR = same product + same month the graph matches
Table 1 **exactly**: 67,395 nodes, 17,486,608 edges (RUR 220,224 · RSR 16,333,680 · RTR 932,704), 8,919 frauds; split
1,348 / 1,348 / 13,479. Every other scoping misses by orders of magnitude.

```bash
python scripts/convert_yelpchi_mirror.py
python scripts/prepare_data.py --dataset yelpchi --probe     # RSR product / RTR product: 17,486,608 = paper
python scripts/prepare_data.py --dataset yelpchi
DATASET=yelpchi bash scripts/ctl_queue.sh YELP configs/experiments/dgp_yelp.yaml dgp_yelpchi   # summaries, 5 seeds
```

Same model config as the AmazonVideo main run (`dgp_yelp.yaml` differs from `dgp_amazon.yaml` only in the dataset).

**Results (2026-11-02, 5 seeds, one A100).** Summaries: 60,013 node and 162,296 metapath summaries, 2026-11-01T16:15Z →
2026-11-02T03:13Z (one process to ~31k, then 3 hash shards sharing one store, `scripts/yelp_nomdk_chain.sh`). Training:
03:14Z → 14:41Z, two lanes, 2.6–4.8 h per seed, peak 19.3 GB, mean final prompt 792 tokens (max 1,845).
`python scripts/yelp_analysis.py` → [`results/controls/yelpchi.md`](results/controls/yelpchi.md).

| Run | Macro-F1 | AUROC | AUPRC | Paper (F1 / AUROC / AUPRC) |
|---|---|---|---|---|
| DGP (`dgp_yelpchi`) | 61.76 ± 1.86 | 78.38 ± 0.88 | 31.05 ± 1.31 | 69.07 / 84.28 / 48.87 |
| MLP (`mlp_yelpchi`, CPU) | 63.64 ± 0.43 | 74.66 ± 0.71 | 33.67 ± 1.01 | 62.09 / 75.00 / 32.24 |
| DGP − MLP (paired) | −1.88 (1/5) | **+3.72** (5/5), CI [+2.36, +5.21] | **−2.62** (0/5), CI [−4.61, −0.42] | +6.98 / +9.28 / +16.63 |

Per seed DGP (F1 / AUROC / AUPRC): 61.97 / 78.80 / 31.60 · 58.45 / 76.67 / 28.48 · 62.72 / 78.40 / 31.30 ·
61.58 / 79.08 / 32.03 · 64.09 / 78.94 / 31.83. CI = paired bootstrap over the 13,479 test reviews; seed-level paired t
with Holm in `results/stats/all_comparisons.md` (E16: AUROC Holm p = 0.004, AUPRC 0.039, Macro-F1 n.s.).

- **The MLP reproduces, DGP does not.** The MLP lands within 0.34 AUROC of the paper, so data, split and graph are
  plausible. DGP is −5.90 AUROC and −17.82 AUPRC below the paper (AmazonVideo: −2.25 / −2.30).
- **DGP ranks better than the MLP but is worse at the top of the ranking.** AUROC is higher on every seed, AUPRC lower on
  every seed. In the paper DGP beats the MLP by +16.63 AUPRC.
- **Model selection is a candidate cause (being tested as E19, §23).** Early stopping on validation loss (patience 2, the paper's
  rule) kept epoch 3 / 1 / 2 / 3 / 2. Validation loss rises after that while validation AUROC keeps rising: in every
  seed the best validation AUROC is at the last epoch run (81.6 / 77.6 / 81.6 / 81.9 / 86.6 vs the kept epochs).
  The YelpChi training set is only 1,348 reviews. Selecting on validation AUROC is a hyperparameter-search item.
- The paper's other YelpReviews rows (ConsisGAD 82.12 AUROC, LLM-only 71.18, FLAG 73.98) were not run here.

## 12. Amazon reproduction

```bash
python scripts/reproduce.py --dataset amazonvideo --device cuda:0 --seeds 0 1 2 3 4
```

Run the ~1 GPU-hour calibration in [`docs/reproduction.md`](docs/reproduction.md) first. One seed of the main
configuration takes about 1.1 h of LoRA training on one A100 once the summaries are cached.

## 13. Baselines

MLP and the target-only LLM are implemented in this repository. The ten official baselines are fetched at
verified commits (`python scripts/fetch_baselines.py`) and receive our exact graph and split through an export
adapter. ConsisGAD's official code already runs unmodified on our AmazonVideo graph and split (a full 5-seed run
costs ≈35 CPU-hours and has not been done); the others are not yet adapted. PMP, GAAP and FLAG publish **no licence**
and must never be redistributed. See [`methods/README.md`](methods/README.md).

**Target-only LLM control (`ctl_T0`, 2026-11-01).** The paper's "LLM" row: the same Qwen3-8B + LoRA classifier,
recipe, split, early stopping and validation-tuned threshold as DGP (dr arm R0), but the prompt holds only
`Target: <review>` and the question, with no neighbour or metapath summaries
(`configs/experiments/controls/T0_llm_target_only.yaml`, `scripts/ctl_queue.sh`, 5 seeds, 0.67 h/seed). Analysis:
`python scripts/ctl_analysis.py` → [`results/controls/controls.md`](results/controls/controls.md).

| AmazonVideo, 5 seeds | Macro-F1 | AUROC | AUPRC | paper |
|---|---|---|---|---|
| DGP (dr_R0) | 64.74 ± 0.45 | 74.97 ± 0.51 | 32.37 ± 0.57 | 66.91 / 77.32 / 34.63 |
| LLM, target only (ctl_T0) | 63.35 ± 2.84 | 73.29 ± 0.36 | 30.54 ± 0.60 | 59.90 / 71.78 / 27.03 |
| MLP | 59.83 ± 0.50 | 68.33 ± 0.51 | 25.11 ± 1.09 | 61.74 / 70.47 / 26.55 |

- **The graph context adds +1.69 AUROC and +1.83 AUPRC** (DGP − T0, 5/5 seeds, Holm-corrected paired t p = 0.011
  for both, bootstrap CIs [+0.92, +2.45] and [+0.67, +3.03]). The paper's DGP − LLM gap is **+5.54 / +7.60**, about
  three to four times larger. The Macro-F1 gap (+1.40) is not significant: T0's seed 3 picked a validation threshold
  of 0.05 and scored 57.81.
- **Our target-only LLM is far stronger than the paper's:** +1.51 AUROC and +3.51 AUPRC above the paper's LLM row,
  while DGP − MLP matches the paper (+6.64 vs +6.85 AUROC, +7.26 vs +8.08 AUPRC). So on AmazonVideo about three
  quarters of DGP's gain over the MLP comes from fine-tuning the LLM on the review itself (T0 − MLP: +4.96 AUROC),
  and one quarter from the graph summaries. The paper does not say how its LLM baseline was trained; a weaker recipe
  there (e.g. no LoRA, or a different threshold) would explain the gap but cannot be checked.

## 14. Ablations

```bash
python scripts/run_ablation.py --dataset amazonvideo --device cuda:0
```

`w/o MDK`, `w/o PathSumm`, `w/o TextSumm`, `w/o NumSumm` (Figure 4). The paper prints no numbers for this figure,
so only its stated ordering can be checked. The other three are not yet run on GPU.

**`w/o MDK` (2026-11-02):** M = 4 random neighbours per metapath (seed 0) instead of MDK's Top-M, everything else as dr
R0 (`configs/experiments/controls/A1_without_mdk.yaml`; the paper never says what replaces MDK, decision D4). 5 seeds,
paired with R0; `python scripts/nomdk_analysis.py` → [`results/controls/nomdk.md`](results/controls/nomdk.md).

| run | Macro-F1 | AUROC | AUPRC |
|---|---|---|---|
| R0 (MDK) | 64.74 ± 0.45 | 74.97 ± 0.51 | 32.37 ± 0.57 |
| w/o MDK (random) | 65.28 ± 0.80 | 74.61 ± 0.54 | 32.51 ± 0.61 |
| Δ R0 − noMDK (seeds better) | −0.53 (1/5) | +0.36 (4/5) | −0.14 (1/5) |

- **MDK adds nothing measurable here.** No difference is significant (paired t p 0.27 / 0.07 / 0.18; Holm family ≥ 0.20);
  the bootstrap 95% CIs over test reviews include 0 (AUROC [−0.32, +1.05], AUPRC [−1.12, +0.81]). The paper states
  that removing MDK "leads to a performance decline" but prints no numbers.
- **The two selections really differ.** Of the random neighbours, MDK also chose 0.3–2.4% on the cross-relation
  metapaths, 9% on RSR, 21% on RPR and 66% on RUR (where 28% of targets have ≤ M candidates, so both keep all). The
  predictions differ too (label agreement 94–96%, McNemar p < 0.05 in 5/5 seeds), but not in quality.
- Cost: 762 prompt tokens on average (R0: 782), 1.96 h per run with two runs sharing the GPU, peak
  19.6 GB. Kept epochs 1–2.

## 15. Budget sensitivity

```bash
python scripts/run_sensitivity.py --study budget --dataset amazonvideo --device cuda:0      # B ∈ {5,10,20,40,80}
python scripts/run_sensitivity.py --study task_aware --dataset amazonvideo --device cuda:0  # Table 3
```

**Not yet run on GPU.** (The K sweep in §17 is a separate, completed study.)

## 16. Results

**DGP, AmazonVideo, 5 seeds (GPU_REPRODUCTION, real Qwen3-8B + LoRA):**

| Metric | Paper | Ours | Diff | Status |
|---|---|---|---|---|
| Macro-F1 | 66.91 ± 0.13 | 65.16 ± 0.52 | −1.75 (−2.6%) | CLOSE |
| AUROC | 77.32 ± 0.11 | 75.07 ± 0.50 | −2.25 (−2.9%) | DEVIATES |
| AUPRC | 34.63 ± 0.24 | 32.33 ± 0.65 | −2.30 (−6.6%) | DEVIATES |

Per seed (test): F1 65.70 / 64.90 / 64.26 / 65.46 / 65.49, AUROC 75.13 / 75.54 / 74.72 / 74.33 / 75.65,
AUPRC 32.78 / 32.47 / 31.70 / 31.49 / 33.21; best epoch 2 / 2 / 1 / 1 / 3 of 10. All ± values in this README are the
std across seeds with `ddof=0` (population std, `src/dgp_repro/metrics/classification.py`); the sample std (`ddof=1`)
is about 12% larger (e.g. 0.58 instead of 0.52 for F1). Macro-F1 uses the threshold that maximises validation F1;
Macro-F1 at 0.5 is 49.45 ± 2.83.

Reconstructed defaults from `configs/models/dgp.yaml` (K=2, M=4, B_node=B_meta=10, LoRA rank=8, alpha=16,
dropout=0.05, lr=1e-4), **no hyperparameter search**. The paper publishes its search grid but not the winning
values (`research/compute_plan.md` §6). Our std across seeds (0.5–0.65) is notably higher than the paper's
(0.11–0.24), consistent with an untuned config rather than a pipeline bug, given Macro-F1 lands within 2.6%.
Full manifests (config, git commit, GPU, peak memory, runtime) in
`results/raw/dgp_amazonvideo_seed0_seed{0..4}_*/run_manifest.json`; details in
`results/tables/comparison_with_paper.md`. (The `_seed0` in the experiment name is the split seed.)

**MLP baseline, AmazonVideo, 5 seeds (CPU):** Macro-F1 59.86 ± 0.51 · AUROC 68.35 ± 0.49 · AUPRC 25.14 ± 1.06, against
the paper's 61.74 / 70.47 / 26.55 → `PARTIALLY_REPRODUCED`. Re-run on 2026-11-01 because its result files were missing
(`results/raw/mlp_amazonvideo_seed*_3cc7104faa/`): 59.83 ± 0.50 · 68.33 ± 0.51 · 25.11 ± 1.09, the same within 0.03.
The paper's target-only **LLM** baseline is in §13 (`ctl_T0`).

`results/tables/main_results.md` is written only from real runs; CPU debug runs are kept in
`results/tables/debug_runs.md` and are never compared with the paper.

**Tokens** (AmazonVideo, real data, real Qwen3-8B): mean review length **115.7 Qwen3 tokens**; average degree
**133.1** (paper: 133); at K=2 a full-neighbour prompt averages **~1.5M raw tokens** per review versus a **measured
mean final DGP prompt of 781.5 tokens** (p95 1,206, max 1,902; compression ≈1922×, 99.9% token reduction;
`results/token_usage/token_usage_dgp_amazonvideo_seed0.json`). This is higher than the paper-formula estimate of ~236
tokens that appeared in the earlier (pre-GPU) complexity analysis. `results/complexity/`.

Paper targets for comparison only: [`research/reported_results.csv`](research/reported_results.csv).

## 17. Extension study: K sweep and FLAG ideas inside DGP

**Question.** Does DGP benefit from (a) a different metapath depth K, or (b) FLAG's two ideas: LLM-written
*discriminative* text (signals that separate the classes) and *residual* text (background that should not), plus
FLAG's residual auxiliary loss?

**Design.** Everything is fixed at the main configuration except the one change listed. The 5 seeds are the same as the
baseline's, so each arm is compared with the baseline **seed by seed**. Extension prompts are adapted from FLAG's
causal / non-causal prompts (BUPT-GAMMA/FLAG `chat.py`) to a per-node, one-line, 40-token setting, are **not** from the
DGP paper, and never see the label (`prompts/dgp/PROVENANCE_EXT.md`). Template names enter every cache key, so baseline
caches stay valid.

| Arm | Config (`configs/experiments/ext/`) | What changes |
|---|---|---|
| K1 | `K1.yaml` | K=1: 3 metapaths, 1 diffusion step |
| K3 | `K3.yaml` | K=3: 39 metapaths, 3 diffusion steps; `max_seq_len` raised to 8192 (recorded deviation) |
| B | `armB_disc_neighbors.yaml` | neighbour node-level text = discriminative line instead of the task-agnostic summary |
| C | `armC_disc_target.yaml` | target's own discriminative line added next to its raw text (`Key signals:`) |
| B+C | `armBC_disc_both.yaml` | B and C together |
| D | `armD_resid_prompt.yaml` | negative control: target's residual (non-causal) line placed in the prompt (`Background:`) |
| E1 | `armE1_resid_aux.yaml` | baseline prompt + FLAG-style residual auxiliary loss (KL-to-uniform + orthogonality, α=β=0.1) |
| E2 | `armE2_disc_resid_aux.yaml` | arm C prompt + residual auxiliary loss: the closest analogue of FLAG's training |

**Results** (AmazonVideo test set, 5 seeds, mean ± std with ddof=0; Δ = mean paired difference to the baseline, with
the number of seeds where the arm is better):

| Arm | Macro-F1 | AUROC | AUPRC | Δ F1 | Δ AUROC | Δ AUPRC | prompt tokens (mean / max) | train time / seed |
|---|---|---|---|---|---|---|---|---|
| baseline (K=2) | 65.16 ± 0.52 | 75.07 ± 0.50 | 32.33 ± 0.65 | — | — | — | 782 / 1,902 | 1.1 h |
| K1 | 65.02 ± 0.69 | 74.89 ± 0.20 | 32.40 ± 0.62 | −0.15 (3/5) | −0.18 (2/5) | +0.07 (4/5) | 281 / 1,251 | 1.1 h |
| K3 | 65.38 ± 0.26 | 75.16 ± 0.40 | 32.32 ± 0.75 | +0.22 (3/5) | +0.09 (3/5) | −0.01 (4/5) | 2,358 / 3,852 | 3.4 h |
| B | 65.55 ± 0.22 | 75.12 ± 0.28 | 33.04 ± 0.45 | +0.38 (2/5) | +0.05 (3/5) | **+0.71 (4/5)** | 681 / 1,742 | 2.0 h |
| **B+C** | **65.63 ± 0.34** | 74.87 ± 0.26 | **33.32 ± 0.47** | +0.47 (3/5) | −0.21 (2/5) | **+0.99 (5/5)** | 705 / 1,768 | 1.7 h |
| C | 65.23 ± 0.64 | 74.59 ± 0.38 | 31.80 ± 0.66 | +0.06 (3/5) | −0.48 (0/5) | −0.53 (0/5) | 806 / 1,938 | 2.3 h |
| D (control) | 64.64 ± 0.94 | 75.10 ± 0.46 | 31.62 ± 0.48 | −0.52 (2/5) | +0.02 (2/5) | −0.71 (0/5) | 799 / 1,930 | 2.3 h |
| E1 | 64.78 ± 1.61 | 74.97 ± 0.31 | 32.19 ± 0.37 | −0.38 (3/5) | −0.10 (3/5) | −0.14 (2/5) | 782 / 1,902 | 2.3 h |
| E2 | 64.06 ± 2.30 | 74.64 ± 0.54 | 31.72 ± 0.94 | −1.11 (3/5) | −0.44 (0/5) | −0.61 (1/5) | 806 / 1,938 | 2.4 h |

Peak GPU memory is 18.3 GB (K1), 19.3–19.9 GB (K=2 arms) and 24.1 GB (K3). Train times are wall-clock per seed on
the extension instance, where two runs usually shared one GPU, so they are not directly comparable with the
baseline's 1.1 h.

**What it shows.**

1. **No arm is a clear win.** Every mean is within about one point of the baseline, and with 5 seeds most differences
   are inside seed noise. None of them closes the gap to the paper (66.91 / 77.32 / 34.63).
2. **K matters little beyond K=2.** K=3 triples the prompt and the training time for +0.2 F1 and no AUPRC gain; K=1
   is about as good as K=2 with a third of the prompt. The neighbour analysis below explains why.
3. **Discriminative text helps on the neighbour side, not on the target side.** Arms B and B+C give the only consistent
   gain: AUPRC +0.71 (4/5 seeds) and +0.99 (5/5 seeds), with lower variance than the baseline. They also make the
   prompt *shorter* (one 40-token line instead of a summary per neighbour). Adding the target's own discriminative line
   alone (C) lowers AUROC and AUPRC on every seed.
4. **FLAG's residual loss does not transfer.** E1 and E2 lower the mean F1 and raise its std 3–4× (E2: 2.30); E2,
   the closest analogue of FLAG's full training, is the worst arm.
5. **The negative control behaves as expected.** Putting residual (non-indicative) text in the prompt (D) lowers F1 and
   AUPRC (AUPRC worse on 5/5 seeds).

**Neighbour analysis** (LLM-free, `analysis/neighbor_analysis.py` → `analysis/out/neighbor_analysis.{json,csv}`):

| K | metapaths | neighbours kept per target | of which unique | raw tokens of kept neighbours | neighbour fraud-rate gap* |
|---|---|---|---|---|---|
| 1 | 3 | 11.9 | 11.8 | 1,682 | 0.118 |
| 2 | 12 | 47.7 | 32.6 | 7,397 | 0.097 |
| 3 | 39 | 155.4 | 51.0 | 25,451 | 0.095 |

\*fraud rate of a target's kept neighbours when the target is fraud minus when it is benign (train + val targets only;
labels are read only for this diagnostic). For a 1-hop metapath, the kept neighbours are **identical** for K=1, 2, 3
(Jaccard 1.0): more diffusion steps do not change who is selected there. Deeper K mostly adds **duplicate** neighbours
(duplicate ratio 0.01 → 0.32 → 0.67) and dilutes the label signal slightly, which matches the flat K sweep.
Decoupling diffusion steps from metapath length (analysis only) changes the selection on composite metapaths
(Jaccard 0.78–0.93 between 1 and 2 steps) but not on 1-hop ones.

**How it was run.** `scripts/ext_queue.py` is a resumable scheduler: each job is a subprocess pinned to one GPU,
finished jobs leave `results/ext/done/<job>`, events go to `results/ext/status.jsonl`. It ran from 2026-10-12 05:47 to
2026-10-13 08:20 UTC, first on 2 GPUs, then (after the hand-off in `HANDOFF_4GPU.md`) on 4. All 46 jobs finished:
5 summary-generation jobs (`gen_T`, `gen_B`, `gen_K1`, `gen_BC`, `gen_K3`), 40 training runs (8 arms × 5 seeds) and the
parity run.

```bash
mkdir -p logs/ext
GPUS=0,1,2,3 SLOTS_PER_GPU=2 nohup python scripts/ext_queue.py > logs/ext/queue.log 2>&1 &
python tools/dgp-dashboard/server.py 18780      # read-only dashboard on 127.0.0.1
python analysis/neighbor_analysis.py            # CPU, needs the K=1,2,3 MDK caches
```

Per-run outputs: `results/raw/ext_<arm>_seed<s>_<hash>/{metrics.json,predictions.csv,run_manifest.json}` and LoRA
adapters in `checkpoints/`. **Note:** until 2026-11-01 `results/aggregated/*.json` were overwritten by each
single-seed job and held one seed only. `run_experiment` now aggregates every finished seed of the same experiment and
config hash on disk, and `scripts/rebuild_aggregates.py` rebuilt all 34 existing files (means unchanged from the tables
in this README, which were always computed from the five `metrics.json` files per arm).

## 18. Markov diffusion: DGP vs FLAG-MD

The companion FLAG project (`../FLAG-Reproduce`) borrowed DGP's diffusion distance as a replacement neighbour sampler
("FLAG-MD"). `analysis/mdk_comparison.py` verifies that the two implementations are **the same operator**
`Z(K) = (1/K) Σ_{k=0..K} T^k`, `T = D⁻¹A`, distance `‖H_w − H_u‖₂` with `H = Z(K)X`:

- both match a dense NumPy reference to ~1e-16 for K = 1, 2, 3, and agree with each other on a single-relation graph
  (same distances, same top-4 sets);
- the 1/K vs 1/(K+1) factor never changes a ranking; dropping the identity term does (13 of 40 top-4 sets changed);
- what differs is the input: DGP uses **count-weighted metapath products** (binarising changed 9 of 38 top-4 sets),
  DeBERTa-v3 ⊕ rating features, and the metapath neighbourhood as candidates; FLAG-MD uses one binary adjacency,
  L2-normalised MiniLM features, and the frontier of a 2-hop expansion.

Write-ups: [`analysis/MDK_COMPARISON.md`](analysis/MDK_COMPARISON.md) and `docs/Markov_Diffusion_DGP_vs_FLAG.docx`.

## 19. dr study: FLAG's published prompts inside DGP

**Question.** Do FLAG's prompts as published, a variant without a class verdict, and FLAG's residual branch improve
DGP against a baseline re-run on the **same** software stack? (§17 compared its arms with the September baseline.)

**Design.** Two pilots of 200 reviews showed that FLAG's discriminative prompt states a class verdict in 188 of 200
lines, so a **no-verdict** variant was added (`prompts/dgp/PROVENANCE_DR.md`). One line per target for each kind,
5 arms × 5 seeds (`configs/experiments/disc_resid/`):

| Arm | What changes from R0 |
|---|---|
| R0 | none: the main configuration re-run on this stack (reference) |
| R1 | FLAG-prompt discriminative line next to the raw text (`Key signals:`) |
| R1b | the same with the no-verdict prompt |
| R2 / R2b | R1 / R1b + FLAG's residual branch and loss (CE + 0.1·KL-to-uniform + 0.1·squared cosine) |

**Results** (test, 5 seeds, Δ vs R0 with the paired bootstrap 95% CI; `results/disc_resid/deviation_final.md`):

| Arm | Macro-F1 | AUROC | AUPRC | Δ AUROC (better) | CI Δ AUROC | Δ AUPRC | CI Δ AUPRC |
|---|---|---|---|---|---|---|---|
| R0 | 64.74 ± 0.45 | 74.97 ± 0.51 | 32.37 ± 0.57 | — | — | — | — |
| R1 | 64.79 ± 1.30 | 75.18 ± 0.56 | 32.42 ± 1.17 | +0.21 (5/5) | [−0.30, +0.72] | +0.05 | [−0.57, +0.68] |
| R1b | 65.33 ± 0.78 | 75.42 ± 0.58 | 32.41 ± 0.80 | +0.45 (5/5) | [+0.11, +0.79] | +0.04 | [−0.48, +0.56] |
| R2 | 64.28 ± 2.60 | 75.23 ± 0.47 | 32.52 ± 0.96 | +0.25 (5/5) | [−0.27, +0.78] | +0.15 | [−0.54, +0.85] |
| R2b | 64.78 ± 1.44 | 75.24 ± 0.51 | 32.37 ± 0.79 | +0.27 (5/5) | [−0.10, +0.67] | +0.00 | [−0.56, +0.57] |

**What it shows.** Every variant raises AUROC on all five seeds, but only R1b's interval excludes 0; AUPRC does not
move. FLAG's lines carry label signal (a TF-IDF probe on the FLAG-prompt line: AUROC 71.62 vs 63.43 on raw text,
`results/disc_resid/text_diagnostics.md`), but the classifier gains little from it. The residual loss again widens the
Macro-F1 spread.

**How it was run.** `scripts/dr_generate_all.sh` (generation, 2026-10-15 02:13 → 03:31 UTC), then `scripts/dr_queue.sh`
(25 runs one after another on one GPU, 2026-10-15 03:31 → 10-16 07:35 UTC), then
`python scripts/dr_analysis.py --stem deviation_final`. Dashboard: `tools/dr-dashboard/v2/`. Report:
`results/2026-10-17-dgp-disc-resid-report.docx`.

## 20. Neighbours at K = 1, 2, 3, measured against K = 3

An LLM-free extension of the §17 neighbour analysis (`analysis/neighbors_k123/`, outputs in
`analysis/out/neighbors_k123/`, summary in `SUMMARY.md`). It recomputes 32,400 cached selections live, then measures
each K against K = 3 (Top-M = 4 per metapath, 10,023 targets):

| K | Distinct neighbours per target | Share of K = 3's covered | Label gap | Gap − K = 3 gap (95% CI) |
|---|---|---|---|---|
| 1 | 11.78 | 0.239 | 0.118 | 0.022 [0.011, 0.034] |
| 2 | 32.59 | 0.640 | 0.097 | 0.002 [−0.004, 0.007] |
| 3 | 51.03 | 1.000 | 0.095 | — |

K = 2 finds 64% of K = 3's distinct neighbours and has the same label gap; K changes the selection only through the
composite metapaths. Report: `results/2026-10-18-dgp-neighbors-k123-report.docx`.

## 21. FLAG-architecture study (stopped after its baseline)

**Question.** Does discriminative text help when it is generated the way FLAG generates it: **after** sampling, one
prompt per trimmed target subgraph (target + its MDK Top-M neighbours), one output line per node?
(`prompts/dgp/PROVENANCE_FA.md`, `configs/experiments/flagarch/`.)

**Arms.** F0 = main configuration (reference). F1 = FLAG-style discriminative lines (neighbours' lines replace node
summaries; the target's line is added as `Key signals:`). F2 (+ residual text and loss) was taken out of scope on
2026-10-23. Text model: Qwen3-8B; FLAG's layout, parser and decoding (greedy, 550 new tokens, reviews cut to 1,200
characters).

**What happened.**

- Pilot (200 subgraphs): prompts averaged 4,076 tokens and 196 of 200 answers hit the 550-token cap. FLAG's parser
  (exactly one line per node, about 33 nodes per subgraph) accepted 6 of 200.
- Full F1 generation: **277 of 10,023 subgraphs parsed (2.76%)**, so 97% of targets fell back to raw text.
- Training (2026-10-25): all five **F0** runs finished — Macro-F1 65.10 ± 0.36, AUROC 74.92 ± 0.49, AUPRC 32.28 ± 0.58,
  in line with R0 (§19). All five **F1** runs failed at the first batch: `prompt of 4143 tokens exceeds max_seq_len=4096`.
- R text was generated afterwards for the record (1,590 of 10,023 subgraphs parsed, 15.9%); no arm uses it.

The study gives no evidence for or against FLAG's architecture inside DGP: the generation step does not work at DGP's
subgraph size with this model and output budget. Re-running F1 would need smaller subgraphs or per-node prompts, plus a
lower `dgp.max_target_tokens`. Commands: `scripts/fa_pilot.py`, `scripts/fa_queue.sh`, `scripts/fa_r_after_queue.sh`;
records in `results/flag_arch/` and `logs/flag_arch/`.

## 22. Gemma swap and the structured zero-shot prompt

Three experiments on the five dr arms (0, 1, 1b, 2, 2b = §19's R0 … R2b), 5 seeds each, paired by seed with §19:

| Exp | Cell | Text writer + classifier | Final classification prompt | Configs |
|---|---|---|---|---|
| — | R | Qwen3-8B | original | `disc_resid/` (§19) |
| 1 | G | **Gemma-2-9B-it** | original | `gemma_swap/` |
| 2 | S | Qwen3-8B | **structured zero-shot** | `zs_qwen/` (+ `P_R0_parity.yaml`) |
| 3 | GS | **Gemma-2-9B-it** | **structured zero-shot** | `zs_gemma/` |

- **Gemma** (`google/gemma-2-9b-it`, revision `11c9b30`) replaces Qwen3-8B in both roles. It needs eager attention, and
  generation closes a batch before prompts × longest² exceeds 16 × 2,048² (`summarizer.max_batch_tokens_sq`), after two
  out-of-memory stops on 2026-10-29. Neighbour selection is identical to §19 for all 12 metapaths.
- **Structured zero-shot prompt** (`prompts/dgp/zs_*.txt`, `PROVENANCE_ZS.md`): the best Gemma zero-shot prompt from
  the 2026-10-20 prompt test (`../experiments/prompt_testing/`), with `<task>`, `<definition>`, `<target_review>`,
  `<key_signals>`, `<neighbourhood>`, `<constraints>`, `<output_format>` sections. It replaces only the final and the
  residual-branch prompts; every field carries the same text as before (checked for all 10,023 targets). It adds a
  definition of "fraud" (a review other shoppers judge unhelpful) and asks for one word, `Yes` or `No`.

**Results** (test, 5 seeds; Δ = mean paired difference, with the paired bootstrap 95% CI of ΔAUPRC):

| Arm | R: AUROC / AUPRC | G: AUROC / AUPRC | S: AUROC / AUPRC | GS: AUROC / AUPRC | G − R: Δ AUPRC [CI] | GS − G: Δ AUPRC [CI] |
|---|---|---|---|---|---|---|
| 0 | 74.97 / 32.37 | **76.38** / 34.31 | 75.52 / 32.76 | 75.37 / 32.92 | +1.94 [+0.70, +3.10] | −1.39 [−2.23, −0.52] |
| 1 | 75.18 / 32.42 | 76.00 / 34.30 | 75.44 / 33.02 | 75.04 / 32.30 | +1.88 [+0.52, +3.20] | −2.00 [−2.90, −1.12] |
| 1b | 75.42 / 32.41 | 75.82 / **34.52** | 75.68 / 32.72 | 75.34 / 33.12 | +2.10 [+0.75, +3.41] | −1.39 [−2.27, −0.46] |
| 2 | 75.23 / 32.52 | 76.35 / 34.20 | 75.49 / 33.10 | 75.13 / 31.92 | +1.68 [+0.42, +2.81] | −2.29 [−3.20, −1.36] |
| 2b | 75.24 / 32.37 | 75.98 / 34.24 | 75.68 / 32.95 | 75.01 / 32.24 | +1.87 [+0.65, +3.07] | −2.00 [−3.04, −0.94] |

**What it shows.**

1. **Gemma is the better backbone** with the original prompt: AUPRC +1.68 to +2.10 in every arm and AUROC +0.40 to
   +1.41 (CI excludes 0 in 4 of 5 arms). Within Gemma, no D/R arm beats G0. The best values come within 0.1–1 point of
   the paper on AUROC and AUPRC (G0 AUROC 76.38 vs 77.32; G1b AUPRC 34.52 vs 34.63); Macro-F1 stays about one point
   below (best 65.97 vs 66.91).
2. **The zero-shot prompt does nothing measurable for Qwen** (every S − R CI for AUROC and AUPRC includes 0). An R0 seed-0
   re-run on the same stack moved Macro-F1 by +0.58, as much as most arm effects.
3. **The zero-shot prompt hurts Gemma** (GS − G: AUPRC −1.39 to −2.29, AUROC −0.48 to −1.22) and removes its advantage
   over Qwen. The prompt × model interaction (GS − G) − (S − R) is negative on all 25 paired seeds for AUPRC.
4. **Cost.** Gemma uses about 25–27 GB at training time against about 20 GB for Qwen. Its prompts are shorter (540 vs
   782 tokens with the original prompt) because its summaries are shorter. Two Gemma runs sharing one GPU were much slower
   than one run alone (2.1–4.3 h per pair vs 68–81 min), so Exp 2 and 3 ran one training job per GPU.

**How it was run.** One resumable queue on 4 A100 80 GB, the three experiments one after another (2026-10-29 12:59 →
10-30 22:57 UTC, 76 training runs, all rc 0); Exp 3 reuses Exp 1's Gemma text caches:

```bash
nohup ../envs/dgp-fa/bin/python scripts/gs_queue.py > logs/gemma_swap/queue.log 2>&1 &
# one training run per GPU, as used from Exp 2 on:
QWEN_TRAIN_WEIGHT=2 GEMMA_WEIGHT=2 setsid nohup ../envs/dgp-fa/bin/python scripts/gs_queue.py >> logs/gemma_swap/queue.log 2>&1 &
```

Analyses (written by the queue): `results/gemma_swap/exp{1,2,3}.md`; checks: `results/gemma_swap/checks/`. Full write-up
with figures: `reports/2026-10-23-DGP-experiment-log.docx` (experiments E11–E13) in the parent folder.

### Exp 4: Gemma's text with the Qwen classifier (writer vs classifier, 2026-11-01)

Exp 1 swapped Gemma into both roles at once. Exp 4 adds the two missing cells for arm 0 (original prompt): **X0 =
Gemma's text + Qwen3-8B classifier** (2026-11-01) and **QG0 = Qwen's text + Gemma-2-9B-it classifier** (2026-11-02 →
11-03), 5 seeds each, paired with R0 (Qwen writes and classifies) and G0 (Gemma writes and classifies). Each role's
effect is then measured twice, once with each model in the other role, and the interaction
(G0 − QG0) − (X0 − R0) tests whether the two effects add up.

- **Same text as G0, word for word.** `configs/experiments/gemma_swap/X0.yaml` = R0 + `summarizer.backend: cached`, a
  read-only backend (`CachedOnlySummarizer`) that serves G0's cached Gemma summaries, never loads Gemma and stops on any
  cache miss. The neighbour-trimming cache key includes the classifier's tokenizer, which changes only the stored raw-token
  counts (neighbours identical for all 12 metapaths), so `summarizer.source_mdk_cache` points X0 at G0's trimming cache.
  9,931 of 10,023 X0 prompts equal G0's; the other 92 differ only where the target review is cut at 1,024 tokens.
- **Same text as R0, word for word (QG0).** `configs/experiments/gemma_swap/QG0.yaml` = R0 with the cached backend
  pointed at R0's Qwen summaries (`source_mdk_cache: 30447a444d0aa6e2`) and Gemma (`11c9b30`, eager attention) as the
  classifier. All 35,747 node and 119,829 metapath summaries came from cache; 9,931 of 10,023 prompts equal R0's, the
  other 92 differ only where the target review is cut at 1,024 Gemma tokens (as in G0). One training lane: two Gemma
  runs sharing the GPU are 2–3× slower.
- **Stack drift.** The new 1-GPU instance runs torch 2.14.1 (R0: 2.14.0), so R0 seed 0 was re-run as `par2_R0`.

| Cell | Writer | Classifier | Macro-F1 | AUROC | AUPRC |
|---|---|---|---|---|---|
| R0 | Qwen | Qwen | 64.74 ± 0.45 | 74.97 ± 0.51 | 32.37 ± 0.57 |
| QG0 | Qwen | Gemma | 64.70 ± 1.35 | 76.18 ± 0.34 | 33.74 ± 0.44 |
| X0 | Gemma | Qwen | 65.30 ± 0.51 | 75.37 ± 0.35 | 32.86 ± 0.37 |
| G0 | Gemma | Gemma | 65.04 ± 1.01 | **76.38** ± 0.34 | **34.31** ± 1.00 |

| Comparison | Δ AUROC [95% CI] · Holm p | Δ AUPRC [95% CI] · Holm p | Share of G0 − R0 (AUROC / AUPRC) |
|---|---|---|---|
| QG0 − R0 (Gemma as classifier, Qwen's text) | +1.21 [+0.53, +1.92] · 0.074 | +1.37 [+0.25, +2.41] · 0.115 | 86% / 71% |
| G0 − X0 (Gemma as classifier, Gemma's text) | +1.02 [+0.45, +1.59] · 0.015 | +1.44 [+0.39, +2.45] · 0.270 | 72% / 75% |
| X0 − R0 (Gemma's text, Qwen classifier) | +0.39 [−0.10, +0.91] · 0.104 | +0.49 [−0.34, +1.30] · 0.315 | 28% / 25% |
| G0 − QG0 (Gemma's text, Gemma classifier) | +0.20 [−0.19, +0.60] · 1.000 | +0.57 [−0.04, +1.15] · 1.000 | 14% / 29% |
| G0 − R0 (both, Exp 1) | +1.41 [+0.68, +2.14] | +1.94 [+0.70, +3.10] | |
| Interaction (G0 − QG0) − (X0 − R0) | −0.19 (1/5 seeds > 0) · t p 0.43 | +0.08 (2/5) · t p 0.89 | |

**What it shows.** About three quarters of Gemma's gain comes from Gemma as the classifier, in both orders: +1.02 to
+1.21 AUROC and +1.37 to +1.44 AUPRC, better on 5/5 seeds in AUROC and both CIs above 0. Gemma's text adds +0.20 to
+0.39 AUROC and +0.49 to +0.57 AUPRC, with CIs that include 0. The interaction is near zero (p 0.43 / 0.89), so the two
roles add up and the order of the swaps barely matters. Macro-F1 separates none of the cells. Gemma's text also gives
Qwen 33% shorter prompts (521 vs 782 tokens) at no loss. Stack drift on seed 0: Macro-F1 +0.51, AUROC −0.03, AUPRC
−0.19. Holm = paired-t p over the 4 role comparisons × 3 metrics; CIs are paired bootstrap over the 7,425 test reviews.
QG0 trained alone in 1.5 h per run (peak 27.5 GB).

```bash
nohup bash scripts/x0_queue.sh >> logs/gemma_swap/x0_queue.log 2>&1 &   # 5 X0 seeds + par2_R0 seed 0, 2 lanes on 1 GPU
(cd scripts && ../../envs/dgp-fa/bin/python x0_analysis.py)              # -> results/gemma_swap/exp4.{md,json}
setsid nohup bash scripts/qg0_queue.sh > /dev/null 2>&1 < /dev/null &     # QG0: prompt build + 5 seeds, one lane
(cd scripts && ../../envs/dgp-fa/bin/python qg0_analysis.py)             # -> results/gemma_swap/exp4_2x2.{md,json}
python tools/x0-dashboard/server.py 18790                                 # read-only progress page (127.0.0.1)
```

The queue uses `HF_HOME=/workspace/hf_cache` (override with `HF_HOME_DGP`) because the instance's default Hugging Face
cache was damaged; it holds Qwen3-8B `b968826` (the revision R0 used) and DeBERTa-v3-base. Write-up: experiment E14 in
the DGP report; the fourth cell is E18.

## 23. Hyperparameter search (E19 running since 2026-11-03; E20 dropped)

**Scope (owner decision, 2026-11-03).** The paper tunes B × K × M × LoRA rank × dropout × lr (1,728 points per dataset,
arXiv v1 §5.1), selects by mean validation AUROC and reports none of the chosen values. The proposal
([`research/hparam_search_plan.md`](research/hparam_search_plan.md)) staged it; the owner approved:

| Stage | What runs | Runs | GPU-h (est.) |
|---|---|---|---|
| **S1 = E19** epoch-selection rule | default configuration, seeds 0–4, YelpChi and AmazonVideo | 10 | ~43 |
| ~~S2-Y = E20~~ YelpChi learning-rate sweep | approved, then **dropped by the owner at 10:54Z on 2026-11-03** (no fine-tuning settings to be tuned; finish sooner) before any of its runs started | 0 | 0 |
| not run (owner decision) | S2 rank × lr grid on AmazonVideo, S3 confirmation, M sweep, dropout | — | — |

K was already swept (§17, flat) and B is the budget study (§15).

**Why the epoch rule first.** All 178 existing training logs were re-read (no GPU): on each, the epoch kept by the
paper's rule (lowest validation loss, early stopping with patience 2) was compared with the epoch of best validation AUROC.

| Runs | n | Best val-AUROC epoch ≠ kept epoch | Mean Δ val AUROC | Mean Δ val AUPRC | Best val AUROC at the last epoch run |
|---|---|---|---|---|---|
| AmazonVideo, Qwen3-8B classifier | 118 | 56 (47%) | +0.40 | +0.60 | 5 |
| AmazonVideo, Gemma-2-9B classifier | 55 | 33 (60%) | +0.41 | +0.21 | 5 |
| **YelpChi, Qwen3-8B classifier** | **5** | **5 (100%)** | **+4.02** | **+6.67** | **5** |

On YelpChi training stopped while validation AUROC was still rising in every seed. These are validation numbers on
the set that picks the epoch, so they overstate any test gain; the test set decides it in E19.

**Protocol.** `training.selection_rules` (new, `src/dgp_repro/training/trainer.py`) trains all 10 epochs and tracks several
rules on one trajectory: `loss_es2` (the paper's rule as reconstructed, = every earlier result), `auroc_all` (best val
AUROC over 10 epochs), `auroc_es2` / `auroc_es3` (early stopping on val AUROC). The adapter state of each kept epoch is
held in memory; the test set is scored once per distinct kept epoch (`predictions_epoch<e>.csv`), and `metrics.json`
holds every rule under `selection`. Top-level metrics stay the paper rule's, so E19 runs remain comparable with
everything above. With `selection_rules` unset training is unchanged (unit test: the paper rule reproduces the legacy
run's epoch, history and predictions exactly; 108 tests pass). Configs `configs/experiments/sel/`; queue
`scripts/sel_queue.sh`, from 10:54Z `scripts/sel_queue2.sh` (one A100, two lanes, YelpChi first); events `results/sel/status.jsonl`; dashboard
`tools/x0-dashboard/` (queue "Hyperparameter search").

```bash
setsid nohup bash scripts/sel_queue.sh > /dev/null 2>&1 < /dev/null &      # resumable
```

**Reporting.** The paper-rule numbers stay the reproduction result; validation-AUROC epoch selection departs from the
paper's stated "early stopping on validation loss" and is reported as a recorded deviation (decision D14). No choice
uses the test set. E19 (rule vs paper rule, per dataset) will enter
`scripts/stats_standard.py` as its own family.

**Status.** Started 2026-11-03T07:13Z; the instance restarted at about 07:41Z and killed the first two runs in epoch 1
(no output lost beyond them), so the queue was relaunched at 08:23Z. At 10:54Z the owner dropped E20 to finish sooner;
`scripts/sel_queue2.sh` took over the two running YelpChi seeds and runs the rest of E19 (8 runs left). From then on the
validation set goes through the model once per epoch instead of twice (loss and scores from the same forward pass,
identical numbers, unit-tested), which shortens the queued runs. Measured: 0.46 h per YelpChi epoch with two lanes.
Expected to finish about 2026-11-04T08:00Z (live ETA on the dashboard). Results go here
and into the report as E19–E20.

## 24. Reproducibility

Every run writes `results/raw/<run_id>/run_manifest.json` (git commit, versions, GPU, full config, prompt version,
K, M, budgets, LoRA settings, seed, runtime, peak memory) and `predictions.csv`, from which `scripts/evaluate.py`
recomputes all metrics. Caches carry metadata sidecars; summaries are cached per node, so a restarted generation job
skips existing lines. Training has no mid-run checkpoint: a stopped run restarts from epoch 0. Seeds are set centrally.
Test labels are read only by the split builder and evaluation.

**One statistics standard (2026-11-01).** The studies were analysed with different tools as they ran (bootstrap CIs
over test reviews, McNemar, Holm only from Exp 4 on). `python scripts/stats_standard.py` re-tests every paired
comparison of E2–E18 the same way: seed as the unit (5 paired seeds), two-sided paired t-test, 95% t-interval, Holm over
the 3 metrics × the comparisons of each experiment ("family") and over all 114 tests ("global") →
[`results/stats/all_comparisons.md`](results/stats/all_comparisons.md).

| Experiment | Comparisons | Tests | p < 0.05 | Holm family < 0.05 | Holm global < 0.05 |
|---|---|---|---|---|---|
| E2 FLAG text / residual arms vs Sept. baseline | 6 | 18 | 5 | 0 | 0 |
| E3 K = 1, 3 vs K = 2 | 2 | 6 | 0 | 0 | 0 |
| E8 dr prompts vs R0 | 4 | 12 | 2 | 0 | 0 |
| E10 F0 re-run vs R0 | 1 | 3 | 0 | 0 | 0 |
| E11 Gemma vs Qwen, 5 arms | 5 | 15 | 8 | 3 (G0 AUROC +1.41; G1b AUPRC +2.10; G2b AUPRC +1.87) | 0 |
| E12 zero-shot prompt, Qwen | 5 | 15 | 0 | 0 | 0 |
| E13 zero-shot prompt, Gemma | 5 | 15 | 9 | 1 (GS2b AUPRC −2.00) | 0 |
| E14 writer vs classifier | 3 | 9 | 6 | 3 (G0 − X0 AUROC; G0 − R0 AUROC, AUPRC) | 0 |
| E15 target-only LLM and MLP | 3 | 9 | 7 | 7 | 4 (all vs the MLP) |
| E16 DGP vs MLP on YelpChi | 1 | 3 | 2 | 2 (AUROC +3.72; AUPRC −2.62) | 0 |
| E17 MDK vs random neighbours (w/o MDK) | 1 | 3 | 0 | 0 | 0 |
| E18 fourth writer × classifier cell (QG0) | 2 | 6 | 2 | 1 (QG0 − R0 AUROC +1.21) | 0 |

Read with this standard, no prompt-side effect (E2, E8, E12) survives correction, the K sweep, the F0 re-run and the w/o MDK ablation (E17) are
null, and Gemma's gain (E11) and the zero-shot penalty for Gemma (E13) survive in some arms but not all: they rest on
the consistent sign across arms (E11 AUPRC better on 24 of 25 paired seeds) and on the bootstrap CIs of the studies'
own analyses rather than on any single corrected seed-level test. Five seeds give little power: a seed-level test
detects only differences several times the seed-to-seed std of the difference. Seed-level tests leave out test-set
sampling noise, which the bootstrap CIs include, so the two answer different questions and both are reported.

## 25. Decisions and known gaps

Reconstruction decisions (full evidence in [`research/evidence_matrix.md`](research/evidence_matrix.md),
[`research/reproduction_gaps.md`](research/reproduction_gaps.md) and
[`research/FINAL_REPRODUCTION_REPORT.md`](research/FINAL_REPRODUCTION_REPORT.md)):

| # | Gap in the paper | Decision |
|---|---|---|
| D1 | AmazonVideo R-S-R: prose says same product + rating + week | Drop "same product"; only then does the graph match Table 1 |
| D2 | Unvoted reviews: unlabelled or benign? | Benign; only this fits the paper's AUROC/AUPRC pairs |
| D3 | Qwen3 thinking mode never mentioned | Disabled; the first-token classifier is impossible otherwise |
| D4 | "w/o MDK" replacement never described | Random M neighbours at the same budget |
| D5 | MDK input differs between paper versions | Camera-ready: DeBERTaV3 ⊕ numeric |
| D6 | Selected K, M, B, LoRA rank/dropout, lr not published | Reconstructed defaults K=2, M=4, B=10, rank 8, α 16, dropout 0.05, lr 1e-4 |
| D7 | Threshold for Macro-F1 not stated | Threshold that maximises validation F1 (0.5 gives ≈49.5) |
| D8 | Software stack changed for the extension study | One parity run on the new stack; arms compared seed by seed with the original baseline |
| D9 | K=3 prompts exceed the default sequence length | `max_seq_len` 8192 for K3 only |
| D10 | FLAG's discriminative prompt mostly states a class verdict (§19) | A no-verdict variant (R1b, R2b) next to FLAG's own prompt |
| D11 | Gemma-2 with eager attention runs out of memory on batches of long reviews (§22) | Length-capped batches (`max_batch_tokens_sq`); greedy decoding, so only padding changes |
| D12 | DGP reads the first answer token; the prototype zero-shot prompt asked for `<label>…</label>` (§22) | The zero-shot prompt asks for one word, `Yes` or `No` |
| D13 | Exp 4 must reuse G0's Gemma text exactly, but cache keys include the classifier's tokenizer (§22) | Read-only `cached` summarizer backend that fails on a miss; `source_mdk_cache` reuses G0's trimming cache (neighbours checked identical) |
| D14 | The paper early-stops on validation loss but selects hyperparameters by validation AUROC, and gives no patience (§23) | Paper rule (patience 2) stays primary; E19 adds validation-AUROC epoch selection as a recorded deviation, on the same training runs |

The paper also never states LoRA alpha, the maximum sequence length, the metapath-summary or final classification
prompts, or how the split was sampled. The two paper versions differ in method details:
[`research/version_divergence.md`](research/version_divergence.md).

**Still open:** the hyperparameter search is running (E19, §23; E20 and the rest of the grid were not run), the
paper's other ablations (w/o PathSumm / TextSumm / NumSumm) and budget / task-aware studies on GPU, why DGP trails the paper on
YelpChi by 18 AUPRC (§11; E19 tests the first candidate), ConsisGAD 5-seed run and the other
baselines. From the follow-up studies:
which part of the zero-shot prompt hurts Gemma (§22), and why two Gemma runs sharing a GPU are so slow.

## 26. Citation

```bibtex
@inproceedings{li2026dgp,
  title     = {DGP: A Dual-Granularity Prompting Framework for Fraud Detection with Graph-Enhanced LLMs},
  author    = {Li, Yuan and Hu, Jun and Hooi, Bryan and He, Bingsheng and Chen, Cheng},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026}
}
```

Cite the original datasets as well: Rayana & Akoglu (KDD 2015) for YelpChi; McAuley & Leskovec (RecSys 2013)
for the Amazon reviews.
