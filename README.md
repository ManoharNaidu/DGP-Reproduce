# DGP Reproduction

A reproduction of **"DGP: A Dual-Granularity Prompting Framework for Fraud Detection with Graph-Enhanced LLMs"**
(Yuan Li, Jun Hu, Bryan Hooi, Bingsheng He, Cheng Chen — AAAI 2026;
[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/38541) · [arXiv:2507.21653](https://arxiv.org/abs/2507.21653)).

> **Status (2026-10-15): DGP reproduced on AmazonVideo, and the extension study is finished.**
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
- Every experiment of the paper wired to a script: main table, ablations, budget sweep, task-aware study,
  token accounting, complexity analysis, comparison with the paper.
- 84 unit tests and a 14-check CPU smoke test, including a check against the real Qwen3 tokenizer.
- A read-only progress dashboard (`tools/dgp-dashboard/`, v0.2.0).

What is outstanding: the paper's ablations / budget / task-aware studies and the hyperparameter grid search on GPU
(compute decision pending, `research/compute_plan.md` §6), the YelpReviews data (available only by email request), and the
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
src/dgp_repro/
  data/ mdk/ embeddings/ summarization/ prompts/ models/ training/ metrics/ evaluation/ baselines/ utils/
  pipeline.py      graph -> prompts (the paper's Figure 3)
  experiment.py    prompts -> train -> evaluate -> records
prompts/dgp/     versioned prompt templates + PROVENANCE.md; extension templates + PROVENANCE_EXT.md
scripts/         prepare_data, build_mdk, generate_summaries, train, evaluate, run_ablation, run_sensitivity,
                 run_grid, run_mlp, reproduce, compare_with_paper, analyze_complexity, fetch_baselines,
                 export_baseline_data, smoke_test, ext_queue (extension scheduler), ext_drain, make_handoff
analysis/        mdk_comparison.py + MDK_COMPARISON.md (DGP vs FLAG-MD), neighbor_analysis.py, out/
methods/         official baselines, fetched at pinned commits (never committed) + README.md provenance
research/        paper specification, version divergence, evidence matrix, provenance, gaps, compute plan,
                 FINAL_REPRODUCTION_REPORT.md
docs/            installation, reproduction, datasets, architecture, troubleshooting,
                 Markov_Diffusion_DGP_vs_FLAG.docx
tools/           dgp-dashboard/ (read-only progress dashboard)
tests/           unit/ (84 tests)
results/         raw/<run_id>/, aggregated/, tables/, figures/, token_usage/, complexity/, ext/ (queue state)
checkpoints/     LoRA adapters, one per run (adapter_model.safetensors + adapter_config.json)
HANDOFF_4GPU.md  hand-off note from the 2-GPU to the 4-GPU instance (now historical, see §17)
```

## 4. Datasets

| Dataset | Status |
|---|---|
| AmazonVideo | **Ready.** Automatic download; graph matches Table 1 exactly |
| YelpReviews | **Blocked.** Original YelpChi release with text, available only by email request. A user-supplied `YelpChi.mat` was rejected: it is the CARE-GNN variant and has no text |
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

YelpReviews: see the email template in [`docs/datasets.md`](docs/datasets.md).

## 10. Quick smoke test

```bash
python scripts/smoke_test.py --real-tokenizer    # CPU, ~30 s, DEBUG_ONLY
python -m pytest tests/unit -q                   # 84 tests
```

Checks graph loading, metapath adjacency, MDK, Top-M, both summarization levels, numeric aggregation,
prompt construction, all four ablation variants, Yes/No token ids at the real Qwen3 chat boundary,
first-token training and readout, metrics and token accounting.

## 11. Yelp reproduction

Blocked until the data arrives. Once `datasets/raw/yelpchi/reviews.jsonl` exists:

```bash
python scripts/prepare_data.py --dataset yelpchi --probe
python scripts/reproduce.py --dataset yelpchi --device cuda:0 --seeds 0 1 2 3 4
```

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

## 14. Ablations

```bash
python scripts/run_ablation.py --dataset amazonvideo --device cuda:0
```

`w/o MDK`, `w/o PathSumm`, `w/o TextSumm`, `w/o NumSumm` (Figure 4). The paper prints no numbers for this figure,
so only its stated ordering can be checked. **Not yet run on GPU.**

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
the paper's 61.74 / 70.47 / 26.55 → `PARTIALLY_REPRODUCED`.

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
adapters in `checkpoints/`. **Note:** `results/aggregated/ext_*.json` and `parity_dgp.json` were overwritten by
each single-seed job, so they hold one seed only; the table above is computed from the five `metrics.json` files per
arm. Re-run `scripts/evaluate.py` per experiment to rebuild the 5-seed aggregates.

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

## 19. Reproducibility

Every run writes `results/raw/<run_id>/run_manifest.json` (git commit, versions, GPU, full config, prompt version,
K, M, budgets, LoRA settings, seed, runtime, peak memory) and `predictions.csv`, from which `scripts/evaluate.py`
recomputes all metrics. Caches carry metadata sidecars; summaries are cached per node, so a restarted generation job
skips existing lines. Training has no mid-run checkpoint: a stopped run restarts from epoch 0. Seeds are set centrally.
Test labels are read only by the split builder and evaluation.

## 20. Decisions and known gaps

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

The paper also never states LoRA alpha, the maximum sequence length, the metapath-summary or final classification
prompts, or how the split was sampled. The two paper versions differ in method details:
[`research/version_divergence.md`](research/version_divergence.md).

**Still open:** hyperparameter grid search (≈60–90 GPU-hours, staged option in `research/compute_plan.md` §6), the
paper's ablation / budget / task-aware studies on GPU, YelpReviews (data access), ConsisGAD 5-seed run and the other
baselines, and rebuilding the 5-seed aggregate files for the extension arms (§17).

## 21. Citation

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
