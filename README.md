# DGP Reproduction

A reproduction of **"DGP: A Dual-Granularity Prompting Framework for Fraud Detection with Graph-Enhanced LLMs"**
(Yuan Li, Jun Hu, Bryan Hooi, Bingsheng He, Cheng Chen — AAAI 2026;
[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/38541) · [arXiv:2507.21653](https://arxiv.org/abs/2507.21653)).

> **Status (2026-09-17): DGP reproduced on AmazonVideo (5 seeds, real Qwen3-8B + LoRA on GPU).**
> The official repository ([Xtra-Computing/DGP](https://github.com/Xtra-Computing/DGP)) contains only a README —
> code release is pending industry-partner approval — so **every DGP component here is a `PAPER_RECONSTRUCTION`**,
> built from both paper versions and traced to evidence in [`research/`](research/).
> A full 5-seed GPU run (2× A100 80GB) is now complete: Macro-F1 65.16 ± 0.52 vs. the paper's 66.91 ± 0.13
> (**CLOSE**, −2.6%), AUROC 75.07 ± 0.50 vs. 77.32 ± 0.11 (**DEVIATES**, −2.9%), AUPRC 32.33 ± 0.65 vs. 34.63 ± 0.24
> (**DEVIATES**, −6.6%) — using reconstructed defaults (K=2, M=4, B=10, LoRA rank=8) with **no hyperparameter
> search**, which is the leading suspect for the remaining gap (`research/compute_plan.md` §6). The MLP baseline
> remains partially reproduced. See [`research/reproduction_matrix.md`](research/reproduction_matrix.md).

---

## 1. DGP Reproduction

What exists and works today:

- The **AmazonVideo** graph rebuilt from raw data, matching the paper's Table 1 **exactly** (37,126 nodes,
  9,883,406 edges, 4,379 frauds, 1,299/1,299/7,425 split).
- The full DGP pipeline — metapaths, Markov-diffusion trimming, bi-level summarization, numeric summaries,
  prompt construction, Qwen3-8B + LoRA first-token classification — run end to end on GPU, not just smoke-tested.
- **DGP's main result reproduced**: 5 seeds, real Qwen3-8B summarization (119,829 node + metapath summaries)
  and LoRA fine-tuning on 2× A100 80GB. See §16.
- Every experiment of the paper wired to a script: main table, ablations, budget sweep, task-aware study,
  token accounting, complexity analysis, comparison with the paper.
- 77 unit tests and a 14-check CPU smoke test, including a check against the real Qwen3 tokenizer.

What is outstanding: the ablations/budget/task-aware studies and the hyperparameter grid search on GPU
(compute decision pending, `research/compute_plan.md` §6), the YelpReviews data (email-gated), and the
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
src/dgp_repro/
  data/ mdk/ embeddings/ summarization/ prompts/ models/ training/ metrics/ evaluation/ baselines/ utils/
  pipeline.py      graph -> prompts (the paper's Figure 3)
  experiment.py    prompts -> train -> evaluate -> records
prompts/dgp/     versioned prompt templates + PROVENANCE.md
scripts/         prepare_data, build_mdk, generate_summaries, train, evaluate, run_ablation,
                 run_sensitivity, reproduce, compare_with_paper, analyze_complexity, fetch_baselines, smoke_test
methods/         official baselines, fetched at pinned commits (never committed) + README.md provenance
research/        paper specification, version divergence, evidence matrix, provenance, gaps, compute plan
docs/            installation, reproduction, datasets, architecture, troubleshooting
tests/           unit/ (77 tests)
results/         raw/<run_id>/, aggregated/, tables/, figures/, token_usage/, complexity/
```

## 4. Datasets

| Dataset | Status |
|---|---|
| AmazonVideo | **Ready.** Automatic download; graph matches Table 1 exactly |
| YelpReviews | **Blocked.** Original YelpChi release with text, available only by email request |
| E-Commerce, LifeService | **Not reproducible.** Proprietary ByteDance data |

Details, including two places where the data contradict the paper's prose: [`docs/datasets.md`](docs/datasets.md).

## 5. Environment

Python 3.11, PyTorch ≥ 2.4, transformers ≥ 4.56 (Qwen3 support and the `dtype=` loading argument),
PEFT, scikit-learn (the paper computes all metrics with scikit-learn). DGP's own graph code uses
`scipy.sparse`; DGL is needed only by some baselines. Each baseline gets its own environment.

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

One A100 80GB is recommended (the authors' follow-up paper used exactly one with the same recipe).
**Verified**: 2× A100-SXM4-80GB, CUDA 12.8, torch 2.11.0+cu128, transformers 5.17.0, peft 0.21.0 — one seed's
LoRA training peaked at **19.7 GB** GPU memory, so a single A100 80GB (or a 48 GB card) has comfortable headroom;
the second GPU was only used to run two seeds in parallel. Full notes: [`docs/installation.md`](docs/installation.md).

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
python -m pytest
```

Checks graph loading, metapath adjacency, MDK, Top-M, both summarization levels, numeric aggregation,
prompt construction, all four ablation variants, Yes/No token ids at the real Qwen3 chat boundary,
first-token training and readout, metrics and token accounting.

## 11. Yelp reproduction

Blocked on data access. Once `datasets/raw/yelpchi/reviews.jsonl` exists:

```bash
python scripts/prepare_data.py --dataset yelpchi --probe
python scripts/reproduce.py --dataset yelpchi --device cuda:0 --seeds 0 1 2 3 4
```

## 12. Amazon reproduction

```bash
python scripts/reproduce.py --dataset amazonvideo --device cuda:0 --seeds 0 1 2 3 4
```

Run the ~1 GPU-hour calibration in [`docs/reproduction.md`](docs/reproduction.md) first.

## 13. Baselines

MLP and the target-only LLM are implemented in this repository. The ten official baselines are fetched at
verified commits (`python scripts/fetch_baselines.py`) and receive our exact graph and split through an export
adapter. ConsisGAD's official code already runs unmodified on our AmazonVideo graph and split; the others are not yet adapted. PMP, GAAP and FLAG publish **no licence** and must never be
redistributed. See [`methods/README.md`](methods/README.md).

## 14. Ablations

```bash
python scripts/run_ablation.py --dataset amazonvideo --device cuda:0
```

`w/o MDK`, `w/o PathSumm`, `w/o TextSumm`, `w/o NumSumm` (Figure 4). The paper prints no numbers for this figure,
so only its stated ordering can be checked.

## 15. Budget sensitivity

```bash
python scripts/run_sensitivity.py --study budget --dataset amazonvideo --device cuda:0      # B ∈ {5,10,20,40,80}
python scripts/run_sensitivity.py --study task_aware --dataset amazonvideo --device cuda:0  # Table 3
```

## 16. Results

**DGP, AmazonVideo, 5 seeds (GPU_REPRODUCTION, real Qwen3-8B + LoRA):**

| Metric | Paper | Ours | Diff | Status |
|---|---|---|---|---|
| Macro-F1 | 66.91 ± 0.13 | 65.16 ± 0.52 | −1.75 (−2.6%) | CLOSE |
| AUROC | 77.32 ± 0.11 | 75.07 ± 0.50 | −2.25 (−2.9%) | DEVIATES |
| AUPRC | 34.63 ± 0.24 | 32.33 ± 0.65 | −2.30 (−6.6%) | DEVIATES |

Reconstructed defaults from `configs/models/dgp.yaml` (K=2, M=4, B_node=B_meta=10, LoRA rank=8, alpha=16,
dropout=0.05, lr=1e-4), **no hyperparameter search** — the paper publishes its search grid but not the winning
values (`research/compute_plan.md` §6). Our std across seeds (0.5–0.65) is notably higher than the paper's
(0.11–0.24), consistent with an untuned config rather than a pipeline bug, given Macro-F1 lands within 2.6%.
Full manifests (config, git commit, GPU, peak memory, runtime) in
`results/raw/dgp_amazonvideo_seed0_seed{0..4}_*/run_manifest.json`; details in
`results/tables/comparison_with_paper.md`.

**MLP baseline, AmazonVideo, 5 seeds (CPU):** Macro-F1 59.86 ± 0.51 · AUROC 68.35 ± 0.49 · AUPRC 25.14 ± 1.06, against the paper's 61.74 / 70.47 / 26.55 → `PARTIALLY_REPRODUCED`.

`results/tables/main_results.md` is written only from real runs; CPU debug runs are kept in `results/tables/debug_runs.md` and are never compared with the paper.

Measured (AmazonVideo, real data, real Qwen3-8B): mean review length **115.7 Qwen3 tokens**; average
degree **133.1** (paper: 133); at K=2 a full-neighbour prompt averages **~1.5M raw tokens** per review versus a
**measured mean final DGP prompt of 781.5 tokens** (compression ratio ≈1922×, 99.9% token reduction;
`results/token_usage/token_usage_dgp_amazonvideo_seed0.json`) — higher than the paper-formula estimate of ~236
tokens that appeared in earlier (pre-GPU-run) complexity analysis. `results/complexity/`.

Paper targets for comparison only: [`research/reported_results.csv`](research/reported_results.csv).

## 17. Reproducibility

Every run writes `results/raw/<run_id>/run_manifest.json` (git commit, versions, GPU, full config, prompt version,
K, M, budgets, LoRA settings, seed, runtime, peak memory) and `predictions.csv`, from which `scripts/evaluate.py`
recomputes all metrics. Caches carry metadata sidecars. Seeds are set centrally. Test labels are read only by
the split builder and evaluation.

## 18. Known gaps

The paper publishes its search grid but not the selected K, M, B, LoRA rank/dropout or learning rate. It never
states LoRA alpha, maximum sequence length, the metapath-summary or final classification prompts, how the
split was sampled, or that Qwen3's thinking mode must be off. Complete register with evidence and chosen defaults:
[`research/evidence_matrix.md`](research/evidence_matrix.md) and
[`research/reproduction_gaps.md`](research/reproduction_gaps.md). The two paper versions differ in method details:
[`research/version_divergence.md`](research/version_divergence.md).

## 19. Citation

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
