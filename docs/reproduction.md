# Reproduction guide

All commands run from the repository root. Every stage caches its output and every finished seed is
skipped on rerun, so any command can be interrupted and restarted.

## 0. Check the installation (CPU, ~30 s)

```bash
python scripts/smoke_test.py --real-tokenizer
python -m pytest
```

## 1. Calibrate on the GPU first (~1 GPU-hour)

Measures real throughput so the cost plan in `research/compute_plan.md` can be updated from measurements:

```bash
python scripts/prepare_data.py --dataset amazonvideo
python scripts/build_mdk.py --dataset amazonvideo --device cuda:0
python scripts/generate_summaries.py --dataset amazonvideo --device cuda:0 --until node_summaries
python scripts/train.py --dataset amazonvideo --device cuda:0 --seeds 0 --set training.max_epochs=1 \
       --experiment calibration_amazonvideo
```

Delete `results/raw/calibration_*` afterwards; it is not a result.

## 2. Hyperparameters

`configs/models/dgp.yaml` marks every value as PAPER, GRID or RECON. GRID values are placeholders:
the paper publishes the search grid but not the chosen values. Decide the search strategy first
(`research/compute_plan.md` §6), then override with `--set`, for example

```bash
python scripts/train.py --dataset amazonvideo --device cuda:0 --seeds 0 \
       --set dgp.K=3 dgp.M=8 classifier.lora.rank=16 training.learning_rate=3e-5 --experiment grid_amazon_K3_M8_r16_lr3e-5
```

Select by **mean validation AUROC** (the paper's criterion): `results/aggregated/<experiment>.json → val.auroc_mean`.
Never select on test metrics.

## 3. Main result (Table 2 row "DGP"), AmazonVideo

```bash
python scripts/reproduce.py --dataset amazonvideo --device cuda:0 --seeds 0 1 2 3 4
```

This runs: prepare → MDK → summaries → 5 × (LoRA training, validation, test) → tables → comparison.
Equivalent individual steps:

```bash
python scripts/prepare_data.py --dataset amazonvideo
python scripts/build_mdk.py --dataset amazonvideo --device cuda:0
python scripts/generate_summaries.py --dataset amazonvideo --device cuda:0
python scripts/train.py --dataset amazonvideo --device cuda:0 --seeds 0 1 2 3 4
python scripts/evaluate.py
python scripts/compare_with_paper.py
```

## 4. Everything for one dataset

```bash
python scripts/reproduce.py --dataset amazonvideo --device cuda:0 --seeds 0 1 2 3 4 \
       --with-llm-baseline --with-ablation --with-budget --with-task-aware
```

| Paper artefact | Script | Output |
|---|---|---|
| Table 2, DGP and LLM rows | `train.py` (+ `--overlay configs/experiments/baselines/llm_target_only.yaml`) | `results/tables/main_results.{csv,md}` |
| Figure 4 ablations | `run_ablation.py` | `results/tables/ablation_<ds>.*`, `results/figures/ablation_<ds>.png` |
| Figure 5 budget | `run_sensitivity.py --study budget` | `results/budget_sensitivity.csv`, `results/figures/budget_sensitivity_<ds>.png` |
| Table 3 task-aware | `run_sensitivity.py --study task_aware` | `results/task_aware_comparison.csv` |
| Token efficiency | produced by every run | `results/token_usage/token_usage_<experiment>.{csv,json}` |
| Complexity analysis | `analyze_complexity.py` | `results/complexity/` |
| Comparison with paper | `compare_with_paper.py` | `results/tables/comparison_with_paper.*` |

## 5. YelpReviews

Blocked until the dataset is obtained (`docs/datasets.md`). Afterwards the same commands work with
`--dataset yelpchi`.

## 6. Per-run records

`results/raw/<run_id>/run_manifest.json` records git commit, library versions, CUDA and GPU, the full
configuration (models, prompt version, K, M, B_node, B_meta, LoRA, seed, epochs, learning rate, batch size,
accumulation), runtime and peak memory. `predictions.csv` holds per-node probabilities, from which
`evaluate.py` recomputes every metric.

## CPU_DEBUG equivalents

Add `--mode smoke` to any script (mock summarizer, hashing embedder, mock LLM). Results go into
`results/tables/debug_runs.*` and are never compared with the paper.
