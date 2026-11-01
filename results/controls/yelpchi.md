# YelpChi: DGP against the MLP and the paper

YelpChi (67,395 reviews, 17,486,608 edges, the paper's Table 1 exactly), test set 13,479 reviews, seeds 0-4. Mean ± std over seeds (ddof=0). Δ = paired per-seed difference; CI = paired bootstrap 95% interval over test reviews. Statistics: scripts/dr_analysis.py:compare.

| run | what | seeds | Macro-F1 | AUROC | AUPRC | paper (F1 / AUROC / AUPRC) | ours − paper (AUROC / AUPRC) |
|---|---|---|---|---|---|---|---|
| dgp_yelpchi | DGP (Qwen3-8B summaries + Qwen3-8B LoRA classifier) | 0 | | | | 69.07 / 84.28 / 48.87 | |
| mlp_yelpchi | MLP baseline | 5 | 63.64 ± 0.43 | 74.66 ± 0.71 | 33.67 ± 1.01 | 62.09 / 75.00 / 32.24 | -0.34 / +1.43 |

Other paper rows for reference (not run here): ConsisGAD (paper's best baseline) 67.33 / 82.12 / 42.11; LLM (target only) 60.79 / 71.18 / 30.45; FLAG 64.79 / 73.98 / 38.45.

