# Disc/resid study: deviation from the reference run

Test set, AmazonVideo. Mean ± std over seeds (ddof=0). Δ = paired per-seed difference to the reference; 'better' = seeds where the arm is higher. CI = paired bootstrap 95% interval of the seed-averaged Δ (2000 resamples of the 7,425 test reviews).

Seeds finished: R0 [0, 1], R1 [0], R1b [0], R2 [0], R2b [0]

## Arms vs R0 (same stack, same seeds)

| arm | seeds | Macro-F1 | AUROC | AUPRC | Δ F1 (better) | Δ AUROC (better) | Δ AUPRC (better) | CI Δ AUROC | CI Δ AUPRC |
|---|---|---|---|---|---|---|---|---|---|
| R1 | 1 | 65.39 ± 0.00 | 75.39 ± 0.00 | 32.37 ± 0.00 | +0.61 (1/1) | +0.26 (1/1) | -0.41 (0/1) | [-0.33, +0.89] | [-1.17, +0.38] |
| R1b | 1 | 65.97 ± 0.00 | 75.45 ± 0.00 | 33.07 ± 0.00 | +1.18 (1/1) | +0.32 (1/1) | +0.29 (1/1) | [-0.14, +0.76] | [-0.43, +0.98] |
| R2 | 1 | 65.06 ± 0.00 | 75.19 ± 0.00 | 32.64 ± 0.00 | +0.28 (1/1) | +0.06 (1/1) | -0.14 (0/1) | [-0.55, +0.68] | [-0.95, +0.72] |
| R2b | 1 | 65.44 ± 0.00 | 75.34 ± 0.00 | 32.32 ± 0.00 | +0.66 (1/1) | +0.21 (1/1) | -0.46 (0/1) | [-0.22, +0.65] | [-1.21, +0.22] |

## R0 vs the original September run (stack parity)

| arm | seeds | Macro-F1 | AUROC | AUPRC | Δ F1 (better) | Δ AUROC (better) | Δ AUPRC (better) | CI Δ AUROC | CI Δ AUPRC |
|---|---|---|---|---|---|---|---|---|---|
| R0 | 2 | 65.08 ± 0.29 | 75.34 ± 0.21 | 32.91 ± 0.13 | -0.23 (1/2) | +0.01 (2/2) | +0.28 (1/2) | [-0.26, +0.29] | [-0.11, +0.69] |

## Arms vs the original September run

| arm | seeds | Macro-F1 | AUROC | AUPRC | Δ F1 (better) | Δ AUROC (better) | Δ AUPRC (better) | CI Δ AUROC | CI Δ AUPRC |
|---|---|---|---|---|---|---|---|---|---|
| R1 | 1 | 65.39 ± 0.00 | 75.39 ± 0.00 | 32.37 ± 0.00 | -0.31 (0/1) | +0.27 (1/1) | -0.42 (0/1) | [-0.31, +0.90] | [-1.20, +0.36] |
| R1b | 1 | 65.97 ± 0.00 | 75.45 ± 0.00 | 33.07 ± 0.00 | +0.26 (1/1) | +0.33 (1/1) | +0.28 (1/1) | [-0.12, +0.76] | [-0.43, +0.95] |
| R2 | 1 | 65.06 ± 0.00 | 75.19 ± 0.00 | 32.64 ± 0.00 | -0.64 (0/1) | +0.06 (1/1) | -0.15 (0/1) | [-0.52, +0.70] | [-0.97, +0.73] |
| R2b | 1 | 65.44 ± 0.00 | 75.34 ± 0.00 | 32.32 ± 0.00 | -0.26 (0/1) | +0.21 (1/1) | -0.47 (0/1) | [-0.21, +0.66] | [-1.17, +0.23] |

## Prediction agreement with R0 (mean over seeds)

| arm | Spearman p_fraud | label agreement | wrong→right | right→wrong | seeds with McNemar p < 0.05 |
|---|---|---|---|---|---|
| R1 | 0.920 | 97.9% | 68 | 88 | 0/1 |
| R1b | 0.960 | 94.8% | 110 | 279 | 1/1 |
| R2 | 0.922 | 98.0% | 77 | 72 | 0/1 |
| R2b | 0.962 | 90.7% | 177 | 513 | 1/1 |

## Cost

| arm | prompt tokens mean / max | train h / seed | peak GPU GB |
|---|---|---|---|
| R0 | 781.5 / 1902 | 1.12 | 19.7 |
| R1 | 815.2 / 1981 | 1.16 | 19.8 |
| R1b | 811.2 / 1981 | 1.16 | 19.8 |
| R2 | 815.2 / 1981 | 1.27 | 19.9 |
| R2b | 811.2 / 1981 | 1.27 | 19.9 |
