# Disc/resid study: deviation from the reference run

Test set, AmazonVideo. Mean ± std over seeds (ddof=0). Δ = paired per-seed difference to the reference; 'better' = seeds where the arm is higher. CI = paired bootstrap 95% interval of the seed-averaged Δ (2000 resamples of the 7,425 test reviews).

Seeds finished: R0 [0, 1, 2, 3, 4], R1 [0, 1, 2, 3, 4], R1b [0, 1, 2, 3, 4], R2 [0, 1, 2, 3, 4], R2b [0, 1, 2, 3, 4]

## Arms vs R0 (same stack, same seeds)

| arm | seeds | Macro-F1 | AUROC | AUPRC | Δ F1 (better) | Δ AUROC (better) | Δ AUPRC (better) | CI Δ AUROC | CI Δ AUPRC |
|---|---|---|---|---|---|---|---|---|---|
| R1 | 5 | 64.79 ± 1.30 | 75.18 ± 0.56 | 32.42 ± 1.17 | +0.04 (4/5) | +0.21 (5/5) | +0.05 (3/5) | [-0.30, +0.72] | [-0.57, +0.68] |
| R1b | 5 | 65.33 ± 0.78 | 75.42 ± 0.58 | 32.41 ± 0.80 | +0.58 (4/5) | +0.45 (5/5) | +0.04 (4/5) | [+0.11, +0.79] | [-0.48, +0.56] |
| R2 | 5 | 64.28 ± 2.60 | 75.23 ± 0.47 | 32.52 ± 0.96 | -0.47 (4/5) | +0.25 (5/5) | +0.15 (3/5) | [-0.27, +0.78] | [-0.54, +0.85] |
| R2b | 5 | 64.78 ± 1.44 | 75.24 ± 0.51 | 32.37 ± 0.79 | +0.03 (4/5) | +0.27 (5/5) | +0.00 (3/5) | [-0.10, +0.67] | [-0.56, +0.57] |

## R0 vs the original September run (stack parity)

| arm | seeds | Macro-F1 | AUROC | AUPRC | Δ F1 (better) | Δ AUROC (better) | Δ AUPRC (better) | CI Δ AUROC | CI Δ AUPRC |
|---|---|---|---|---|---|---|---|---|---|
| R0 | 5 | 64.74 ± 0.45 | 74.97 ± 0.51 | 32.37 ± 0.57 | -0.42 (1/5) | -0.10 (2/5) | +0.04 (3/5) | [-0.28, +0.09] | [-0.19, +0.29] |

## Arms vs the original September run

| arm | seeds | Macro-F1 | AUROC | AUPRC | Δ F1 (better) | Δ AUROC (better) | Δ AUPRC (better) | CI Δ AUROC | CI Δ AUPRC |
|---|---|---|---|---|---|---|---|---|---|
| R1 | 5 | 64.79 ± 1.30 | 75.18 ± 0.56 | 32.42 ± 1.17 | -0.38 (3/5) | +0.11 (4/5) | +0.09 (3/5) | [-0.38, +0.62] | [-0.55, +0.74] |
| R1b | 5 | 65.33 ± 0.78 | 75.42 ± 0.58 | 32.41 ± 0.80 | +0.16 (3/5) | +0.35 (5/5) | +0.08 (3/5) | [+0.01, +0.69] | [-0.43, +0.60] |
| R2 | 5 | 64.28 ± 2.60 | 75.23 ± 0.47 | 32.52 ± 0.96 | -0.89 (3/5) | +0.15 (4/5) | +0.19 (2/5) | [-0.34, +0.68] | [-0.50, +0.93] |
| R2b | 5 | 64.78 ± 1.44 | 75.24 ± 0.51 | 32.37 ± 0.79 | -0.39 (2/5) | +0.17 (3/5) | +0.04 (2/5) | [-0.20, +0.54] | [-0.51, +0.61] |

## Prediction agreement with R0 (mean over seeds)

| arm | Spearman p_fraud | label agreement | wrong→right | right→wrong | seeds with McNemar p < 0.05 |
|---|---|---|---|---|---|
| R1 | 0.917 | 94.8% | 280 | 109 | 4/5 |
| R1b | 0.950 | 95.8% | 190 | 120 | 5/5 |
| R2 | 0.917 | 94.5% | 294 | 112 | 4/5 |
| R2b | 0.945 | 92.8% | 187 | 346 | 3/5 |

## Cost

| arm | prompt tokens mean / max | train h / seed | peak GPU GB |
|---|---|---|---|
| R0 | 781.5 / 1902 | 1.03 | 19.7 |
| R1 | 815.2 / 1981 | 1.08 | 19.8 |
| R1b | 811.2 / 1981 | 1.07 | 19.8 |
| R2 | 815.2 / 1981 | 1.17 | 19.9 |
| R2b | 811.2 / 1981 | 1.17 | 19.9 |
