# Gemma-swap / zero-shot study · Experiment 2

AmazonVideo test set (7,425 reviews). Mean ± std over seeds (ddof=0). Δ = paired per-seed difference (arm − reference); 'better' = seeds where the arm is higher. CI = paired bootstrap 95% interval of the seed-averaged Δ (2000 resamples of the test reviews). McNemar: seeds whose correctness differs at p < 0.05 (each run at its own validation threshold). Statistics: scripts/dr_analysis.py:compare.

## Exp 2 · zero-shot vs original prompt, Qwen (S vs R)

| arm | reference | seeds | Macro-F1 | AUROC | AUPRC | Δ F1 (better) | Δ AUROC (better) | Δ AUPRC (better) | CI Δ AUROC | CI Δ AUPRC | McNemar p<0.05 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S0 | R0 | 5 | 65.68 ± 0.53 | 75.52 ± 0.14 | 32.76 ± 0.44 | +0.94 (5/5) | +0.55 (4/5) | +0.39 (4/5) | [-0.12, +1.27] | [-0.95, +1.71] | 4/5 |
| S1 | R1 | 5 | 65.75 ± 0.42 | 75.44 ± 0.67 | 33.02 ± 0.72 | +0.97 (5/5) | +0.26 (2/5) | +0.60 (2/5) | [-0.36, +0.90] | [-0.55, +1.78] | 4/5 |
| S1b | R1b | 5 | 65.28 ± 0.47 | 75.68 ± 0.12 | 32.72 ± 0.55 | -0.04 (3/5) | +0.26 (3/5) | +0.30 (4/5) | [-0.44, +1.00] | [-0.98, +1.56] | 3/5 |
| S2 | R2 | 5 | 65.07 ± 1.54 | 75.49 ± 0.28 | 33.10 ± 0.83 | +0.79 (5/5) | +0.26 (4/5) | +0.58 (3/5) | [-0.34, +0.87] | [-0.61, +1.67] | 5/5 |
| S2b | R2b | 5 | 65.14 ± 1.03 | 75.68 ± 0.15 | 32.95 ± 0.51 | +0.37 (4/5) | +0.43 (4/5) | +0.58 (4/5) | [-0.28, +1.16] | [-0.74, +1.88] | 5/5 |

## Exp 2 · within zero-shot Qwen (S arms vs S0)

| arm | reference | seeds | Macro-F1 | AUROC | AUPRC | Δ F1 (better) | Δ AUROC (better) | Δ AUPRC (better) | CI Δ AUROC | CI Δ AUPRC | McNemar p<0.05 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | S0 | 5 | 65.75 ± 0.42 | 75.44 ± 0.67 | 33.02 ± 0.72 | +0.07 (3/5) | -0.08 (3/5) | +0.26 (4/5) | [-0.56, +0.39] | [-0.38, +0.98] | 5/5 |
| S1b | S0 | 5 | 65.28 ± 0.47 | 75.68 ± 0.12 | 32.72 ± 0.55 | -0.40 (1/5) | +0.16 (4/5) | -0.04 (2/5) | [-0.16, +0.46] | [-0.56, +0.48] | 4/5 |
| S2 | S0 | 5 | 65.07 ± 1.54 | 75.49 ± 0.28 | 33.10 ± 0.83 | -0.62 (2/5) | -0.03 (2/5) | +0.34 (4/5) | [-0.48, +0.42] | [-0.46, +1.14] | 4/5 |
| S2b | S0 | 5 | 65.14 ± 1.03 | 75.68 ± 0.15 | 32.95 ± 0.51 | -0.54 (2/5) | +0.16 (4/5) | +0.19 (3/5) | [-0.22, +0.56] | [-0.58, +0.99] | 4/5 |

## Stack drift: R0 seed 0 re-run on this stack (par_R0) vs the dr study's R0

| arm | reference | seeds | Macro-F1 | AUROC | AUPRC | Δ F1 (better) | Δ AUROC (better) | Δ AUPRC (better) | CI Δ AUROC | CI Δ AUPRC | McNemar p<0.05 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| par_R0 | R0 | 1 | 65.37 ± 0.00 | 75.09 ± 0.00 | 32.83 ± 0.00 | +0.58 (1/1) | -0.04 (0/1) | +0.05 (1/1) | [-0.24, +0.16] | [-0.32, +0.41] | 1/1 |

## Cost

| run | prompt tokens mean / max | hours / run (wall) | peak GPU GB |
|---|---|---|---|
| zs_S0 | 930.5 / 2051 | 1.19 | 20.0 |
| zs_S1 | 969.2 / 2135 | 1.39 | 20.1 |
| zs_S1b | 965.3 / 2135 | 1.33 | 20.1 |
| zs_S2 | 969.2 / 2135 | 1.35 | 20.3 |
| zs_S2b | 965.3 / 2135 | 1.46 | 20.2 |
| dr_R0 | 781.5 / 1902 | 1.03 | 19.7 |
| dr_R1 | 815.2 / 1981 | 1.08 | 19.8 |
| dr_R1b | 811.2 / 1981 | 1.07 | 19.8 |
| dr_R2 | 815.2 / 1981 | 1.17 | 19.9 |
| dr_R2b | 811.2 / 1981 | 1.17 | 19.9 |
