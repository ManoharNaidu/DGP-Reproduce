# Gemma-swap / zero-shot study · Experiment 1

AmazonVideo test set (7,425 reviews). Mean ± std over seeds (ddof=0). Δ = paired per-seed difference (arm − reference); 'better' = seeds where the arm is higher. CI = paired bootstrap 95% interval of the seed-averaged Δ (2000 resamples of the test reviews). McNemar: seeds whose correctness differs at p < 0.05 (each run at its own validation threshold). Statistics: scripts/dr_analysis.py:compare.

## Exp 1 · Gemma vs Qwen, original prompt (G vs R)

| arm | reference | seeds | Macro-F1 | AUROC | AUPRC | Δ F1 (better) | Δ AUROC (better) | Δ AUPRC (better) | CI Δ AUROC | CI Δ AUPRC | McNemar p<0.05 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G0 | R0 | 5 | 65.04 ± 1.01 | 76.38 ± 0.34 | 34.31 ± 1.00 | +0.30 (3/5) | +1.41 (5/5) | +1.94 (5/5) | [+0.68, +2.14] | [+0.70, +3.10] | 5/5 |
| G1 | R1 | 5 | 65.65 ± 1.41 | 76.00 ± 0.27 | 34.30 ± 0.61 | +0.86 (5/5) | +0.82 (5/5) | +1.88 (5/5) | [+0.08, +1.52] | [+0.52, +3.20] | 3/5 |
| G1b | R1b | 5 | 65.60 ± 1.60 | 75.82 ± 0.28 | 34.52 ± 0.38 | +0.27 (4/5) | +0.40 (4/5) | +2.10 (5/5) | [-0.28, +1.08] | [+0.75, +3.41] | 4/5 |
| G2 | R2 | 5 | 65.43 ± 0.95 | 76.35 ± 0.58 | 34.20 ± 1.82 | +1.15 (4/5) | +1.13 (5/5) | +1.68 (4/5) | [+0.34, +1.86] | [+0.42, +2.81] | 4/5 |
| G2b | R2b | 5 | 65.97 ± 0.83 | 75.98 ± 0.35 | 34.24 ± 0.38 | +1.20 (4/5) | +0.74 (5/5) | +1.87 (5/5) | [+0.06, +1.42] | [+0.65, +3.07] | 5/5 |

## Exp 1 · within Gemma (G arms vs G0)

| arm | reference | seeds | Macro-F1 | AUROC | AUPRC | Δ F1 (better) | Δ AUROC (better) | Δ AUPRC (better) | CI Δ AUROC | CI Δ AUPRC | McNemar p<0.05 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G1 | G0 | 5 | 65.65 ± 1.41 | 76.00 ± 0.27 | 34.30 ± 0.61 | +0.60 (4/5) | -0.39 (0/5) | -0.01 (2/5) | [-0.83, +0.06] | [-0.80, +0.82] | 5/5 |
| G1b | G0 | 5 | 65.60 ± 1.60 | 75.82 ± 0.28 | 34.52 ± 0.38 | +0.55 (3/5) | -0.56 (0/5) | +0.21 (3/5) | [-1.02, -0.10] | [-0.53, +0.98] | 5/5 |
| G2 | G0 | 5 | 65.43 ± 0.95 | 76.35 ± 0.58 | 34.20 ± 1.82 | +0.38 (4/5) | -0.03 (3/5) | -0.11 (3/5) | [-0.47, +0.40] | [-0.91, +0.70] | 4/5 |
| G2b | G0 | 5 | 65.97 ± 0.83 | 75.98 ± 0.35 | 34.24 ± 0.38 | +0.93 (4/5) | -0.40 (1/5) | -0.06 (2/5) | [-0.81, -0.01] | [-0.82, +0.71] | 4/5 |

## Cost

| run | prompt tokens mean / max | hours / run (wall) | peak GPU GB |
|---|---|---|---|
| gs_G0 | 540.0 / 1598 | 2.49 | 25.1 |
| gs_G1 | 570.7 / 1637 | 2.54 | 25.4 |
| gs_G1b | 567.1 / 1623 | 2.33 | 25.3 |
| gs_G2 | 570.7 / 1637 | 2.79 | 25.4 |
| gs_G2b | 567.1 / 1623 | 3.25 | 25.3 |
| dr_R0 | 781.5 / 1902 | 1.03 | 19.7 |
| dr_R1 | 815.2 / 1981 | 1.08 | 19.8 |
| dr_R1b | 811.2 / 1981 | 1.07 | 19.8 |
| dr_R2 | 815.2 / 1981 | 1.17 | 19.9 |
| dr_R2b | 811.2 / 1981 | 1.17 | 19.9 |
