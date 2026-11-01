# Gemma-swap study · Experiment 4: writer vs classifier

Which role explains Gemma-2-9B-it's gain over Qwen3-8B in DGP (Exp 1, arm 0)? X0 keeps Gemma's text (gs_G0's cached summaries, reused word for word) and swaps the classifier back to Qwen3-8B. AmazonVideo test set (7,425 reviews), dr arm R0 (original DGP prompt). Mean ± std over seeds (ddof=0). Δ = paired per-seed difference; 'better' = seeds where the first run is higher. CI = paired bootstrap 95% interval (2000 resamples of the test reviews). Holm: paired-t p-values corrected over the 3 comparisons × 3 metrics below. Statistics: scripts/dr_analysis.py:compare.

## Cells

| run | roles | seeds | Macro-F1 | AUROC | AUPRC |
|---|---|---|---|---|---|
| dr_R0 | Qwen writes · Qwen classifies | 5 | 64.74 ± 0.45 | 74.97 ± 0.51 | 32.37 ± 0.57 |
| gs_X0 | Gemma writes · Qwen classifies | 5 | 65.30 ± 0.51 | 75.37 ± 0.35 | 32.86 ± 0.37 |
| gs_G0 | Gemma writes · Gemma classifies | 5 | 65.04 ± 1.01 | 76.38 ± 0.34 | 34.31 ± 1.00 |
| paper | Qwen · Qwen | 5 | 66.91 | 77.32 | 34.63 |

## Paired comparisons

| comparison | what changes | seeds | Δ macro_f1 (better) · t p · Holm p | Δ auroc (better) · t p · Holm p | Δ auprc (better) · t p · Holm p | CI Δ AUROC | CI Δ AUPRC | McNemar p<0.05 |
|---|---|---|---|---|---|---|---|---|
| X0 vs R0 | Gemma's text (classifier: Qwen) | 5 | +0.55 (4/5) · 0.155 · 0.466 | +0.39 (5/5) · 0.010 · 0.062 | +0.49 (5/5) · 0.045 · 0.180 | [-0.10, +0.91] | [-0.34, +1.30] | 4/5 |
| G0 vs X0 | Gemma as classifier (text: Gemma's) | 5 | -0.25 (2/5) · 0.679 · 1.000 | +1.02 (5/5) · 0.001 · 0.010 | +1.44 (4/5) · 0.034 · 0.169 | [+0.45, +1.59] | [+0.39, +2.45] | 4/5 |
| G0 vs R0 | both roles (Exp 1 reference) | 5 | +0.30 (3/5) · 0.659 · 1.000 | +1.41 (5/5) · 0.001 · 0.009 | +1.94 (5/5) · 0.005 · 0.035 | [+0.68, +2.14] | [+0.70, +3.10] | 5/5 |

## Decomposition of Gemma's gain (G0 − R0 = writer + classifier)

Seeds [0, 1, 2, 3, 4]. Writer = X0 − R0, classifier = G0 − X0; they sum to G0 − R0 exactly. Share = part / total of the seed-averaged gain (only meaningful when the total is clearly non-zero).

| metric | total G0 − R0 | writer X0 − R0 | classifier G0 − X0 | writer share |
|---|---|---|---|---|
| macro_f1 | +0.30 | +0.55 | -0.25 | 184% |
| auroc | +1.41 | +0.39 | +1.02 | 28% |
| auprc | +1.94 | +0.49 | +1.44 | 25% |

## Stack drift (this machine)

dr_R0's exact config (hash a1e63bf98c), seed 0, re-run on this machine's stack (par2_R0) vs dr_R0 seed 0. A Δ of this size arises from the software stack alone; X0 differences smaller than it are not attributable to the writer.

| metric | par2_R0 | dr_R0 | Δ |
|---|---|---|---|
| macro_f1 | 65.29 | 64.78 | +0.51 |
| auroc | 75.10 | 75.13 | -0.03 |
| auprc | 32.59 | 32.78 | -0.19 |

## Cost

| run | prompt tokens mean / max | hours / run (wall) | peak GPU GB |
|---|---|---|---|
| dr_R0 | 781.5 / 1902 | 1.03 | 19.7 |
| gs_X0 | 521.4 / 1585 | 1.64 | 18.9 |
| gs_G0 | 540.0 / 1598 | 2.49 | 25.1 |

