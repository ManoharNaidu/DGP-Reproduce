# Gemma-swap study · Experiment 4, fourth cell: the 2 × 2 writer × classifier design

QG0 keeps Qwen3-8B's text (dr_R0's cached summaries, reused word for word) and makes Gemma-2-9B-it the classifier. With R0, X0 and G0 it completes the 2 × 2 design. AmazonVideo test set (7,425 reviews), dr arm R0 (original DGP prompt). Mean ± std over seeds (ddof=0). Δ = paired per-seed difference; 'better' = seeds where the first run is higher. CI = paired bootstrap 95% interval (2000 resamples of the test reviews). Holm: paired-t p-values corrected over the 4 comparisons × 3 metrics below. Statistics: scripts/dr_analysis.py:compare.

## Cells

| run | roles | seeds | Macro-F1 | AUROC | AUPRC |
|---|---|---|---|---|---|
| dr_R0 | Qwen writes · Qwen classifies | 5 | 64.74 ± 0.45 | 74.97 ± 0.51 | 32.37 ± 0.57 |
| gs_QG0 | Qwen writes · Gemma classifies | 5 | 64.70 ± 1.35 | 76.18 ± 0.34 | 33.74 ± 0.44 |
| gs_X0 | Gemma writes · Qwen classifies | 5 | 65.30 ± 0.51 | 75.37 ± 0.35 | 32.86 ± 0.37 |
| gs_G0 | Gemma writes · Gemma classifies | 5 | 65.04 ± 1.01 | 76.38 ± 0.34 | 34.31 ± 1.00 |

## Paired comparisons

| comparison | what changes | seeds | Δ macro_f1 (better) · t p · Holm p | Δ auroc (better) · t p · Holm p | Δ auprc (better) · t p · Holm p | CI Δ AUROC | CI Δ AUPRC | McNemar p<0.05 |
|---|---|---|---|---|---|---|---|---|
| QG0 vs R0 | Gemma as classifier (text: Qwen's) | 5 | -0.04 (3/5) · 0.965 · 1.000 | +1.21 (5/5) · 0.007 · 0.074 | +1.37 (5/5) · 0.013 · 0.115 | [+0.53, +1.92] | [+0.25, +2.41] | 5/5 |
| G0 vs X0 | Gemma as classifier (text: Gemma's) | 5 | -0.25 (2/5) · 0.679 · 1.000 | +1.02 (5/5) · 0.001 · 0.015 | +1.44 (4/5) · 0.034 · 0.270 | [+0.45, +1.59] | [+0.39, +2.45] | 4/5 |
| X0 vs R0 | Gemma's text (classifier: Qwen) | 5 | +0.55 (4/5) · 0.155 · 0.932 | +0.39 (5/5) · 0.010 · 0.104 | +0.49 (5/5) · 0.045 · 0.315 | [-0.10, +0.91] | [-0.34, +1.30] | 4/5 |
| G0 vs QG0 | Gemma's text (classifier: Gemma) | 5 | +0.34 (3/5) · 0.618 · 1.000 | +0.20 (4/5) · 0.305 · 1.000 | +0.57 (4/5) · 0.211 · 1.000 | [-0.19, +0.60] | [-0.04, +1.15] | 5/5 |

## Interaction and the two orders of decomposition

Seeds [0, 1, 2, 3, 4]. Interaction = (G0 − QG0) − (X0 − R0) per seed (0 = writer and classifier effects add up); p = one-sample t over seeds. Writer-first order (E14): writer X0 − R0, then classifier G0 − X0. Classifier-first order: classifier QG0 − R0, then writer G0 − QG0. Both sum to G0 − R0.

| metric | G0 − R0 | interaction (seeds > 0) · p | writer-first: writer / classifier | classifier-first: classifier / writer |
|---|---|---|---|---|
| macro_f1 | +0.30 | -0.21 (2/5) · 0.823 | +0.55 / -0.25 | -0.04 / +0.34 |
| auroc | +1.41 | -0.19 (1/5) · 0.428 | +0.39 / +1.02 | +1.21 / +0.20 |
| auprc | +1.94 | +0.08 (2/5) · 0.892 | +0.49 / +1.44 | +1.37 / +0.57 |

## Cost

| run | prompt tokens mean / max | hours / run (wall) | peak GPU GB |
|---|---|---|---|
| dr_R0 | 781.5 / 1902 | 1.03 | 19.7 |
| gs_QG0 | 798.7 / 1918 | 1.50 | 27.5 |
| gs_X0 | 521.4 / 1585 | 1.64 | 18.9 |
| gs_G0 | 540.0 / 1598 | 2.49 | 25.1 |

Prompt tokens are counted with each cell's classifier tokenizer, so Qwen and Gemma counts of the same text differ. Training hours are not comparable across machines or lane counts.

