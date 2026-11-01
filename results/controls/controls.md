# Controls: what the graph context adds

DGP (dr_R0) against the same Qwen3-8B + LoRA classifier given only the target review (ctl_T0), and the MLP baseline. AmazonVideo test set (7,425 reviews), seeds 0-4. Mean ± std over seeds (ddof=0). Δ = paired per-seed difference; 'better' = seeds where the first run is higher. CI = paired bootstrap 95% interval (2000 resamples of the test reviews). Holm: paired-t p-values corrected over the 3 comparisons × 3 metrics. Statistics: scripts/dr_analysis.py:compare.

## Cells

| run | what | seeds | Macro-F1 | AUROC | AUPRC | paper (F1 / AUROC / AUPRC) |
|---|---|---|---|---|---|---|
| dr_R0 | DGP: target + graph summaries (Qwen3-8B) | 5 | 64.74 ± 0.45 | 74.97 ± 0.51 | 32.37 ± 0.57 | 66.91 / 77.32 / 34.63 |
| ctl_T0 | LLM: target review only (Qwen3-8B) | 5 | 63.35 ± 2.84 | 73.29 ± 0.36 | 30.54 ± 0.60 | 59.90 / 71.78 / 27.03 |
| mlp_amazonvideo | MLP baseline | 5 | 59.83 ± 0.50 | 68.33 ± 0.51 | 25.11 ± 1.09 | 61.74 / 70.47 / 26.55 |

## Paired comparisons

| comparison | what changes | seeds | Δ macro_f1 (better) · t p · Holm p | Δ auroc (better) · t p · Holm p | Δ auprc (better) · t p · Holm p | CI Δ AUROC | CI Δ AUPRC | McNemar p<0.05 |
|---|---|---|---|---|---|---|---|---|
| R0 vs T0 | graph context (classifier fixed) | 5 | +1.40 (2/5) · 0.405 · 0.405 | +1.69 (5/5) · 0.003 · 0.011 | +1.83 (5/5) · 0.004 · 0.011 | [+0.92, +2.45] | [+0.67, +3.03] | 5/5 |
| T0 vs MLP | fine-tuned LLM vs MLP, no graph text | 5 | +3.52 (4/5) · 0.076 · 0.151 | +4.96 (5/5) · 0.000 · 0.002 | +5.43 (5/5) · 0.002 · 0.011 | [+3.43, +6.48] | [+3.38, +7.55] | 4/5 |
| R0 vs MLP | DGP vs MLP | 5 | +4.92 (5/5) · 0.000 · 0.001 | +6.64 (5/5) · 0.000 · 0.001 | +7.26 (5/5) · 0.000 · 0.002 | [+5.03, +8.17] | [+5.06, +9.49] | 4/5 |

## Gaps: ours vs the paper

| gap | metric | ours | paper |
|---|---|---|---|
| R0 vs T0 | macro_f1 | +1.40 | +7.01 |
| R0 vs T0 | auroc | +1.69 | +5.54 |
| R0 vs T0 | auprc | +1.83 | +7.60 |
| T0 vs MLP | macro_f1 | +3.52 | -1.84 |
| T0 vs MLP | auroc | +4.96 | +1.31 |
| T0 vs MLP | auprc | +5.43 | +0.48 |
| R0 vs MLP | macro_f1 | +4.92 | +5.17 |
| R0 vs MLP | auroc | +6.64 | +6.85 |
| R0 vs MLP | auprc | +7.26 | +8.08 |

## Cost

| run | prompt tokens mean / max | hours / run (wall) | peak GPU GB |
|---|---|---|---|
| dr_R0 | 781.5 / 1902 | 1.03 | 19.7 |
| ctl_T0 | 121.3 / 1034 | 0.67 | 17.8 |

