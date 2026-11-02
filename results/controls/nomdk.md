# Ablation w/o MDK: MDK neighbour selection vs random neighbours

AmazonVideo test set (7,425 reviews), seeds 0-4, mean ± std over seeds (ddof=0). Both cells use the same summarizer, classifier, budgets and recipe; only the choice of the M = 4 neighbours per metapath differs. Δ = R0 − noMDK (positive = MDK helps); CI = paired bootstrap 95% interval over test reviews.

| run | what | seeds | Macro-F1 | AUROC | AUPRC |
|---|---|---|---|---|---|
| dr_R0 | DGP: MDK selects the M = 4 neighbours | 5 | 64.74 ± 0.45 | 74.97 ± 0.51 | 32.37 ± 0.57 |
| ctl_noMDK | w/o MDK: M = 4 random neighbours | 5 | 65.28 ± 0.80 | 74.61 ± 0.54 | 32.51 ± 0.61 |

## R0 − noMDK (what MDK adds)

| metric | Δ (better) | t p |
|---|---|---|
| macro_f1 | -0.53 (1/5) | 0.2665 |
| auroc | +0.36 (4/5) | 0.0666 |
| auprc | -0.14 (1/5) | 0.1829 |

Bootstrap 95% CI: AUROC [-0.32, +1.05], AUPRC [-1.12, +0.81]. McNemar p < 0.05 in 5/5 seeds.

## How different are the two selections?

MDK cache `30447a444d0aa6e2` vs random cache `b697e61d334368f3` (10,023 targets, same K, M and split). 'shared' = share of the random neighbours that MDK also chose; 'identical' = targets whose two sets are equal; 'all kept' = targets with at most M candidates (including none), where both keep every candidate; such targets also count as identical.

| metapath | shared | identical | all kept |
|---|---|---|---|
| RPR-RPR | 20.6% | 4.4% | 3.5% |
| RPR-RSR | 0.3% | 0.0% | 0.0% |
| RPR-RUR | 2.4% | 0.0% | 0.0% |
| RPR | 20.6% | 4.4% | 3.5% |
| RSR-RPR | 0.6% | 1.0% | 0.9% |
| RSR-RSR | 9.1% | 4.5% | 4.3% |
| RSR-RUR | 1.4% | 1.1% | 1.1% |
| RSR | 9.3% | 4.5% | 4.3% |
| RUR-RPR | 1.5% | 0.0% | 0.0% |
| RUR-RSR | 1.0% | 0.0% | 0.0% |
| RUR-RUR | 65.9% | 33.1% | 28.3% |
| RUR | 65.9% | 33.1% | 28.3% |

## Cost (noMDK)

| prompt tokens mean / max | hours / run (wall) | peak GPU GB |
|---|---|---|
| 762.4 / 1845 | 1.96 | 19.6 |

