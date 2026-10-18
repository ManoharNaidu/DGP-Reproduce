# Generated-text diagnostics (disc/resid study)

10023 targets; probe on train + val only (2598 reviews, 11.78% fraud), TF-IDF 1-2 grams + logistic regression, 5-fold CV.

| text | coverage | empty | multi-line | hit cap | tokens mean / p95 | mentions class word | Jaccard w/ raw |
|---|---|---|---|---|---|---|---|
| disc_flag | 10023 | 0 | 54 | 3 | 30.7 / 41 | 9258 | 0.091 |
| disc_noverdict | 10023 | 0 | 10 | 7 | 26.7 / 40 | 26 | 0.16 |
| residual | 10023 | 0 | 72 | 24 | 22.6 / 37 | 1 | 0.182 |

| probe input | AUROC | AUPRC |
|---|---|---|
| raw | 63.43 ± 1.79 | 21.85 ± 2.96 |
| disc_flag | 71.62 ± 3.99 | 30.47 ± 4.51 |
| disc_noverdict | 70.54 ± 3.35 | 26.62 ± 5.04 |
| residual | 66.21 ± 2.93 | 24.92 ± 4.12 |
| raw+disc_flag | 72.62 ± 2.27 | 30.31 ± 3.93 |
| raw+disc_noverdict | 68.75 ± 1.94 | 25.91 ± 4.42 |

disc_flag: says 'unhelpful' for 52.0% of fraud vs 18.5% of benign train+val reviews; 'helpful' for 40.8% vs 73.5%.

disc_noverdict: says 'unhelpful' for 0.0% of fraud vs 0.0% of benign train+val reviews; 'helpful' for 0.3% vs 0.2%.
