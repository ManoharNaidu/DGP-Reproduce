# Comparison with paper (public datasets)

| dataset | method | variant | mode | metric | paper | paper_std | ours | ours_std | abs_diff | rel_diff_pct | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| YelpReviews | MLP | main | CPU_REPRODUCTION | macro_f1 | 62.09 | 0.05 | 63.64 | 0.43 | 1.55 | 2.5 | CLOSE |
| YelpReviews | MLP | main | CPU_REPRODUCTION | auroc | 75.0 | 0.06 | 74.58 | 0.86 | -0.42 | -0.56 | MATCH |
| YelpReviews | MLP | main | CPU_REPRODUCTION | auprc | 32.24 | 0.13 | 33.58 | 1.16 | 1.34 | 4.16 | MATCH |
| YelpReviews | SAGE | main |  | all |  |  |  |  |  |  | NO_RESULT |
| YelpReviews | HGT | main |  | all |  |  |  |  |  |  | NO_RESULT |
| YelpReviews | ConsisGAD | main |  | all |  |  |  |  |  |  | NO_RESULT |
| YelpReviews | PMP | main |  | all |  |  |  |  |  |  | NO_RESULT |
| YelpReviews | GAAP | main |  | all |  |  |  |  |  |  | NO_RESULT |
| YelpReviews | LLM | main |  | all |  |  |  |  |  |  | NO_RESULT |
| YelpReviews | TAPE | main |  | all |  |  |  |  |  |  | NO_RESULT |
| YelpReviews | FLAG | main |  | all |  |  |  |  |  |  | NO_RESULT |
| YelpReviews | GraphGPT | main |  | all |  |  |  |  |  |  | NO_RESULT |
| YelpReviews | HiGPT | main |  | all |  |  |  |  |  |  | NO_RESULT |
| YelpReviews | InstructGLM | main |  | all |  |  |  |  |  |  | NO_RESULT |
| YelpReviews | DGP | main | GPU_REPRODUCTION | macro_f1 | 69.07 | 0.23 | 61.76 | 1.86 | -7.31 | -10.58 | DEVIATES |
| YelpReviews | DGP | main | GPU_REPRODUCTION | auroc | 84.28 | 0.11 | 78.38 | 0.88 | -5.9 | -7.0 | DEVIATES |
| YelpReviews | DGP | main | GPU_REPRODUCTION | auprc | 48.87 | 0.82 | 31.05 | 1.31 | -17.82 | -36.46 | DEVIATES |
| AmazonVideo | MLP | main | CPU_REPRODUCTION | macro_f1 | 61.74 | 0.39 | 59.83 | 0.5 | -1.91 | -3.09 | CLOSE |
| AmazonVideo | MLP | main | CPU_REPRODUCTION | auroc | 70.47 | 0.18 | 68.33 | 0.51 | -2.14 | -3.04 | DEVIATES |
| AmazonVideo | MLP | main | CPU_REPRODUCTION | auprc | 26.55 | 0.48 | 25.11 | 1.09 | -1.44 | -5.42 | MATCH |
| AmazonVideo | SAGE | main |  | all |  |  |  |  |  |  | NO_RESULT |
| AmazonVideo | HGT | main |  | all |  |  |  |  |  |  | NO_RESULT |
| AmazonVideo | ConsisGAD | main |  | all |  |  |  |  |  |  | NO_RESULT |
| AmazonVideo | PMP | main |  | all |  |  |  |  |  |  | NO_RESULT |
| AmazonVideo | GAAP | main |  | all |  |  |  |  |  |  | NO_RESULT |
| AmazonVideo | LLM | main |  | all |  |  |  |  |  |  | NO_RESULT |
| AmazonVideo | TAPE | main |  | all |  |  |  |  |  |  | NO_RESULT |
| AmazonVideo | FLAG | main |  | all |  |  |  |  |  |  | NO_RESULT |
| AmazonVideo | GraphGPT | main |  | all |  |  |  |  |  |  | NO_RESULT |
| AmazonVideo | HiGPT | main |  | all |  |  |  |  |  |  | NO_RESULT |
| AmazonVideo | InstructGLM | main |  | all |  |  |  |  |  |  | NO_RESULT |
| AmazonVideo | DGP | main | GPU_REPRODUCTION | macro_f1 | 66.91 | 0.13 | 65.16 | 0.52 | -1.75 | -2.62 | CLOSE |
| AmazonVideo | DGP | main | GPU_REPRODUCTION | auroc | 77.32 | 0.11 | 75.07 | 0.5 | -2.25 | -2.91 | DEVIATES |
| AmazonVideo | DGP | main | GPU_REPRODUCTION | auprc | 34.63 | 0.24 | 32.33 | 0.65 | -2.3 | -6.64 | DEVIATES |
| YelpReviews | DGP | task_agnostic |  | all |  |  |  |  |  |  | NO_RESULT |
| YelpReviews | DGP | task_aware |  | all |  |  |  |  |  |  | NO_RESULT |
| AmazonVideo | DGP | task_agnostic |  | all |  |  |  |  |  |  | NO_RESULT |
| AmazonVideo | DGP | task_aware |  | all |  |  |  |  |  |  | NO_RESULT |
