"""YelpChi · DGP against the MLP baseline and the paper's YelpReviews rows (Table 2).

    python scripts/yelp_analysis.py         # CPU; reads finished runs only; writes results/controls/yelpchi.{json,md}

Cells (YelpChi from the FLAG-Reproduce release mirror, graph identical to the paper's Table 1; split 1,348 / 1,348 /
13,479; seeds 0-4):
    dgp_yelpchi   DGP, the AmazonVideo main-run configuration (Qwen3-8B summarizer + Qwen3-8B LoRA classifier)
                  (scripts/yelp_nomdk_chain.sh -> scripts/ctl_queue.sh, configs/experiments/dgp_yelp.yaml)
    mlp_yelpchi   the MLP baseline on DeBERTa text embeddings + numeric features (scripts/run_mlp.py, CPU)

Statistics: scripts/dr_analysis.py:compare unchanged (paired per-seed Δ, paired t, paired bootstrap 95% CI over the
13,479 test reviews, McNemar).
"""

import json
from pathlib import Path

import numpy as np

from dr_analysis import METRICS, compare, cost, fmt, runs
from x0_analysis import test_metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "controls"
CELLS = {"dgp_yelpchi": "DGP (Qwen3-8B summaries + Qwen3-8B LoRA classifier)", "mlp_yelpchi": "MLP baseline"}
PAPER = {"dgp_yelpchi": {"macro_f1": 69.07, "auroc": 84.28, "auprc": 48.87},     # research/reported_results.csv
         "mlp_yelpchi": {"macro_f1": 62.09, "auroc": 75.00, "auprc": 32.24}}
PAPER_OTHER = {"ConsisGAD (paper's best baseline)": (67.33, 82.12, 42.11), "LLM (target only)": (60.79, 71.18, 30.45),
               "FLAG": (64.79, 73.98, 38.45)}


def main():
    R = {k: runs(k) for k in CELLS}
    L = ["# YelpChi: DGP against the MLP and the paper", "",
         "YelpChi (67,395 reviews, 17,486,608 edges, the paper's Table 1 exactly), test set 13,479 reviews, seeds 0-4. "
         "Mean ± std over seeds (ddof=0). Δ = paired per-seed difference; CI = paired bootstrap 95% interval over test "
         "reviews. Statistics: scripts/dr_analysis.py:compare.", "",
         "| run | what | seeds | Macro-F1 | AUROC | AUPRC | paper (F1 / AUROC / AUPRC) | ours − paper (AUROC / AUPRC) |",
         "|---|---|---|---|---|---|---|---|"]
    payload = {"cells": {}, "paper": PAPER, "paper_other": PAPER_OTHER}
    for k, label in CELLS.items():
        ms = list(test_metrics(R[k]).values())
        paper = " / ".join(f"{PAPER[k][m]:.2f}" for m in METRICS)
        if not ms:
            L.append(f"| {k} | {label} | 0 | | | | {paper} | |")
            continue
        agg = {m: (float(np.mean([x[m] for x in ms])), float(np.std([x[m] for x in ms]))) for m in METRICS}
        payload["cells"][k] = {"seeds": sorted(R[k]), **{m: {"mean": v[0], "std": v[1]} for m, v in agg.items()},
                               "threshold_per_seed": [x.get("threshold") for x in ms]}
        L.append(f"| {k} | {label} | {len(ms)} | " + " | ".join(f"{agg[m][0]:.2f} ± {agg[m][1]:.2f}" for m in METRICS)
                 + f" | {paper} | {agg['auroc'][0] - PAPER[k]['auroc']:+.2f} / {agg['auprc'][0] - PAPER[k]['auprc']:+.2f} |")
    L += ["", "Other paper rows for reference (not run here): "
          + "; ".join(f"{k} {v[0]:.2f} / {v[1]:.2f} / {v[2]:.2f}" for k, v in PAPER_OTHER.items()) + ".", ""]

    r = compare(R["dgp_yelpchi"], R["mlp_yelpchi"])
    if r:
        n = len(r["seeds"])
        L += ["## DGP − MLP", "", "| metric | Δ (better) | t p | paper Δ |", "|---|---|---|---|"]
        for m in METRICS:
            x = r["metrics"][m]
            p = x.get("t_p")
            L.append(f"| {m} | {x['delta_mean']:+.2f} ({x['seeds_better']}/{n}) | "
                     f"{'—' if p is None else f'{p:.4f}'} | {PAPER['dgp_yelpchi'][m] - PAPER['mlp_yelpchi'][m]:+.2f} |")
        ci = r["bootstrap_ci95"]
        L += ["", f"Bootstrap 95% CI: AUROC [{ci['auroc'][0]:+.2f}, {ci['auroc'][1]:+.2f}], "
              f"AUPRC [{ci['auprc'][0]:+.2f}, {ci['auprc'][1]:+.2f}]. McNemar p < 0.05 in "
              f"{sum(x['mcnemar_p'] < 0.05 for x in r['agreement'])}/{n} seeds.", ""]
        payload["dgp_minus_mlp"] = r

    c = cost("dgp_yelpchi", R["dgp_yelpchi"]) if R["dgp_yelpchi"] else None
    payload["cost"] = c
    if c:
        L += ["## Cost", "", "| prompt tokens mean / max | hours / run (wall) | peak GPU GB |", "|---|---|---|",
              f"| {fmt(c['final_prompt_tokens_mean'], 1)} / {fmt(c['final_prompt_tokens_max'], 0)} | "
              f"{fmt(c['train_hours_mean'])} | {fmt(c['peak_gpu_gb'], 1)} |", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "yelpchi.json").write_text(json.dumps(payload, indent=2, default=float))
    (OUT / "yelpchi.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
