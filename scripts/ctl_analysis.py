"""Controls · what the graph context adds: DGP (R0) vs the same LLM on the target review only (T0) vs the MLP baseline.

    python scripts/ctl_analysis.py          # CPU; reads finished runs only; writes results/controls/controls.{json,md}

Cells (AmazonVideo, seeds 0-4, the same split, validation-tuned threshold and early stopping for the two LLM cells):
    dr_R0            DGP prompt (target + MDK-trimmed neighbour summaries), Qwen3-8B + LoRA  (dr study, 2026-10-15)
    ctl_T0           "Target: <review> / Question: Is this fraud?", same model and recipe     (2026-11-01, scripts/ctl_queue.sh)
    mlp_amazonvideo  the paper's MLP baseline (CPU)                                           (re-run 2026-11-01)

    R0 - T0  = what the graph context adds with the classifier held fixed (the paper's 'DGP vs LLM' gap)
    T0 - MLP = what the fine-tuned LLM adds over the MLP when neither sees the graph text

Statistics: scripts/dr_analysis.py:compare unchanged (paired per-seed Δ, paired t, Wilcoxon, paired bootstrap 95% CI over
the 7,425 test reviews, McNemar), with Holm correction of the paired-t p-values over the 3 comparisons x 3 metrics.
"""

import json
from pathlib import Path

import numpy as np

from dr_analysis import METRICS, compare, cost, fmt, runs
from x0_analysis import holm, test_metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "controls"
CELLS = {"dr_R0": "DGP: target + graph summaries (Qwen3-8B)", "ctl_T0": "LLM: target review only (Qwen3-8B)",
         "mlp_amazonvideo": "MLP baseline"}
PAPER = {"dr_R0": {"macro_f1": 66.91, "auroc": 77.32, "auprc": 34.63},          # Table 2 of the DGP paper, AmazonVideo
         "ctl_T0": {"macro_f1": 59.90, "auroc": 71.78, "auprc": 27.03},
         "mlp_amazonvideo": {"macro_f1": 61.74, "auroc": 70.47, "auprc": 26.55}}
COMPARE = [("R0 vs T0", "dr_R0", "ctl_T0", "graph context (classifier fixed)"),
           ("T0 vs MLP", "ctl_T0", "mlp_amazonvideo", "fine-tuned LLM vs MLP, no graph text"),
           ("R0 vs MLP", "dr_R0", "mlp_amazonvideo", "DGP vs MLP")]


def main():
    R = {k: runs(k) for k in CELLS}
    L = ["# Controls: what the graph context adds", "",
         "DGP (dr_R0) against the same Qwen3-8B + LoRA classifier given only the target review (ctl_T0), and the MLP "
         "baseline. AmazonVideo test set (7,425 reviews), seeds 0-4. Mean ± std over seeds (ddof=0). Δ = paired "
         "per-seed difference; 'better' = seeds where the first run is higher. CI = paired bootstrap 95% interval "
         "(2000 resamples of the test reviews). Holm: paired-t p-values corrected over the 3 comparisons × 3 metrics. "
         "Statistics: scripts/dr_analysis.py:compare.", "",
         "## Cells", "", "| run | what | seeds | Macro-F1 | AUROC | AUPRC | paper (F1 / AUROC / AUPRC) |",
         "|---|---|---|---|---|---|---|"]
    payload = {"cells": {}, "comparisons": {}, "paper": PAPER}
    for k, label in CELLS.items():
        ms = list(test_metrics(R[k]).values())
        paper = " / ".join(f"{PAPER[k][m]:.2f}" for m in METRICS)
        if not ms:
            L.append(f"| {k} | {label} | 0 | | | | {paper} |")
            continue
        agg = {m: (float(np.mean([x[m] for x in ms])), float(np.std([x[m] for x in ms]))) for m in METRICS}
        payload["cells"][k] = {"seeds": sorted(R[k]), **{m: {"mean": v[0], "std": v[1]} for m, v in agg.items()},
                               "threshold_per_seed": [x.get("threshold") for x in ms]}
        L.append(f"| {k} | {label} | {len(ms)} | " + " | ".join(f"{agg[m][0]:.2f} ± {agg[m][1]:.2f}" for m in METRICS)
                 + f" | {paper} |")
    L.append("")

    res = {name: compare(R[a], R[b]) for name, a, b, _ in COMPARE}
    pvals = {(name, m): res[name]["metrics"][m]["t_p"] for name, *_ in COMPARE if res[name]
             for m in METRICS if "t_p" in res[name]["metrics"][m]}
    adj = holm(pvals) if pvals else {}
    L.extend(["## Paired comparisons", "",
              "| comparison | what changes | seeds | " + " | ".join(f"Δ {m} (better) · t p · Holm p" for m in METRICS)
              + " | CI Δ AUROC | CI Δ AUPRC | McNemar p<0.05 |", "|---|---|---|" + "---|" * (len(METRICS) + 3)])
    for name, a, b, what in COMPARE:
        r = res[name]
        if not r:
            L.append(f"| {name} | {what} | 0 |" + " |" * (len(METRICS) + 3))
            continue
        n = len(r["seeds"])
        cells = []
        for m in METRICS:
            x = r["metrics"][m]
            p = x.get("t_p")
            cells.append(f"{x['delta_mean']:+.2f} ({x['seeds_better']}/{n}) · "
                         + (f"{p:.3f} · {adj[(name, m)]:.3f}" if p is not None else "— · —"))
        ci = r["bootstrap_ci95"]
        L.append(f"| {name} | {what} | {n} | " + " | ".join(cells)
                 + f" | [{ci['auroc'][0]:+.2f}, {ci['auroc'][1]:+.2f}] | [{ci['auprc'][0]:+.2f}, {ci['auprc'][1]:+.2f}] | "
                 f"{sum(x['mcnemar_p'] < 0.05 for x in r['agreement'])}/{n} |")
        payload["comparisons"][name] = {"a": a, "b": b, "what": what, **r,
                                        "holm_t_p": {m: adj.get((name, m)) for m in METRICS}}
    L.append("")

    # The same gaps in the paper, next to ours.
    if all(payload["cells"].get(k) for k in CELLS):
        L.extend(["## Gaps: ours vs the paper", "",
                  "| gap | metric | ours | paper |", "|---|---|---|---|"])
        payload["gaps"] = {}
        for name, a, b, _ in COMPARE:
            for m in METRICS:
                ours = payload["cells"][a][m]["mean"] - payload["cells"][b][m]["mean"]
                paper = PAPER[a][m] - PAPER[b][m]
                payload["gaps"].setdefault(name, {})[m] = {"ours": ours, "paper": paper}
                L.append(f"| {name} | {m} | {ours:+.2f} | {paper:+.2f} |")
        L.append("")

    L.extend(["## Cost", "", "| run | prompt tokens mean / max | hours / run (wall) | peak GPU GB |", "|---|---|---|---|"])
    payload["cost"] = {}
    for k in ("dr_R0", "ctl_T0"):
        c = cost(k, R[k]) if R[k] else None
        payload["cost"][k] = c
        if c:
            L.append(f"| {k} | {fmt(c['final_prompt_tokens_mean'], 1)} / {fmt(c['final_prompt_tokens_max'], 0)} | "
                     f"{fmt(c['train_hours_mean'])} | {fmt(c['peak_gpu_gb'], 1)} |")
    L.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "controls.json").write_text(json.dumps(payload, indent=2, default=float))
    (OUT / "controls.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
