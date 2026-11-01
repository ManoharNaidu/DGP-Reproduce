"""Exp 4 · writer/classifier split: which role explains Gemma's gain over Qwen, the text writer or the classifier?

    python scripts/x0_analysis.py           # CPU; reads finished runs only; writes results/gemma_swap/exp4.{json,md}

Cells (all dr arm R0 = original DGP prompt, AmazonVideo, seeds 0-4):
    dr_R0   Qwen writes,  Qwen classifies    (dr study, 2026-10-15)
    gs_X0   Gemma writes, Qwen classifies    (this experiment; gs_G0's cached Gemma text, scripts/x0_queue.sh)
    gs_G0   Gemma writes, Gemma classifies   (Gemma-swap Exp 1, 2026-10-29)
    par2_R0 dr_R0's config, seed 0, re-run on this machine's stack (2026-11-01): size of stack drift alone

    X0 - R0 = effect of Gemma's text (classifier fixed: Qwen)
    G0 - X0 = effect of Gemma as classifier (text fixed: Gemma's)
    The two add up to G0 - R0 per seed exactly, so the writer share = (X0 - R0) / (G0 - R0) of the mean gain.

Statistics: scripts/dr_analysis.py:compare unchanged (paired per-seed Δ, paired t, Wilcoxon, paired bootstrap 95% CI over
the 7,425 test reviews, McNemar), plus Holm correction of the paired-t p-values over the 3 role comparisons x 3 metrics.
"""

import json
from pathlib import Path

import numpy as np

from dr_analysis import METRICS, compare, cost, fmt, runs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "gemma_swap"
CELLS = {"dr_R0": "Qwen writes · Qwen classifies", "gs_X0": "Gemma writes · Qwen classifies",
         "gs_G0": "Gemma writes · Gemma classifies"}
ROLE = [("X0 vs R0", "gs_X0", "dr_R0", "Gemma's text (classifier: Qwen)"),
        ("G0 vs X0", "gs_G0", "gs_X0", "Gemma as classifier (text: Gemma's)"),
        ("G0 vs R0", "gs_G0", "dr_R0", "both roles (Exp 1 reference)")]
PAPER = {"macro_f1": 66.91, "auroc": 77.32, "auprc": 34.63}


def holm(ps: dict) -> dict:
    keys = sorted(ps, key=ps.get)
    out, running = {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (len(keys) - i) * ps[k]))
        out[k] = running
    return out


def test_metrics(r: dict) -> dict:
    return {s: json.loads((p / "metrics.json").read_text())["test"] for s, p in r.items()}


def main():
    R = {k: runs(k) for k in CELLS}
    L = ["# Gemma-swap study · Experiment 4: writer vs classifier", "",
         "Which role explains Gemma-2-9B-it's gain over Qwen3-8B in DGP (Exp 1, arm 0)? X0 keeps Gemma's text (gs_G0's "
         "cached summaries, reused word for word) and swaps the classifier back to Qwen3-8B. AmazonVideo test set "
         "(7,425 reviews), dr arm R0 (original DGP prompt). Mean ± std over seeds (ddof=0). Δ = paired per-seed "
         "difference; 'better' = seeds where the first run is higher. CI = paired bootstrap 95% interval (2000 resamples "
         "of the test reviews). Holm: paired-t p-values corrected over the 3 comparisons × 3 metrics below. "
         "Statistics: scripts/dr_analysis.py:compare.", "",
         "## Cells", "", "| run | roles | seeds | Macro-F1 | AUROC | AUPRC |", "|---|---|---|---|---|---|"]
    payload = {"cells": {}, "comparisons": {}}
    for k, label in CELLS.items():
        ms = list(test_metrics(R[k]).values())
        if not ms:
            L.append(f"| {k} | {label} | 0 | | | |")
            continue
        agg = {m: (float(np.mean([x[m] for x in ms])), float(np.std([x[m] for x in ms]))) for m in METRICS}
        payload["cells"][k] = {"seeds": sorted(R[k]), **{m: {"mean": v[0], "std": v[1]} for m, v in agg.items()}}
        L.append(f"| {k} | {label} | {len(ms)} | " + " | ".join(f"{agg[m][0]:.2f} ± {agg[m][1]:.2f}" for m in METRICS) + " |")
    L.append(f"| paper | Qwen · Qwen | 5 | {PAPER['macro_f1']} | {PAPER['auroc']} | {PAPER['auprc']} |")
    L.append("")

    res = {name: compare(R[a], R[b]) for name, a, b, _ in ROLE}
    pvals = {(name, m): res[name]["metrics"][m]["t_p"] for name, *_ in ROLE if res[name]
             for m in METRICS if "t_p" in res[name]["metrics"][m]}
    adj = holm(pvals) if pvals else {}
    L.extend(["## Paired comparisons", "",
              "| comparison | what changes | seeds | " + " | ".join(f"Δ {m} (better) · t p · Holm p" for m in METRICS)
              + " | CI Δ AUROC | CI Δ AUPRC | McNemar p<0.05 |", "|---|---|---|" + "---|" * (len(METRICS) + 3)])
    for name, a, b, what in ROLE:
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

    # decomposition of the mean G0 - R0 gain (identical seeds in all three cells)
    seeds = sorted(set(R["dr_R0"]) & set(R["gs_X0"]) & set(R["gs_G0"]))
    if seeds:
        t = {k: test_metrics({s: R[k][s] for s in seeds}) for k in CELLS}
        L.extend(["## Decomposition of Gemma's gain (G0 − R0 = writer + classifier)", "",
                  f"Seeds {seeds}. Writer = X0 − R0, classifier = G0 − X0; they sum to G0 − R0 exactly. Share = part / "
                  "total of the seed-averaged gain (only meaningful when the total is clearly non-zero).", "",
                  "| metric | total G0 − R0 | writer X0 − R0 | classifier G0 − X0 | writer share |", "|---|---|---|---|---|"])
        payload["decomposition"] = {"seeds": seeds}
        for m in METRICS:
            tot = np.mean([t["gs_G0"][s][m] - t["dr_R0"][s][m] for s in seeds])
            wr = np.mean([t["gs_X0"][s][m] - t["dr_R0"][s][m] for s in seeds])
            cl = tot - wr
            share = wr / tot if abs(tot) > 1e-9 else None
            payload["decomposition"][m] = {"total": tot, "writer": wr, "classifier": cl, "writer_share": share}
            L.append(f"| {m} | {tot:+.2f} | {wr:+.2f} | {cl:+.2f} | {'—' if share is None else f'{share:.0%}'} |")
        L.append("")

    par = compare(runs("par2_R0"), runs("dr_R0"))
    payload["stack_parity"] = par
    if par:
        L.extend(["## Stack drift (this machine)", "",
                  "dr_R0's exact config (hash a1e63bf98c), seed 0, re-run on this machine's stack (par2_R0) vs dr_R0 seed 0. "
                  "A Δ of this size arises from the software stack alone; X0 differences smaller than it are not "
                  "attributable to the writer.", "", "| metric | par2_R0 | dr_R0 | Δ |", "|---|---|---|---|"])
        for m in METRICS:
            x = par["metrics"][m]
            L.append(f"| {m} | {x['arm_mean']:.2f} | {x['ref_mean']:.2f} | {x['delta_mean']:+.2f} |")
        L.append("")

    L.extend(["## Cost", "", "| run | prompt tokens mean / max | hours / run (wall) | peak GPU GB |", "|---|---|---|---|"])
    payload["cost"] = {}
    for k in CELLS:
        c = cost(k, R[k]) if R[k] else None
        payload["cost"][k] = c
        if c:
            L.append(f"| {k} | {fmt(c['final_prompt_tokens_mean'], 1)} / {fmt(c['final_prompt_tokens_max'], 0)} | "
                     f"{fmt(c['train_hours_mean'])} | {fmt(c['peak_gpu_gb'], 1)} |")
    L.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exp4.json").write_text(json.dumps(payload, indent=2, default=float))
    (OUT / "exp4.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
