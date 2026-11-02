"""Exp 4, fourth cell: the full 2 x 2 writer x classifier design (Qwen3-8B vs Gemma-2-9B-it in each role).

    python scripts/qg0_analysis.py          # CPU; reads finished runs only; writes results/gemma_swap/exp4_2x2.{json,md}

Cells (all dr arm R0 = original DGP prompt, AmazonVideo, seeds 0-4):
                     classifier Qwen     classifier Gemma
    writer Qwen      dr_R0               gs_QG0  (this cell; dr_R0's cached Qwen text, scripts/qg0_queue.sh)
    writer Gemma     gs_X0 (E14)         gs_G0   (E11)

    Classifier effect:  QG0 - R0 (on Qwen's text),     G0 - X0 (on Gemma's text, E14)
    Writer effect:      X0 - R0 (with Qwen classifier, E14), G0 - QG0 (with Gemma classifier)
    Interaction:        (G0 - QG0) - (X0 - R0) = (G0 - X0) - (QG0 - R0), per seed; 0 = the two roles add up.
    E14 split G0 - R0 in one order only (writer first); the two orders agree exactly when the interaction is 0.

Statistics: scripts/dr_analysis.py:compare unchanged (paired per-seed Δ, paired t, paired bootstrap 95% CI over the
7,425 test reviews, McNemar), Holm over the 4 role comparisons x 3 metrics; the interaction gets a paired t over seeds.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp

from dr_analysis import METRICS, compare, cost, fmt, runs
from x0_analysis import holm, test_metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "gemma_swap"
CELLS = {"dr_R0": "Qwen writes · Qwen classifies", "gs_QG0": "Qwen writes · Gemma classifies",
         "gs_X0": "Gemma writes · Qwen classifies", "gs_G0": "Gemma writes · Gemma classifies"}
ROLE = [("QG0 vs R0", "gs_QG0", "dr_R0", "Gemma as classifier (text: Qwen's)"),
        ("G0 vs X0", "gs_G0", "gs_X0", "Gemma as classifier (text: Gemma's)"),
        ("X0 vs R0", "gs_X0", "dr_R0", "Gemma's text (classifier: Qwen)"),
        ("G0 vs QG0", "gs_G0", "gs_QG0", "Gemma's text (classifier: Gemma)")]


def main():
    R = {k: runs(k) for k in CELLS}
    L = ["# Gemma-swap study · Experiment 4, fourth cell: the 2 × 2 writer × classifier design", "",
         "QG0 keeps Qwen3-8B's text (dr_R0's cached summaries, reused word for word) and makes Gemma-2-9B-it the "
         "classifier. With R0, X0 and G0 it completes the 2 × 2 design. AmazonVideo test set (7,425 reviews), dr arm R0 "
         "(original DGP prompt). Mean ± std over seeds (ddof=0). Δ = paired per-seed difference; 'better' = seeds where "
         "the first run is higher. CI = paired bootstrap 95% interval (2000 resamples of the test reviews). Holm: paired-t "
         "p-values corrected over the 4 comparisons × 3 metrics below. Statistics: scripts/dr_analysis.py:compare.", "",
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
    L.append("")

    res = {name: compare(R[a], R[b]) for name, a, b, _ in ROLE}
    pvals = {(name, m): res[name]["metrics"][m]["t_p"] for name, *_ in ROLE if res[name]
             for m in METRICS if res[name]["metrics"][m].get("t_p") is not None}
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

    seeds = sorted(set.intersection(*(set(R[k]) for k in CELLS)))
    if seeds:
        t = {k: test_metrics({s: R[k][s] for s in seeds}) for k in CELLS}
        d = lambda a, b, m: np.array([t[a][s][m] - t[b][s][m] for s in seeds])  # noqa: E731
        L.extend(["## Interaction and the two orders of decomposition", "",
                  f"Seeds {seeds}. Interaction = (G0 − QG0) − (X0 − R0) per seed (0 = writer and classifier effects add up); "
                  "p = one-sample t over seeds. Writer-first order (E14): writer X0 − R0, then classifier G0 − X0. "
                  "Classifier-first order: classifier QG0 − R0, then writer G0 − QG0. Both sum to G0 − R0.", "",
                  "| metric | G0 − R0 | interaction (seeds > 0) · p | writer-first: writer / classifier | "
                  "classifier-first: classifier / writer |", "|---|---|---|---|---|"])
        payload["interaction"] = {"seeds": seeds}
        for m in METRICS:
            tot = d("gs_G0", "dr_R0", m)
            inter = d("gs_G0", "gs_QG0", m) - d("gs_X0", "dr_R0", m)
            p = float(ttest_1samp(inter, 0).pvalue) if len(seeds) > 1 and inter.std() > 0 else None
            w1, c1 = d("gs_X0", "dr_R0", m).mean(), d("gs_G0", "gs_X0", m).mean()
            c2, w2 = d("gs_QG0", "dr_R0", m).mean(), d("gs_G0", "gs_QG0", m).mean()
            payload["interaction"][m] = {"total": float(tot.mean()), "interaction_mean": float(inter.mean()),
                                         "interaction_per_seed": inter.round(3).tolist(), "t_p": p,
                                         "writer_first": {"writer": float(w1), "classifier": float(c1)},
                                         "classifier_first": {"classifier": float(c2), "writer": float(w2)}}
            L.append(f"| {m} | {tot.mean():+.2f} | {inter.mean():+.2f} ({int((inter > 0).sum())}/{len(seeds)}) · "
                     f"{'—' if p is None else f'{p:.3f}'} | {w1:+.2f} / {c1:+.2f} | {c2:+.2f} / {w2:+.2f} |")
        L.append("")

    L.extend(["## Cost", "", "| run | prompt tokens mean / max | hours / run (wall) | peak GPU GB |", "|---|---|---|---|"])
    payload["cost"] = {}
    for k in CELLS:
        c = cost(k, R[k]) if R[k] else None
        payload["cost"][k] = c
        if c:
            L.append(f"| {k} | {fmt(c['final_prompt_tokens_mean'], 1)} / {fmt(c['final_prompt_tokens_max'], 0)} | "
                     f"{fmt(c['train_hours_mean'])} | {fmt(c['peak_gpu_gb'], 1)} |")
    L.extend(["", "Prompt tokens are counted with each cell's classifier tokenizer, so Qwen and Gemma counts of the same "
              "text differ. Training hours are not comparable across machines or lane counts.", ""])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exp4_2x2.json").write_text(json.dumps(payload, indent=2, default=float))
    (OUT / "exp4_2x2.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
