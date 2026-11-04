"""One statistics standard for every paired comparison in the DGP follow-up experiments (E2-E18).

    python scripts/stats_standard.py        # CPU, seconds; reads results/raw/*/metrics.json only
                                            # writes results/stats/all_comparisons.{md,json}

Each experiment was analysed with its own tools when it ran (some with bootstrap CIs over test reviews, some with
no test at all, Holm only in E14-E15). This script applies the same seed-level test to all of them:

    unit          the seed (5 per cell; arms and references share seeds 0-4, so every comparison is paired)
    statistic     mean paired difference arm - reference, in percentage points, per test metric
    test          two-sided paired t-test over seeds (df = n - 1), with its 95% t-interval for the mean difference
    correction    Holm over the 3 metrics x all comparisons of one experiment (family), and once more over all
                  comparisons of all experiments together (global, the strictest reading)

The seed-level test treats the 5 training runs as the replicates, so it captures run-to-run noise but not test-set
sampling noise (the bootstrap CIs in each study's own analysis cover that instead). Wilcoxon signed-rank is not used:
with 5 pairs its smallest two-sided p-value is 0.0625, so it can never reject at 0.05.
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

from dr_analysis import METRICS, runs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "stats"
SEPT = "dgp_amazonvideo_seed0"          # the September 5-seed reproduction (original stack)

# (experiment, label, arm, reference, what the difference measures)
COMPARISONS = [
    *[("E2", f"arm {a} − Sept. baseline", f"ext_arm{a}", SEPT, "FLAG text / residual-loss arm vs baseline (other stack)")
      for a in ("B", "C", "BC", "D", "E1", "E2")],
    *[("E3", f"{k} − Sept. baseline", f"ext_{k}", SEPT, "metapath depth vs K = 2 (other stack)") for k in ("K1", "K3")],
    *[("E8", f"{a} − R0", f"dr_{a}", "dr_R0", "FLAG prompt variant vs same-stack baseline")
      for a in ("R1", "R1b", "R2", "R2b")],
    ("E10", "F0 − R0", "fa_F0", "dr_R0", "same configuration re-run (stack / noise check)"),
    *[("E11", f"G{a} − R{a}", f"gs_G{a}", f"dr_R{a}", "Gemma-2-9B-it instead of Qwen3-8B (writer + classifier)")
      for a in ("0", "1", "1b", "2", "2b")],
    *[("E12", f"S{a} − R{a}", f"zs_S{a}", f"dr_R{a}", "structured zero-shot prompt, Qwen")
      for a in ("0", "1", "1b", "2", "2b")],
    *[("E13", f"GS{a} − G{a}", f"zs_GS{a}", f"gs_G{a}", "structured zero-shot prompt, Gemma")
      for a in ("0", "1", "1b", "2", "2b")],
    ("E14", "X0 − R0", "gs_X0", "dr_R0", "Gemma's text, Qwen classifier"),
    ("E14", "G0 − X0", "gs_G0", "gs_X0", "Gemma as classifier, Gemma's text"),
    ("E14", "G0 − R0", "gs_G0", "dr_R0", "both roles"),
    ("E15", "R0 − T0", "dr_R0", "ctl_T0", "graph context, classifier fixed"),
    ("E15", "T0 − MLP", "ctl_T0", "mlp_amazonvideo", "fine-tuned LLM vs MLP"),
    ("E15", "R0 − MLP", "dr_R0", "mlp_amazonvideo", "DGP vs MLP"),
    ("E16", "DGP − MLP (YelpChi)", "dgp_yelpchi", "mlp_yelpchi", "DGP vs MLP on a second dataset"),
    ("E17", "R0 − noMDK", "dr_R0", "ctl_noMDK", "MDK vs random neighbours"),
    ("E18", "QG0 − R0", "gs_QG0", "dr_R0", "Gemma as classifier, Qwen's text"),
    ("E18", "G0 − QG0", "gs_G0", "gs_QG0", "Gemma's text, Gemma classifier"),
    # E19 (2026-11-04): epoch-selection rules on one trajectory per seed; "<exp>@<rule>" reads metrics.json "selection".
    # YelpChi only: AmazonVideo has one complete seed (the search was stopped by the owner).
    ("E19", "best val AUROC − paper rule (YelpChi)", "sel_yelpchi@auroc_all", "sel_yelpchi@loss_es2", "epoch = best val AUROC of 10"),
    ("E19", "AUROC patience 2 − paper rule (YelpChi)", "sel_yelpchi@auroc_es2", "sel_yelpchi@loss_es2", "early stop on val AUROC, patience 2"),
    ("E19", "AUROC patience 3 − paper rule (YelpChi)", "sel_yelpchi@auroc_es3", "sel_yelpchi@loss_es2", "early stop on val AUROC, patience 3"),
]
LABEL = {"macro_f1": "Macro-F1", "auroc": "AUROC", "auprc": "AUPRC"}


def holm(p: dict) -> dict:
    keys = sorted(p, key=p.get)
    out, running = {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (len(keys) - i) * p[k]))
        out[k] = running
    return out


def test_metrics(prefix: str) -> dict[int, dict]:
    if "@" in prefix:                                   # E19: one selection rule of a multi-rule run
        exp, rule = prefix.split("@")
        out = {}
        for s, d in runs(exp).items():
            m = json.loads((d / "metrics.json").read_text())
            out[s] = m["selection"][rule]["test"]
        return out
    return {s: json.loads((d / "metrics.json").read_text())["test"] for s, d in runs(prefix).items()}


def main():
    cache, rows = {}, []
    for exp, label, arm, ref, what in COMPARISONS:
        for prefix in (arm, ref):
            cache.setdefault(prefix, test_metrics(prefix))
        seeds = sorted(set(cache[arm]) & set(cache[ref]))
        for m in METRICS:
            d = np.array([cache[arm][s][m] - cache[ref][s][m] for s in seeds])
            n = len(d)
            t = stats.ttest_rel([cache[arm][s][m] for s in seeds], [cache[ref][s][m] for s in seeds]) if n > 1 else None
            half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
            rows.append({"experiment": exp, "comparison": label, "arm": arm, "reference": ref, "what": what,
                         "metric": m, "seeds": seeds, "delta_mean": float(d.mean()), "delta_per_seed": d.round(4).tolist(),
                         "seeds_better": int((d > 0).sum()), "ci95": [float(d.mean() - half), float(d.mean() + half)],
                         "t": float(t.statistic) if t else None, "p": float(t.pvalue) if t else None})
    fam = {}
    for exp in dict.fromkeys(r["experiment"] for r in rows):
        idx = [i for i, r in enumerate(rows) if r["experiment"] == exp and r["p"] is not None]
        fam.update(holm({i: rows[i]["p"] for i in idx}))
    glob = holm({i: r["p"] for i, r in enumerate(rows) if r["p"] is not None})
    for i, r in enumerate(rows):
        r["holm_family"], r["holm_global"] = fam.get(i), glob.get(i)

    L = ["# One statistics standard for E2-E19", "",
         "Every paired comparison of the DGP follow-up experiments, re-tested the same way (`scripts/stats_standard.py`). "
         "Test set, AmazonVideo (E16, E19: YelpChi), seeds 0-4. Δ = mean paired difference arm − reference in pp; 'better' = seeds where the "
         "arm is higher; CI = 95% t-interval of the mean seed difference; p = two-sided paired t over seeds (df 4). Holm "
         "(family) corrects over 3 metrics × the comparisons of one experiment; Holm (global) over all "
         f"{len(glob)} tests. Bold = Holm (family) < 0.05. Seed-level only: test-set sampling noise is not included "
         "(see each study's bootstrap CIs).", ""]
    for exp in dict.fromkeys(r["experiment"] for r in rows):
        L += [f"## {exp}", "", "| comparison | metric | Δ (better) | 95% CI | p | Holm family | Holm global |",
              "|---|---|---|---|---|---|---|"]
        for r in (x for x in rows if x["experiment"] == exp):
            sig = r["holm_family"] is not None and r["holm_family"] < 0.05
            b = "**" if sig else ""
            L.append(f"| {r['comparison']} | {LABEL[r['metric']]} | {b}{r['delta_mean']:+.2f}{b} ({r['seeds_better']}/"
                     f"{len(r['seeds'])}) | [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] | {r['p']:.4f} | "
                     f"{r['holm_family']:.4f} | {r['holm_global']:.4f} |")
        L.append("")
    survivors = [r for r in rows if r["holm_global"] is not None and r["holm_global"] < 0.05]
    L += ["## Survives the global Holm correction", "",
          "| experiment | comparison | metric | Δ | Holm global |", "|---|---|---|---|---|"]
    L += [f"| {r['experiment']} | {r['comparison']} | {LABEL[r['metric']]} | {r['delta_mean']:+.2f} | {r['holm_global']:.4f} |"
          for r in survivors] or ["| — | none | | | |"]
    L.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "all_comparisons.json").write_text(json.dumps({"comparisons": rows, "n_tests": len(glob)}, indent=2))
    (OUT / "all_comparisons.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
