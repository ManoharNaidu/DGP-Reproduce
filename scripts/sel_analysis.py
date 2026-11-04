"""E19: epoch-selection rule (2026-11-04). The owner stopped the search after the 5 YelpChi seeds and AmazonVideo seed 0.

    python scripts/sel_analysis.py          # CPU; writes results/sel/e19.{json,md}

Each run trained all 10 epochs; metrics.json "selection" holds the test metrics at the epoch each rule kept, and
predictions_epoch<e>.csv the per-review scores. Paired per seed against the paper rule (loss_es2): mean Δ, seeds better,
paired t (Holm over 3 rules × 3 metrics), and a paired bootstrap 95% CI over the test reviews (2000 resamples; the same
resample for both rules, Δ averaged over seeds). Also compares loss_es2 with the original E16 runs (dgp_yelpchi): the
same configuration and rule, so their difference is run-to-run drift (GPU non-determinism, other software stack).
"""

import csv
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "results" / "raw", ROOT / "results" / "sel"
RULES = {"loss_es2": "paper rule: lowest val loss, early stop patience 2", "auroc_all": "best val AUROC of 10 epochs",
         "auroc_es2": "early stop on val AUROC, patience 2", "auroc_es3": "early stop on val AUROC, patience 3"}
METRICS = ("macro_f1", "auroc", "auprc")
PAPER = {"sel_yelpchi": (69.07, 84.28, 48.87), "sel_amazonvideo": (66.91, 77.32, 34.63)}
ORIG = {"sel_yelpchi": "dgp_yelpchi", "sel_amazonvideo": "dgp_amazonvideo"}
B = 2000


def runs(exp):
    out = {}
    for d in glob.glob(str(RAW / f"{exp}_seed*")):
        p = Path(d)
        if (p / "metrics.json").exists() and p.name.rsplit("_seed", 1)[0] == exp:
            out[json.loads((p / "metrics.json").read_text())["seed"]] = p
    return dict(sorted(out.items()))


def test_scores(run, epoch=None):
    f = run / (f"predictions_epoch{epoch}.csv" if epoch else "predictions.csv")
    rows = [r for r in csv.DictReader(open(f, encoding="utf-8")) if r["split"] == "test"]
    rows.sort(key=lambda r: int(r["node"]))
    return np.array([int(r["label"]) for r in rows]), np.array([float(r["p_fraud"]) for r in rows])


def boot_ci(pairs):
    """pairs: per seed (y, p_rule, p_ref) on the same test reviews -> 95% CI of mean Δ AUROC and Δ AUPRC (pp)."""
    rng = np.random.default_rng(0)
    n = len(pairs[0][0])
    d = np.zeros((B, 2))
    for b in range(B):
        idx = rng.integers(0, n, n)
        acc = []
        for y, a, r in pairs:
            yy = y[idx]
            acc.append([roc_auc_score(yy, a[idx]) - roc_auc_score(yy, r[idx]),
                        average_precision_score(yy, a[idx]) - average_precision_score(yy, r[idx])])
        d[b] = np.mean(acc, axis=0) * 100
    return {"auroc": np.percentile(d[:, 0], [2.5, 97.5]).tolist(), "auprc": np.percentile(d[:, 1], [2.5, 97.5]).tolist()}


def holm(p):
    keys = sorted(p, key=p.get)
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (len(keys) - i) * p[k]))
        out[k] = run
    return out


def main():
    payload, L = {}, ["# E19 · epoch-selection rule: validation loss (paper) vs validation AUROC", "",
                      "Default DGP configuration (Qwen3-8B, K=2, M=4, B=10, LoRA r=8, lr 1e-4); every run trained all 10 epochs and "
                      "the test set was scored at the epoch each rule kept, so all rules are compared on the same seed and the same "
                      "training trajectory. Stopped by the owner on 2026-11-04 after YelpChi seeds 0-4 and AmazonVideo seed 0 "
                      "(AmazonVideo seed 1 was stopped during its last test pass; seeds 2-4 were not run). Δ = rule − paper rule, pp; "
                      "(better) = seeds where the rule is higher; t p = paired t over seeds, Holm over 3 rules × 3 metrics; CI = paired "
                      "bootstrap 95% over test reviews (2000 resamples). `scripts/sel_analysis.py`.", ""]
    for exp in ("sel_yelpchi", "sel_amazonvideo"):
        R = runs(exp)
        if not R:
            continue
        M = {s: json.loads((d / "metrics.json").read_text())["selection"] for s, d in R.items()}
        seeds = list(M)
        name = "YelpChi" if "yelp" in exp else "AmazonVideo"
        L += [f"## {name} ({len(seeds)} seed{'s' if len(seeds) > 1 else ''}: {seeds})", "",
              "| rule | kept epochs | Macro-F1 | AUROC | AUPRC |", "|---|---|---|---|---|"]
        res = {"seeds": seeds, "rules": {}}
        for r, what in RULES.items():
            v = {m: np.array([M[s][r]["test"][m] for s in seeds]) for m in METRICS}
            eps = [M[s][r]["epoch"] for s in seeds]
            res["rules"][r] = {"epochs": eps, **{m: {"mean": float(v[m].mean()), "std": float(v[m].std()),
                                                     "per_seed": v[m].round(4).tolist()} for m in METRICS}}
            L.append(f"| {r} ({what}) | {', '.join(map(str, eps))} | " +
                     " | ".join(f"{v[m].mean():.2f} ± {v[m].std():.2f}" for m in METRICS) + " |")
        L.append(f"| paper (Table 2) | — | " + " | ".join(f"{x:.2f}" for x in PAPER[exp]) + " |")
        L.append("")
        if len(seeds) > 1:
            pv, comp = {}, {}
            for r in list(RULES)[1:]:
                comp[r] = {}
                for m in METRICS:
                    d = np.array([M[s][r]["test"][m] - M[s]["loss_es2"]["test"][m] for s in seeds])
                    t = stats.ttest_rel([M[s][r]["test"][m] for s in seeds], [M[s]["loss_es2"]["test"][m] for s in seeds])
                    p = float(t.pvalue) if np.isfinite(t.pvalue) else None
                    comp[r][m] = {"delta_mean": float(d.mean()), "per_seed": d.round(4).tolist(),
                                  "seeds_better": int((d > 0).sum()), "t_p": p}
                    if p is not None:
                        pv[(r, m)] = p
                pairs = []
                for s in seeds:
                    y, a = test_scores(R[s], M[s][r]["epoch"])
                    _, b = test_scores(R[s], M[s]["loss_es2"]["epoch"])
                    pairs.append((y, a, b))
                comp[r]["bootstrap_ci95"] = boot_ci(pairs)
            adj = holm(pv)
            L += ["| rule − paper rule | " + " | ".join(f"Δ {m} (better) · t p · Holm" for m in METRICS)
                  + " | CI Δ AUROC | CI Δ AUPRC |", "|---|" + "---|" * (len(METRICS) + 2)]
            for r, c in comp.items():
                cells = []
                for m in METRICS:
                    x = c[m]
                    x["holm"] = adj.get((r, m))
                    cells.append(f"{x['delta_mean']:+.2f} ({x['seeds_better']}/{len(seeds)}) · "
                                 + (f"{x['t_p']:.3f} · {x['holm']:.3f}" if x["t_p"] is not None else "— · —"))
                ci = c["bootstrap_ci95"]
                L.append(f"| {r} | " + " | ".join(cells) + f" | [{ci['auroc'][0]:+.2f}, {ci['auroc'][1]:+.2f}] | "
                         f"[{ci['auprc'][0]:+.2f}, {ci['auprc'][1]:+.2f}] |")
            res["comparisons"] = comp
            L.append("")
        orig = runs(ORIG[exp])
        common = [s for s in seeds if s in orig]
        if common:
            om = {s: json.loads((orig[s] / "metrics.json").read_text()) for s in common}
            L += [f"Drift check, paper rule here vs the original runs `{ORIG[exp]}` (same configuration and rule):", "",
                  "| seed | kept epoch (orig → here) | AUROC orig → here | AUPRC orig → here |", "|---|---|---|---|"]
            drift = {}
            for s in common:
                a, b = om[s]["test"], M[s]["loss_es2"]["test"]
                drift[s] = {"epoch": [om[s]["best_epoch"], M[s]["loss_es2"]["epoch"]], "auroc": [a["auroc"], b["auroc"]],
                            "auprc": [a["auprc"], b["auprc"]]}
                L.append(f"| {s} | {om[s]['best_epoch']} → {M[s]['loss_es2']['epoch']} | {a['auroc']:.2f} → {b['auroc']:.2f} | "
                         f"{a['auprc']:.2f} → {b['auprc']:.2f} |")
            dA = np.mean([drift[s]["auroc"][1] - drift[s]["auroc"][0] for s in common])
            dP = np.mean([drift[s]["auprc"][1] - drift[s]["auprc"][0] for s in common])
            L += ["", f"Mean drift: AUROC {dA:+.2f}, AUPRC {dP:+.2f} pp.", ""]
            res["drift_vs_original"] = {"per_seed": drift, "auroc_mean": float(dA), "auprc_mean": float(dP)}
        payload[exp] = res
    p1 = sorted(RAW.glob("sel_amazonvideo_seed1_*"))
    if p1:
        h = [json.loads(x) for x in open(p1[0] / "training_log.jsonl")]
        L += ["## AmazonVideo seed 1 (incomplete)", "",
              f"Trained all {len(h)} epochs; the paper rule keeps epoch {min(h, key=lambda r: r['val_loss'])['epoch']}, best val "
              f"AUROC is epoch {max(h, key=lambda r: r['val_auroc'])['epoch']}. Only the latter's test pass finished "
              "(`predictions_epoch1.csv`), so the seed has no paired comparison and is left out above.", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e19.json").write_text(json.dumps(payload, indent=2, default=float))
    (OUT / "e19.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
