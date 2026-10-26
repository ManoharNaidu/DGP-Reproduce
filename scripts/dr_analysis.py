"""Disc/resid study: how each arm deviates from the reference run (CPU, reads finished runs only).

    python scripts/dr_analysis.py                        # writes results/disc_resid/deviation.{json,md}
    python scripts/dr_analysis.py --stem deviation_final # same analysis, other file name (keeps earlier snapshots)

Reference = R0 (the original DGP configuration re-run on this stack); R0 is also compared with the original
September run (results/raw/dgp_amazonvideo_seed0_seed{s}_f12e6c750d). Per arm, using only seeds finished for both:
  * test Macro-F1 / AUROC / AUPRC: mean ± std (ddof=0), paired per-seed delta, seeds better, paired t and Wilcoxon p
    (with 5 seeds the smallest two-sided Wilcoxon p is 0.0625);
  * paired bootstrap 95% CI of the seed-averaged delta in AUROC / AUPRC, resampling test reviews (same resample for
    every seed and both runs);
  * prediction agreement: Spearman correlation of p_fraud, agreement of hard labels at each run's own threshold,
    McNemar exact p on correctness, and flips (wrong -> right, right -> wrong);
  * cost: mean final prompt tokens, training time, peak GPU memory.
Writes results/disc_resid/<stem>.{json,md} (default stem: deviation).
"""

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, spearmanr, ttest_rel, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
ARMS = {"R0": "dr_R0", "R1": "dr_R1", "R1b": "dr_R1b", "R2": "dr_R2", "R2b": "dr_R2b"}
ORIGINAL = "dgp_amazonvideo_seed0"
METRICS = ("macro_f1", "auroc", "auprc")
N_BOOT = 2000


def runs(prefix: str) -> dict[int, Path]:
    out = {}
    for d in glob.glob(str(RAW / f"{prefix}_seed*")):
        p = Path(d)
        if (p / "metrics.json").exists() and p.name.rsplit("_seed", 1)[0] == prefix:
            out[json.loads((p / "metrics.json").read_text())["seed"]] = p
    return out


def preds(run: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes, y, p = [], [], []
    with open(run / "predictions.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                nodes.append(int(row["node"])); y.append(int(row["label"])); p.append(float(row["p_fraud"]))
    order = np.argsort(nodes)
    return np.asarray(nodes)[order], np.asarray(y)[order], np.asarray(p)[order]


def compare(a: dict[int, Path], b: dict[int, Path]) -> dict:
    """Deviation of runs `a` from reference runs `b` over their common seeds."""
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return {}
    ma = {s: json.loads((a[s] / "metrics.json").read_text())["test"] for s in seeds}
    mb = {s: json.loads((b[s] / "metrics.json").read_text())["test"] for s in seeds}
    res = {"seeds": seeds, "metrics": {}}
    for m in METRICS:
        va, vb = np.array([ma[s][m] for s in seeds]), np.array([mb[s][m] for s in seeds])
        d = va - vb
        r = {"arm_mean": va.mean(), "arm_std": va.std(), "ref_mean": vb.mean(), "ref_std": vb.std(),
             "delta_mean": d.mean(), "delta_per_seed": d.round(3).tolist(), "seeds_better": int((d > 0).sum())}
        if len(seeds) >= 2 and np.any(d != 0):
            r["t_p"] = float(ttest_rel(va, vb).pvalue)
            r["wilcoxon_p"] = float(wilcoxon(va, vb).pvalue)
        res["metrics"][m] = r

    # paired bootstrap over test reviews, same indices for every seed and both runs
    P = {s: (preds(a[s]), preds(b[s])) for s in seeds}
    y = P[seeds[0]][0][1]
    for s in seeds:
        assert (P[s][0][0] == P[s][1][0]).all() and (P[s][0][1] == y).all(), "test sets differ"
    rng = np.random.default_rng(0)
    boot = {"auroc": [], "auprc": []}
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        for m, fn in (("auroc", roc_auc_score), ("auprc", average_precision_score)):
            boot[m].append(np.mean([fn(y[idx], P[s][0][2][idx]) - fn(y[idx], P[s][1][2][idx]) for s in seeds]) * 100)
    res["bootstrap_ci95"] = {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for m, v in boot.items()}

    # prediction-level agreement at each run's own (validation-chosen) threshold
    agree = []
    for s in seeds:
        (_, _, pa), (_, _, pb) = P[s]
        ha, hb = pa >= ma[s]["threshold"], pb >= mb[s]["threshold"]
        ca, cb = ha == y.astype(bool), hb == y.astype(bool)
        fixed, broken = int((ca & ~cb).sum()), int((~ca & cb).sum())
        agree.append({"seed": s, "spearman": float(spearmanr(pa, pb).statistic), "label_agreement": float((ha == hb).mean()),
                      "wrong_to_right": fixed, "right_to_wrong": broken,
                      "mcnemar_p": float(binomtest(fixed, fixed + broken, 0.5).pvalue) if fixed + broken else 1.0})
    res["agreement"] = agree
    return res


def cost(prefix: str, r: dict[int, Path]) -> dict:
    man = [json.loads((p / "run_manifest.json").read_text()) for p in r.values() if (p / "run_manifest.json").exists()]
    tok = list((ROOT / "results" / "token_usage").glob(f"token_usage_{prefix}.json"))
    t = json.loads(tok[0].read_text()) if tok else {}
    t = t.get("summary", t)
    return {"final_prompt_tokens_mean": t.get("mean_final_prompt_tokens"), "final_prompt_tokens_max": t.get("max_prompt_length"),
            "train_hours_mean": float(np.mean([m["runtime_seconds"] for m in man])) / 3600 if man else None,
            "peak_gpu_gb": max((m.get("peak_memory") or {}).get("gpu_peak_gb", 0) for m in man) if man else None}


def fmt(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="deviation", help="output file name without extension")
    stem = ap.parse_args().stem
    R = {k: runs(v) for k, v in ARMS.items()}
    orig = runs(ORIGINAL)
    out = {"vs_R0": {k: compare(R[k], R["R0"]) for k in ARMS if k != "R0"},
           "R0_vs_original": compare(R["R0"], orig),
           "vs_original": {k: compare(R[k], orig) for k in ARMS if k != "R0"},
           "cost": {k: cost(v, R[k]) for k, v in ARMS.items()},
           "seeds_done": {k: sorted(v) for k, v in R.items()}}
    dst = ROOT / "results" / "disc_resid"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / f"{stem}.json").write_text(json.dumps(out, indent=2, default=float))

    L = ["# Disc/resid study: deviation from the reference run", "",
         "Test set, AmazonVideo. Mean ± std over seeds (ddof=0). Δ = paired per-seed difference to the reference; "
         "'better' = seeds where the arm is higher. CI = paired bootstrap 95% interval of the seed-averaged Δ "
         f"({N_BOOT} resamples of the 7,425 test reviews).", "",
         f"Seeds finished: " + ", ".join(f"{k} {v}" for k, v in out["seeds_done"].items()), ""]

    def table(title, block):
        L.extend([f"## {title}", "",
                  "| arm | seeds | Macro-F1 | AUROC | AUPRC | Δ F1 (better) | Δ AUROC (better) | Δ AUPRC (better) | CI Δ AUROC | CI Δ AUPRC |",
                  "|---|---|---|---|---|---|---|---|---|---|"])
        for k, r in block.items():
            if not r:
                L.append(f"| {k} | 0 | | | | | | | | |"); continue
            m = r["metrics"]; ci = r["bootstrap_ci95"]
            L.append(f"| {k} | {len(r['seeds'])} | " + " | ".join(f"{m[x]['arm_mean']:.2f} ± {m[x]['arm_std']:.2f}" for x in METRICS)
                     + " | " + " | ".join(f"{m[x]['delta_mean']:+.2f} ({m[x]['seeds_better']}/{len(r['seeds'])})" for x in METRICS)
                     + f" | [{ci['auroc'][0]:+.2f}, {ci['auroc'][1]:+.2f}] | [{ci['auprc'][0]:+.2f}, {ci['auprc'][1]:+.2f}] |")
        L.append("")

    table("Arms vs R0 (same stack, same seeds)", out["vs_R0"])
    table("R0 vs the original September run (stack parity)", {"R0": out["R0_vs_original"]})
    table("Arms vs the original September run", out["vs_original"])

    L.extend(["## Prediction agreement with R0 (mean over seeds)", "",
              "| arm | Spearman p_fraud | label agreement | wrong→right | right→wrong | seeds with McNemar p < 0.05 |",
              "|---|---|---|---|---|---|"])
    for k, r in out["vs_R0"].items():
        if r:
            a = r["agreement"]
            L.append(f"| {k} | {np.mean([x['spearman'] for x in a]):.3f} | {np.mean([x['label_agreement'] for x in a]) * 100:.1f}% | "
                     f"{np.mean([x['wrong_to_right'] for x in a]):.0f} | {np.mean([x['right_to_wrong'] for x in a]):.0f} | "
                     f"{sum(x['mcnemar_p'] < 0.05 for x in a)}/{len(a)} |")
    L.extend(["", "## Cost", "", "| arm | prompt tokens mean / max | train h / seed | peak GPU GB |", "|---|---|---|---|"])
    for k, c in out["cost"].items():
        L.append(f"| {k} | {fmt(c['final_prompt_tokens_mean'], 1)} / {fmt(c['final_prompt_tokens_max'], 0)} | "
                 f"{fmt(c['train_hours_mean'])} | {fmt(c['peak_gpu_gb'], 1)} |")
    (dst / f"{stem}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
