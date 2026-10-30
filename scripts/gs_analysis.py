"""Gemma-swap / structured zero-shot study: paired comparisons per experiment (CPU, reads finished runs only).

    python scripts/gs_analysis.py --exp 1     # G (Gemma, original prompt)   vs R (dr study, Qwen, original prompt)
    python scripts/gs_analysis.py --exp 2     # S (Qwen, zero-shot prompt)    vs R;  par_R0 vs dr_R0 (stack drift)
    python scripts/gs_analysis.py --exp 3     # GS (Gemma, zero-shot prompt)  vs G and vs S; prompt x model interaction
    python scripts/gs_analysis.py --exp all   # all of the above + the 2x2 table

Statistics are dr_analysis.compare unchanged (paired per-seed Δ, seeds better, paired t / Wilcoxon, paired bootstrap 95% CI
over the 7,425 test reviews, prediction agreement and McNemar), so the numbers are comparable with
results/disc_resid/deviation_final.md. Also rebuilds 5-seed aggregates (results/gemma_swap/aggregates.json), replacing
results/aggregated/<experiment>.json, which single-seed jobs overwrite.
Writes results/gemma_swap/exp<N>.{json,md}.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dr_analysis import METRICS, compare, cost, fmt, runs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "gemma_swap"
ARMS = ["0", "1", "1b", "2", "2b"]
WHAT = {"0": "baseline", "1": "disc · FLAG prompt", "1b": "disc · no-verdict", "2": "disc + residual loss",
        "2b": "no-verdict disc + residual loss"}
CELL = {"R": ("dr_R{}", "Qwen · original prompt"), "G": ("gs_G{}", "Gemma · original prompt"),
        "S": ("zs_S{}", "Qwen · zero-shot prompt"), "GS": ("zs_GS{}", "Gemma · zero-shot prompt")}
PAPER = {"macro_f1": 66.91, "auroc": 77.32, "auprc": 34.63}


def cell_runs(cell, arm):
    return runs(CELL[cell][0].format(arm))


def aggregates():
    out = {}
    for cell, (pat, label) in CELL.items():
        for a in ARMS:
            r = cell_runs(cell, a)
            ms = [json.loads((p / "metrics.json").read_text())["test"] for p in r.values()]
            if ms:
                out[pat.format(a)] = {"cell": cell, "label": label, "arm": a, "seeds": sorted(r),
                                      **{f"{m}_{s}": float(f(np.array([x[m] for x in ms])))
                                         for m in METRICS for s, f in (("mean", np.mean), ("std", np.std))}}
    par = runs("par_R0")
    if par:
        out["par_R0"] = {"cell": "parity", "seeds": sorted(par),
                         **{m: json.loads((p / "metrics.json").read_text())["test"][m] for p in par.values() for m in METRICS}}
    return out


def comparisons(exp):
    """(section title, {row label: (runs a, runs b)}) for one experiment."""
    if exp == "1":
        return [("Exp 1 · Gemma vs Qwen, original prompt (G vs R)", {f"G{a}": ("G", "R", a) for a in ARMS}),
                ("Exp 1 · within Gemma (G arms vs G0)", {f"G{a}": ("G", "G", a, "0") for a in ARMS if a != "0"})]
    if exp == "2":
        return [("Exp 2 · zero-shot vs original prompt, Qwen (S vs R)", {f"S{a}": ("S", "R", a) for a in ARMS}),
                ("Exp 2 · within zero-shot Qwen (S arms vs S0)", {f"S{a}": ("S", "S", a, "0") for a in ARMS if a != "0"})]
    return [("Exp 3 · zero-shot vs original prompt, Gemma (GS vs G)", {f"GS{a}": ("GS", "G", a) for a in ARMS}),
            ("Exp 3 · Gemma vs Qwen, zero-shot prompt (GS vs S)", {f"GS{a}": ("GS", "S", a) for a in ARMS}),
            ("Exp 3 · within zero-shot Gemma (GS arms vs GS0)", {f"GS{a}": ("GS", "GS", a, "0") for a in ARMS if a != "0"})]


def run_pair(spec):
    ca, cb, a = spec[:3]
    b = spec[3] if len(spec) > 3 else a
    return compare(cell_runs(ca, a), cell_runs(cb, b)), f"{cb}{b}"


def interaction():
    """(GS - G) - (S - R) per arm and seed: does the zero-shot prompt help one model more than the other?"""
    out = {}
    for a in ARMS:
        R = {c: cell_runs(c, a) for c in CELL}
        seeds = sorted(set.intersection(*(set(v) for v in R.values()))) if all(R.values()) else []
        if not seeds:
            continue
        t = {c: {s: json.loads((R[c][s] / "metrics.json").read_text())["test"] for s in seeds} for c in CELL}
        out[a] = {"seeds": seeds, **{m: {"per_seed": [round((t["GS"][s][m] - t["G"][s][m]) - (t["S"][s][m] - t["R"][s][m]), 3)
                                                      for s in seeds]} for m in METRICS}}
        for m in METRICS:
            v = np.array(out[a][m]["per_seed"])
            out[a][m].update(mean=float(v.mean()), positive=int((v > 0).sum()))
    return out


def table(L, title, block):
    L.extend([f"## {title}", "",
              "| arm | reference | seeds | Macro-F1 | AUROC | AUPRC | Δ F1 (better) | Δ AUROC (better) | Δ AUPRC (better) | CI Δ AUROC | CI Δ AUPRC | McNemar p<0.05 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"])
    for k, (r, ref) in block.items():
        if not r:
            L.append(f"| {k} | {ref} | 0 | | | | | | | | | |")
            continue
        m, ci, n = r["metrics"], r["bootstrap_ci95"], len(r["seeds"])
        L.append(f"| {k} | {ref} | {n} | " + " | ".join(f"{m[x]['arm_mean']:.2f} ± {m[x]['arm_std']:.2f}" for x in METRICS)
                 + " | " + " | ".join(f"{m[x]['delta_mean']:+.2f} ({m[x]['seeds_better']}/{n})" for x in METRICS)
                 + f" | [{ci['auroc'][0]:+.2f}, {ci['auroc'][1]:+.2f}] | [{ci['auprc'][0]:+.2f}, {ci['auprc'][1]:+.2f}] | "
                 f"{sum(x['mcnemar_p'] < 0.05 for x in r['agreement'])}/{n} |")
    L.append("")


def two_by_two(L, agg):
    L.extend(["## 2 × 2 overview (test, mean ± std over seeds)", "",
              "| arm | " + " | ".join(CELL[c][1] for c in CELL) + " |", "|---|" + "---|" * len(CELL)])
    for metric in METRICS:
        for a in ARMS:
            cells = []
            for c in CELL:
                x = agg.get(CELL[c][0].format(a))
                cells.append(f"{x[metric + '_mean']:.2f} ± {x[metric + '_std']:.2f} (n={len(x['seeds'])})" if x else "—")
            L.append(f"| {a} · {metric} | " + " | ".join(cells) + " |")
    L.extend([f"", f"Paper (DGP, AmazonVideo): Macro-F1 {PAPER['macro_f1']}, AUROC {PAPER['auroc']}, AUPRC {PAPER['auprc']}.", ""])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["1", "2", "3", "all"], required=True)
    exp = ap.parse_args().exp
    OUT.mkdir(parents=True, exist_ok=True)
    agg = aggregates()
    (OUT / "aggregates.json").write_text(json.dumps(agg, indent=2))
    exps = ["1", "2", "3"] if exp == "all" else [exp]
    for e in exps:
        payload, L = {"exp": e, "sections": {}}, [
            f"# Gemma-swap / zero-shot study · Experiment {e}", "",
            "AmazonVideo test set (7,425 reviews). Mean ± std over seeds (ddof=0). Δ = paired per-seed difference "
            "(arm − reference); 'better' = seeds where the arm is higher. CI = paired bootstrap 95% interval of the "
            "seed-averaged Δ (2000 resamples of the test reviews). McNemar: seeds whose correctness differs at p < 0.05 "
            "(each run at its own validation threshold). Statistics: scripts/dr_analysis.py:compare.", ""]
        for title, block in comparisons(e):
            res = {k: run_pair(spec) for k, spec in block.items()}
            payload["sections"][title] = {k: {"reference": ref, **r} for k, (r, ref) in res.items()}
            table(L, title, res)
        if e == "2":
            par = compare(runs("par_R0"), runs("dr_R0"))
            payload["stack_parity"] = par
            table(L, "Stack drift: R0 seed 0 re-run on this stack (par_R0) vs the dr study's R0", {"par_R0": (par, "R0")})
        if e == "3":
            inter = interaction()
            payload["interaction"] = inter
            L.extend(["## Prompt × model interaction: (GS − G) − (S − R), per seed", "",
                      "Positive = the zero-shot prompt helps Gemma more than it helps Qwen.", "",
                      "| arm | seeds | " + " | ".join(f"{m} mean (positive seeds)" for m in METRICS) + " |",
                      "|---|---|" + "---|" * len(METRICS)])
            for a, r in inter.items():
                L.append(f"| {a} | {len(r['seeds'])} | " + " | ".join(
                    f"{r[m]['mean']:+.2f} ({r[m]['positive']}/{len(r['seeds'])})" for m in METRICS) + " |")
            L.append("")
            two_by_two(L, agg)
        cells = {"1": ["G"], "2": ["S"], "3": ["GS"]}[e]
        L.extend(["## Cost", "", "| run | prompt tokens mean / max | hours / run (wall) | peak GPU GB |", "|---|---|---|---|"])
        payload["cost"] = {}
        for c in cells + (["R"] if e != "3" else []):
            for a in ARMS:
                name = CELL[c][0].format(a)
                k = cost(name, cell_runs(c, a)) if cell_runs(c, a) else None
                payload["cost"][name] = k
                if k:
                    L.append(f"| {name} | {fmt(k['final_prompt_tokens_mean'], 1)} / {fmt(k['final_prompt_tokens_max'], 0)} | "
                             f"{fmt(k['train_hours_mean'])} | {fmt(k['peak_gpu_gb'], 1)} |")
        (OUT / f"exp{e}.json").write_text(json.dumps(payload, indent=2, default=float))
        (OUT / f"exp{e}.md").write_text("\n".join(L) + "\n")
        print("\n".join(L))


if __name__ == "__main__":
    main()
