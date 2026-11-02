"""Ablation w/o MDK (paper Figure 4): DGP with MDK neighbour selection (dr R0) vs the same pipeline with M random
neighbours per metapath (ctl_noMDK), AmazonVideo.

    python scripts/nomdk_analysis.py        # CPU; reads finished runs only; writes results/controls/nomdk.{json,md}

Cells (AmazonVideo, seeds 0-4, same split, Qwen3-8B summarizer and LoRA classifier, K=2, M=4, B=10, early stopping on
validation loss, validation-tuned threshold):
    dr_R0       MDK selects the M=4 neighbours per metapath                     (dr study, 2026-10-15)
    ctl_noMDK   M=4 neighbours drawn at random per metapath (mdk.random_seed 0)  (2026-11-02, scripts/yelp_nomdk_chain.sh)
                configs/experiments/controls/A1_without_mdk.yaml; the paper does not say what replaces MDK (decision D4)

Also measures how much the two selections overlap (MDK cache MDK_KEY vs random cache RANDOM_KEY, one .npz per metapath).
The paper prints no numbers for Figure 4; it states only that removing MDK "leads to a performance decline".
Statistics: scripts/dr_analysis.py:compare unchanged (paired per-seed Δ, paired t, paired bootstrap 95% CI over the
7,425 test reviews, McNemar).
"""

import json
from pathlib import Path

import numpy as np

from dr_analysis import METRICS, compare, cost, fmt, runs
from x0_analysis import test_metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "controls"
MDK_DIR = ROOT / "datasets" / "cache" / "mdk"
MDK_KEY, RANDOM_KEY = "30447a444d0aa6e2", "b697e61d334368f3"   # MDK: the Qwen-counted cache of R0 (= 173d9d5c91e37c3f, results/gemma_swap/checks/mdk_identity.md)
CELLS = {"dr_R0": "DGP: MDK selects the M = 4 neighbours", "ctl_noMDK": "w/o MDK: M = 4 random neighbours"}


def overlap() -> dict:
    """Per metapath: share of the random selection that MDK also selected, and of targets with identical sets."""
    meta = {k: json.loads((MDK_DIR / k / "metadata.json").read_text()) for k in (MDK_KEY, RANDOM_KEY)}
    assert meta[MDK_KEY]["use_mdk"] and not meta[RANDOM_KEY]["use_mdk"], "cache keys swapped"
    for f in ("dataset", "split_seed", "targets_hash", "K", "M"):
        assert meta[MDK_KEY][f] == meta[RANDOM_KEY][f], f
    out = {}
    for f in sorted(p.name for p in (MDK_DIR / MDK_KEY).glob("*.npz")):
        a, b = np.load(MDK_DIR / MDK_KEY / f), np.load(MDK_DIR / RANDOM_KEY / f)
        assert (a["targets"] == b["targets"]).all()
        shared = same = total = full = 0
        for i in range(len(a["targets"])):
            na = set(a["neighbors"][a["offsets"][i]:a["offsets"][i + 1]].tolist())
            nb = set(b["neighbors"][b["offsets"][i]:b["offsets"][i + 1]].tolist())
            shared += len(na & nb); total += len(nb); same += na == nb
            full += int(a["sizes"][i]) <= len(na)          # fewer than M+1 candidates: both keep everything
        out[f[:-4]] = {"shared_share": shared / max(total, 1), "identical_sets": same / len(a["targets"]),
                       "all_candidates_kept": full / len(a["targets"])}
    return out


def main():
    R = {k: runs(k) for k in CELLS}
    L = ["# Ablation w/o MDK: MDK neighbour selection vs random neighbours", "",
         "AmazonVideo test set (7,425 reviews), seeds 0-4, mean ± std over seeds (ddof=0). Both cells use the same "
         "summarizer, classifier, budgets and recipe; only the choice of the M = 4 neighbours per metapath differs. "
         "Δ = R0 − noMDK (positive = MDK helps); CI = paired bootstrap 95% interval over test reviews.", "",
         "| run | what | seeds | Macro-F1 | AUROC | AUPRC |", "|---|---|---|---|---|---|"]
    payload = {"cells": {}}
    for k, label in CELLS.items():
        ms = list(test_metrics(R[k]).values())
        if not ms:
            L.append(f"| {k} | {label} | 0 | | | |")
            continue
        agg = {m: (float(np.mean([x[m] for x in ms])), float(np.std([x[m] for x in ms]))) for m in METRICS}
        payload["cells"][k] = {"seeds": sorted(R[k]), **{m: {"mean": v[0], "std": v[1]} for m, v in agg.items()}}
        L.append(f"| {k} | {label} | {len(ms)} | " + " | ".join(f"{agg[m][0]:.2f} ± {agg[m][1]:.2f}" for m in METRICS) + " |")
    L.append("")

    r = compare(R["dr_R0"], R["ctl_noMDK"])
    if r:
        n = len(r["seeds"])
        L += ["## R0 − noMDK (what MDK adds)", "", "| metric | Δ (better) | t p |", "|---|---|---|"]
        for m in METRICS:
            x = r["metrics"][m]
            p = x.get("t_p")
            L.append(f"| {m} | {x['delta_mean']:+.2f} ({x['seeds_better']}/{n}) | {'—' if p is None else f'{p:.4f}'} |")
        ci = r["bootstrap_ci95"]
        L += ["", f"Bootstrap 95% CI: AUROC [{ci['auroc'][0]:+.2f}, {ci['auroc'][1]:+.2f}], "
              f"AUPRC [{ci['auprc'][0]:+.2f}, {ci['auprc'][1]:+.2f}]. McNemar p < 0.05 in "
              f"{sum(x['mcnemar_p'] < 0.05 for x in r['agreement'])}/{n} seeds.", ""]
        payload["r0_minus_nomdk"] = r

    ov = overlap()
    payload["overlap"] = ov
    L += ["## How different are the two selections?", "",
          f"MDK cache `{MDK_KEY}` vs random cache `{RANDOM_KEY}` (10,023 targets, same K, M and split). "
          "'shared' = share of the random neighbours that MDK also chose; 'identical' = targets whose two sets are equal; "
          "'all kept' = targets with at most M candidates (including none), where both keep every candidate; such targets also "
          "count as identical.", "",
          "| metapath | shared | identical | all kept |", "|---|---|---|---|"]
    for k, v in ov.items():
        L.append(f"| {k} | {v['shared_share']:.1%} | {v['identical_sets']:.1%} | {v['all_candidates_kept']:.1%} |")
    L.append("")

    c = cost("ctl_noMDK", R["ctl_noMDK"]) if R["ctl_noMDK"] else None
    payload["cost"] = c
    if c:
        L += ["## Cost (noMDK)", "", "| prompt tokens mean / max | hours / run (wall) | peak GPU GB |", "|---|---|---|",
              f"| {fmt(c['final_prompt_tokens_mean'], 1)} / {fmt(c['final_prompt_tokens_max'], 0)} | "
              f"{fmt(c['train_hours_mean'])} | {fmt(c['peak_gpu_gb'], 1)} |", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "nomdk.json").write_text(json.dumps(payload, indent=2, default=float))
    (OUT / "nomdk.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
