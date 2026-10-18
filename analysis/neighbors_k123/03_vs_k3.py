"""K = 3 as the base: how much of the K = 3 neighbourhood do K = 1 and K = 2 reproduce?

Reads only analysis/out/neighbors_k123/neighbors_long.csv.gz (written by 01_build.py + 02_analyze.py); no model, GPU or cache access.
Writes analysis/out/neighbors_k123/vs_k3.json. Neighbour labels are used only for train + val targets, as in 02_analyze.py.
    PYTHONPATH=<deps> python analysis/neighbors_k123/03_vs_k3.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "out" / "neighbors_k123"
df = pd.read_csv(OUT / "neighbors_long.csv.gz", usecols=["K", "metapath", "hops", "target", "split", "rank", "neighbor",
                                                           "neighbor_label_probe_targets_only"])
targets = np.sort(df["target"].unique()); T = len(targets); tix = {int(t): i for i, t in enumerate(targets)}
sets = {K: {} for K in (1, 2, 3)}; slots = {K: np.zeros(T, int) for K in (1, 2, 3)}
for K, g in df.groupby("K"):
    slots[K] = np.bincount(g["target"].map(tix).to_numpy(), minlength=T)
    u = g.drop_duplicates(["target", "neighbor"])
    for t, nb in u.groupby("target")["neighbor"]:
        sets[K][int(t)] = set(nb.tolist())
res = {"n_targets": T, "per_K": {}}
for K in (1, 2, 3):
    jac, cov, cont, sub, miss, nuniq = [], [], [], [], [], []
    for t in targets:
        a, b = sets[K].get(int(t), set()), sets[3].get(int(t), set())
        inter = len(a & b); uni = len(a | b)
        jac.append(inter / uni if uni else 1.0); cov.append(inter / len(b) if b else 1.0); cont.append(inter / len(a) if a else 1.0)
        sub.append(a <= b); miss.append(len(b - a)); nuniq.append(len(a))
    res["per_K"][str(K)] = {"slots": float(slots[K].mean()), "slots_vs_k3": float(slots[K].mean() / slots[3].mean()),
        "distinct": float(np.mean(nuniq)), "distinct_vs_k3": float(np.mean(nuniq) / np.mean([len(sets[3].get(int(t), ())) for t in targets])),
        "jaccard_with_k3": float(np.mean(jac)), "k3_covered": float(np.mean(cov)), "inside_k3": float(np.mean(cont)),
        "subset_of_k3_pct": float(np.mean(sub)), "k3_missing": float(np.mean(miss))}
# Top-4 per (target, metapath): identical at K and at K = 3?  1-hop paths exist at every K; 2-hop at K = 2, 3.
top = {}
for (K, mp, h), g in df.groupby(["K", "metapath", "hops"]):
    top[(K, mp)] = {int(t): frozenset(x.tolist()) for t, x in g.groupby("target")["neighbor"]}
same = {}
for K in (1, 2):
    for hops in (1, 2):
        pairs = [(mp, top[(K, mp)], top[(3, mp)]) for (k, mp) in top if k == K and (mp.count("-") + 1) == hops and (3, mp) in top]
        if not pairs: continue
        eq = n = 0; jj = []
        for mp, a, b in pairs:
            for t in set(a) | set(b):
                x, y = a.get(t, frozenset()), b.get(t, frozenset()); n += 1; eq += x == y; jj.append(len(x & y) / len(x | y) if x | y else 1.0)
        same[f"{K}|{hops}"] = {"paths": len(pairs), "identical_pct": 100 * eq / n, "jaccard": float(np.mean(jj))}
res["top4_same_as_k3"] = same
# label signal relative to K = 3 (paired bootstrap over train + val targets; rate over kept slots as in 02_analyze.py)
d = df[df["split"].isin(["train", "val"])]
lab = d.groupby(["K", "target"])["neighbor_label_probe_targets_only"].mean().unstack(0)
y = d.drop_duplicates("target").set_index("target")
# target class is not in the long file: take it from the summary's probe order via 02's labels is unavailable here, so read it from the audit file
import pickle, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import context  # noqa: E402
c = context(); yv = c.labels
cls = pd.Series({int(t): int(yv[int(t)]) for t in lab.index})
rng = np.random.default_rng(0); idx = lab.index.to_numpy(); n = len(idx)
rates = {K: lab[K].to_numpy() for K in (1, 2, 3)}; c1 = cls.loc[idx].to_numpy() == 1
def gap(r, m): return np.nanmean(r[m & c1]) - np.nanmean(r[m & ~c1])
full = np.ones(n, bool); g0 = {K: gap(rates[K], full) for K in (1, 2, 3)}
boot = {K: [] for K in (1, 2)}
for _ in range(2000):
    s = rng.integers(0, n, n); m = np.zeros(n, bool); cnt = np.bincount(s, minlength=n)
    w = cnt.astype(float)
    def wg(r):
        ok = ~np.isnan(r); a = (w * ok * c1); b = (w * ok * ~c1)
        return (np.nansum(a * np.nan_to_num(r)) / a.sum()) - (np.nansum(b * np.nan_to_num(r)) / b.sum())
    g3 = wg(rates[3])
    for K in (1, 2): boot[K].append(wg(rates[K]) - g3)
res["label_gap"] = {str(K): {"gap": float(g0[K]), "diff_vs_k3": float(g0[K] - g0[3]),
                             "diff_ci": [float(np.percentile(boot[K], 2.5)), float(np.percentile(boot[K], 97.5))] if K != 3 else None} for K in (1, 2, 3)}
json.dump(res, open(OUT / "vs_k3.json", "w"), indent=1)
print(json.dumps(res, indent=1))
