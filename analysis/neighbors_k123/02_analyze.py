"""Phases 2-5: aggregate the per-metapath work files into the tables, tests and case studies of the K = 1, 2, 3 neighbour inspection.

Reads analysis/out/neighbors_k123/work/*.pkl (made by 01_build.py). Writes, next to them:
  neighbors_k123_summary.json, per_metapath.csv, neighbors_long.csv.gz (every kept neighbour of every target, all K).
Labels: neighbour labels are used only for train+val targets (their labels are the diagnostic); the audit-sample baselines use
the 400 train+val audit targets. Test targets contribute only structure, distance, rating and text statistics.
"""
import csv
import gzip
import json
import pickle
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from common import KS, M, OUT, context
from dgp_repro.mdk import metapath_name

WORK = OUT / "work"


def jac(a, b):
    a, b = set(a), set(b)
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def boot_gap(rate_f, rate_b, n=2000, seed=0):
    rng = np.random.default_rng(seed); rate_f, rate_b = np.asarray(rate_f), np.asarray(rate_b)
    g = [rng.choice(rate_f, len(rate_f)).mean() - rng.choice(rate_b, len(rate_b)).mean() for _ in range(n)]
    return float(rate_f.mean() - rate_b.mean()), [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))]


def main():
    c = context(); y = c.labels; T = len(c.targets); tpos = {int(v): i for i, v in enumerate(c.targets)}
    probe_idx = np.array([tpos[int(v)] for v in c.probe]); probe_lab = y[c.probe]
    split_of = np.array(["train"] * len(c.split.train) + ["val"] * len(c.split.val) + ["test"] * len(c.split.test))
    R = {}
    for K in KS:
        for p in c.paths[K]:
            n = metapath_name(p)
            with open(WORK / f"K{K}__{n}.pkl", "rb") as f: R[(K, n)] = pickle.load(f)
    out = {"base_fraud_rate": float((y == 1).mean()), "n_targets": T, "n_probe": int(len(probe_idx)), "n_probe_fraud": int((probe_lab == 1).sum())}

    # ---------- Phase 0: verification ----------
    out["verify"] = {str(K): {"tasks": sum(1 for k in R if k[0] == K),
                              "audited_pairs": int(sum(len(R[k]["A"]["target"]) for k in R if k[0] == K)),
                              "mismatches": int(sum(np.sum(R[k]["A"]["mismatch"]) for k in R if k[0] == K))} for K in KS}

    # ---------- A. structure per K x hop, and per metapath ----------
    per_path, struct = [], {}
    for (K, n), r in sorted(R.items(), key=lambda kv: (kv[0][0], len(kv[0][1]), kv[0][1])):
        kept = (r["kid"] >= 0).sum(1); sz = r["sizes"]
        tokk = np.array([c.tok[row[row >= 0]].sum() for row in r["kid"]])
        rt = np.where(r["kid"] >= 0, c.rating[np.where(r["kid"] >= 0, r["kid"], 0)] == c.rating[c.targets][:, None], np.nan)
        rt = np.where(r["kid"] >= 0, rt, np.nan)
        A = r["A"]
        row = dict(K=K, metapath=n, hops=r["hops"], mean_full_nbrs=float(sz.mean()), median_full_nbrs=float(np.median(sz)), p95_full_nbrs=float(np.percentile(sz, 95)),
                   empty_pct=100 * float((sz == 0).mean()), lt_M_pct=100 * float((kept < M).mean()), mean_kept=float(kept.mean()), mean_kept_raw_tokens=float(tokk.mean()),
                   rating_match_kept=float(np.nanmean(rt)), rating_match_cands=float(np.nanmean(A["cand_rating"])),
                   text_cos_kept=float(np.nanmean(r["kc"])), text_cos_cands=float(np.nanmean(A["cand_cos_mean"])),
                   raw_dist_kept=float(np.nanmean(r["kr"])),
                   sel_ratio=float(np.nanmean(np.array(A["d4"]) / np.array(A["cand_med"]))) if len(A["d4"]) else float("nan"),
                   exact_ties_pct=100 * float(np.nanmean(np.array(A["margin"])[~np.isnan(A["margin"])] == 0)) if np.any(~np.isnan(A["margin"])) else float("nan"),
                   mdk_eq_raw_pct=100 * float(np.mean([np.array_equal(np.sort(a), np.sort(b)) for a, b in zip(A["mdk"], A["raw"])])),
                   mdk_eq_cos_pct=100 * float(np.mean([np.array_equal(np.sort(a), np.sort(b)) for a, b in zip(A["mdk"], A["cos"])])),
                   jac_mdk_raw=float(np.mean([jac(a, b) for a, b in zip(A["mdk"], A["raw"])])), jac_mdk_cos=float(np.mean([jac(a, b) for a, b in zip(A["mdk"], A["cos"])])))
        per_path.append(row)
    with open(OUT / "per_metapath.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_path[0])); w.writeheader(); w.writerows(per_path)
    out["per_path"] = per_path
    agg = {}
    for K in KS:
        for h in range(1, K + 1):
            rows = [r for r in per_path if r["K"] == K and r["hops"] == h]
            agg[f"{K}|{h}"] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ("K", "metapath", "hops")} | {"n_paths": len(rows)}
    out["agg_by_K_hop"] = agg

    # ---------- B. unions, duplicates, growth across K ----------
    U = {K: [None] * T for K in KS}; Uh = {K: {h: [None] * T for h in range(1, K + 1)} for K in KS}
    for K in KS:
        keys = [k for k in R if k[0] == K]
        for i in range(T):
            allb = [R[k]["kid"][i][R[k]["kid"][i] >= 0] for k in keys]
            cat = np.concatenate(allb) if allb else np.zeros(0, int)
            U[K][i] = set(cat.tolist())
            for h in range(1, K + 1):
                hh = [R[k]["kid"][i][R[k]["kid"][i] >= 0] for k in keys if R[k]["hops"] <= h]
                Uh[K][h][i] = set(np.concatenate(hh).tolist()) if hh else set()
    gr = {}
    for K in KS:
        tot = np.array([sum((R[k]["kid"][i] >= 0).sum() for k in R if k[0] == K) for i in range(T)])
        uq = np.array([len(U[K][i]) for i in range(T)])
        gr[str(K)] = {"mean_kept_total": float(tot.mean()), "mean_unique": float(uq.mean()), "dup_ratio": float(1 - uq.sum() / tot.sum()),
                      "unique_by_hop_le": {str(h): float(np.mean([len(Uh[K][h][i]) for i in range(T)])) for h in range(1, K + 1)},
                      "raw_tokens_unique": float(np.mean([c.tok[list(U[K][i])].sum() if U[K][i] else 0 for i in range(0, T, 5)]) )}
    gr["new_2_vs_1"] = float(np.mean([len(U[2][i] - U[1][i]) for i in range(T)])); gr["new_3_vs_2"] = float(np.mean([len(U[3][i] - U[2][i]) for i in range(T)]))
    gr["jac_union_1_2"] = float(np.mean([jac(U[1][i], U[2][i]) for i in range(T)])); gr["jac_union_2_3"] = float(np.mean([jac(U[2][i], U[3][i]) for i in range(T)]))
    gr["u1_subset_u2"] = float(np.mean([U[1][i] <= U[2][i] for i in range(T)])); gr["u2_subset_u3"] = float(np.mean([U[2][i] <= U[3][i] for i in range(T)]))
    out["union"] = gr

    # ---------- C. stability of the Top-4 across K, per metapath ----------
    stab = []
    for n in sorted({k[1] for k in R}):
        ks = [K for K in KS if (K, n) in R]
        for a in ks:
            for b in ks:
                if a < b:
                    j = float(np.mean([jac(R[(a, n)]["kid"][i][R[(a, n)]["kid"][i] >= 0], R[(b, n)]["kid"][i][R[(b, n)]["kid"][i] >= 0]) for i in range(T)]))
                    stab.append({"metapath": n, "hops": len(n.split("-")), "Ka": a, "Kb": b, "jaccard": j})
    out["stability"] = stab

    # ---------- D. direct relation to the target (2000 random targets) ----------
    rng = np.random.default_rng(1); samp = rng.choice(T, 2000, replace=False)
    rel = c.graph.relations
    direct = {}
    for K in KS:
        res = defaultdict(list)
        for i in samp:
            v = int(c.targets[i]); nb = np.array(sorted(U[K][i]), dtype=np.int64)
            if len(nb) == 0: continue
            flags = {r: np.isin(nb, rel[r].indices[rel[r].indptr[v]:rel[r].indptr[v + 1]]) for r in rel}
            anyd = np.logical_or.reduce(list(flags.values()))
            res["any"].append(anyd.mean()); [res[r].append(flags[r].mean()) for r in rel]
        direct[str(K)] = {k: float(np.mean(v)) for k, v in res.items()}
    out["direct"] = direct

    # ---------- E. labels: neighbour fraud rate by target class ----------
    lab = {}
    for K in KS:
        for h in list(range(1, K + 1)) + ["all"]:
            keys = [k for k in R if k[0] == K and (h == "all" or R[k]["hops"] == h)]
            rates = []
            for i in probe_idx:
                allb = np.concatenate([R[k]["kid"][i][R[k]["kid"][i] >= 0] for k in keys])
                rates.append(float(np.mean(y[allb] == 1)) if len(allb) else np.nan)
            rates = np.array(rates); ok = ~np.isnan(rates)
            gap, ci = boot_gap(rates[ok & (probe_lab == 1)], rates[ok & (probe_lab == 0)])
            lab[f"{K}|{h}"] = {"rate_if_fraud": float(np.nanmean(rates[probe_lab == 1])), "rate_if_benign": float(np.nanmean(rates[probe_lab == 0])), "gap": gap, "ci": ci}
    out["labels"] = lab
    # baselines on the audit sample (train+val audit targets)
    base = {}
    for K in KS:
        for h in range(1, K + 1):
            keys = [k for k in R if k[0] == K and R[k]["hops"] == h]
            S = R[keys[0]]["A"]["target"]; sel = [j for j, v in enumerate(S) if int(v) in set(c.probe.tolist())]
            comp = {m: [] for m in ("mdk", "raw", "cos", "rand")}
            tl = []
            for j in sel:
                v = int(S[j]); tl.append(y[v])
                for m in ("mdk", "raw", "cos"):
                    allb = np.concatenate([R[k]["A"][m][j] for k in keys]); comp[m].append(float(np.mean(y[allb] == 1)) if len(allb) else np.nan)
                cf = [R[k]["A"]["cand_fraud"][j] for k in keys if not np.isnan(R[k]["A"]["cand_fraud"][j])]
                comp["rand"].append(float(np.mean(cf)) if cf else np.nan)
            tl = np.array(tl); b = {}
            for m, v in comp.items():
                v = np.array(v); ok = ~np.isnan(v)
                gap, ci = boot_gap(v[ok & (tl == 1)], v[ok & (tl == 0)]); b[m] = {"gap": gap, "ci": ci, "rate_if_fraud": float(np.nanmean(v[tl == 1])), "rate_if_benign": float(np.nanmean(v[tl == 0]))}
            b["n_targets"] = int(len(tl)); b["n_fraud"] = int((tl == 1).sum())
            base[f"{K}|{h}"] = b
    out["baselines"] = base

    # ---------- F. clique test for the 1-hop relations ----------
    cl = {}
    for r, A in rel.items():
        nc, comp = connected_components(A, directed=False); sizes = np.bincount(comp); deg = np.diff(A.indptr)
        is_clique = deg == (sizes[comp] - 1)
        cl[r] = {"components": int(nc), "nodes_in_clique_components_pct": 100 * float(is_clique.mean()), "singleton_pct": 100 * float((sizes[comp] == 1).mean()),
                 "max_component": int(sizes.max()), "median_component": float(np.median(sizes[sizes > 1])) if (sizes > 1).any() else 0.0}
    prop = {}
    for K in KS:
        errs = []
        for r in rel:
            res = R[(K, r)]; m = res["kid"] >= 0
            n_cl = (res["sizes"] + 1).astype(float)                      # clique size = candidates + target
            rr = -1.0 / np.maximum(n_cl - 1, 1)
            cK = np.abs(sum(rr ** k for k in range(K + 1)) / K)
            pred = np.broadcast_to(cK[:, None], m.shape) * res["kr"]
            ok = m & (res["kd"] > 0) & (np.isfinite(pred)) & (pred > 0)
            errs.append(np.abs(res["kd"][ok] - pred[ok]) / pred[ok])
        e = np.concatenate(errs); prop[str(K)] = {"pairs": int(len(e)), "median_rel_err": float(np.median(e)), "p99_rel_err": float(np.percentile(e, 99)), "max_rel_err": float(e.max())}
    one_hop_eq = {str(K): {r: next(p for p in per_path if p["K"] == K and p["metapath"] == r)["mdk_eq_raw_pct"] for r in rel} for K in KS}
    out["clique"] = {"relations": cl, "proportionality": prop, "mdk_equals_raw_top4_pct_1hop": one_hop_eq}
    out["scale"] = {"mean_text_emb_norm": float(np.linalg.norm(c.emb, axis=1).mean()), "rating_std": float(c.rating.std()),
                    "mean_rating_share_of_sq_dist": float(np.nanmean([np.nanmean(np.where(R[k]["kr"] > 0, (c.rating[np.where(R[k]["kid"] >= 0, R[k]["kid"], 0)] - c.rating[c.targets][:, None]) ** 2 / np.where(R[k]["kr"] > 0, R[k]["kr"] ** 2, np.nan), np.nan)) for k in R]))}

    # ---------- G. case studies ----------
    rng = np.random.default_rng(3)
    pool = [int(v) for v in c.probe if len(U[3][tpos[int(v)]]) > 0 and len(c.graph.texts[int(v)]) < 400 and len(c.graph.texts[int(v)]) > 80]
    pf = [v for v in pool if y[v] == 1]; pb = [v for v in pool if y[v] == 0]
    chosen = list(rng.choice(pf, 2, replace=False)) + list(rng.choice(pb, 2, replace=False))
    def snip(u, n=150):
        t = c.graph.texts[int(u)].replace("\n", " "); return t[:n] + ("…" if len(t) > n else "")
    cases = []
    for v in chosen:
        v = int(v); i = tpos[v]; item = {"id": v, "label": int(y[v]), "rating": float(c.rating[v]), "split": str(split_of[i]), "text": snip(v, 260),
                                         "union_sizes": {str(K): len(U[K][i]) for K in KS}, "k1": [], "new2": [], "new3": []}
        for n in ("RUR", "RPR", "RSR"):
            r = R[(1, n)]
            for rank, u in enumerate(r["kid"][i]):
                if u >= 0: item["k1"].append({"path": n, "rank": rank + 1, "id": int(u), "rating": float(c.rating[u]), "label": int(y[u]), "d": float(r["kd"][i][rank]), "text": snip(u)})
        seen = set(U[1][i])
        for K, key in ((2, "new2"), (3, "new3")):
            for k in sorted((k for k in R if k[0] == K and R[k]["hops"] == K), key=lambda k: k[1]):
                r = R[k]
                for rank, u in enumerate(r["kid"][i]):
                    if u >= 0 and int(u) not in seen and len([x for x in item[key]]) < 6:
                        item[key].append({"path": k[1], "rank": rank + 1, "id": int(u), "rating": float(c.rating[u]), "label": int(y[u]), "text": snip(u, 110)}); seen.add(int(u))
            seen |= U[K][i]
        item["new2_total"] = len(U[2][i] - U[1][i]); item["new3_total"] = len(U[3][i] - U[2][i])
        cases.append(item)
    out["cases"] = cases

    (OUT / "neighbors_k123_summary.json").write_text(json.dumps(out, indent=1, default=float))
    # ---------- long table ----------
    with gzip.open(OUT / "neighbors_long.csv.gz", "wt", newline="") as f:
        w = csv.writer(f); w.writerow(["K", "metapath", "hops", "target", "split", "rank", "neighbor", "diffusion_dist", "raw_feature_dist", "text_cosine", "target_rating", "neighbor_rating", "neighbor_label_probe_targets_only", "neighbor_tokens"])
        for (K, n), r in sorted(R.items()):
            for i in range(T):
                v = int(c.targets[i]); isp = split_of[i] != "test"
                for rank in range(M):
                    u = int(r["kid"][i][rank])
                    if u < 0: break
                    w.writerow([K, n, r["hops"], v, split_of[i], rank + 1, u, f"{r['kd'][i][rank]:.6g}", f"{r['kr'][i][rank]:.6g}", f"{r['kc'][i][rank]:.5f}", c.rating[v], c.rating[u], int(y[u]) if isp else "", int(c.tok[u])])
    print("done")


if __name__ == "__main__":
    main()
