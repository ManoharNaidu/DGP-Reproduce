"""Phases 0 + 1: verify the cached Top-4 against a live recomputation and collect everything later phases need.

One task per (K, metapath), run in parallel (fork, shared read-only graph). Per task:
  * ALL 10,023 targets: the cached kept neighbours, their diffusion distance, raw-feature distance, text cosine.
  * A fixed audit sample S (seed 0; 400 train+val and 200 test targets): the full candidate set N_P(v) is scored three ways
    (diffusion distance H, raw feature distance X, DeBERTa cosine), the live diffusion Top-4 must equal the cache, and the
    Top-4 under the two alternative scores, the rank-5 margin and candidate-set means are stored.
Output: analysis/out/neighbors_k123/work/K{K}__{metapath}.pkl and phase0_verify.json.
"""
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np

from common import KS, M, OUT, context, cached_trims
from dgp_repro.mdk import metapath_name
from dgp_repro.mdk.adjacency import metapath_neighbors
from dgp_repro.mdk.diffusion import compute_diffusion_embeddings
from dgp_repro.mdk.distance import compute_diffusion_distance
from dgp_repro.mdk.trimming import select_top_m_neighbors

WORK = OUT / "work"
N_PROBE, N_TEST = 400, 200


def audit_sample(c):
    rng = np.random.default_rng(0)
    return np.concatenate([rng.choice(c.probe, N_PROBE, replace=False), rng.choice(c.split.test, N_TEST, replace=False)]).astype(np.int64)


def top_m(cands, dist):
    return select_top_m_neighbors(cands, dist, M)


def task(args):
    K, path = args
    t0 = time.time(); c = context(); name = metapath_name(path)
    kept, sizes = cached_trims(K)[name]
    H = compute_diffusion_embeddings(c.graph.relations, path, c.X, K, "paper_eq6", "weighted")
    X, E = c.X, c.emb
    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    T = len(c.targets)
    kid = -np.ones((T, M), np.int64); kd = np.full((T, M), np.nan); kr = np.full((T, M), np.nan); kc = np.full((T, M), np.nan)
    for i, v in enumerate(c.targets.tolist()):
        nb = kept[v]
        if len(nb):
            kid[i, :len(nb)] = nb
            kd[i, :len(nb)] = compute_diffusion_distance(H, v, nb)
            kr[i, :len(nb)] = np.linalg.norm(X[nb] - X[v], axis=1)
            kc[i, :len(nb)] = En[nb] @ En[v]
    S = audit_sample(c)
    cands = dict(zip(S.tolist(), metapath_neighbors(c.graph.relations, path, S)))
    y = c.labels
    A = {k: [] for k in ("target", "size", "mismatch", "margin", "d4", "d5", "cand_med", "cand_fraud", "cand_rating", "cand_cos_mean",
                         "mdk", "raw", "cos")}
    for v in S.tolist():
        cs = cands[v]; A["target"].append(v); A["size"].append(len(cs))
        if len(cs) == 0:
            for k in ("margin", "d4", "d5", "cand_med", "cand_fraud", "cand_rating", "cand_cos_mean"): A[k].append(np.nan)
            A["mismatch"].append(False); A["mdk"].append(np.zeros(0, int)); A["raw"].append(np.zeros(0, int)); A["cos"].append(np.zeros(0, int)); continue
        dh = compute_diffusion_distance(H, v, cs)
        live = top_m(cs, dh)
        A["mismatch"].append(not np.array_equal(live, kept[v]))
        srt = np.sort(dh)
        A["d4"].append(srt[min(M, len(srt)) - 1]); A["d5"].append(srt[M] if len(srt) > M else np.nan)
        A["margin"].append(srt[M] - srt[M - 1] if len(srt) > M else np.nan)
        A["cand_med"].append(np.median(dh))
        dr = np.linalg.norm(X[cs] - X[v], axis=1); cs_cos = En[cs] @ En[v]
        A["raw"].append(top_m(cs, dr)); A["cos"].append(top_m(cs, -cs_cos)); A["mdk"].append(live)
        A["cand_fraud"].append(float(np.mean(y[cs] == 1))); A["cand_rating"].append(float(np.mean(c.rating[cs] == c.rating[v])))
        A["cand_cos_mean"].append(float(cs_cos.mean()))
    res = dict(K=K, name=name, hops=len(path), kid=kid, kd=kd, kr=kr, kc=kc, A=A, seconds=time.time() - t0,
               sizes=np.array([sizes[int(v)] for v in c.targets]))
    WORK.mkdir(parents=True, exist_ok=True)
    with open(WORK / f"K{K}__{name}.pkl", "wb") as f:
        pickle.dump(res, f)
    return K, name, int(np.sum(A["mismatch"])), len(S), round(res["seconds"], 1)


def main():
    c = context()
    jobs = [(K, p) for K in KS for p in c.paths[K]]
    jobs.sort(key=lambda j: -len(j[1]))            # heavy (long) metapaths first
    done = []
    with Pool(int(os.environ.get("NPROC", 24))) as pool:
        for r in pool.imap_unordered(task, jobs):
            done.append(r); print(r, f"{len(done)}/{len(jobs)}", flush=True)
    per_k = {K: {"tasks": sum(d[0] == K for d in done), "audited_pairs": int(sum(d[3] for d in done if d[0] == K)),
                 "mismatches": int(sum(d[2] for d in done if d[0] == K))} for K in KS}
    rep = {"nodes": c.graph.num_nodes, "targets": int(len(c.targets)), "split": c.split.sizes(), "feature_dim": int(c.X.shape[1]),
           "cache_dirs": {str(K): c.cache_dirs[K].name for K in KS}, "audit_sample": {"train_val": N_PROBE, "test": N_TEST},
           "per_K": per_k, "pass": all(v["mismatches"] == 0 for v in per_k.values())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase0_verify.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2)); print("PASS" if rep["pass"] else "FAIL")


if __name__ == "__main__":
    main()
