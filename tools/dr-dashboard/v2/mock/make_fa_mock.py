"""Mock data tree for the FLAG-architecture section of the dashboard (all synthetic; numbers mean nothing).

    python tools/dr-dashboard/v2/mock/make_fa_mock.py tools/dr-dashboard/v2/mock/data/fa
    python tools/dr-dashboard/v2/server.py 18784 --root tools/dr-dashboard/v2/mock/data/fa --no-insights

Covers: D generation finished with some format failures, R generation half-way with a half-written last line,
metapath summaries started, F0/F1 done for seed 0, F2 seed 0 running in its +FLAG* phase, one failed run,
a pilot summary.
"""
import json
import os
import sys
import time

root = os.path.abspath(sys.argv[1])
now = time.time()


def w(rel, text):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)


def stamp(t):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t))


for kind, key, n, partial in (("discriminative", "aaaa000000000001", 10023, False), ("residual", "aaaa000000000002", 4100, True)):
    w(f"datasets/cache/fa_subgraph_text/{key}/metadata.json", json.dumps({"kind": kind, "cache_key": key, "created": stamp(now - 7200)}))
    rows = []
    for v in range(n):
        size = 5 + v % 40
        ok = size <= 25 or v % 3 == 0
        rows.append(json.dumps({"key": str(v), "text": "\n".join(f"{i}. mock line {i}" for i in range(1, size + 1)) if ok else "1. short",
                                "nodes": list(range(size)), "ok": ok, "overflow": False, "prompt_tokens": 600 + 150 * size,
                                "generated_tokens": min(550, 15 * size), "hit_cap": 15 * size >= 550}))
    text = "\n".join(rows) + "\n" + ('{"key": "9999", "te' if partial else "")
    w(f"datasets/cache/fa_subgraph_text/{key}/answers.jsonl", text)
w("datasets/cache/metapath_summaries/bbbb000000000001/metadata.json",
  json.dumps({"node_text": "fa_subgraph_text/aaaa000000000001", "template": "metapath_summary"}))
w("datasets/cache/metapath_summaries/bbbb000000000001/summaries.jsonl", "".join(json.dumps({"key": f"{i}|RUR|x", "text": "s"}) + "\n" for i in range(30000)))
w("logs/flag_arch/gen_all.log", f"[{stamp(now - 7200)}] generate fa_F2\nFA discriminative: {{'subgraphs': 10023}}\n")
w("logs/flag_arch/queue.log", "")
w("mock_liveness.json", json.dumps({"fa_queue.sh": True}))
w("mock_gpu.json", json.dumps([{"name": "NVIDIA A100-SXM4-80GB", "util": 97.0, "mem_used": 41000.0, "mem_total": 81920.0, "temp": 61.0}]))
w("results/flag_arch/pilot/summary.json", json.dumps({
    "n": 200, "usable_both_kinds": 71,
    "kinds": {"discriminative": {"subgraph_success_rate": 0.43, "node_coverage": 0.31, "hit_cap": 88, "overflow": 0,
                                 "prompt_tokens": {"mean": 5410, "p95": 9020}, "nodes_per_subgraph": {"mean": 33.4}},
              "residual": {"subgraph_success_rate": 0.51, "node_coverage": 0.38, "hit_cap": 70, "overflow": 0}},
    "classifier_check": {"per_example_mean": {"ce": 0.71, "kl": 0.004, "squared_dot": 3.2e7, "squared_cosine": 0.88}},
    "estimate": {"generation_h": {"discriminative": 9.1, "residual": 8.8}, "metapath_summaries_h": {"discriminative": 3.0, "residual": 3.0},
                 "F2_plus_flag_star_phase_h_per_seed": 11.5}}))


def run(exp, seed, test, manifest_extra=None, log=None):
    d = f"results/raw/{exp}_seed{seed}_cafe{seed}"
    if test:
        w(f"{d}/metrics.json", json.dumps({"seed": seed, "best_epoch": 2, "test": test}))
    w(f"{d}/run_manifest.json", json.dumps({"written": stamp(now), **(manifest_extra or {})}))
    w(f"{d}/training_log.jsonl", "".join(json.dumps(r) + "\n" for r in (log or [])))


q = []
t = now - 5000
for job, ok in (("fa_F0_s0", True), ("fa_F1_s0", True), ("fa_F2_s0", None)):
    q.append(f"[{stamp(t)}] start {job}")
    if ok:
        q.append(f"[{stamp(t + 1500)}] done  {job} (25 min)")
        w(f"results/flag_arch/done/{job}", "1500")
    t += 1600
run("fa_F0", 0, {"macro_f1": 64.9, "auroc": 75.0, "auprc": 32.4})
run("fa_F1", 0, {"macro_f1": 65.4, "auroc": 75.3, "auprc": 33.0})
run("fa_F2", 0, None, log=[{"epoch": e, "train_loss": 0.3, "val_loss": 0.31} for e in (1, 2, 3)] +
    [{"phase": "flag_finetune", "epoch": e, "train_loss": 1.1, "train_ce": 0.4, "train_kl": 0.01, "train_orth": 7.0, "val_macro_f1": 64.0} for e in (1, 2, 3, 4)])
w("logs/flag_arch/queue.log", "\n".join(q) + f"\n[{stamp(now - 100)}] FAILED fa_F0_s9 (bogus job name, ignored)\n")
w("logs/flag_arch/train_fa_F2_s0.log", "epoch 4 of the +FLAG* phase\n")
print("mock written to", root)
