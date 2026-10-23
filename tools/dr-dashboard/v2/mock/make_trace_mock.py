"""Adds the traceability files to mock trees made by make_mock.py / make_fa_mock.py (all synthetic; numbers mean nothing).

    python tools/dr-dashboard/v2/mock/make_mock.py <out>              # writes <out>/full and <out>/empty
    python tools/dr-dashboard/v2/mock/make_fa_mock.py <out>/fa
    python tools/dr-dashboard/v2/mock/make_trace_mock.py <out>/full <out>/fa

Per tree: validation metrics + run_manifest.json (git commit, config) for every finished run, a prompt JSONL for one arm,
split.json / labels.npy / numeric.npy / graph.json, the token-usage CSV of that arm. The FA tree also gets a batch
record (hypothesis, branch, command) and a failed run whose log ends in a parse-error traceback; the dr tree gets none,
so its hypothesis / branch / command must come back as "unknown".
"""
import glob
import json
import os
import sys

import numpy as np

N_NODES = 60
SHA = "0123456789abcdef0123456789abcdef01234567"


def w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def augment(root, study):
    arm_cfg = {"dr_R0": {}, "dr_R1": {"templates": {"final": "dr_final_disc"}},
               "dr_R2": {"templates": {"final": "dr_final_disc"}, "residual": {"alpha": 0.1, "beta": 0.1}},
               "fa_F0": {}, "fa_F1": {"flag_arch": {"enabled": True, "kinds": ["discriminative"]}}}
    for i, mpath in enumerate(sorted(glob.glob(os.path.join(root, "results", "raw", "*", "metrics.json")))):
        run = os.path.dirname(mpath)
        exp = os.path.basename(run).rsplit("_seed", 1)[0]
        m = json.load(open(mpath))
        if "test" in m and "val" not in m:
            m["val"] = {k: round(v + 1.0 + (i % 3) * 0.1, 4) for k, v in m["test"].items() if isinstance(v, (int, float))}
            w(mpath, json.dumps(m))
        cfg = {"experiment": exp, "seeds": [m.get("seed", 0)],
               "classifier": {"lora": {"r": 8, "alpha": 16}, "lr": 1e-4}, **arm_cfg.get(exp, {})}
        w(os.path.join(run, "run_manifest.json"), json.dumps({
            "written": f"2026-10-2{i % 3}T0{i % 9}:00:00", "git_commit": SHA + ("-dirty" if i % 2 else ""),
            "run_id": os.path.basename(run), "experiment": exp, "config": cfg,
            "prompt_metadata": {"prompt_version": "v1-mock"}}))
    rng = np.random.default_rng(0)
    base = os.path.join(root, "datasets", "processed", "amazonvideo")
    os.makedirs(base, exist_ok=True)
    np.save(os.path.join(base, "labels.npy"), (rng.random(N_NODES) < 0.3).astype("int8"))
    np.save(os.path.join(base, "numeric.npy"), rng.integers(1, 6, size=(N_NODES, 1)).astype("float32"))
    w(os.path.join(base, "graph.json"), json.dumps({"numeric_names": ["rating"]}))
    w(os.path.join(base, "split.json"), json.dumps({"seed": 0, "train": list(range(0, 20)), "val": list(range(20, 40)),
                                                    "test": list(range(40, N_NODES))}))
    exp = "dr_R1" if study == "dr" else "fa_F1"
    lines = [json.dumps({"node": n, "prompt": f"Target: mock review {n} <b>&amp;</b> " + "x" * (200 + 40 * n)})
             for n in range(N_NODES)]
    w(os.path.join(root, "datasets", "cache", "prompts", f"{exp}_abc123.jsonl"), "\n".join(lines) + "\n")
    rows = ["node,split,final_prompt_tokens,num_neighbors_full,num_neighbors_trimmed,num_metapaths,target_truncated"]
    rows += [f"{n},x,{500 + n},{1000 + n},48,12,False" for n in range(N_NODES)]
    w(os.path.join(root, "results", "token_usage", f"token_usage_{exp}.csv"), "\n".join(rows) + "\n")
    if study == "fa":
        w(os.path.join(root, "results", "flag_arch", "batch_20261023T000000Z.json"), json.dumps({
            "schema": "batch-record-v1", "study": "flag_arch", "recorded_at": "2026-10-23T00:00:00Z",
            "git_commit": SHA, "git_branch": "mock-branch", "git_dirty": False, "git_dirty_files": [],
            "command": "nohup scripts/fa_queue.sh > logs/flag_arch/queue.log 2>&1 &",
            "run_hypothesis": "Mock hypothesis: FLAG-style D lines help."}))
        q = os.path.join(root, "logs", "flag_arch", "queue.log")
        w(q, open(q).read().replace("FAILED fa_F0_s9 (bogus job name, ignored)",
                                    "FAILED fa_F0_s9 (bogus job name, ignored)\n[2026-10-23T21:49:00] start fa_F0_s1\n"
                                    "[2026-10-23T21:49:30] FAILED fa_F0_s1 (see logs/flag_arch/train_fa_F0_s1.log)"))
        w(os.path.join(root, "logs", "flag_arch", "train_fa_F0_s1.log"),
          "loading\nTraceback (most recent call last):\n  File \"x.py\", line 3, in <module>\n"
          "json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0) <script>alert(1)</script>\n")


if __name__ == "__main__":
    for tree in sys.argv[1:]:
        augment(os.path.abspath(tree), "fa" if os.path.exists(os.path.join(tree, "results", "flag_arch")) else "dr")
        print("traceability mock added to", tree)
