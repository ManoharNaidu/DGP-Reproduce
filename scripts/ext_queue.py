"""Two-GPU job scheduler for the extension experiments (tasks 2 and 3). Resumable.

    nohup /venv/main/bin/python scripts/ext_queue.py > logs/ext/queue.log 2>&1 &

Each job is a subprocess pinned to one GPU via CUDA_VISIBLE_DEVICES. A GPU has `SLOTS` slots; a job takes `weight`
slots. Finished jobs leave results/ext/done/<job> and are skipped on restart. Status events are appended to
results/ext/status.jsonl (read by the dashboard). Nothing here touches the baseline results.
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = "/venv/main/bin/python"
GPUS = [int(g) for g in os.environ.get("GPUS", "0,1").split(",")]   # e.g. GPUS=0,1,2,3 on a 4-GPU box
SLOTS = int(os.environ.get("SLOTS_PER_GPU", 2))
EXT = ROOT / "results" / "ext"
(EXT / "done").mkdir(parents=True, exist_ok=True)
(ROOT / "logs" / "ext").mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
CFG = "configs/experiments/ext/{}.yaml"


def mdk_complete(k: int, n_paths: int):
    def check() -> bool:
        import json as j
        for meta in (ROOT / "datasets" / "cache" / "mdk").glob("*/metadata.json"):
            m = j.loads(meta.read_text())
            if m.get("K") == k and m.get("M") == 4 and m.get("use_mdk") and "Qwen3" in str(m.get("counter")):
                busy = subprocess.run(["pgrep", "-f", "scripts/build_mdk.py"], capture_output=True).returncode == 0
                return len(list(meta.parent.glob("*.npz"))) >= n_paths and not busy
        return False
    return check


def job(name, args, deps=(), weight=1, ready=None, group="train"):
    return {"name": name, "cmd": [PY, "-W", "ignore", *args], "deps": list(deps), "weight": weight, "ready": ready,
            "group": group}


def build_jobs():
    J = []
    gen = lambda n, cfg, deps=(), **kw: J.append(job(f"gen_{n}", ["scripts/generate_summaries.py", "--dataset", "amazonvideo",
                                                              "--config", CFG.format(cfg), "--device", "cuda:0"], deps, group="gen", **kw))
    train = lambda arm, cfg, deps=(), **kw: [J.append(job(f"train_{arm}_s{s}", [
        "scripts/train.py", "--dataset", "amazonvideo", "--config", CFG.format(cfg), "--experiment", f"ext_{arm}",
        "--seeds", str(s), "--device", "cuda:0"], deps, **kw)) for s in SEEDS]
    # stack-parity check: baseline seed 0 on the new software stack
    J.append(job("train_parity_s0", ["scripts/train.py", "--dataset", "amazonvideo", "--experiment", "parity_dgp",
                                     "--seeds", "0", "--device", "cuda:0"]))
    gen("T", "armE2_disc_resid_aux")                       # target discriminative + residual text (shared by C, D, E1, E2, BC)
    gen("B", "armB_disc_neighbors")                        # neighbour discriminative text + metapath summaries over it
    gen("K1", "K1")
    train("armC", "armC_disc_target", ["gen_T"])
    train("armD", "armD_resid_prompt", ["gen_T"])
    train("armE1", "armE1_resid_aux", ["gen_T"])
    train("armE2", "armE2_disc_resid_aux", ["gen_T"])
    train("K1", "K1", ["gen_K1"])
    train("armB", "armB_disc_neighbors", ["gen_B"])
    gen("BC", "armBC_disc_both", ["gen_B", "gen_T"])
    train("armBC", "armBC_disc_both", ["gen_BC"])
    gen("K3", "K3", ready=mdk_complete(3, 39))
    train("K3", "K3", ["gen_K3"], weight=2)
    return J


lock = threading.Lock()
free = {g: SLOTS for g in GPUS}
state = {}          # name -> pending|running|done|failed


def log_event(**kw):
    kw["t"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(EXT / "status.jsonl", "a") as f:
        f.write(json.dumps(kw) + "\n")


def run(j, gpu):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}
    t0 = time.time()
    log_event(job=j["name"], event="start", gpu=gpu, group=j["group"])
    with open(ROOT / "logs" / "ext" / f"{j['name']}.log", "a") as lf:
        rc = subprocess.run(j["cmd"], cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    with lock:
        free[gpu] += j["weight"]
        state[j["name"]] = "done" if rc == 0 else "failed"
    if rc == 0:
        (EXT / "done" / j["name"]).write_text(str(round(time.time() - t0)))
    log_event(job=j["name"], event="end", rc=rc, gpu=gpu, seconds=round(time.time() - t0), group=j["group"])


def main():
    jobs = build_jobs()
    for j in jobs:
        state[j["name"]] = "done" if (EXT / "done" / j["name"]).exists() else "pending"
    threads = []
    while any(s in ("pending", "running") for s in state.values()):
        launched = False
        with lock:
            for j in jobs:
                if state[j["name"]] != "pending":
                    continue
                if any(state[d] == "failed" for d in j["deps"]):
                    state[j["name"]] = "failed"; log_event(job=j["name"], event="skipped_dep_failed"); continue
                if any(state[d] != "done" for d in j["deps"]) or (j["ready"] and not j["ready"]()):
                    continue
                gpu = next((g for g in GPUS if free[g] >= j["weight"]), None)
                if gpu is None:
                    continue
                free[gpu] -= j["weight"]
                state[j["name"]] = "running"
                t = threading.Thread(target=run, args=(j, gpu), daemon=True); t.start(); threads.append(t)
                launched = True
                time.sleep(20)       # stagger model loads
        if not launched:
            time.sleep(15)
    for t in threads:
        t.join()
    print("queue finished:", {n: s for n, s in state.items() if s != "done"} or "all done")


if __name__ == "__main__":
    main()
