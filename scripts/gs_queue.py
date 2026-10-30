"""Scheduler for the Gemma-swap / structured zero-shot study: three experiments, one after the other, each spread over
all GPUs. Resumable.

    nohup envs/dgp-fa/bin/python scripts/gs_queue.py > logs/gemma_swap/queue.log 2>&1 &

    Exp 1  gs_G*   Gemma-2-9B writes all text and is the LoRA classifier; dr arms R0, R1, R1b, R2, R2b
    Exp 2  zs_S*   Qwen3-8B, structured zero-shot classification prompt; + par_R0 (R0 seed 0 on this stack)
    Exp 3  zs_GS*  Gemma-2-9B, structured zero-shot classification prompt

An experiment starts only when the previous one finished without a failed job. Each job is a subprocess pinned to one
GPU (CUDA_VISIBLE_DEVICES); a GPU has SLOTS slots and a job takes `weight` of them. Text generation runs as hash-sliced
shards (DGP_GEN_SHARD) that fill the same caches in parallel; MDK and prompts are built unsharded.

Gemma training gate: Gemma training jobs start one per GPU. When the first one finishes, its peak memory decides
whether two fit on a GPU, and its runtime projects the remaining time. If the projection exceeds GATE_HOURS, nothing
new is launched and the queue exits with code 3 after the running jobs finish (GATE_STOP), so a person decides.
A job is also never started with less than MIN_FREE_GB of free disk (DISK_STOP when nothing else is running).

Files (read by tools/dr-dashboard/v2): results/gemma_swap/{status.jsonl, queue_state.json, done/<job>},
logs/gemma_swap/<job>.log. Finished jobs are skipped on restart; a stopped training run restarts from epoch 0.
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT.parent / "envs" / "dgp-fa" / "bin" / "python")
GPUS = [int(g) for g in os.environ.get("GPUS", "0,1,2,3").split(",")]
SLOTS = 2
SHARDS = int(os.environ.get("GEN_SHARDS", 2 * len(GPUS)))
GATE_HOURS = float(os.environ.get("GATE_HOURS", 24))
MIN_FREE_GB = float(os.environ.get("MIN_FREE_GB", 1.5))
TWO_PER_GPU_MAX_GB = 34.0          # two Gemma runs share an 80 GB card only if one peaks at or below this
QWEN_RUN_S = 1.1 * 3600            # dr study: 1.03-1.17 h per Qwen run (run manifests), used only for the projection
OUT = ROOT / "results" / "gemma_swap"
LOGS = ROOT / "logs" / "gemma_swap"
(OUT / "done").mkdir(parents=True, exist_ok=True)
(OUT / "checks").mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
ENV = {"HF_HOME": "/workspace/.hf_home", "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
# Gemma-2 needs eager attention, whose memory grows with input length squared: one node-summary worker peaked at
# 66.6 GB on a batch with a ~4k-token review (2026-10-29 13:06 OOM, two workers per GPU). Generators get a whole GPU.
GEN_WEIGHT = int(os.environ.get("GEN_WEIGHT", 2))
# The dr study trained Qwen one run at a time (68-78 min each); QWEN_TRAIN_WEIGHT=2 keeps that, 1 lets two share a GPU.
QWEN_TRAIN_WEIGHT = int(os.environ.get("QWEN_TRAIN_WEIGHT", 1))
SEEDS = [0, 1, 2, 3, 4]
ARMS = ["0", "1", "1b", "2", "2b"]
TT_CONFIGS = {"2": "disc + residual (FLAG prompt)", "1b": "disc (no-verdict prompt)"}   # target D/R line caches


def job(name, exp, group, args, deps=(), weight=1, gpu=True, gemma_train=False):
    return {"name": name, "exp": exp, "group": group, "cmd": [PY, "-W", "ignore", *args], "deps": list(deps),
            "weight": weight, "gpu": gpu, "gemma_train": gemma_train}


def gen(name, exp, cfg, until, deps=(), shard=None, group="gen"):
    j = job(name, exp, group, ["scripts/generate_summaries.py", "--dataset", "amazonvideo", "--config", cfg,
                                "--until", until, "--device", "cuda:0"], deps,
            weight=GEN_WEIGHT if shard else 1)       # prompt builds generate nothing and share a GPU
    j["shard"] = shard
    return j


def train(name, exp, cfg, experiment, seed, deps, gemma):
    return job(name, exp, "train", ["scripts/train.py", "--dataset", "amazonvideo", "--config", cfg, "--experiment",
                                    experiment, "--seeds", str(seed), "--device", "cuda:0"], deps,
               weight=2 if gemma else QWEN_TRAIN_WEIGHT, gemma_train=gemma)


def build_jobs():
    J = []
    G = "configs/experiments/gemma_swap/G{}.yaml"
    S = "configs/experiments/zs_qwen/S{}.yaml"
    GS = "configs/experiments/zs_gemma/GS{}.yaml"
    # ---- Exp 1: Gemma writes every text, then is the classifier
    J.append(gen("e1_mdk", 1, G.format("0"), "mdk"))
    J.append(job("e1_check_mdk", 1, "check", ["scripts/gs_checks.py", "mdk"], ["e1_mdk"], gpu=False))
    node = [f"e1_node_{i}" for i in range(SHARDS)]
    meta = [f"e1_meta_{i}" for i in range(SHARDS)]
    tt = {a: [f"e1_tt{a}_{i}" for i in range(SHARDS)] for a in TT_CONFIGS}
    for i in range(SHARDS):
        J.append(gen(node[i], 1, G.format("0"), "node_summaries", ["e1_mdk"], shard=f"{i}/{SHARDS}"))
    for a in TT_CONFIGS:
        for i in range(SHARDS):
            J.append(gen(tt[a][i], 1, G.format(a), "target_text", ["e1_mdk"], shard=f"{i}/{SHARDS}"))
    for i in range(SHARDS):
        J.append(gen(meta[i], 1, G.format("0"), "metapath_summaries", node, shard=f"{i}/{SHARDS}"))
    J.append(job("e1_check_text", 1, "check", ["scripts/gs_checks.py", "text"], tt["2"] + tt["1b"] + meta, gpu=False))
    for a in ARMS:
        J.append(gen(f"e1_prompts_G{a}", 1, G.format(a), "prompts", meta + tt["2"] + tt["1b"], group="prompts"))
    for s in SEEDS:                        # seed-major: all arms of a seed finish together, for paired comparisons
        for a in ARMS:
            J.append(train(f"e1_train_G{a}_s{s}", 1, G.format(a), f"gs_G{a}", s,
                           [f"e1_prompts_G{a}", "e1_check_mdk"], gemma=True))
    J.append(job("e1_analysis", 1, "analysis", ["scripts/gs_analysis.py", "--exp", "1"],
                 [f"e1_train_G{a}_s{s}" for a in ARMS for s in SEEDS], gpu=False))
    # ---- Exp 2: Qwen, structured zero-shot prompt (Qwen text caches reused)
    for a in ARMS:
        J.append(gen(f"e2_prompts_S{a}", 2, S.format(a), "prompts", group="prompts"))
    J.append(job("e2_check_layout", 2, "check", ["scripts/gs_checks.py", "layout", "--exp", "2"],
                 [f"e2_prompts_S{a}" for a in ARMS], gpu=False))
    J.append(train("e2_train_par_R0_s0", 2, "configs/experiments/zs_qwen/P_R0_parity.yaml", "par_R0", 0, [], gemma=False))
    for s in SEEDS:
        for a in ARMS:
            J.append(train(f"e2_train_S{a}_s{s}", 2, S.format(a), f"zs_S{a}", s, [f"e2_prompts_S{a}", "e2_check_layout"],
                           gemma=False))
    J.append(job("e2_analysis", 2, "analysis", ["scripts/gs_analysis.py", "--exp", "2"],
                 ["e2_train_par_R0_s0"] + [f"e2_train_S{a}_s{s}" for a in ARMS for s in SEEDS], gpu=False))
    # ---- Exp 3: Gemma, structured zero-shot prompt (Gemma text caches from Exp 1 reused)
    for a in ARMS:
        J.append(gen(f"e3_prompts_GS{a}", 3, GS.format(a), "prompts", group="prompts"))
    J.append(job("e3_check_layout", 3, "check", ["scripts/gs_checks.py", "layout", "--exp", "3"],
                 [f"e3_prompts_GS{a}" for a in ARMS], gpu=False))
    for s in SEEDS:
        for a in ARMS:
            J.append(train(f"e3_train_GS{a}_s{s}", 3, GS.format(a), f"zs_GS{a}", s,
                           [f"e3_prompts_GS{a}", "e3_check_layout"], gemma=True))
    J.append(job("e3_analysis", 3, "analysis", ["scripts/gs_analysis.py", "--exp", "3"],
                 [f"e3_train_GS{a}_s{s}" for a in ARMS for s in SEEDS], gpu=False))
    return J


lock = threading.Lock()
free = {g: SLOTS for g in GPUS}
state, info = {}, {}                       # name -> pending|running|done|failed|skipped ; name -> {gpu, started}
gate = {"gemma_weight": None, "measured_s": None, "peak_gb": None, "projection_h": None, "stopped": None}


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log_event(**kw):
    kw["t"] = now()
    with open(OUT / "status.jsonl", "a") as f:
        f.write(json.dumps(kw) + "\n")
    print(json.dumps(kw), flush=True)


def write_state(jobs, current_exp):
    snap = {"updated": now(), "current_exp": current_exp, "gpus": GPUS, "slots_per_gpu": SLOTS, "shards": SHARDS,
            "free_slots": dict(free), "gate": gate, "gate_hours": GATE_HOURS, "min_free_gb": MIN_FREE_GB,
            "disk_free_gb": round(shutil.disk_usage("/").free / 1e9, 2),
            "jobs": [{"name": j["name"], "exp": j["exp"], "group": j["group"], "state": state[j["name"]],
                      "weight": j["weight"], "deps": j["deps"], **info.get(j["name"], {})} for j in jobs]}
    tmp = OUT / "queue_state.json.tmp"
    tmp.write_text(json.dumps(snap, indent=1))
    os.replace(tmp, OUT / "queue_state.json")


def run(j, gpu):
    env = {**os.environ, **ENV, "CUDA_VISIBLE_DEVICES": "" if gpu is None else str(gpu)}
    if j.get("shard"):
        env["DGP_GEN_SHARD"] = j["shard"]
    t0 = time.time()
    log_event(job=j["name"], exp=j["exp"], event="start", gpu=gpu, group=j["group"])
    with open(LOGS / f"{j['name']}.log", "a") as lf:
        lf.write(f"\n===== {now()} start on GPU {gpu}: {' '.join(j['cmd'][3:])}\n")
        lf.flush()
        rc = subprocess.run(j["cmd"], cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    secs = round(time.time() - t0)
    with lock:
        if gpu is not None:
            free[gpu] += j["weight"]
        state[j["name"]] = "done" if rc == 0 else "failed"
        info[j["name"]].update(seconds=secs, rc=rc)
    if rc == 0:
        (OUT / "done" / j["name"]).write_text(str(secs))
    log_event(job=j["name"], exp=j["exp"], event="end", rc=rc, gpu=gpu, seconds=secs, group=j["group"])
    if rc == 0 and j["gemma_train"] and gate["gemma_weight"] is None:
        decide_gemma(j, secs)


def manifest_of(experiment, seed):
    hits = sorted((ROOT / "results" / "raw").glob(f"{experiment}_seed{seed}_*/run_manifest.json"),
                  key=lambda p: p.stat().st_mtime)
    return json.loads(hits[-1].read_text()) if hits else {}


def decide_gemma(j, secs):
    """First finished Gemma run: slots per GPU from its peak memory, remaining-time projection, gate."""
    args = j["cmd"]
    m = manifest_of(args[args.index("--experiment") + 1], args[args.index("--seeds") + 1])
    peak = (m.get("peak_memory") or {}).get("gpu_peak_gb")
    with lock:
        weight = 1 if isinstance(peak, (int, float)) and peak <= TWO_PER_GPU_MAX_GB else 2
        # Exp 1 measured two Gemma runs per GPU at ~25% less throughput than one (pairs 2.1-4.3 h vs 68-81 min solo)
        weight = int(os.environ.get("GEMMA_WEIGHT", weight))
        gate.update(gemma_weight=weight, measured_s=secs, peak_gb=peak)
        for other in JOBS:
            if other["gemma_train"] and state[other["name"]] == "pending":
                other["weight"] = weight
        left_gemma = sum(1 for o in JOBS if o["gemma_train"] and state[o["name"]] in ("pending", "running"))
        left_qwen = sum(1 for o in JOBS if o["group"] == "train" and not o["gemma_train"] and state[o["name"]] == "pending")
        # conservative: Gemma at the measured one-per-GPU speed even if two share a GPU; Qwen SLOTS // weight per GPU
        hours = (left_gemma * secs / len(GPUS)
                 + left_qwen * QWEN_RUN_S / (len(GPUS) * max(1, SLOTS // QWEN_TRAIN_WEIGHT))) / 3600
        gate["projection_h"] = round(hours, 1)
        if hours > GATE_HOURS:
            gate["stopped"] = f"projected {hours:.1f} h of training left > {GATE_HOURS:g} h"
    log_event(job=j["name"], event="gemma_gate", peak_gb=peak, weight=weight, measured_s=secs,
              projection_h=gate["projection_h"], stop=bool(gate["stopped"]))


def disk_ok():
    return shutil.disk_usage("/").free / 1e9 >= MIN_FREE_GB


def main():
    global JOBS
    JOBS = build_jobs()
    names = {j["name"] for j in JOBS}
    assert len(names) == len(JOBS) and all(d in names for j in JOBS for d in j["deps"])
    for j in JOBS:
        state[j["name"]] = "done" if (OUT / "done" / j["name"]).exists() else "pending"
        info[j["name"]] = {}
    done_gemma = [j for j in JOBS if j["gemma_train"] and state[j["name"]] == "done"]
    if done_gemma:                          # restart after the gate was passed: reuse the first measurement
        first = done_gemma[0]
        decide_gemma(first, int((OUT / "done" / first["name"]).read_text() or 0))
        gate["stopped"] = None if os.environ.get("GATE_OVERRIDE") else gate["stopped"]
    log_event(event="queue_start", jobs=len(JOBS), done=sum(s == "done" for s in state.values()), gpus=GPUS,
              shards=SHARDS)
    exit_code = 0
    for exp in (1, 2, 3):
        mine = [j for j in JOBS if j["exp"] == exp]
        log_event(event="experiment_start", exp=exp)
        while True:
            with lock:
                running = [j for j in mine if state[j["name"]] == "running"]
                pending = [j for j in mine if state[j["name"]] == "pending"]
                if not running and not pending:
                    break
                blocked = gate["stopped"] or not disk_ok()
                if blocked and not running:
                    break
                for j in pending:
                    if blocked:
                        break
                    if any(state[d] in ("failed", "skipped") for d in j["deps"]):
                        state[j["name"]] = "skipped"
                        log_event(job=j["name"], exp=exp, event="skipped_dep_failed")
                        continue
                    if any(state[d] != "done" for d in j["deps"]):
                        continue
                    if j["gemma_train"] and gate["gemma_weight"] is None and any(
                            state[o["name"]] == "done" for o in JOBS if o["gemma_train"]):
                        continue            # the gate decision is being computed
                    if not j["gpu"]:
                        gpu = None
                    else:
                        gpu = next((g for g in sorted(GPUS, key=lambda g: -free[g]) if free[g] >= j["weight"]), None)
                        if gpu is None:
                            continue
                        free[gpu] -= j["weight"]
                    state[j["name"]] = "running"
                    info[j["name"]] = {"gpu": gpu, "started": now()}
                    threading.Thread(target=run, args=(j, gpu), daemon=True).start()
                write_state(JOBS, exp)
            time.sleep(10)
        with lock:
            write_state(JOBS, exp)
        failed = [j["name"] for j in mine if state[j["name"]] in ("failed", "skipped")]
        if gate["stopped"]:
            log_event(event="GATE_STOP", exp=exp, reason=gate["stopped"])
            exit_code = 3
            break
        if not disk_ok() and any(state[j["name"]] == "pending" for j in mine):
            log_event(event="DISK_STOP", exp=exp, free_gb=round(shutil.disk_usage("/").free / 1e9, 2))
            exit_code = 4
            break
        if failed:
            log_event(event="EXPERIMENT_FAILED", exp=exp, failed=failed)
            exit_code = 1
            break
        log_event(event="experiment_done", exp=exp)
    else:
        log_event(event="QUEUE_DONE")
    with lock:
        write_state(JOBS, None)
    sys.exit(exit_code)


JOBS = []
if __name__ == "__main__":
    main()
