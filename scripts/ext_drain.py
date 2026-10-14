"""Drain helper: used after the ext_queue scheduler was stopped while jobs were still running.

Waits for the given orphaned job PIDs, decides success from their outputs (the exit code is not available for
non-child processes), then writes results/ext/done/<job> and an `end` event exactly as ext_queue.py does.
    python scripts/ext_drain.py PID [PID ...]
"""
import glob, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "results" / "ext"
GEN_TOTAL = {"gen_B": None}


def job_of(cmd):
    a = cmd.split()
    if "scripts/train.py" in cmd:
        exp = a[a.index("--experiment") + 1]; seed = a[a.index("--seeds") + 1]
        return f"train_{exp[4:]}_s{seed}", ("train", exp, int(seed))
    cfg = Path(a[a.index("--config") + 1]).stem
    name = {"armB_disc_neighbors": "gen_B", "armE2_disc_resid_aux": "gen_T", "K1": "gen_K1", "armBC_disc_both": "gen_BC", "K3": "gen_K3"}[cfg]
    return name, ("gen", cfg, None)


def train_ok(exp, seed):
    for m in glob.glob(str(ROOT / "results" / "raw" / "*" / "metrics.json")):
        d = os.path.dirname(m)
        try:
            man = json.load(open(os.path.join(d, "run_manifest.json")))
            if man.get("experiment") == exp and json.load(open(m)).get("seed") == seed:
                return True
        except (OSError, ValueError):
            pass
    return False


def gen_ok(name):
    log = (ROOT / "logs" / "ext" / f"{name}.log").read_text(errors="ignore").splitlines()[-40:]
    return not any("Traceback" in l or "Error" in l for l in log)


def alive(pid):
    try:
        os.kill(pid, 0)
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return False


def main():
    jobs = {}
    for pid in map(int, sys.argv[1:]):
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        jobs[pid] = (*job_of(cmd), time.time())
    while jobs:
        for pid in list(jobs):
            if alive(pid): continue
            name, (kind, key, seed), t0 = jobs.pop(pid)
            ok = train_ok(key, seed) if kind == "train" else gen_ok(name)
            if ok: (EXT / "done" / name).write_text("drained")
            with open(EXT / "status.jsonl", "a") as f:
                f.write(json.dumps({"job": name, "event": "end", "rc": 0 if ok else 1, "group": kind if kind == "gen" else "train",
                                    "note": "finished after scheduler stop (drain)", "t": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
        time.sleep(20)
    print("drained", flush=True)


if __name__ == "__main__":
    main()
