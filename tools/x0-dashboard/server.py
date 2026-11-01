"""Live, read-only progress dashboard for Exp 4 (writer/classifier split: gs_X0 + the par2_R0 drift run).

    python tools/x0-dashboard/server.py [port]        # default 18790, binds 127.0.0.1 only

Plain standard library. Reads (never writes) what scripts/x0_queue.sh and scripts/train.py produce:
    results/gemma_swap/x0/status.jsonl            queue events (start / end / rc / secs)
    results/raw/<run>/training_log.jsonl          one line per finished epoch (val loss / AUROC / AUPRC / F1)
    results/raw/<run>/metrics.json                test metrics once a run is finished
    logs/gemma_swap/<job>.log                     tails, whitelisted job names only
    results/gemma_swap/exp4.{json,md}             analysis, once scripts/x0_analysis.py has been run
    nvidia-smi                                    GPU load
Reference cells dr_R0 (Qwen writes + classifies) and gs_G0 (Gemma writes + classifies) are read from results/raw too.
"""
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
STATUS = os.path.join(ROOT, "results", "gemma_swap", "x0", "status.jsonl")
LOGS = os.path.join(ROOT, "logs", "gemma_swap")
RAW = os.path.join(ROOT, "results", "raw")
EXP4 = os.path.join(ROOT, "results", "gemma_swap", "exp4")
METRICS = ("macro_f1", "auroc", "auprc")
MAX_EPOCHS, PATIENCE = 10, 2                       # configs/models/dgp.yaml (early stopping on val loss)
# job -> (lane, experiment, seed); order = queue order in scripts/x0_queue.sh
JOBS = {"x0_prompts": ("prep", None, None),
        "x0_s0": ("A", "gs_X0", 0), "x0_s2": ("A", "gs_X0", 2), "x0_s4": ("A", "gs_X0", 4),
        "x0_s1": ("B", "gs_X0", 1), "x0_s3": ("B", "gs_X0", 3), "par2_R0_s0": ("B", "par2_R0", 0)}
CELLS = {"dr_R0": "Qwen text · Qwen classifier", "gs_X0": "Gemma text · Qwen classifier",
         "gs_G0": "Gemma text · Gemma classifier"}
PAPER = {"macro_f1": 66.91, "auroc": 77.32, "auprc": 34.63}
FALLBACK_RUN_S = 1.8 * 3600                       # before any X0 run has finished (2 runs share the GPU)


def epoch_of(t):
    try:
        return datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def read_json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def jsonl(p):
    out = []
    try:
        with open(p) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass                           # half-written line
    except OSError:
        pass
    return out


def run_dir(exp, seed):
    ds = sorted(glob.glob(os.path.join(RAW, f"{exp}_seed{seed}_*")), key=os.path.getmtime)
    return ds[-1] if ds else None


def test_metrics(exp, seed):
    d = run_dir(exp, seed)
    m = read_json(os.path.join(d, "metrics.json")) if d else None
    t = (m or {}).get("test") or {}
    return ({k: t[k] for k in METRICS}, m.get("best_epoch")) if all(k in t for k in METRICS) else (None, None)


def alive(pattern):
    try:
        return subprocess.run(["pgrep", "-f", pattern], capture_output=True, timeout=3).returncode == 0
    except Exception:
        return False


def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,"
                              "power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        rows = []
        for ln in out.strip().splitlines():
            n, u, mu, mt, t, p = [x.strip() for x in ln.split(",")]
            rows.append({"name": n, "util": float(u), "mem_used": float(mu), "mem_total": float(mt), "temp": float(t),
                         "power": float(p) if p.replace(".", "").isdigit() else None})
        return rows
    except Exception:
        return []


def job_rows(now):
    ev = jsonl(STATUS)
    last = {}
    for e in ev:                                   # the latest attempt of each job wins (the queue is resumable)
        j = e.get("job")
        if e.get("event") == "start":
            last[j] = {"start": e["t"]}
        elif j in last:
            last[j].update({"end": e["t"], "event": e["event"], "rc": e.get("rc"), "secs": e.get("secs")})
    rows = []
    for name, (lane, exp, seed) in JOBS.items():
        x = last.get(name, {})
        st = ("queued" if not x else "running" if "end" not in x else
              "done" if x.get("event") == "end" and (x.get("rc") or 0) == 0 else "failed")
        r = {"job": name, "lane": lane, "experiment": exp, "seed": seed, "state": st, "start": x.get("start"),
             "end": x.get("end"), "rc": x.get("rc"),
             "elapsed_s": (x.get("secs") or (epoch_of(x.get("end")) or now) - epoch_of(x["start"])) if x.get("start") else None}
        if exp:
            d = run_dir(exp, seed) if st != "queued" else None
            hist = jsonl(os.path.join(d, "training_log.jsonl")) if d else []
            r["history"] = [{k: h.get(k) for k in ("epoch", "train_loss", "val_loss", "val_auroc", "val_auprc", "val_macro_f1")}
                            for h in hist]
            best = min(range(len(hist)), key=lambda i: hist[i]["val_loss"]) if hist else None
            r["best_epoch"] = hist[best]["epoch"] if hist else None
            since = len(hist) - 1 - best if hist else 0
            r["test"], _ = test_metrics(exp, seed)
            if st == "running":
                r["phase"] = (("testing on 7,425 reviews" if hist and (since >= PATIENCE or len(hist) >= MAX_EPOCHS)
                               else f"training epoch {len(hist) + 1}") if hist or r["elapsed_s"] > 600
                              else "loading model / epoch 1")
                r["phase_note"] = (f"early stop after {len(hist) + PATIENCE - since} epochs at most"
                                   if hist and since < PATIENCE and len(hist) < MAX_EPOCHS else None)
        rows.append(r)
    return rows, ev


def eta(rows, now):
    done = [r["elapsed_s"] for r in rows if r["experiment"] and r["state"] == "done" and r["elapsed_s"]]
    per = sum(done) / len(done) if done else FALLBACK_RUN_S
    lanes = {}
    for r in rows:
        if r["lane"] not in ("A", "B") or r["state"] in ("done", "failed"):
            continue
        left = max(per - (r["elapsed_s"] or 0), 300) if r["state"] == "running" else per
        lanes[r["lane"]] = lanes.get(r["lane"], 0) + left
    secs = max(lanes.values()) if lanes else 0
    return {"per_run_s": per, "measured": bool(done), "remaining_s": secs,
            "finish": datetime.fromtimestamp(now + secs, timezone.utc).strftime("%H:%M UTC") if secs else None}


def results():
    cells = {}
    for exp, label in CELLS.items():
        per = {}
        for s in range(5):
            t, be = test_metrics(exp, s)
            if t:
                per[s] = {**t, "best_epoch": be}
        cells[exp] = {"label": label, "per_seed": per}
    par, _ = test_metrics("par2_R0", 0)
    return {"cells": cells, "paper": PAPER, "par2_R0": par}


def state():
    now = time.time()
    rows, ev = job_rows(now)
    q_alive = alive("scripts/x0_queue.sh")
    st = ("done" if any(e.get("job") == "queue" and e.get("event") == "done" for e in ev[-1:]) else
          "failed" if any(r["state"] == "failed" for r in rows) and not q_alive else
          "running" if q_alive else "stopped" if ev else "not started")
    first = epoch_of(ev[0]["t"]) if ev else None
    return {"now": datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"), "queue": st,
            "queue_alive": q_alive, "started": ev[0]["t"] if ev else None,
            "elapsed_s": now - first if first else None, "jobs": rows, "eta": eta(rows, now),
            "results": results(), "analysis": read_json(EXP4 + ".json"), "analysis_md": os.path.exists(EXP4 + ".md"),
            "gpu": gpu(), "events": ev[-14:][::-1]}


class H(BaseHTTPRequestHandler):
    def send(self, code, body, ct):
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path, _, query = self.path.partition("?")
        if path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self.send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/api/state":
            self.send(200, json.dumps(state(), default=lambda o: None).encode(), "application/json")
        elif path == "/api/log":
            m = re.search(r"(?:^|&)job=([a-z0-9_]+)(?:&|$)", query)
            if not m or m.group(1) not in JOBS:       # whitelist: the queue's own job names
                self.send(400, b"bad job", "text/plain")
                return
            try:
                with open(os.path.join(LOGS, m.group(1) + ".log"), "rb") as f:
                    f.seek(0, 2)
                    f.seek(max(0, f.tell() - 16384))
                    data = f.read().decode("utf-8", "replace")
                lines = [ln.split("\r")[-1] for ln in data.splitlines()][-60:]   # keep the last tqdm frame only
                self.send(200, ("\n".join(lines) or "(log is empty)").encode(), "text/plain; charset=utf-8")
            except OSError:
                self.send(200, b"(no log yet)", "text/plain; charset=utf-8")
        elif path == "/files/exp4.md" and os.path.exists(EXP4 + ".md"):
            with open(EXP4 + ".md", "rb") as f:
                self.send(200, f.read(), "text/plain; charset=utf-8")
        else:
            self.send(404, b"not found", "text/plain")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18790
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
