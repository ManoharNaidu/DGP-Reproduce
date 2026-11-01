"""Live, read-only progress dashboard for the DGP follow-up work of 2026-11-01: Exp 4 (writer vs classifier), the
control queues (scripts/ctl_queue.sh), baselines, and the overall plan.

    python tools/x0-dashboard/server.py [port]        # default 18790, binds 127.0.0.1 only

Plain standard library. Reads (never writes):
    tools/x0-dashboard/plan.json                   the plan and each item's state (edited by hand)
    results/gemma_swap/x0/status.jsonl             Exp 4 queue events (scripts/x0_queue.sh)
    results/controls/<NAME>/status.jsonl           control queue events (scripts/ctl_queue.sh), found automatically
    results/raw/<run>/{training_log.jsonl,metrics.json}
    logs/gemma_swap/<job>.log, logs/controls/<job>.log   tails, whitelisted job names only
    results/gemma_swap/exp4.{json,md}              Exp 4 analysis
    nvidia-smi                                     GPU load
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
RAW = os.path.join(ROOT, "results", "raw")
EXP4 = os.path.join(ROOT, "results", "gemma_swap", "exp4")
CONTROLS_MD = os.path.join(ROOT, "results", "controls", "controls.md")
METRICS = ("macro_f1", "auroc", "auprc")
MAX_EPOCHS, PATIENCE = 10, 2                       # configs/models/dgp.yaml (early stopping on val loss)
# control queues: NAME -> (experiment, title, fallback hours per run while two runs share the GPU)
CONTROLS = {"T0": ("ctl_T0", "Control · LLM on the target review only", 0.8),
            "YELP": ("dgp_yelpchi", "DGP on YelpChi (summaries, then 5 seeds)", 1.2),
            "NOMDK": ("ctl_noMDK", "Ablation · w/o MDK (random neighbours, AmazonVideo)", 1.0)}
# reference cells shown in the results tables: experiment -> (short, label)
CELLS = {"dr_R0": ("R0", "Qwen text · Qwen classifier"), "gs_X0": ("X0", "Gemma text · Qwen classifier"),
         "gs_G0": ("G0", "Gemma text · Gemma classifier")}
BASELINES = [  # short, label, experiment (ours) or None, paper values
    ("DGP", "DGP (R0 recipe, original prompt)", "dr_R0", {"macro_f1": 66.91, "auroc": 77.32, "auprc": 34.63}),
    ("DGP·Gemma", "DGP with Gemma-2-9B-it (G0)", "gs_G0", None),
    ("LLM", "LLM, target review only (T0)", "ctl_T0", {"macro_f1": 59.90, "auroc": 71.78, "auprc": 27.03}),
    ("MLP", "MLP on DeBERTa + numeric features", "mlp_amazonvideo", {"macro_f1": 61.74, "auroc": 70.47, "auprc": 26.55}),
]
PAPER = {"macro_f1": 66.91, "auroc": 77.32, "auprc": 34.63}


def studies():
    """Every queue the dashboard knows: Exp 4 plus each results/controls/<NAME>/ that has a status file."""
    out = [{"key": "x0", "title": "Exp 4 · Gemma text with a Qwen classifier", "status": os.path.join(ROOT, "results", "gemma_swap", "x0", "status.jsonl"),
            "logs": os.path.join(ROOT, "logs", "gemma_swap"), "proc": "scripts/x0_queue.sh", "fallback_s": 1.8 * 3600,
            "jobs": [("x0_prompts", "prep", None, None), ("x0_s0", "A", "gs_X0", 0), ("x0_s2", "A", "gs_X0", 2),
                     ("x0_s4", "A", "gs_X0", 4), ("x0_s1", "B", "gs_X0", 1), ("x0_s3", "B", "gs_X0", 3),
                     ("par2_R0_s0", "B", "par2_R0", 0)]}]
    for st in sorted(glob.glob(os.path.join(ROOT, "results", "controls", "*", "status.jsonl")), key=os.path.getmtime):
        name = os.path.basename(os.path.dirname(st))
        exp, title, hours = CONTROLS.get(name, ("ctl_" + name, f"Control · {name}", 1.0))
        out.append({"key": name, "title": title, "status": st, "logs": os.path.join(ROOT, "logs", "controls"),
                    "proc": f"ctl_queue.sh {name} ", "fallback_s": hours * 3600,
                    "jobs": [(f"{name}_prompts", "prep", None, None)] +
                            [(f"{name}_s{s}", "A" if s % 2 == 0 else "B", exp, s) for s in (0, 2, 4, 1, 3)]})
    return out


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


def per_seed(exp):
    out = {}
    for s in range(5):
        t, be = test_metrics(exp, s)
        if t:
            out[s] = {**t, "best_epoch": be}
    return out


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


def job_rows(study, ev, now):
    last = {}
    for e in ev:                                   # the latest attempt of each job wins (the queues are resumable)
        j = e.get("job")
        if e.get("event") == "start":
            last[j] = {"start": e["t"]}
        elif j in last:
            last[j].update({"end": e["t"], "event": e["event"], "rc": e.get("rc"), "secs": e.get("secs")})
    rows = []
    for name, lane, exp, seed in study["jobs"]:
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
                               else f"training epoch {len(hist) + 1}") if hist or (r["elapsed_s"] or 0) > 600
                              else "loading model / epoch 1")
                r["phase_note"] = (f"early stop after {len(hist) + PATIENCE - since} epochs at most"
                                   if hist and since < PATIENCE and len(hist) < MAX_EPOCHS else None)
        rows.append(r)
    return rows


def eta(rows, now, fallback_s):
    done = [r["elapsed_s"] for r in rows if r["experiment"] and r["state"] == "done" and r["elapsed_s"]]
    per = sum(done) / len(done) if done else fallback_s
    lanes = {}
    for r in rows:
        if r["lane"] not in ("A", "B") or r["state"] in ("done", "failed"):
            continue
        left = max(per - (r["elapsed_s"] or 0), 120) if r["state"] == "running" else per
        lanes[r["lane"]] = lanes.get(r["lane"], 0) + left
    secs = max(lanes.values()) if lanes else 0
    return {"per_run_s": per, "measured": bool(done), "remaining_s": secs,
            "finish": datetime.fromtimestamp(now + secs, timezone.utc).strftime("%H:%M UTC") if secs else None}


def queue_state(study, now):
    ev = jsonl(study["status"])
    rows = job_rows(study, ev, now)
    q_alive = alive(study["proc"])
    st = ("done" if ev and ev[-1].get("job") == "queue" and ev[-1].get("event") == "done" else
          "failed" if any(r["state"] == "failed" for r in rows) and not q_alive else
          "running" if q_alive else "stopped" if ev else "not started")
    first = epoch_of(ev[0]["t"]) if ev else None
    end = epoch_of(ev[-1]["t"]) if st == "done" else now
    return {"key": study["key"], "title": study["title"], "state": st, "alive": q_alive,
            "started": ev[0]["t"] if ev else None, "ended": ev[-1]["t"] if st == "done" else None,
            "elapsed_s": end - first if first else None, "jobs": rows, "eta": eta(rows, now, study["fallback_s"]),
            "events": [{**e, "queue": study["key"]} for e in ev[-14:]]}


def baselines():
    out = []
    for short, label, exp, paper in BASELINES:
        out.append({"short": short, "label": label, "experiment": exp, "paper": paper, "per_seed": per_seed(exp)})
    r0, t0 = per_seed("dr_R0"), per_seed("ctl_T0")
    shared = sorted(set(r0) & set(t0))
    ctx = ({m: sum(r0[s][m] - t0[s][m] for s in shared) / len(shared) for m in METRICS} |
           {"seeds": shared, "better": {m: sum(r0[s][m] > t0[s][m] for s in shared) for m in METRICS}}) if shared else None
    return {"rows": out, "graph_context": ctx}


def state():
    now = time.time()
    queues = [queue_state(s, now) for s in studies()]
    events = sorted((e for q in queues for e in q["events"]), key=lambda e: e.get("t", ""), reverse=True)[:18]
    par, _ = test_metrics("par2_R0", 0)
    return {"now": datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "plan": read_json(os.path.join(HERE, "plan.json")), "queues": queues,
            "exp4": {"cells": {k: {"short": v[0], "label": v[1], "per_seed": per_seed(k)} for k, v in CELLS.items()},
                     "paper": PAPER, "par2_R0": par, "analysis": read_json(EXP4 + ".json"),
                     "analysis_md": os.path.exists(EXP4 + ".md")},
            "baselines": baselines(), "gpu": gpu(), "events": events}


def log_path(job):
    for s in studies():
        for name, *_ in s["jobs"]:
            if name == job:
                return os.path.join(s["logs"], job + ".log")
    return None


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
            m = re.search(r"(?:^|&)job=([A-Za-z0-9_]+)(?:&|$)", query)
            p = log_path(m.group(1)) if m else None      # whitelist: job names of the known queues only
            if not p:
                self.send(400, b"bad job", "text/plain")
                return
            try:
                with open(p, "rb") as f:
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
        elif path == "/files/controls.md" and os.path.exists(CONTROLS_MD):
            with open(CONTROLS_MD, "rb") as f:
                self.send(200, f.read(), "text/plain; charset=utf-8")
        else:
            self.send(404, b"not found", "text/plain")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18790
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
