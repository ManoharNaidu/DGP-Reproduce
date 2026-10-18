#!/usr/bin/env python3
"""dr-dashboard: read-only progress page for the disc/resid study (FLAG's discriminative / residual text in DGP).

    python tools/dr-dashboard/server.py 18781        # binds 127.0.0.1 only; exposed through the Caddy portal

Reads, never writes: generation caches, logs/disc_resid/*, results/disc_resid/*, results/raw/dr_*, nvidia-smi.
"""
import glob, json, os, re, subprocess, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

D = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(D, "..", ".."))
N_TARGETS = 10023
SEEDS = [0, 1, 2, 3, 4]
ARMS = [  # (key, experiment, label, what changes)
    ("R0", "dr_R0", "R0 baseline", "original DGP configuration, re-run on this stack"),
    ("R1", "dr_R1", "R1 disc (FLAG prompt)", "+ discriminative line next to the raw text"),
    ("R1b", "dr_R1b", "R1b disc (no-verdict)", "+ discriminative line, prompt forbids stating the class"),
    ("R2", "dr_R2", "R2 disc + residual loss", "R1 + FLAG residual branch (KL-to-uniform + orthogonality)"),
    ("R2b", "dr_R2b", "R2b no-verdict + residual loss", "R1b + FLAG residual branch"),
]
TEXTS = [("dr_discriminative", "Discriminative (FLAG prompt)"), ("dr_residual", "Residual"),
         ("dr_discriminative_nv", "Discriminative (no-verdict)")]
ORIGINAL = "dgp_amazonvideo_seed0"
_cache = {"t": 0, "v": None}


def lines(path):
    try:
        with open(path, "rb") as f:
            return sum(1 for ln in f if ln.strip())
    except OSError:
        return 0


def generation():
    out = []
    for tmpl, label in TEXTS:
        row = {"template": tmpl, "label": label, "done": 0, "total": N_TARGETS, "rate_per_min": None}
        for meta in glob.glob(os.path.join(R, "datasets/cache/node_summaries/*/metadata.json")):
            m = json.load(open(meta))
            if m.get("template") == tmpl and m.get("truncate_chars") == 1200:
                p = os.path.join(os.path.dirname(meta), "summaries.jsonl")
                row["done"] = lines(p)
                if os.path.exists(p) and row["done"]:
                    start = time.mktime(time.strptime(m["created"], "%Y-%m-%dT%H:%M:%S"))
                    end = os.path.getmtime(p)
                    if end > start:
                        row["rate_per_min"] = row["done"] / ((end - start) / 60)
                    row["last_write_s"] = time.time() - end
        out.append(row)
    log = os.path.join(R, "logs/disc_resid/gen_all.log")
    stages = [ln.strip() for ln in open(log)] if os.path.exists(log) else []
    return {"texts": out, "log": stages[-12:], "finished": any("GEN_DONE" in s for s in stages)}


def run_dir(exp, seed):
    hits = [d for d in glob.glob(os.path.join(R, f"results/raw/{exp}_seed{seed}_*"))
            if os.path.basename(d).rsplit("_seed", 1)[0] == exp]
    return max(hits, key=os.path.getmtime) if hits else None


def metrics(exp, seed):
    d = run_dir(exp, seed)
    if d and os.path.exists(os.path.join(d, "metrics.json")):
        return json.load(open(os.path.join(d, "metrics.json")))
    return None


def queue():
    done_dir = os.path.join(R, "results/disc_resid/done")
    qlog = os.path.join(R, "logs/disc_resid/queue.log")
    ev = [ln.strip() for ln in open(qlog)] if os.path.exists(qlog) else []
    started = {}
    for ln in ev:
        m = re.match(r"\[(.+?)\] start (\S+)", ln)
        if m:
            started[m.group(2)] = m.group(1)
    failed = {re.search(r"FAILED (\S+)", ln).group(1) for ln in ev if "FAILED" in ln}
    jobs, durations = [], {}
    for seed in SEEDS:
        for key, exp, label, _ in ARMS:
            job = f"{exp}_s{seed}"
            row = {"job": job, "arm": key, "seed": seed, "state": "waiting"}
            dpath = os.path.join(done_dir, job)
            if os.path.exists(dpath):
                row["state"] = "done"
                row["seconds"] = int(open(dpath).read().strip() or 0)
                durations.setdefault(key, []).append(row["seconds"])
            elif job in failed and job not in started:
                row["state"] = "failed"
            elif job in started:
                row["state"] = "running" if ev and started[job] == max(started.values()) and not any(
                    "QUEUE_DONE" in e for e in ev) else ("failed" if job in failed else "stopped")
                row["started"] = started[job]
                d = run_dir(exp, seed)
                tl = os.path.join(d, "training_log.jsonl") if d else None
                if tl and os.path.exists(tl):
                    hist = [json.loads(x) for x in open(tl) if x.strip()]
                    row["epochs"] = len(hist)
                    if hist:
                        row["last_epoch"] = {k: hist[-1].get(k) for k in ("epoch", "train_loss", "val_loss", "val_auroc", "val_auprc")}
            jobs.append(row)
    # ETA: mean duration per arm where known; residual arms fall back to 2x the plain mean
    plain = [s for k in ("R0", "R1", "R1b") for s in durations.get(k, [])]
    resid = [s for k in ("R2", "R2b") for s in durations.get(k, [])]
    est = {"plain": sum(plain) / len(plain) if plain else 1.1 * 3600}
    est["resid"] = sum(resid) / len(resid) if resid else 2 * est["plain"]
    remaining = sum(est["resid" if j["arm"].startswith("R2") else "plain"] for j in jobs if j["state"] in ("waiting", "running", "stopped"))
    return {"jobs": jobs, "eta_seconds": remaining, "est": est, "log": ev[-8:], "finished": any("QUEUE_DONE" in e for e in ev)}


def results():
    rows = []
    base = {s: metrics("dr_R0", s) for s in SEEDS}
    orig = {s: metrics(ORIGINAL, s) for s in SEEDS}
    for key, exp, label, what in ARMS:
        per = {}
        for s in SEEDS:
            m = metrics(exp, s)
            if m:
                t = m["test"]
                ref = base.get(s) if key != "R0" else orig.get(s)
                per[s] = {k: t[k] for k in ("macro_f1", "auroc", "auprc")}
                per[s]["best_epoch"] = m.get("best_epoch")
                if ref:
                    per[s]["delta"] = {k: t[k] - ref["test"][k] for k in ("macro_f1", "auroc", "auprc")}
        rows.append({"arm": key, "label": label, "what": what, "per_seed": per,
                     "ref": "original Sept. run" if key == "R0" else "R0"})
    paper = {"macro_f1": 66.91, "auroc": 77.32, "auprc": 34.63}
    original = {k: (sum(orig[s]["test"][k] for s in SEEDS if orig[s]) / max(1, sum(1 for s in SEEDS if orig[s])))
                for k in paper}
    return {"arms": rows, "paper": paper, "original_mean": original}


def diagnostics():
    p = os.path.join(R, "results/disc_resid/text_diagnostics.json")
    return json.load(open(p)) if os.path.exists(p) else None


def gpus():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        return [dict(zip(("name", "util", "mem_used", "mem_total", "temp"), [x.strip() for x in ln.split(",")]))
                for ln in out.strip().splitlines()]
    except Exception:
        return []


def state():
    if time.time() - _cache["t"] > 5:
        _cache["v"] = {"now": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), "generation": generation(),
                       "queue": queue(), "results": results(), "diagnostics": diagnostics(), "gpus": gpus()}
        _cache["t"] = time.time()
    return _cache["v"]


class H(BaseHTTPRequestHandler):
    def send(self, code, body, ct):
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send(200, open(os.path.join(D, "index.html"), "rb").read(), "text/html; charset=utf-8")
        elif self.path.startswith("/api/state"):
            self.send(200, json.dumps(state(), default=float).encode(), "application/json")
        else:
            self.send(404, b"not found", "text/plain")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18781
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
