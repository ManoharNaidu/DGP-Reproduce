#!/usr/bin/env python3
"""dgp-dashboard: read-only loopback service. Aggregates the extension queue, run results, caches, GPUs."""
import glob, json, os, subprocess, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

D = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(D, "..", ".."))                 # DGP-Reproduce root
sys.path.insert(0, os.path.join(R, "scripts"))
cache = {"t": 0, "v": None}

ARMS = {  # experiment name -> (label, what changes)
    "dgp_amazonvideo_seed0": ("A baseline", "DGP, K=2 (existing 5-seed run)"),
    "parity_dgp": ("parity", "baseline re-run on the new software stack"),
    "ext_armB": ("B disc-neighbours", "neighbour text = discriminative lines"),
    "ext_armC": ("C disc-target", "+ target discriminative line"),
    "ext_armBC": ("B+C", "both"),
    "ext_armD": ("D resid-in-prompt", "negative control: residual text in prompt"),
    "ext_armE1": ("E1 resid-aux", "baseline prompt + residual auxiliary loss"),
    "ext_armE2": ("E2 disc+resid-aux", "arm C prompt + residual auxiliary loss"),
    "ext_K1": ("K=1", "3 metapaths, 1 diffusion step"),
    "ext_K3": ("K=3", "39 metapaths, 3 diffusion steps"),
}
BASE = "dgp_amazonvideo_seed0"
EXPECTED = {"target": 10023, "node:node_discriminative": 35747, "metapath:node_discriminative": 119829,
            "metapath:node_summary": 29978}


def read_jsonl(p):
    out = []
    if os.path.exists(p):
        for line in open(p):
            try: out.append(json.loads(line))
            except ValueError: pass
    return out


def jobs_state():
    try:
        import ext_queue
        order = [(j["name"], j["group"], j["deps"]) for j in ext_queue.build_jobs()]
    except Exception:
        order = []
    ev = read_jsonl(os.path.join(R, "results", "ext", "status.jsonl"))
    last = {}
    for e in ev:
        last[e["job"]] = e
    done = {os.path.basename(p) for p in glob.glob(os.path.join(R, "results", "ext", "done", "*"))}
    rows = []
    for name, group, deps in order:
        e = last.get(name)
        if name in done: st = "done"
        elif e and e["event"] == "start": st = "running"
        elif e and e["event"] in ("end", "skipped_dep_failed"): st = "failed"
        else: st = "waiting"
        row = {"name": name, "group": group, "state": st, "gpu": e.get("gpu") if e else None}
        if e and e["event"] == "start":
            row["started"] = e["t"]
        if e and e["event"] == "end":
            row["seconds"] = e.get("seconds")
        rows.append(row)
    return rows


def stores():
    kmap = {}
    for m in glob.glob(os.path.join(R, "datasets", "cache", "mdk", "*", "metadata.json")):
        meta = json.load(open(m)); kmap[meta.get("cache_key")] = meta.get("K")
    out = []
    for m in glob.glob(os.path.join(R, "datasets", "cache", "*_summaries", "*", "metadata.json")):
        meta = json.load(open(m)); d = os.path.dirname(m); f = os.path.join(d, "summaries.jsonl")
        n = sum(1 for _ in open(f)) if os.path.exists(f) else 0
        tpl = meta.get("template", "")
        if meta.get("stage") == "node_summaries":
            target = meta.get("role") == "target"
            label = f"{'target' if target else 'neighbour'} · {tpl}"
            exp = EXPECTED["target"] if target else (EXPECTED["node:node_discriminative"] if tpl == "node_discriminative" else None)
        else:
            k = kmap.get(meta.get("mdk_cache"))
            label = f"metapath · K={k} · over {meta.get('node_template', '')}"
            exp = {1: 29978, 2: 119829}.get(k)
        out.append({"label": label, "lines": n, "expected": exp})
    return sorted(out, key=lambda x: x["label"])


def results():
    groups = {}
    for mp in glob.glob(os.path.join(R, "results", "raw", "*", "metrics.json")):
        d = os.path.dirname(mp)
        try:
            man = json.load(open(os.path.join(d, "run_manifest.json"))); met = json.load(open(mp))
        except (OSError, ValueError):
            continue
        exp = man.get("experiment", "")
        if exp.startswith(BASE): exp = BASE
        if exp not in ARMS: continue
        t = met["test"]
        groups.setdefault(exp, []).append({"seed": met.get("seed"), "f1": t["macro_f1"], "auroc": t["auroc"], "auprc": t["auprc"],
                                           "best_epoch": met.get("best_epoch"), "minutes": round((man.get("runtime_seconds") or 0) / 60)})
    def ms(v):
        n = len(v); m = sum(v) / n; s = (sum((x - m) ** 2 for x in v) / n) ** .5
        return m, s
    rows = []
    base = None
    for exp, (label, desc) in ARMS.items():
        runs = groups.get(exp, [])
        row = {"exp": exp, "label": label, "desc": desc, "n": len(runs), "seeds": sorted(r["seed"] for r in runs)}
        for k in ("f1", "auroc", "auprc"):
            if runs: row[k], row[k + "_sd"] = ms([r[k] for r in runs])
        if runs: row["minutes"] = round(sum(r["minutes"] for r in runs) / len(runs))
        if exp == BASE: base = row
        rows.append(row)
    for row in rows:
        if base and row["n"] and base["n"] and row["exp"] != BASE:
            for k in ("f1", "auroc", "auprc"): row["d_" + k] = row[k] - base[k]
    return rows


def gpus():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5).stdout
        return [dict(zip(("index", "used", "total", "util"), map(int, (x.strip() for x in l.split(","))))) for l in o.strip().splitlines()]
    except Exception:
        return []


def neighbours():
    p = os.path.join(R, "analysis", "out", "neighbor_analysis.json")
    return json.load(open(p)) if os.path.exists(p) else None


def mdk3():
    for m in glob.glob(os.path.join(R, "datasets", "cache", "mdk", "*", "metadata.json")):
        meta = json.load(open(m))
        if meta.get("K") == 3 and meta.get("use_mdk"):
            return len(glob.glob(os.path.join(os.path.dirname(m), "*.npz")))
    return 0


def state():
    if time.time() - cache["t"] > 3:
        cache.update(t=time.time(), v={"updated": time.strftime("%H:%M:%S"), "jobs": jobs_state(), "stores": stores(),
                                       "results": results(), "gpus": gpus(), "neighbours": neighbours(), "mdk_k3": mdk3()})
    return cache["v"]


def css():
    p = json.load(open(os.path.join(D, "palette.json")))
    v = lambda c: ";".join(f"--{k}:{x}" for k, x in c.items())
    return f":root{{{v(p['light'])}}}@media(prefers-color-scheme:dark){{:root{{{v(p['dark'])}}}}}"


class H(BaseHTTPRequestHandler):
    def send(self, code, body, ct="application/json"):
        b = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body)).encode()
        self.send_response(code); self.send_header("Content-Type", ct); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/": self.send(200, open(os.path.join(D, "index.html"), "rb").read(), "text/html")
        elif self.path == "/palette.css": self.send(200, css(), "text/css")
        elif self.path == "/health": self.send(200, {"ok": True, "version": open(os.path.join(D, "VERSION")).read().strip() if os.path.exists(os.path.join(D, "VERSION")) else "dev"})
        elif self.path == "/api/state": self.send(200, state())
        else: self.send(404, {"error": "not found"})
    def log_message(self, *a): pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8780
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()   # loopback only; read-only
