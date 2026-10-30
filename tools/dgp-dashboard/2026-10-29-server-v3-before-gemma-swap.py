#!/usr/bin/env python3
"""dgp-dashboard: read-only loopback service for the DGP research overview.

    python tools/dgp-dashboard/server.py 8780            # binds 127.0.0.1 only

Three studies, kept separate and labelled: the main DGP reproduction (historical), the completed extension
experiments (Task 2 arms, Task 3 K sweep, disc/resid study; historical) and the FLAG-architecture study (live).
Reads, never writes: results/raw/*/{metrics,run_manifest}.json, results/ext/*, results/disc_resid/*,
results/flag_arch/*, logs/flag_arch/*, datasets/cache/*/metadata.json, analysis/out/*, research/paper_specification.md,
nvidia-smi. The FLAG-architecture and disc/resid job states come from tools/dr-dashboard/v2/server.py (imported,
read-only functions only). A value that was never recorded is null (shown as "—"), never zero.

Endpoints: / · /palette.css · /health · /api/state · /api/results.csv · /api/run?id=<run dir> · /api/log?name=<log>
· /files/<report>.
"""
import csv, glob, hmac, importlib.util, io, json, math, os, re, subprocess, sys, time, types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

D = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(D, "..", ".."))                 # DGP-Reproduce root
sys.path.insert(0, os.path.join(R, "scripts"))
cache = {"t": 0, "v": None}
METRICS = ("macro_f1", "auroc", "auprc")
STALL_S = 1800                    # a running job whose log has not moved for 30 min is shown as stalled
TOKEN = os.environ.get("DGP_DASH_TOKEN") or None   # when set (public tunnel): every request needs ?token=, the cookie or a Bearer header

ARMS = {  # experiment name -> (label, what changes)   (v0.2 names, kept for /api/state "results")
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

# study -> arms (key, experiment, label, what changes, job prefix in the ext queue or None)
STUDIES = [
    {"key": "main", "group": "Main DGP reproduction", "name": "DGP reproduction · AmazonVideo", "kind": "historical",
     "baseline": "A", "note": "5 seeds, Qwen3-8B summaries + LoRA; parity = seed 0 re-run on the extension software stack",
     "arms": [("A", BASE, "A baseline", "DGP, K=2, reconstructed defaults", None),
              ("parity", "parity_dgp", "Parity", "baseline re-run on the new software stack", "train_parity")]},
    {"key": "t2", "group": "Extension experiments", "name": "Task 2 · discriminative / residual text", "kind": "historical",
     "baseline": "A", "baseline_from": "main", "note": "arms vs baseline A, paired by seed",
     "arms": [("B", "ext_armB", "B disc-neighbours", "neighbour text = discriminative lines", "train_armB_"),
              ("C", "ext_armC", "C disc-target", "+ target discriminative line", "train_armC_"),
              ("BC", "ext_armBC", "B+C", "both", "train_armBC_"),
              ("D", "ext_armD", "D resid-in-prompt", "negative control: residual text in prompt", "train_armD_"),
              ("E1", "ext_armE1", "E1 resid-aux", "baseline prompt + residual auxiliary loss", "train_armE1_"),
              ("E2", "ext_armE2", "E2 disc+resid-aux", "arm C prompt + residual auxiliary loss", "train_armE2_")]},
    {"key": "t3", "group": "Extension experiments", "name": "Task 3 · neighbours at K = 1, 2, 3", "kind": "historical",
     "baseline": "A", "baseline_from": "main", "note": "K=2 is baseline A",
     "arms": [("K1", "ext_K1", "K=1", "3 metapaths, 1 diffusion step", "train_K1_"),
              ("K3", "ext_K3", "K=3", "39 metapaths, 3 diffusion steps", "train_K3_")]},
    {"key": "dr", "group": "Extension experiments", "name": "Disc/resid study · FLAG text per review", "kind": "historical",
     "baseline": "R0", "note": "R0 = baseline re-run on the same stack; arms vs R0, paired by seed",
     "arms": [("R0", "dr_R0", "R0 baseline", "DGP baseline re-run on this stack", None),
              ("R1", "dr_R1", "R1 disc · FLAG prompt", "+ discriminative line next to the raw text", None),
              ("R1b", "dr_R1b", "R1b disc · no-verdict", "+ discriminative line; prompt forbids stating the class", None),
              ("R2", "dr_R2", "R2 disc + residual loss", "R1 + FLAG residual branch", None),
              ("R2b", "dr_R2b", "R2b no-verdict + residual", "R1b + FLAG residual branch", None)]},
    {"key": "fa", "group": "FLAG-architecture study", "name": "FLAG-architecture study · D per trimmed subgraph", "kind": "live",
     "baseline": "F0", "note": "F0 = baseline re-run; F1 = FLAG's D written once per trimmed target subgraph (+FLAG); "
                               "residual loss / fine-tuning out of scope",
     "arms": [("F0", "fa_F0", "F0 baseline", "main DGP configuration, re-run", None),
              ("F1", "fa_F1", "F1 FLAG-arch D", "D per trimmed subgraph feeds the metapath summaries; target line as Key signals", None)]},
]
FILES = {  # /files/<name> -> path (read-only reports and evidence)
    "deviation.md": ("results", "disc_resid", "deviation.md"),
    "text_diagnostics.md": ("results", "disc_resid", "text_diagnostics.md"),
    "comparison_with_paper.md": ("results", "tables", "comparison_with_paper.md"),
    "FINAL_REPRODUCTION_REPORT.md": ("research", "FINAL_REPRODUCTION_REPORT.md"),
    "neighbor_analysis.json": ("analysis", "out", "neighbor_analysis.json"),
    "fa_pilot_summary.json": ("results", "flag_arch", "pilot", "summary.json"),
}
LOGS = {  # /api/log?name=<key> -> path (tails only)
    "fa_queue": ("logs", "flag_arch", "queue.log"), "fa_gen_all": ("logs", "flag_arch", "gen_all.log"),
    "fa_gen_F1": ("logs", "flag_arch", "gen_fa_F1.log"), "fa_gen_F0": ("logs", "flag_arch", "gen_fa_F0.log"),
    "fa_r_text": ("logs", "flag_arch", "r_text.log"), "fa_gen_R": ("logs", "flag_arch", "gen_fa_R.log"),
    "dr_queue": ("logs", "disc_resid", "queue.log"), "neighbor_analysis": ("logs", "neighbor_analysis.log"),
}


def P(*parts):
    return os.path.join(R, *parts)


def read_json(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_jsonl(p):
    out = []
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            try: out.append(json.loads(line))
            except ValueError: pass
    return out


def tail(p, n=40, max_bytes=65536):
    try:
        with open(p, "rb") as f:
            f.seek(0, 2); size = f.tell(); f.seek(max(0, size - max_bytes))
            data = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    lines = [ln for ln in re.split(r"[\r\n]+", data) if ln.strip()]
    return lines[-n:]


def mtime(p):
    try: return os.path.getmtime(p)
    except OSError: return None


def iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)) if t else None


def safe(fn, default=None):
    try: return fn()
    except Exception as e:                          # one broken section never takes the page down
        return default if default is not None else {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------- dr-dashboard readers (FLAG-arch + disc/resid jobs)
_dr = {"m": None, "err": None}


def dr():
    """tools/dr-dashboard/v2/server.py as a module, pointed at this project root; only its read functions are used."""
    if _dr["m"] is None and _dr["err"] is None:
        try:
            spec = importlib.util.spec_from_file_location("dr_dashboard_server", P("tools", "dr-dashboard", "v2", "server.py"))
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
            m.R = R
            m.ARGS = types.SimpleNamespace(root=None, no_insights=True, manifest=None)
            _dr["m"] = m
        except Exception as e:
            _dr["err"] = f"{type(e).__name__}: {e}"
    return _dr["m"]


# ---------------------------------------------------------------- v0.2 sections (unchanged shapes; stores relabelled)
def jobs_state():
    try:
        import ext_queue
        order = [(j["name"], j["group"], j["deps"]) for j in ext_queue.build_jobs()]
    except Exception:
        order = []
    ev = read_jsonl(P("results", "ext", "status.jsonl"))
    last = {}
    for e in ev:
        last[e["job"]] = e
    done = {os.path.basename(p) for p in glob.glob(P("results", "ext", "done", "*"))}
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
            row["ended"] = e.get("t")
        rows.append(row)
    return rows


def stores():
    kmap = {}
    for m in glob.glob(P("datasets", "cache", "mdk", "*", "metadata.json")):
        meta = read_json(m) or {}; kmap[meta.get("cache_key")] = meta.get("K")
    fa_kind = {}
    for m in glob.glob(P("datasets", "cache", "fa_subgraph_text", "*", "metadata.json")):
        meta = read_json(m) or {}; fa_kind[meta.get("cache_key")] = meta.get("kind")
    out = []
    for m in glob.glob(P("datasets", "cache", "*_summaries", "*", "metadata.json")):
        meta = read_json(m) or {}; d = os.path.dirname(m); f = os.path.join(d, "summaries.jsonl")
        n = sum(1 for _ in open(f)) if os.path.exists(f) else 0
        tpl = meta.get("template", "")
        study = "fa" if str(meta.get("node_text", "")).startswith("fa_subgraph_text/") else ("dr" if tpl.startswith("dr_") else "main/ext")
        if meta.get("stage") == "node_summaries":
            target = meta.get("role") == "target"
            label = f"{'target' if target else 'neighbour'} · {tpl}"
            exp = EXPECTED["target"] if target else (EXPECTED["node:node_discriminative"] if tpl == "node_discriminative" else None)
        else:
            k = kmap.get(meta.get("mdk_cache"))
            over = meta.get("node_template") or (f"FLAG-arch {fa_kind.get(meta['node_text'].split('/', 1)[1], '?')} lines"
                                                 if study == "fa" else "?")
            label = f"metapath · K={k} · over {over}"
            exp = {1: 29978, 2: 119829}.get(k) if study != "fa" else None    # FA: per-target, varies with neighbours
        out.append({"label": label, "lines": n, "expected": exp, "study": study, "written": iso(mtime(f))})
    for m in glob.glob(P("datasets", "cache", "fa_subgraph_text", "*", "metadata.json")):
        meta = read_json(m) or {}; f = os.path.join(os.path.dirname(m), "answers.jsonl")
        n = sum(1 for ln in open(f) if ln.strip()) if os.path.exists(f) else 0
        out.append({"label": f"FLAG-arch subgraph text · {meta.get('kind')} (one answer per target)", "lines": n,
                    "expected": 10023, "study": "fa", "written": iso(mtime(f))})
    return sorted(out, key=lambda x: (x["study"], x["label"]))


def results():
    groups = {}
    for mp in glob.glob(P("results", "raw", "*", "metrics.json")):
        d = os.path.dirname(mp)
        man = read_json(os.path.join(d, "run_manifest.json")); met = read_json(mp)
        if not man or not met or not isinstance(met.get("test"), dict):
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
    rows, base = [], None
    for exp, (label, desc) in ARMS.items():
        runs = groups.get(exp, [])
        row = {"exp": exp, "label": label, "desc": desc, "n": len(runs), "seeds": sorted(r["seed"] for r in runs),
               "runs": sorted(runs, key=lambda r: r["seed"])}
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
        o = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        out = []
        for line in o.strip().splitlines():
            x = [v.strip() for v in line.split(",")]
            num = lambda v: round(float(v)) if re.fullmatch(r"[\d.]+", v) else None
            out.append({"index": num(x[0]), "name": x[1], "used": num(x[2]), "total": num(x[3]), "util": num(x[4]),
                        "temp": num(x[5]), "power": num(x[6])})
        return out
    except Exception:
        return []


def neighbours():
    return read_json(P("analysis", "out", "neighbor_analysis.json"))


def mdk3():
    for m in glob.glob(P("datasets", "cache", "mdk", "*", "metadata.json")):
        meta = read_json(m) or {}
        if meta.get("K") == 3 and meta.get("use_mdk"):
            return len(glob.glob(os.path.join(os.path.dirname(m), "*.npz")))
    return 0


GEN_EST = {"gen_B": 20000, "gen_BC": 9000, "gen_K3": 36000, "gen_K1": 8000, "gen_T": 6000}
SLOTS = 4
BLOCKS = [  # (name, note, job prefixes, what it answers)
    ("Baseline + parity", "A: existing 5 seeds; parity: seed 0 re-run on the new software stack", ["train_parity"], "parity"),
    ("Task 2: discriminative / residual text", "arms B, C, B+C (prompt) and D, E1, E2 (residual)",
     ["gen_T", "gen_B", "gen_BC", "train_armB_", "train_armC_", "train_armBC_", "train_armD_", "train_armE1_", "train_armE2_"], "t2"),
    ("Task 3: neighbours at K = 1, 2, 3", "K=1 and K=3 runs against the K=2 baseline; plus the LLM-free neighbour analysis",
     ["gen_K1", "gen_K3", "train_K1_", "train_K3_"], "t3"),
]


def eta(jobs):
    """Remaining wall time of the extension queue: max(critical chain, remaining work / slots)."""
    fin = sorted(j["seconds"] for j in jobs if j["group"] == "train" and j["state"] == "done" and j.get("seconds"))
    per = fin[len(fin) // 2] if fin else 9000
    work, chain = 0.0, {}
    now = time.time()
    for j in jobs:
        if j["state"] == "done": continue
        est = GEN_EST.get(j["name"], per) if j["group"] == "gen" else per
        if j["state"] == "running" and j.get("started"):
            el = now - time.mktime(time.strptime(j["started"], "%Y-%m-%dT%H:%M:%S"))
            est = max(est - el, 0.15 * est)
        work += est * (2 if "_K3_" in j["name"] else 1)
        chain[j["name"]] = est
    k3 = chain.get("gen_K3", 0) + (per if any(n.startswith("train_K3_") for n in chain) else 0)
    return {"eta_s": round(max(work / SLOTS, k3)) if chain else 0, "partial": not fin, "critical": "gen_K3 then K=3 training" if k3 >= work / SLOTS and k3 else None}


def blocks(jobs):
    out = []
    for name, note, prefixes, key in BLOCKS:
        js = [j for j in jobs if any(j["name"].startswith(p) for p in prefixes)]
        out.append({"name": name, "note": note, "total": len(js), "done": sum(j["state"] == "done" for j in js),
                    "running": sum(j["state"] == "running" for j in js), "failed": sum(j["state"] == "failed" for j in js)})
    return out


def sample_gpus(g):
    h = cache.setdefault("hist", [])
    h.append({"t": time.time(), "util": {x["index"]: x["util"] for x in g}, "mem": {x["index"]: x["used"] for x in g},
              "temp": {x["index"]: x["temp"] for x in g}})
    del h[:-360]


# ---------------------------------------------------------------- statistics (stdlib only)
def mean_sd(v):
    v = [x for x in v if isinstance(x, (int, float))]
    if not v:
        return None
    m = sum(v) / len(v)
    return {"n": len(v), "mean": m, "sd": (sum((x - m) ** 2 for x in v) / len(v)) ** .5}   # population sd (ddof=0)


def _betacf(a, b, x):
    tiny, c, d = 1e-300, 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny); h = d
    for m in range(1, 300):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((a + m2 - 1) * (a + m2)), -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))):
            d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c; c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 3e-14:
            break
    return h


def betainc(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lb) * _betacf(b, a, 1 - x) / b


def paired_t(deltas):
    """Two-sided paired t-test p-value of the seed-wise differences (None when n < 2 or no spread)."""
    n = len(deltas)
    if n < 2:
        return None
    m = sum(deltas) / n
    s = (sum((x - m) ** 2 for x in deltas) / (n - 1)) ** .5
    if s == 0:
        return None
    t = m / (s / math.sqrt(n)); df = n - 1
    return betainc(df / 2, 0.5, df / (df + t * t))


def verdict(paired, base_sd):
    """Direction words never depend on colour: improved / regressed need p < 0.05 (uncorrected); otherwise
    'within noise' when |mean Δ| is below the baseline's seed sd, else a non-significant trend."""
    if not paired or paired["n"] < 2:
        return "single seed" if paired else None
    m, p = paired["mean"], paired["p"]
    if p is not None and p < 0.05:
        return "improved" if m > 0 else "regressed"
    if base_sd is not None and abs(m) < base_sd:
        return "within noise"
    return "trend up (n.s.)" if m > 0 else "trend down (n.s.)"


# ---------------------------------------------------------------- run index + studies
def run_dir(exp, seed):
    hits = [d for d in glob.glob(P("results", "raw", f"{exp}_seed{seed}_*")) if os.path.basename(d).rsplit("_seed", 1)[0] == exp]
    return max(hits, key=lambda d: mtime(d) or 0) if hits else None


def run_record(exp, seed):
    d = run_dir(exp, seed)
    if not d:
        return None
    met = read_json(os.path.join(d, "metrics.json"))
    man = read_json(os.path.join(d, "run_manifest.json")) or {}
    if not isinstance(met, dict) or not isinstance(met.get("test"), dict):
        return {"run_id": os.path.basename(d), "test": None}
    rt = man.get("runtime_seconds")
    g = str(man.get("git_commit") or "")
    return {"run_id": os.path.basename(d), "test": {k: met["test"].get(k) for k in METRICS},
            "val": {k: (met.get("val") or {}).get(k) for k in METRICS} if isinstance(met.get("val"), dict) else None,
            "best_epoch": met.get("best_epoch"), "minutes": round(rt / 60, 1) if isinstance(rt, (int, float)) else None,
            "written": man.get("written"), "git_commit": g[:12] or None, "dirty": g.endswith("-dirty") if g else None}


def job_states(now):
    """Per study: {(arm, seed): job row} normalised to queued / running / stalled / completed / failed / interrupted."""
    out = {"ext": {}, "dr": {}, "fa": {}}
    for j in jobs_state():
        m = re.match(r"train_(.+)_s(\d)$", j["name"])
        if m or j["name"] == "train_parity":
            key = ("parity", 0) if j["name"] == "train_parity" else (m.group(1), int(m.group(2)))
            out["ext"][key] = {"state": {"done": "completed", "waiting": "queued"}.get(j["state"], j["state"]),
                               "seconds": j.get("seconds"), "ended": j.get("ended")}
    m = dr()
    if m:
        for study, q, folder in (("dr", safe(lambda: m.queue(now), {}), "disc_resid"), ("fa", safe(lambda: m.fa_queue(now), {}), "flag_arch")):
            for j in (q or {}).get("jobs", []) if isinstance(q, dict) else []:
                st = {"done": "completed", "waiting": "queued", "stopped": "interrupted"}.get(j["state"], j["state"])
                row = {"state": st, "seconds": j.get("seconds"), "elapsed_s": j.get("elapsed_s"), "error": j.get("error"),
                       "phase": j.get("phase"), "epochs_done": j.get("epochs_done"), "max_epochs": j.get("max_epochs"),
                       "last": j.get("last"), "log": f"logs/{folder}/train_{j['job']}.log"}
                if st == "running":
                    ages = [now - t for t in (mtime(P(row["log"])),) if t]
                    row["log_age_s"] = min(ages) if ages else None
                    if row["log_age_s"] is not None and row["log_age_s"] > STALL_S:
                        row["state"] = "stalled"
                out[study][(j["arm"], j["seed"])] = row
    return out


def paper_targets():
    """AmazonVideo row of the paper's main table, as transcribed in research/paper_specification.md."""
    src = P("research", "paper_specification.md")
    try:
        for i, line in enumerate(open(src, encoding="utf-8"), 1):
            if line.startswith("| AmazonVideo |"):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                vals = [re.match(r"([\d.]+)\s*±\s*([\d.]+)", c) for c in cells]
                vals = [v for v in vals if v]
                if len(vals) >= 3:
                    return {"source": f"research/paper_specification.md:{i}",
                            **{k: {"mean": float(v.group(1)), "sd": float(v.group(2))} for k, v in zip(METRICS, vals[:3])}}
    except OSError:
        pass
    return None


def studies(now):
    js = job_states(now)
    recs = {}
    out = []
    for st in STUDIES:
        jobsrc = js["ext"] if st["key"] in ("main", "t2", "t3") else js.get(st["key"], {})
        arms = []
        for akey, exp, label, what, prefix in st["arms"]:
            seeds = {}
            for s in range(5):
                r = recs.get((exp, s)) or run_record(exp, s)
                recs[(exp, s)] = r
                jkey = ("parity", 0) if prefix == "train_parity" else ((prefix[6:-1] if prefix else akey), s)
                job = jobsrc.get(jkey) if (prefix or st["key"] in ("dr", "fa")) else None
                has = bool(r and r.get("test"))
                state = (job or {}).get("state") or ("completed" if has else None)
                if state == "completed" and not has:
                    state = "completed (no metrics)"
                if prefix == "train_parity" and s > 0:
                    state = "not planned"
                seeds[str(s)] = {"state": state or "unavailable", "run": r if has else None, "job": job}
            per = {s: v["run"]["test"] for s, v in seeds.items() if v["run"]}
            arms.append({"arm": akey, "exp": exp, "label": label, "what": what, "is_baseline": akey == st["baseline"],
                         "seeds": seeds, "n": len(per),
                         "stats": {k: mean_sd([t.get(k) for t in per.values()]) for k in METRICS}})
        out.append({**{k: st[k] for k in ("key", "group", "name", "kind", "baseline", "note")}, "arms": arms})
    # paired deltas vs each study's baseline (Task 2/3 use baseline A of the main study)
    by_key = {s["key"]: s for s in out}
    for s in out:
        src = by_key.get(next((x.get("baseline_from") for x in STUDIES if x["key"] == s["key"]), None) or s["key"])
        base = next((a for a in src["arms"] if a["arm"] == s["baseline"]), None)
        s["baseline_label"] = base["label"] if base else None
        s["baseline_stats"] = base["stats"] if base else None
        for a in s["arms"]:
            if a["is_baseline"] or not base:
                a["paired"] = None
                continue
            a["paired"] = {}
            for k in METRICS:
                ds = {sd: a["seeds"][sd]["run"]["test"][k] - base["seeds"][sd]["run"]["test"][k]
                      for sd in a["seeds"] if a["seeds"][sd]["run"] and base["seeds"][sd]["run"]
                      and isinstance(a["seeds"][sd]["run"]["test"].get(k), (int, float))
                      and isinstance(base["seeds"][sd]["run"]["test"].get(k), (int, float))}
                if not ds:
                    a["paired"][k] = None
                    continue
                v = list(ds.values()); ms = mean_sd(v)
                pr = {"n": len(v), "mean": ms["mean"], "sd": ms["sd"], "improved": sum(x > 0 for x in v),
                      "per_seed": ds, "p": paired_t(v)}
                pr["verdict"] = verdict(pr, (base["stats"][k] or {}).get("sd"))
                a["paired"][k] = pr
    return out


# ---------------------------------------------------------------- overview (live study), insights, diagnostics, infra
def fa_batches():
    """Launch records of the live study; a later launch of the same command supersedes the earlier one."""
    out = []
    for path in sorted(glob.glob(P("results", "flag_arch", "batch_*.json"))):
        b = read_json(path)
        if isinstance(b, dict):
            out.append({"file": os.path.relpath(path, R), "recorded_at": b.get("recorded_at"), "command": b.get("command"),
                        "configs": [os.path.basename(c).replace(".yaml", "") for c in b.get("configs") or []],
                        "git_commit": (b.get("git_commit") or "")[:7] or None})
    for i, b in enumerate(out):
        later = [x for x in out[i + 1:] if x["command"] == b["command"]]
        b["superseded_by"] = later[0]["recorded_at"] if later else None
    return out


def latest_events(n=8, since=None):
    ev = []
    for name in ("fa_gen_all", "fa_queue", "fa_r_text"):
        for ln in tail(P(*LOGS[name]), 30):
            m = re.match(r"\[(\d{4}-\d\d-\d\dT[\d:]+)Z?\]\s*(.*)", ln)
            if m:
                ev.append({"t": m.group(1) + "Z", "text": m.group(2), "source": "/".join(LOGS[name])})
    ev = [e for e in ev if not since or e["t"] >= since]          # only the current launch, not earlier runs
    return sorted(ev, key=lambda e: e["t"])[-n:][::-1]


def fa_overview(now, studies_):
    m = dr()
    fa = safe(lambda: m.fa_state(now), {}) if m else {}
    if not isinstance(fa, dict) or "error" in fa:
        return {"error": (fa or {}).get("error") if isinstance(fa, dict) else "unavailable", "study": "fa"}
    pl = fa.get("pilot") or {}
    est, kinds = pl.get("estimate") or {}, pl.get("kinds") or {}
    gen = fa.get("generation") or {}
    tasks = {t["kind"]: t for t in gen.get("tasks", [])}
    fa_st = next(s for s in studies_ if s["key"] == "fa")
    jobs = [v for a in fa_st["arms"] for v in a["seeds"].values()]
    done_runs = [v for v in jobs if v["run"]]
    total_runs = len(jobs)
    run_s = [v["job"]["seconds"] for v in jobs if v["job"] and v["job"].get("seconds")]
    per_run_s = sum(run_s) / len(run_s) if run_s else None
    items = []

    def item(key, label, done, total, hours, state, note, eta_s=None):
        frac = min(1.0, done / total) if (total and done is not None) else None
        items.append({"key": key, "label": label, "done": done, "total": total, "fraction": frac, "hours_est": hours,
                      "state": state, "note": note, "eta_s": eta_s})

    D, Rk = tasks.get("discriminative") or {}, tasks.get("residual") or {}

    def gen_eta(kind, t):
        """Seconds left for a running text kind: rate from this dashboard's own samples over >= 10 min, else the
        pilot's measured seconds per subgraph (the cache's own start time includes the pilot and restarts)."""
        if not t or t.get("state") != "running" or not t.get("total") or t.get("done") is None:
            return None, None
        h = cache.setdefault("gen_hist", {}).setdefault(kind, [])
        if not h or h[-1][1] != t["done"] or now - h[-1][0] > 60:
            h.append((now, t["done"]))
        del h[:-400]
        win = [x for x in h if now - x[0] <= 3600]
        left = t["total"] - t["done"]
        if len(win) >= 2 and win[-1][0] - win[0][0] >= 600 and win[-1][1] > win[0][1]:
            return left / ((win[-1][1] - win[0][1]) / (win[-1][0] - win[0][0])), "measured over the last " + dur_s(win[-1][0] - win[0][0])
        sps = (kinds.get(kind) or {}).get("seconds_per_subgraph")
        return (left * sps, "pilot rate") if sps else (None, None)
    mp_req = ((kinds.get("discriminative") or {}).get("metapath_summaries") or {})
    mp_total = round(mp_req["requests"] / pl["n"] * 10023) if mp_req.get("requests") and pl.get("n") else None
    mp_done = (D.get("metapath_summaries") or {}).get("done")
    gen_log = tail(P(*LOGS["fa_gen_all"]), 50)
    d_eta, d_basis = gen_eta("discriminative", D)
    item("d_text", "D text per subgraph", D.get("done"), D.get("total"), (est.get("generation_h") or {}).get("discriminative"),
         D.get("state"), f"parsed {pct(D.get('success_rate'))} of subgraphs · node coverage {pct(D.get('node_coverage'))}"
         + (f" · {dur_s(d_eta)} left ({d_basis})" if d_eta else ""), d_eta)
    item("d_meta", "D metapath summaries", mp_done, mp_total, (est.get("metapath_summaries_h") or {}).get("discriminative"),
         "completed" if mp_total and mp_done and mp_done >= mp_total else
         ("running" if D.get("state") == "completed" and gen.get("state") == "running" else "queued"),
         "starts when the D text is complete · expected total scaled from the pilot (varies with neighbours)")
    prompts_done = sum(any(glob.glob(P("datasets", "cache", "prompts", f"{e}_*.jsonl"))) for e in ("fa_F1", "fa_F0"))
    item("prompts", "Final prompts F1, F0", prompts_done, 2, None, "completed" if prompts_done == 2 else "queued", "one file per arm")
    r_eta, _ = gen_eta("residual", Rk)
    item("r_text", "R text per subgraph (text only)", Rk.get("done"), Rk.get("total"), (est.get("generation_h") or {}).get("residual"),
         Rk.get("state"), f"runs after training · parsed {pct(Rk.get('success_rate'))} so far", r_eta)
    running = [dict(arm=a["arm"], seed=int(s), **{k: v["job"].get(k) for k in ("state", "phase", "epochs_done", "max_epochs", "elapsed_s", "log_age_s")})
               for a in fa_st["arms"] for s, v in a["seeds"].items() if v["job"] and v["job"]["state"] in ("running", "stalled")]
    per_h = (per_run_s / 3600) if per_run_s else 1.1
    gen_h = sum(x["hours_est"] or 0 for x in items)
    gen_done_h = sum((x["hours_est"] or 0) * (x["fraction"] or 0) for x in items)
    train_done_h = len(done_runs) * per_h + sum(min(per_h * .95, (r.get("elapsed_s") or 0) / 3600) for r in running)
    total_h = gen_h + total_runs * per_h
    rem = []
    for x in items:
        if x["fraction"] is not None and x["fraction"] < 1 and x["hours_est"]:
            live = x["eta_s"] if x["state"] == "running" and x["eta_s"] else None    # a live rate only while it runs
            rem.append(live if live else (1 - x["fraction"]) * x["hours_est"] * 3600)
    rem.append(max(0.0, total_runs * per_h - train_done_h) * 3600)
    batches = fa_batches()
    current = [b for b in batches if not b["superseded_by"]]
    since = min((b["recorded_at"] for b in current if b["recorded_at"]), default=None)
    return {
        "study": "fa", "name": fa_st["name"], "kind": "live", "batches": batches, "current_since": since,
        "overall": (gen_done_h + train_done_h) / total_h if total_h else None,
        "overall_basis": "hours-weighted: pilot estimates for generation, " + ("measured" if per_run_s else "1.1 h (estimate)") + " per training run",
        "stages": [
            {"key": "generation", "label": "Generation", "items": items,
             "fraction": gen_done_h / gen_h if gen_h else None, "state": gen.get("state")},
            {"key": "training", "label": "Training", "done": len(done_runs), "total": total_runs, "running": running,
             "fraction": len(done_runs) / total_runs if total_runs else None, "per_run_s": per_run_s},
            {"key": "evaluation", "label": "Evaluation", "done": len(done_runs), "total": total_runs,
             "fraction": len(done_runs) / total_runs if total_runs else None,
             "note": "test metrics are written by each run when it finishes"},
        ],
        "runs": {"done": len(done_runs), "total": total_runs, "active": len(running)},
        "eta_s": sum(rem) if rem else None, "eta_basis": "live generation rate where measured, else pilot rates",
        "events": latest_events(since=since), "pilot": {"n": pl.get("n"), "kinds": {k: {x: v.get(x) for x in ("subgraph_success_rate", "node_coverage", "hit_cap")}
                                                                          for k, v in kinds.items()}} if pl else None,
        "gen_log": gen_log[-12:], "queue_log": tail(P(*LOGS["fa_queue"]), 12), "r_log": tail(P(*LOGS["fa_r_text"]), 6),
    }


def dur_s(s):
    s = int(s); h, m = divmod(s // 60, 60)
    return f"{h // 24}d {h % 24}h" if h >= 24 else f"{h}h {m:02d}m" if h else f"{m}m"


def pct(x):
    return f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "—"


def insights(studies_, paper):
    comps = []
    for s in studies_:
        for a in s["arms"]:
            for k, pr in (a.get("paired") or {}).items():
                if pr and pr["n"] >= 2:
                    comps.append({"study": s["key"], "study_name": s["name"], "arm": a["arm"], "label": a["label"], "metric": k,
                                  "mean": pr["mean"], "n": pr["n"], "improved": pr["improved"], "p": pr["p"], "verdict": pr["verdict"],
                                  "baseline": s["baseline_label"]})
    out = {"comparisons": len({(c["study"], c["arm"]) for c in comps}), "by_metric": {}, "stability": [], "notes": []}
    for k in METRICS:
        cs = [c for c in comps if c["metric"] == k]
        if not cs:
            out["by_metric"][k] = None
            continue
        best, worst = max(cs, key=lambda c: c["mean"]), min(cs, key=lambda c: c["mean"])
        sig = [c for c in cs if c["p"] is not None and c["p"] < 0.05]
        out["by_metric"][k] = {"strongest": best, "largest_regression": worst if worst["mean"] < 0 else None,
                               "max_abs": max(abs(c["mean"]) for c in cs), "n_arms": len(cs), "n_p05": len(sig),
                               "significant": sig}
    for s in studies_:
        b = s.get("baseline_stats") or {}
        for a in s["arms"]:
            if a["is_baseline"]:
                continue
            for k in METRICS:
                st, bs = a["stats"].get(k), b.get(k)
                if st and bs and st["n"] >= 3 and bs["sd"] and st["sd"] >= 1.5 * bs["sd"]:
                    out["stability"].append({"study_name": s["name"], "arm": a["arm"], "label": a["label"], "metric": k,
                                             "sd": st["sd"], "baseline_sd": bs["sd"], "n": st["n"]})
    main = next(s for s in studies_ if s["key"] == "main")
    A = next(a for a in main["arms"] if a["arm"] == "A")
    if paper:
        out["paper_gap"] = {k: {"ours": (A["stats"][k] or {}).get("mean"), "paper": paper[k]["mean"],
                                "delta": (A["stats"][k]["mean"] - paper[k]["mean"]) if A["stats"][k] else None,
                                "rel": ((A["stats"][k]["mean"] - paper[k]["mean"]) / paper[k]["mean"]) if A["stats"][k] else None}
                            for k in METRICS}
    out["notes"].append("p-values: paired t-test over seeds, two-sided, uncorrected for the "
                        f"{out['comparisons']} arm comparisons; with 5 seeds a p < 0.05 is weak evidence, not proof.")
    return out


def diagnostics(now):
    m = dr()
    td = read_json(P("results", "disc_resid", "text_diagnostics.json"))
    nb = neighbours()
    nb_log = P(*LOGS["neighbor_analysis"])
    nb_state = ("completed" if nb else ("log present, no output" if os.path.exists(nb_log) else "not run"))
    fa = safe(lambda: m.fa_state(now), {}) if m else {}
    return {
        "stores": stores(),
        "dr_text": {"written": iso(mtime(P("results", "disc_resid", "text_diagnostics.json"))), **(td or {})} if td else None,
        "fa_text": {"tasks": [{k: t.get(k) for k in ("kind", "label", "done", "total", "state", "success_rate", "node_coverage",
                                                    "hit_cap", "overflow", "prompt_tokens_mean", "generated_tokens_mean")}
                              for t in ((fa or {}).get("generation") or {}).get("tasks", [])]} if isinstance(fa, dict) else None,
        "neighbours": {"state": nb_state, "written": iso(mtime(P("analysis", "out", "neighbor_analysis.json"))),
                       "all": [r for r in (nb or {}).get("stats", []) if r.get("metapath") == "ALL"]},
        "reports": [{"name": k, "path": "/".join(v), "written": iso(mtime(P(*v)))} for k, v in FILES.items() if os.path.exists(P(*v))],
    }


def token_accounting():
    rows = []
    for st in STUDIES:
        for akey, exp, label, *_ in st["arms"]:
            name = "token_usage_dgp_amazonvideo" if exp == BASE else f"token_usage_{exp}"
            d = read_json(P("results", "token_usage", name + ".json"))
            if not d:
                continue
            rows.append({"study": st["key"], "arm": akey, "label": label, "file": f"results/token_usage/{name}.json",
                         **{k: d.get(k) for k in ("nodes", "mean_final_prompt_tokens", "p95_prompt_length", "max_prompt_length",
                                                  "mean_node_summary_tokens", "mean_metapath_summary_tokens", "mean_num_neighbors_trimmed",
                                                  "token_reduction")}})
    return rows


def infra(studies_, g):
    return {"gpus": g, "history": cache.get("hist", [])[-180:],
            "history_note": "sampled by this dashboard while it runs (about every 5 s); empty after a restart",
            "runtimes": [{"study": s["key"], "arm": a["arm"], "label": a["label"],
                          "minutes": [v["run"]["minutes"] for v in a["seeds"].values() if v["run"] and v["run"]["minutes"] is not None]}
                         for s in studies_ for a in s["arms"]],
            "tokens": token_accounting(),
            "logs": {k: "/".join(v) for k, v in LOGS.items() if os.path.exists(P(*v))},
            "ext_events": [e for e in read_jsonl(P("results", "ext", "status.jsonl"))[-12:]][::-1],
            "meta": {"version": open(os.path.join(D, "VERSION")).read().strip() if os.path.exists(os.path.join(D, "VERSION")) else "dev",
                     "root": R, "python": sys.version.split()[0], "dr_reader": "ok" if dr() else _dr["err"]}}


def state():
    if time.time() - cache["t"] > 5:
        now = time.time()
        j, g = jobs_state(), gpus()
        sample_gpus(g)
        sts = safe(lambda: studies(now), [])
        paper = safe(paper_targets, {}) or None
        cache.update(t=now, v={
            # v0.2 keys (unchanged)
            "updated": time.strftime("%H:%M"), "jobs": j, "stores": safe(stores, []), "eta": eta(j), "blocks": blocks(j),
            "results": safe(results, []), "gpus": g, "gpu_history": cache["hist"][-60:], "neighbours": neighbours(),
            "mdk_k3": mdk3(), "analysis_running": False,   # v0.2 inferred this from a leftover log file; never claimed now
            # v0.3
            "generated_at": iso(now), "server_time": now, "paper": paper, "studies": sts,
            "overview": safe(lambda: fa_overview(now, sts)) if isinstance(sts, list) and sts else None,
            "insights": safe(lambda: insights(sts, paper)) if isinstance(sts, list) and sts else None,
            "diagnostics": safe(lambda: diagnostics(now)), "infra": safe(lambda: infra(sts if isinstance(sts, list) else [], g)),
        })
    return cache["v"]


def results_csv():
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["study", "arm", "experiment", "seed", "run_id", "macro_f1", "auroc", "auprc", "val_auroc", "best_epoch",
                "minutes", "baseline", "delta_macro_f1", "delta_auroc", "delta_auprc", "status"])
    for s in state().get("studies") or []:
        for a in s["arms"]:
            for sd, v in sorted(a["seeds"].items()):
                r = v["run"] or {}
                t = r.get("test") or {}
                pr = a.get("paired") or {}
                d = [(pr.get(k) or {}).get("per_seed", {}).get(sd) for k in METRICS]
                w.writerow([s["key"], a["arm"], a["exp"], sd, r.get("run_id", "")] +
                           [f"{t[k]:.4f}" if isinstance(t.get(k), (int, float)) else "" for k in METRICS] +
                           [f"{(r.get('val') or {}).get('auroc'):.4f}" if isinstance((r.get("val") or {}).get("auroc"), (int, float)) else "",
                            r.get("best_epoch", ""), r.get("minutes", ""), "" if a["is_baseline"] else s["baseline"]] +
                           [f"{x:.4f}" if isinstance(x, (int, float)) else "" for x in d] + [v["state"]])
    return buf.getvalue()


def css():
    p = json.load(open(os.path.join(D, "palette.json")))
    v = lambda c: ";".join(f"--{k}:{x}" for k, x in c.items())
    return f":root{{{v(p['dark'])}}}:root[data-theme=light]{{{v(p['light'])}}}"


class H(BaseHTTPRequestHandler):
    def send(self, code, body, ct="application/json", extra=None):
        b = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body, default=str)).encode()
        self.send_response(code); self.send_header("Content-Type", ct); self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        if getattr(self, "cookie", None):
            self.send_header("Set-Cookie", self.cookie); self.cookie = None
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers(); self.wfile.write(b)

    def authorised(self, q):
        if not TOKEN:
            return True, None
        cookie = dict(c.strip().split("=", 1) for c in (self.headers.get("Cookie") or "").split(";") if "=" in c)
        bearer = (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
        for cand, from_query in ((q.get("token", ""), True), (cookie.get("dgp_token", ""), False), (bearer, False)):
            if cand and hmac.compare_digest(cand, TOKEN):
                return True, from_query
        return False, None

    def do_GET(self):
        path, _, query = self.path.partition("?")
        q = dict(kv.split("=", 1) for kv in query.split("&") if "=" in kv)
        ok, from_query = self.authorised(q)
        if not ok:
            self.send(401, "Token required: open the link with ?token=<dashboard token>.", "text/plain; charset=utf-8"); return
        if from_query:                                   # remember the token so the page's own requests carry it
            self.cookie = f"dgp_token={TOKEN}; Path=/; HttpOnly; SameSite=Strict; Max-Age=2592000"
        if path == "/": self.send(200, open(os.path.join(D, "index.html"), "rb").read(), "text/html; charset=utf-8")
        elif path == "/palette.css": self.send(200, css(), "text/css")
        elif path == "/health": self.send(200, {"ok": True, "version": open(os.path.join(D, "VERSION")).read().strip() if os.path.exists(os.path.join(D, "VERSION")) else "dev"})
        elif path == "/api/state": self.send(200, state())
        elif path == "/api/results.csv":
            self.send(200, results_csv(), "text/csv; charset=utf-8", {"Content-Disposition": "attachment; filename=dgp_results.csv"})
        elif path == "/api/run":
            rid = q.get("id", "")
            d = P("results", "raw", rid)
            if not re.fullmatch(r"[A-Za-z0-9_]+_seed\d+_[0-9a-f]+", rid) or not os.path.isdir(d):
                self.send(400, {"error": "unknown run"})
            else:
                man = read_json(os.path.join(d, "run_manifest.json")) or {}
                self.send(200, {"run_id": rid, "manifest": man, "metrics": read_json(os.path.join(d, "metrics.json"))})
        elif path == "/api/log":
            name = q.get("name", "")
            if name not in LOGS: self.send(400, {"error": "unknown log"})
            else: self.send(200, "\n".join(tail(P(*LOGS[name]), 80)) or "(log is empty or missing)", "text/plain; charset=utf-8")
        elif path.startswith("/files/") and path[7:] in FILES:
            p = P(*FILES[path[7:]])
            if os.path.exists(p): self.send(200, open(p, "rb").read(), "application/json" if p.endswith(".json") else "text/plain; charset=utf-8")
            else: self.send(404, {"error": "not created yet"})
        else: self.send(404, {"error": "not found"})

    def log_message(self, *a): pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8780
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()   # loopback only; read-only
