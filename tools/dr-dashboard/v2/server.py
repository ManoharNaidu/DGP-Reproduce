#!/usr/bin/env python3
"""dr-dashboard v2: read-only progress page for the disc/resid study (FLAG's discriminative / residual text in DGP).

    python tools/dr-dashboard/v2/server.py 18781                      # live data, binds 127.0.0.1 only
    python tools/dr-dashboard/v2/server.py 18782 --root <mock-root>   # same page on a mock data tree

Reads, never writes, the studies' files: generation caches, logs/disc_resid/*, results/disc_resid/*, results/raw/dr_*,
and for the FLAG-architecture study datasets/cache/fa_subgraph_text/*, logs/flag_arch/*, results/flag_arch/*, results/raw/fa_*.
Writes only its own files next to this script: history.json (1 h of sampled counts / GPU readings, so a restart keeps
the sparklines) and cache/text_insights.json (computed once by text_insights.py when generation is complete).
Every section is computed independently; a missing or half-written file yields null for that field, never an error.

Read-only endpoints besides /api/state: /api/results.csv, /api/manifest?study=dr|fa[&download=1] (experiment manifest
built from run manifests, metrics and batch records), /api/samples?study=dr|fa&arm=<arm>&n=1..5 (random prompts with
split, train/val label and graph features), /api/log?job=<job>. `server.py --manifest fa` prints the manifest.
"""
import argparse, csv, glob, io, json, os, re, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ARGS = None
R = None                  # data root (project root, or a mock tree)
N_TARGETS = 10023
SEEDS = [0, 1, 2, 3, 4]
MAX_EPOCHS = 10
HISTORY_S = 3600
ARMS = [  # key, experiment, label, what changes, uses residual branch
    ("R0", "dr_R0", "R0 baseline", "original DGP configuration, re-run on this stack", False),
    ("R1", "dr_R1", "R1 disc · FLAG prompt", "+ discriminative line next to the raw text", False),
    ("R1b", "dr_R1b", "R1b disc · no-verdict", "+ discriminative line; prompt forbids stating the class", False),
    ("R2", "dr_R2", "R2 disc + residual loss", "R1 + FLAG residual branch (KL-to-uniform + orthogonality)", True),
    ("R2b", "dr_R2b", "R2b no-verdict + residual", "R1b + FLAG residual branch", True),
]
TEXTS = [  # template, key, label — in the order scripts/dr_generate_all.sh produces them
    ("dr_discriminative", "disc_flag", "Discriminative · FLAG prompt"),
    ("dr_residual", "residual", "Residual"),
    ("dr_discriminative_nv", "disc_noverdict", "Discriminative · no-verdict"),
]
GEN_CAP = 4 * 40 + 16     # max_new_tokens for budget 40
ORIGINAL = "dgp_amazonvideo_seed0"
REFERENCE = {"original": {"macro_f1": 65.16, "auroc": 75.07, "auprc": 32.33},
             "paper": {"macro_f1": 66.91, "auroc": 77.32, "auprc": 34.63}}
METRICS = ("macro_f1", "auroc", "auprc")
LOCK = threading.Lock()
_cache = {"t": 0.0, "v": None}
HIST = {"gen": {}, "gpu": []}        # gen: key -> [[t, count]], gpu: [[t, util, mem_used, mem_total, temp]]


# ---------------------------------------------------------------- helpers (all failure-tolerant)
def P(*parts):
    return os.path.join(R, *parts)


def safe(fn, default=None):
    try:
        return fn()
    except Exception as e:            # a section never takes the page down
        sys.stderr.write(f"[dr-dashboard] {fn.__name__ if hasattr(fn, '__name__') else fn}: {e!r}\n")
        return default


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_lines(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return [ln.rstrip("\n") for ln in f]
    except OSError:
        return []


def tail_lines(path, n=40, max_bytes=65536):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read().decode("utf-8", errors="replace")
        return data.splitlines()[-n:]
    except OSError:
        return []


def iso_to_epoch(s):
    try:
        return time.mktime(time.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")) - time.timezone
    except Exception:
        return None


def mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def process_alive(pattern):
    if ARGS.root:                      # mock tree: liveness comes from a file instead of the process table
        live = read_json(P("mock_liveness.json")) or {}
        return bool(live.get(pattern))
    try:
        # anchored on the script's own command line, so shells that merely mention the name do not match
        return subprocess.run(["pgrep", "-f", rf"^(/bin/)?bash (\S*/)?scripts/{re.escape(pattern)}( |$)"],
                              capture_output=True, timeout=3).returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------- history (sampled, persisted)
HIST_FILE = None


def load_history():
    h = read_json(HIST_FILE)
    if isinstance(h, dict):
        HIST["gen"] = {k: v for k, v in (h.get("gen") or {}).items() if isinstance(v, list)}
        HIST["gpu"] = h.get("gpu") if isinstance(h.get("gpu"), list) else []


def save_history():
    try:
        tmp = HIST_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(HIST, f)
        os.replace(tmp, HIST_FILE)
    except OSError:
        pass


def push(series, row, now):
    if not series or now - series[-1][0] >= 9:
        series.append(row)
    while series and now - series[0][0] > HISTORY_S:
        series.pop(0)


# ---------------------------------------------------------------- generation
def cache_store(template):
    for meta in glob.glob(P("datasets", "cache", "node_summaries", "*", "metadata.json")):
        m = read_json(meta)
        if m and m.get("template") == template and m.get("truncate_chars") == 1200:
            return os.path.dirname(meta), m
    return None, None


def scan_store(path):
    """count, empty, multi-line, hit-cap, token lengths, last 5 texts. Tolerates a half-written last line."""
    out = {"done": 0, "empty": 0, "multi_line": 0, "hit_cap": 0, "lengths": [], "samples": []}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                text = str(row.get("text", ""))
                out["done"] += 1
                out["empty"] += not text.strip()
                out["multi_line"] += len([x for x in text.splitlines() if x.strip()]) > 1
                n = row.get("generated_tokens")
                if isinstance(n, (int, float)):
                    out["lengths"].append(n)
                    out["hit_cap"] += n >= GEN_CAP
                out["samples"].append(text)
                if len(out["samples"]) > 5:
                    out["samples"].pop(0)
    except OSError:
        pass
    return out


def generation(now):
    gen_log = read_lines(P("logs", "disc_resid", "gen_all.log"))
    finished_all = any("GEN_DONE" in ln for ln in gen_log)
    alive = process_alive("dr_generate_all.sh")
    tasks, first_open = [], None
    for tmpl, key, label in TEXTS:
        d, meta = cache_store(tmpl)
        s = scan_store(os.path.join(d, "summaries.jsonl")) if d else scan_store("")
        t = {"key": key, "label": label, "template": tmpl, "total": N_TARGETS, "done": s["done"],
             "empty": s["empty"], "multi_line": s["multi_line"], "hit_cap": s["hit_cap"],
             "tokens_mean": round(sum(s["lengths"]) / len(s["lengths"]), 1) if s["lengths"] else None,
             "samples": [x[:220] for x in s["samples"]], "started_at": None, "finished_at": None}
        if meta:
            t["started_at"] = iso_to_epoch(meta.get("created", ""))
        f = os.path.join(d, "summaries.jsonl") if d else None
        last_write = mtime(f) if f else None
        t["last_write_age"] = now - last_write if last_write else None
        if s["done"] >= N_TARGETS:
            t["state"] = "completed"
            t["finished_at"] = last_write
        elif s["done"] > 0 or meta:
            t["state"] = "running" if alive and first_open is None else ("failed" if not alive and not finished_all else "queued")
        else:
            t["state"] = "queued"
        if t["state"] != "completed" and first_open is None:
            first_open = key
        series = HIST["gen"].setdefault(key, [])
        push(series, [round(now), s["done"]], now)
        t["rate_series"] = rate_series(series)
        t["rate_now"] = t["rate_series"][-1][1] if t["rate_series"] else None
        if t["state"] == "completed" and t["started_at"] and t["finished_at"] and t["finished_at"] > t["started_at"]:
            t["rate_avg"] = N_TARGETS / ((t["finished_at"] - t["started_at"]) / 60)
        elif t["started_at"] and s["done"] and last_write and last_write > t["started_at"]:
            t["rate_avg"] = s["done"] / ((last_write - t["started_at"]) / 60)
        else:
            t["rate_avg"] = None
        tasks.append(t)
    done_rates = [t["rate_avg"] for t in tasks if t["state"] == "completed" and t.get("rate_avg")]
    fallback = sum(done_rates) / len(done_rates) if done_rates else None
    eta = 0.0
    eta_known = True
    for t in tasks:
        left = t["total"] - t["done"]
        if left <= 0:
            t["eta_s"] = 0
            continue
        if t["state"] == "failed":       # a stalled task has no meaningful ETA
            t["eta_s"] = None
            eta_known = False
            continue
        rate = t["rate_now"] if t["state"] == "running" and t["rate_now"] else (t.get("rate_avg") or fallback)
        t["eta_s"] = left / rate * 60 if rate else None
        if t["eta_s"] is None:
            eta_known = False
        else:
            eta += t["eta_s"]
    state = "completed" if finished_all or all(t["state"] == "completed" for t in tasks) else (
        "running" if alive else ("failed" if any(t["state"] == "failed" for t in tasks) else "queued"))
    return {"state": state, "tasks": tasks, "eta_s": eta if eta_known else None, "alive": alive,
            "rate_fallback": fallback, "log": gen_log[-12:]}


def rate_series(series):
    """items/min between consecutive samples (>= 30 s apart), from the sampled cumulative counts."""
    out, last = [], None
    for t, c in series:
        if last is None:
            last = (t, c)
            continue
        if t - last[0] >= 30:
            out.append([t, max(0.0, (c - last[1]) / ((t - last[0]) / 60))])
            last = (t, c)
    return out[-120:]


# ---------------------------------------------------------------- training queue
def run_dir(exp, seed):
    hits = [d for d in glob.glob(P("results", "raw", f"{exp}_seed{seed}_*"))
            if os.path.basename(d).rsplit("_seed", 1)[0] == exp]
    return max(hits, key=lambda d: mtime(d) or 0) if hits else None


def run_metrics(exp, seed):
    d = run_dir(exp, seed)
    return read_json(os.path.join(d, "metrics.json")) if d else None


def queue(now):
    qlog = read_lines(P("logs", "disc_resid", "queue.log"))
    started, failed, finished = {}, set(), set()
    for ln in qlog:
        m = re.match(r"\[(\S+?)\] (start|done|FAILED)\s+(\S+)", ln)
        if not m:
            continue
        ts, ev, job = m.groups()
        if ev == "start":
            started[job] = iso_to_epoch(ts)
            failed.discard(job)
        elif ev == "FAILED":
            failed.add(job)
        else:
            finished.add(job)
    q_alive = process_alive("dr_queue.sh")
    waiting_gen = process_alive("dr_after_gen.sh") and not qlog
    queue_done = any("QUEUE_DONE" in ln for ln in qlog)
    latest = max(started, key=lambda j: started[j] or 0) if started else None
    jobs = []
    for seed in SEEDS:
        for key, exp, label, _, resid in ARMS:
            job = f"{exp}_s{seed}"
            row = {"job": job, "arm": key, "seed": seed, "resid": resid, "state": "waiting",
                   "log": f"logs/disc_resid/train_{job}.log"}
            done_file = P("results", "disc_resid", "done", job)
            if os.path.exists(done_file):
                row["state"] = "done"
                try:
                    row["seconds"] = int(open(done_file).read().strip() or 0)
                except (OSError, ValueError):
                    row["seconds"] = None
                m = run_metrics(exp, seed)
                if m and isinstance(m.get("test"), dict):
                    row["test"] = {k: m["test"].get(k) for k in METRICS}
                    row["best_epoch"] = m.get("best_epoch")
            elif job in failed:
                row["state"] = "failed"
                row["log_tail"] = tail_lines(P(row["log"]), 6)
                row["error"] = safe(lambda: error_snippet(P(row["log"])))
            elif job in started and job == latest and q_alive and not queue_done:
                row["state"] = "running"
                row["started_at"] = started[job]
                row["elapsed_s"] = now - started[job] if started[job] else None
                d = run_dir(exp, seed)
                hist = []
                for x in read_lines(os.path.join(d, "training_log.jsonl")) if d else []:
                    try:
                        hist.append(json.loads(x))
                    except ValueError:
                        pass
                row["epochs_done"] = len(hist)
                row["max_epochs"] = MAX_EPOCHS
                if hist:
                    last = hist[-1]
                    row["last"] = {k: last.get(k) for k in ("epoch", "train_loss", "val_loss", "val_auroc", "val_auprc", "val_macro_f1")}
            elif job in started:
                row["state"] = "stopped"          # started earlier but never finished (queue restarted / killed)
            jobs.append(row)
    dur = {}
    for j in jobs:
        if j["state"] == "done" and j.get("seconds"):
            dur.setdefault("resid" if j["resid"] else "plain", []).append(j["seconds"])
    est = {"plain": (sum(dur["plain"]) / len(dur["plain"])) if dur.get("plain") else 1.1 * 3600,
           "plain_measured": bool(dur.get("plain"))}
    est["resid"] = (sum(dur["resid"]) / len(dur["resid"])) if dur.get("resid") else 2 * est["plain"]
    est["resid_measured"] = bool(dur.get("resid"))
    remaining = 0.0
    for j in jobs:
        per = est["resid" if j["resid"] else "plain"]
        if j["state"] in ("waiting", "stopped", "failed"):
            remaining += per
        elif j["state"] == "running":
            remaining += max(0.0, per - (j.get("elapsed_s") or 0))
    arms = []
    for key, exp, label, what, resid in ARMS:
        mine = [j for j in jobs if j["arm"] == key]
        secs = [j["seconds"] for j in mine if j["state"] == "done" and j.get("seconds")]
        arms.append({"arm": key, "label": label, "what": what, "resid": resid,
                     "done": sum(j["state"] == "done" for j in mine), "total": len(mine),
                     "failed": sum(j["state"] == "failed" for j in mine),
                     "mean_runtime_s": sum(secs) / len(secs) if secs else None})
    state = "completed" if queue_done else ("running" if q_alive else ("waiting for generation" if waiting_gen else
                                                                       ("stopped" if started else "not started")))
    events = qlog[-10:]
    return {"state": state, "jobs": jobs, "arms": arms, "eta_s": remaining if not queue_done else 0, "est": est,
            "log": events, "failed": [{"job": j["job"], "log": j["log"], "tail": j.get("log_tail", []), "error": j.get("error")}
                                      for j in jobs if j["state"] == "failed"]}


# ---------------------------------------------------------------- results
def results():
    base = {s: run_metrics("dr_R0", s) for s in SEEDS}
    orig = {s: run_metrics(ORIGINAL, s) for s in SEEDS}
    arms = []
    for key, exp, label, what, _ in ARMS:
        per = {}
        for s in SEEDS:
            m = run_metrics(exp, s)
            t = (m or {}).get("test")
            if not isinstance(t, dict) or any(not isinstance(t.get(k), (int, float)) for k in METRICS):
                continue
            ref = (base if key != "R0" else orig).get(s)
            rt = (ref or {}).get("test") if ref else None
            per[str(s)] = {"metrics": {k: t[k] for k in METRICS}, "best_epoch": m.get("best_epoch"),
                           "val": {k: (m.get("val") or {}).get(k) for k in METRICS},
                           "delta": {k: t[k] - rt[k] for k in METRICS} if isinstance(rt, dict) and all(
                               isinstance(rt.get(k), (int, float)) for k in METRICS) else None}
        arms.append({"arm": key, "label": label, "what": what, "ref": "original Sept. run" if key == "R0" else "R0",
                     "per_seed": per})
    orig_seeds = {str(s): {k: v["test"][k] for k in METRICS} for s, v in orig.items() if v and "test" in v}
    return {"arms": arms, "reference": REFERENCE, "original_per_seed": orig_seeds, "noise_band": 0.5,
            "winner": pick_winner(arms)}


def results_csv():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["arm", "experiment", "seed", "macro_f1", "auroc", "auprc", "best_epoch", "reference",
                "delta_macro_f1", "delta_auroc", "delta_auprc"])
    r = results()
    exp_of = {a[0]: a[1] for a in ARMS}
    for a in r["arms"]:
        for s, v in sorted(a["per_seed"].items()):
            d = v["delta"] or {}
            w.writerow([a["arm"], exp_of[a["arm"]], s] + [f"{v['metrics'][k]:.4f}" for k in METRICS] +
                       [v.get("best_epoch"), a["ref"]] + [f"{d[k]:.4f}" if k in d else "" for k in METRICS])
    for s, m in sorted(r["original_per_seed"].items()):
        w.writerow(["original", ORIGINAL, s] + [f"{m[k]:.4f}" for k in METRICS] + ["", "", "", "", ""])
    return buf.getvalue()


# ---------------------------------------------------------------- FLAG-architecture study (D & R per trimmed subgraph)
FA_ARMS = [  # key, experiment, label, what changes, has the +FLAG* phase
    ("F0", "fa_F0", "F0 baseline", "main DGP configuration, re-run on this instance's stack", False),
    ("F1", "fa_F1", "F1 FLAG-arch D", "D written per trimmed subgraph; neighbours' lines feed the metapath summaries; "
                                      "target's line as Key signals", False),
]   # F2 (R text + FLAG's +FLAG* loss / fine-tuning phase) is out of scope for this experiment (user, 2026-10-23)
FA_KINDS = [("discriminative", "Discriminative (D)"), ("residual", "Residual (R) · text only")]   # R: scripts/fa_r_after_queue.sh
FA_CAP = 550
FA_PHASE_EPOCHS = 30


def fa_stores():
    """kind -> (cache dir, metadata) for the FLAG subgraph-text caches."""
    out = {}
    for meta in glob.glob(P("datasets", "cache", "fa_subgraph_text", "*", "metadata.json")):
        m = read_json(meta)
        if m and m.get("kind") in dict(FA_KINDS):
            out[m["kind"]] = (os.path.dirname(meta), m)
    return out


def fa_scan(path):
    """Subgraph answers: done, parsed ok, nodes, covered nodes, cap hits, overflows, token means, last ok lines."""
    out = {"done": 0, "ok": 0, "nodes": 0, "nodes_ok": 0, "hit_cap": 0, "overflow": 0, "ptok": [], "gtok": [], "samples": []}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue                      # half-written last line
                n = len(row.get("nodes") or [])
                out["done"] += 1
                out["nodes"] += n
                out["hit_cap"] += bool(row.get("hit_cap"))
                out["overflow"] += bool(row.get("overflow"))
                if isinstance(row.get("prompt_tokens"), (int, float)):
                    out["ptok"].append(row["prompt_tokens"])
                if isinstance(row.get("generated_tokens"), (int, float)):
                    out["gtok"].append(row["generated_tokens"])
                if row.get("ok"):
                    out["ok"] += 1
                    out["nodes_ok"] += n
                    first = next((x for x in str(row.get("text", "")).split("\n") if x.strip()), "")
                    out["samples"].append(first)
                    if len(out["samples"]) > 4:
                        out["samples"].pop(0)
    except OSError:
        pass
    return out


def fa_generation(now):
    gen_log = read_lines(P("logs", "flag_arch", "gen_all.log"))
    finished = any("GEN_DONE" in ln for ln in gen_log)
    gen_failed = any("GEN_FAILED" in ln for ln in gen_log)
    alive = process_alive("fa_queue.sh") and not finished
    r_log = read_lines(P("logs", "flag_arch", "r_text.log"))       # R text: its own job, after the F0/F1 queue
    r_end = any("R_TEXT_DONE" in ln or "R_TEXT_FAILED" in ln for ln in r_log)
    r_alive = process_alive("fa_r_after_queue.sh") and any("generate fa_R" in ln for ln in r_log) and not r_end
    stores = fa_stores()
    tasks = []
    for kind, label in FA_KINDS:
        d, meta = stores.get(kind, (None, None))
        s = fa_scan(os.path.join(d, "answers.jsonl")) if d else fa_scan("")
        f = os.path.join(d, "answers.jsonl") if d else None
        last = mtime(f) if f else None
        t = {"key": "fa:" + kind, "kind": kind, "label": label, "total": N_TARGETS, "done": s["done"], "ok": s["ok"],
             "success_rate": s["ok"] / s["done"] if s["done"] else None,
             "node_coverage": s["nodes_ok"] / s["nodes"] if s["nodes"] else None,
             "hit_cap": s["hit_cap"], "overflow": s["overflow"],
             "prompt_tokens_mean": round(sum(s["ptok"]) / len(s["ptok"])) if s["ptok"] else None,
             "generated_tokens_mean": round(sum(s["gtok"]) / len(s["gtok"])) if s["gtok"] else None,
             "samples": [x[:220] for x in s["samples"]], "started_at": iso_to_epoch((meta or {}).get("created", "")),
             "last_write_age": now - last if last else None}
        alive_k, failed_k = (r_alive, any("R_TEXT_FAILED" in ln for ln in r_log)) if kind == "residual" else (alive, gen_failed)
        t["state"] = ("completed" if s["done"] >= N_TARGETS else "running" if alive_k and s["done"] and t["last_write_age"] is not None
                      and t["last_write_age"] < 1800 else "failed" if failed_k else "partial" if s["done"] else "queued")
        series = HIST["gen"].setdefault(t["key"], [])
        push(series, [round(now), s["done"]], now)
        t["rate_series"] = rate_series(series)
        t["rate_now"] = t["rate_series"][-1][1] if t["rate_series"] else None
        rate = t["rate_now"] or (s["done"] / ((last - t["started_at"]) / 60) if last and t["started_at"] and last > t["started_at"] else None)
        t["eta_s"] = (t["total"] - s["done"]) / rate * 60 if rate and s["done"] < t["total"] else (0 if s["done"] >= t["total"] else None)
        # metapath summaries of this kind's lines (written after all subgraphs of the kind are done)
        mp = None
        if meta:
            for m_path in glob.glob(P("datasets", "cache", "metapath_summaries", "*", "metadata.json")):
                mm = read_json(m_path)
                if mm and mm.get("node_text") == f"fa_subgraph_text/{meta.get('cache_key')}":
                    mp = {"done": len(read_lines(os.path.join(os.path.dirname(m_path), "summaries.jsonl"))),
                          "upper_bound": N_TARGETS * 12}
        t["metapath_summaries"] = mp
        tasks.append(t)
    state = "completed" if finished else ("failed" if gen_failed else "running" if alive else
                                          ("partial" if any(t["done"] for t in tasks) else "not started"))
    err = None
    if gen_failed:
        m = re.search(r"GEN_FAILED (\S+)", next(ln for ln in reversed(gen_log) if "GEN_FAILED" in ln))
        err = safe(lambda: error_snippet(P("logs", "flag_arch", f"gen_{m.group(1)}.log"))) if m else None
    return {"state": state, "tasks": tasks, "log": gen_log[-12:], "error": err}


def fa_history(exp, seed):
    d = run_dir(exp, seed)
    rows = []
    for x in read_lines(os.path.join(d, "training_log.jsonl")) if d else []:
        try:
            rows.append(json.loads(x))
        except ValueError:
            pass
    return rows


def fa_queue(now):
    qlog = read_lines(P("logs", "flag_arch", "queue.log"))
    started, failed = {}, set()
    for ln in qlog:
        m = re.match(r"\[(\S+?)\] (start|done|FAILED)\s+(\S+)", ln)
        if m:
            ts, ev, job = m.groups()
            if ev == "start":
                started[job] = iso_to_epoch(ts)
                failed.discard(job)
            elif ev == "FAILED":
                failed.add(job)
    alive = process_alive("fa_queue.sh")
    done_all = any("QUEUE_DONE" in ln for ln in qlog)
    latest = max(started, key=lambda j: started[j] or 0) if started else None
    jobs = []
    for seed in SEEDS:
        for key, exp, label, _, phase in FA_ARMS:
            job = f"{exp}_s{seed}"
            row = {"job": job, "arm": key, "seed": seed, "phase_arm": phase, "state": "waiting",
                   "log": f"logs/flag_arch/train_{job}.log"}
            done_file = P("results", "flag_arch", "done", job)
            if os.path.exists(done_file):
                row["state"] = "done"
                try:
                    row["seconds"] = int(open(done_file).read().strip() or 0)
                except (OSError, ValueError):
                    row["seconds"] = None
                m = run_metrics(exp, seed)
                if m and isinstance(m.get("test"), dict):
                    row["test"] = {k: m["test"].get(k) for k in METRICS}
                man = read_json(os.path.join(run_dir(exp, seed) or "", "run_manifest.json")) or {}
                ft = man.get("flag_finetune")
                if ft:
                    row["finetune"] = {k: ft.get(k) for k in ("applied", "best_epoch", "usable_train_targets")}
            elif job in failed:
                row["state"] = "failed"
                row["log_tail"] = tail_lines(P(row["log"]), 6)
                row["error"] = safe(lambda: error_snippet(P(row["log"])))
            elif job in started and job == latest and alive and not done_all:
                row["state"] = "running"
                row["elapsed_s"] = now - started[job] if started[job] else None
                hist = fa_history(exp, seed)
                normal = [h for h in hist if h.get("phase") != "flag_finetune"]
                ft = [h for h in hist if h.get("phase") == "flag_finetune"]
                row["phase"] = "+FLAG* phase" if ft else "training"
                row["epochs_done"] = len(ft) if ft else len(normal)
                row["max_epochs"] = FA_PHASE_EPOCHS if ft else MAX_EPOCHS
                last = (ft or normal or [None])[-1]
                if last:
                    row["last"] = {k: last.get(k) for k in ("epoch", "train_loss", "val_loss", "val_auroc", "val_macro_f1",
                                                            "train_ce", "train_kl", "train_orth")}
            elif job in started:
                row["state"] = "stopped"
            jobs.append(row)
    state = "completed" if done_all else ("running" if alive else ("stopped" if started else "not started"))
    return {"state": state, "jobs": jobs, "log": qlog[-10:],
            "arms": [{"arm": k, "label": l, "what": w, "phase": p} for k, _, l, w, p in FA_ARMS]}


def fa_results():
    base = {s: run_metrics("fa_F0", s) for s in SEEDS}
    arms = []
    for key, exp, label, what, _ in FA_ARMS:
        per = {}
        for s in SEEDS:
            mm = run_metrics(exp, s) or {}
            t = mm.get("test")
            if not isinstance(t, dict) or any(not isinstance(t.get(k), (int, float)) for k in METRICS):
                continue
            rt = (base.get(s) or {}).get("test")
            per[str(s)] = {"metrics": {k: t[k] for k in METRICS}, "val": {k: (mm.get("val") or {}).get(k) for k in METRICS},
                           "delta": {k: t[k] - rt[k] for k in METRICS} if key != "F0" and isinstance(rt, dict) else None}
        arms.append({"arm": key, "label": label, "what": what, "per_seed": per})
    return {"arms": arms, "reference": REFERENCE, "noise_band": 0.5, "winner": pick_winner(arms)}


def fa_state(now):
    pilot = read_json(P("results", "flag_arch", "pilot", "summary.json"))
    return {"pilot": pilot, "generation": safe(lambda: fa_generation(now)), "queue": safe(lambda: fa_queue(now)),
            "results": safe(fa_results), "pilot_log": tail_lines(P("logs", "flag_arch", "pilot.log"), 8)}


# ---------------------------------------------------------------- diagnostics + insights
_insights_lock = threading.Lock()


def diagnostics(gen):
    diag = read_json(P("results", "disc_resid", "text_diagnostics.json"))
    ins_path = os.path.join(INSIGHTS_DIR, "text_insights.json")
    ins = read_json(ins_path)
    complete = gen and all(t["state"] == "completed" for t in gen["tasks"])
    if complete and ins is None and not ARGS.no_insights and _insights_lock.acquire(blocking=False):
        def work():
            try:
                subprocess.run([sys.executable, os.path.join(HERE, "text_insights.py"), "--root", R, "--out", ins_path],
                               capture_output=True, timeout=900, cwd=R)
            finally:
                _insights_lock.release()
        threading.Thread(target=work, daemon=True).start()
    leak = None
    nv = ((diag or {}).get("texts") or {}).get("disc_noverdict")
    probe = (diag or {}).get("probe") or {}
    if nv and diag.get("n_targets"):
        rate = nv["mentions_class_word"] / diag["n_targets"] * 100
        gap = (probe.get("disc_noverdict", {}).get("auroc") or 0) - (probe.get("raw", {}).get("auroc") or 0)
        leak = {"class_word_pct": round(rate, 2), "probe_gap_vs_raw": round(gap, 2),
                "alert": rate > 2.0, "threshold": "class words in > 2% of lines (the probe gap is shown for information: "
                                                  "a discriminative text is meant to be informative)"}
    return {"diagnostics": diag, "insights": ins, "insights_pending": bool(complete and ins is None), "leakage": leak}


# ---------------------------------------------------------------- GPU
def gpus(now):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        out = ""
    rows = []
    for ln in out.strip().splitlines():
        try:
            name, util, used, total, temp = [x.strip() for x in ln.split(",")]
            rows.append({"name": name, "util": float(util), "mem_used": float(used), "mem_total": float(total), "temp": float(temp)})
        except ValueError:
            continue
    if ARGS.root:                       # mock tree: its GPU file replaces the real reading (or none at all)
        rows = read_json(P("mock_gpu.json")) or []
    if rows:
        g = rows[0]
        push(HIST["gpu"], [round(now), g["util"], g["mem_used"], g["mem_total"], g["temp"]], now)
    for g in rows:
        g["warn_temp"] = g["temp"] > 80
        g["warn_mem"] = g["mem_total"] > 0 and g["mem_used"] / g["mem_total"] > 0.9
    return {"gpus": rows, "history": HIST["gpu"][-360:]}


# ---------------------------------------------------------------- events, overview
def events():
    out = []
    for name in ("gen_all.log", "queue.log"):
        for ln in read_lines(P("logs", "disc_resid", name)):
            m = re.match(r"\[(\d{4}-\d\d-\d\dT[\d:]+)\]\s+(.*)", ln)
            if m:
                out.append({"t": m.group(1), "msg": m.group(2)})
    out.sort(key=lambda e: e["t"])
    return out


def overview(gen, q, ev, now):
    starts = [iso_to_epoch(e["t"]) for e in ev[:1]]
    gen_starts = [t["started_at"] for t in (gen or {}).get("tasks", []) if t.get("started_at")]
    t0 = min([x for x in starts + gen_starts if x] or [None]) if (starts or gen_starts) else None
    # overall progress weighted by (estimated) time: generation items at the measured average rate, training by run time
    gen_total = gen_done = 0.0
    rate = (gen or {}).get("rate_fallback") or 400.0
    for t in (gen or {}).get("tasks", []):
        r = t.get("rate_avg") or rate
        gen_total += t["total"] / r
        gen_done += min(t["done"], t["total"]) / r
    gen_total *= 60
    gen_done *= 60
    tr_total = tr_done = 0.0
    if q:
        for j in q["jobs"]:
            per = q["est"]["resid" if j["resid"] else "plain"]
            tr_total += per
            if j["state"] == "done":
                tr_done += per
            elif j["state"] == "running":
                tr_done += min(per * 0.95, j.get("elapsed_s") or 0)
    total = gen_total + tr_total
    return {"started_at": t0, "elapsed_s": now - t0 if t0 else None,
            "progress": (gen_done + tr_done) / total if total else None,
            "gen_progress": gen_done / gen_total if gen_total else None,
            "train_progress": tr_done / tr_total if tr_total else None,
            "eta_gen_s": (gen or {}).get("eta_s"), "eta_train_s": (q or {}).get("eta_s"),
            "eta_total_s": ((gen or {}).get("eta_s") or 0) + ((q or {}).get("eta_s") or 0) if q else None}


def files():
    dev = P("results", "disc_resid", "deviation.md")
    diag = P("results", "disc_resid", "text_diagnostics.md")
    return {"deviation_md": "files/deviation.md" if os.path.exists(dev) else None,
            "text_diagnostics_md": "files/text_diagnostics.md" if os.path.exists(diag) else None}


def state():
    with LOCK:
        now = time.time()
        if now - _cache["t"] < 5 and _cache["v"] is not None:
            return _cache["v"]
        gen = safe(lambda: generation(now))
        q = safe(lambda: queue(now))
        ev = safe(events, [])
        v = {"server_time": now, "now": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
             "mock": bool(ARGS.root), "overview": safe(lambda: overview(gen, q, ev, now)),
             "events": ev[-15:][::-1], "generation": gen, "queue": q, "results": safe(results),
             "diag": safe(lambda: diagnostics(gen)), "gpu": safe(lambda: gpus(now)), "files": safe(files),
             "fa": safe(lambda: fa_state(now)),
             "provenance": {k: safe(lambda k=k: provenance_header(k)) for k in STUDIES}}
        save_history()
        _cache.update(t=now, v=v)
        return v


# ---------------------------------------------------------------- traceability: provenance, manifest, samples, errors
# Everything here is read from files written at launch / run time (run_manifest.json, metrics.json, batch records,
# cache metadata); nothing is reconstructed at display time. A value that was never recorded is the string "unknown".
UNKNOWN = "unknown"
STUDIES = {  # key -> title, arms, results/logs folder, launcher, baseline arm
    "dr": {"title": "Disc/resid study · D & R per review", "arms": ARMS, "folder": "disc_resid",
           "launcher": "scripts/dr_queue.sh", "baseline": "R0"},
    "fa": {"title": "FLAG-architecture study · D & R per trimmed subgraph", "arms": FA_ARMS, "folder": "flag_arch",
           "launcher": "scripts/fa_queue.sh", "baseline": "F0"},
}
WINNER = {"metric": "auroc", "split": "val",
          "rule": "highest mean validation AUROC over the arm's finished seeds (DGP's model-selection metric; "
                  "test metrics are never used to pick the winner)"}
LABEL_NAMES = {1: "unhelpful (fraud)", 0: "helpful (benign)"}    # configs/datasets/amazonvideo.yaml: label.threshold
SAMPLE_TEXT_CAP = 1500
ERR_RE = re.compile(r"Traceback|Error\b|Exception\b|out of memory|OutOfMemory|CUDA error|Killed|JSONDecodeError|"
                    r"could not parse|parse error", re.I)


def run_provenance(exp, seed):
    d = run_dir(exp, seed)
    m = read_json(os.path.join(d, "run_manifest.json")) if d else None
    if not isinstance(m, dict):
        return None
    g = str(m.get("git_commit") or "")
    sha = g[:-len("-dirty")] if g.endswith("-dirty") else g
    ok = bool(re.fullmatch(r"[0-9a-f]{40}", sha))
    cfg = m.get("config") if isinstance(m.get("config"), dict) else {}
    pm = m.get("prompt_metadata") if isinstance(m.get("prompt_metadata"), dict) else {}
    return {"run_id": m.get("run_id") or os.path.basename(d), "manifest": os.path.relpath(os.path.join(d, "run_manifest.json"), R),
            "written": m.get("written") or UNKNOWN, "git_commit": sha if ok else UNKNOWN,
            "git_dirty": g.endswith("-dirty") if ok else UNKNOWN, "git_branch": m.get("git_branch") or UNKNOWN,
            "prompt_version": pm.get("prompt_version") or UNKNOWN, "templates": cfg.get("templates") or UNKNOWN,
            "runtime_seconds": m.get("runtime_seconds"), "config": cfg}


def flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict) and v:
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def batch_records(folder):
    recs = []
    for path in sorted(glob.glob(P("results", folder, "batch_*.json"))):
        b = read_json(path)
        if isinstance(b, dict):
            recs.append({**b, "file": os.path.relpath(path, R)})
    return recs


def mean_std(vals):
    v = [x for x in vals if isinstance(x, (int, float))]
    if not v:
        return None
    m = sum(v) / len(v)
    return {"n": len(v), "mean": m, "std": (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5, "min": min(v), "max": max(v)}


def pick_winner(arms):
    """arms: [{"arm", "per_seed": {seed: {"val": {...}}}}] -> {"arm", "mean", "n", rule...} or None."""
    best = None
    for a in arms:
        st = mean_std([(v.get(WINNER["split"]) or {}).get(WINNER["metric"]) for v in (a.get("per_seed") or {}).values()])
        if st and (best is None or st["mean"] > best["mean"]):
            best = {"arm": a["arm"], "mean": st["mean"], "n": st["n"]}
    return {**best, **WINNER} if best else None


def study_manifest(key):
    st = STUDIES[key]
    batches = batch_records(st["folder"])
    arms, written, commits = [], [], {}
    for akey, exp, label, what, _ in st["arms"]:
        per, prov0 = {}, None
        for s in SEEDS:
            pv = run_provenance(exp, s)
            m = run_metrics(exp, s)
            if pv and pv["written"] != UNKNOWN:
                written.append(pv["written"])
            if pv:
                c = commits.setdefault(f'{pv["git_commit"]}|{pv["git_dirty"]}',
                                       {"commit": pv["git_commit"], "dirty": pv["git_dirty"], "runs": []})
                c["runs"].append(pv["run_id"])
            if not isinstance(m, dict) or not isinstance(m.get("test"), dict):
                continue
            per[str(s)] = {"val": {k: (m.get("val") or {}).get(k) for k in METRICS},
                           "test": {k: m["test"].get(k) for k in METRICS}, "best_epoch": m.get("best_epoch"),
                           "run_id": pv["run_id"] if pv else UNKNOWN, "written": pv["written"] if pv else UNKNOWN,
                           "git_commit": pv["git_commit"] if pv else UNKNOWN, "git_dirty": pv["git_dirty"] if pv else UNKNOWN}
            prov0 = prov0 or pv
        prov0 = prov0 or next((p for p in (run_provenance(exp, s) for s in SEEDS) if p), None)
        arms.append({
            "arm": akey, "experiment": exp, "label": label, "description": what,
            "seeds_finished": sorted(int(s) for s in per),
            "hyperparameters": flatten(prov0["config"]) if prov0 and prov0["config"] else UNKNOWN,
            "hyperparameters_source": prov0["manifest"] if prov0 else UNKNOWN,
            "prompt_version": prov0["prompt_version"] if prov0 else UNKNOWN,
            "templates": prov0["templates"] if prov0 else UNKNOWN,
            "per_seed": per,
            "summary": {f"{sp}_{k}": mean_std([v[sp].get(k) for v in per.values()]) for sp in ("val", "test") for k in METRICS}})
    stamps = sorted(written) + sorted(b.get("recorded_at") for b in batches if b.get("recorded_at"))
    return {
        "schema": "dgp-experiment-manifest-v1",
        "experiment": st["title"], "study": key, "dataset": "amazonvideo",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "date_range": {"first": min(stamps) if stamps else UNKNOWN, "last": max(stamps) if stamps else UNKNOWN,
                       "timezone": "run manifests: container local time, no zone recorded; batch records: UTC"},
        "git": {"commits": list(commits.values()) or UNKNOWN,
                "branches": sorted({b.get("git_branch") for b in batches if b.get("git_branch")}) or UNKNOWN,
                "note": "commit and dirty flag recorded by each run at launch (run_manifest.json); "
                        "branch only in batch records"},
        "run_hypothesis": [b["run_hypothesis"] for b in batches if b.get("run_hypothesis")] or UNKNOWN,
        "cli_commands": [b["command"] for b in batches if b.get("command")] or UNKNOWN,
        "launcher_script": st["launcher"],
        "batches": batches or UNKNOWN,
        "text_generation": text_generation_meta(key),
        "metrics": {"split_reported": "test (val shown for selection)", "std": "population std (ddof=0)", "scale": "percent"},
        "winner": pick_winner(arms),
        "arms": arms,
    }


def text_generation_meta(key):
    out = []
    if key == "fa":
        for kind, (d, meta) in fa_stores().items():
            if meta:
                out.append({"kind": kind, "cache": os.path.relpath(d, R), **{k: meta.get(k, UNKNOWN) for k in (
                    "prompt_set", "prompt_hashes", "summarizer", "max_new_tokens", "truncate_chars", "chat_template",
                    "unit", "git_commit", "created")}, "prompt_version": meta.get("prompt_version", UNKNOWN)})
    else:
        for template, tkey, label in TEXTS:
            d, meta = cache_store(template)
            if meta:
                out.append({"kind": tkey, "label": label, "cache": os.path.relpath(d, R), "template": template,
                            **{k: meta.get(k, UNKNOWN) for k in ("template_hash", "summarizer", "max_new_tokens", "budget",
                                                                 "truncate_chars", "git_commit", "created")},
                            "prompt_version": meta.get("prompt_version", UNKNOWN)})
    return out or UNKNOWN


def provenance_header(key):
    """Small per-study summary for the page header (latest batch record + commits the runs recorded)."""
    st = STUDIES[key]
    b = (batch_records(st["folder"]) or [None])[-1] or {}
    runs = [p for p in (run_provenance(exp, s) for _, exp, *_ in st["arms"] for s in SEEDS) if p]
    stamps = sorted(p["written"] for p in runs if p["written"] != UNKNOWN)
    shas = sorted({(p["git_commit"], p["git_dirty"]) for p in runs}, key=str)
    return {"study": key, "title": st["title"],
            "git_commit": b.get("git_commit") or (shas[-1][0] if len(shas) == 1 else UNKNOWN),
            "git_branch": b.get("git_branch") or UNKNOWN,
            "git_dirty": b["git_dirty"] if "git_dirty" in b else (shas[-1][1] if len(shas) == 1 else UNKNOWN),
            "git_dirty_files": b.get("git_dirty_files", UNKNOWN),
            "run_commits": [{"commit": c, "dirty": d} for c, d in shas],
            "run_hypothesis": b.get("run_hypothesis") or UNKNOWN, "batch_recorded_at": b.get("recorded_at") or UNKNOWN,
            "batch_file": b.get("file") or UNKNOWN,
            "first_run": stamps[0] if stamps else UNKNOWN, "last_run": stamps[-1] if stamps else UNKNOWN}


def error_snippet(path, max_lines=14, max_chars=1600):
    """Last error block of a log: from the last Traceback (if in the tail) to the end, else the last matching line +-2."""
    lines = tail_lines(path, 200)
    hits = [i for i, ln in enumerate(lines) if ERR_RE.search(ln)]
    if not hits:
        return None
    tb = [i for i in hits if "Traceback" in lines[i]]
    block = lines[tb[-1]:] if tb else lines[max(0, hits[-1] - 2): hits[-1] + 3]
    block = block[-max_lines:]
    text = "\n".join(block)[-max_chars:]
    low = text.lower()
    kind = ("OOM" if "out of memory" in low or "outofmemory" in low else
            "parse error" if "jsondecodeerror" in low or "parse" in low else
            "killed" if "killed" in low else "error")
    last = next((ln for ln in reversed(block) if ERR_RE.search(ln) and "Traceback" not in ln), block[-1])
    return {"kind": kind, "summary": last.strip()[:240], "text": text, "log": os.path.relpath(path, R)}


_split_cache = {"m": None, "v": None}


def node_split():
    path = P("datasets", "processed", "amazonvideo", "split.json")
    mt = mtime(path)
    if _split_cache["m"] != mt:
        sp = read_json(path) or {}
        _split_cache.update(m=mt, v={int(n): k for k in ("train", "val", "test") for n in (sp.get(k) or [])})
    return _split_cache["v"] or {}


def node_arrays():
    try:
        import numpy as np
        base = P("datasets", "processed", "amazonvideo")
        labels = np.load(os.path.join(base, "labels.npy"), mmap_mode="r")
        g = read_json(os.path.join(base, "graph.json")) or {}
        npath = os.path.join(base, "numeric.npy")
        num = np.load(npath, mmap_mode="r") if os.path.exists(npath) else None
        return labels, num, g.get("numeric_names") or []
    except Exception:
        return None, None, []


def token_usage_rows(exp, nodes):
    path = P("results", "token_usage", f"token_usage_{exp}.csv")
    want, out = {str(n) for n in nodes}, {}
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("node") in want:
                    out[int(row["node"])] = row
                    if len(out) == len(want):
                        break
    except OSError:
        pass
    return out, os.path.relpath(path, R)


def prompt_samples(study, arm, n=5, rng=None):
    import random
    rng = rng or random.Random()
    st = STUDIES.get(study)
    exp = next((e for k, e, *_ in (st or {}).get("arms", []) if k == arm), None)
    if not exp:
        return {"error": "unknown study or arm"}
    files = [p for p in glob.glob(P("datasets", "cache", "prompts", f"{exp}_*.jsonl"))
             if re.fullmatch(re.escape(exp) + r"_[0-9a-f]+\.jsonl", os.path.basename(p))]
    if not files:
        return {"study": study, "arm": arm, "experiment": exp, "source": UNKNOWN, "samples": [],
                "note": "no prompt file for this arm yet (written by scripts/generate_summaries.py)"}
    path = max(files, key=lambda p: mtime(p) or 0)
    size, picked = os.path.getsize(path), {}
    with open(path, "rb") as fh:                    # random byte offsets: never reads the whole file
        for _ in range(n * 6):
            if len(picked) >= n or size == 0:
                break
            fh.seek(rng.randrange(size))
            fh.readline()                            # skip the partial line we landed in
            ln = fh.readline()
            if not ln:                               # landed in the last line: wrap to the first
                fh.seek(0)
                ln = fh.readline()
            try:
                rec = json.loads(ln)
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(rec, dict) and isinstance(rec.get("node"), int) and rec["node"] not in picked:
                picked[rec["node"]] = rec
    split = node_split()
    labels, num, num_names = node_arrays()
    tu, tu_path = token_usage_rows(exp, picked)
    out = []
    for node, rec in picked.items():
        text = str(rec.get("prompt", ""))
        sp = split.get(node, UNKNOWN)
        lab = UNKNOWN
        if sp in ("train", "val") and labels is not None and node < len(labels):
            lab = {"value": int(labels[node]), "name": LABEL_NAMES.get(int(labels[node]), UNKNOWN)}
        elif sp == "test":
            lab = "hidden (test split)"
        feats = {}
        if num is not None and node < len(num):
            feats.update({name: float(num[node][i]) for i, name in enumerate(num_names)})
        row = tu.get(node) or {}
        for k in ("num_neighbors_full", "num_neighbors_trimmed", "num_metapaths", "final_prompt_tokens", "target_truncated"):
            feats[k] = row.get(k, UNKNOWN)
        out.append({"node": node, "split": sp, "label": lab, "features": feats, "prompt_chars": len(text),
                    "truncated": len(text) > SAMPLE_TEXT_CAP, "prompt": text[:SAMPLE_TEXT_CAP]})
    return {"study": study, "arm": arm, "experiment": exp, "source": os.path.relpath(path, R), "features_source": tu_path,
            "labels": "train/val labels from datasets/processed/amazonvideo/labels.npy; test labels never shown",
            "sampling": "random byte offsets (longer records slightly more likely)", "samples": out}


# ---------------------------------------------------------------- HTTP
SERVE = {"deviation.md": ("results", "disc_resid", "deviation.md"),
         "text_diagnostics.md": ("results", "disc_resid", "text_diagnostics.md")}


class H(BaseHTTPRequestHandler):
    def send(self, code, body, ct, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        query = self.path.split("?", 1)[1] if "?" in self.path else ""
        if path in ("/", "/index.html"):
            self.send(200, open(os.path.join(HERE, "index.html"), "rb").read(), "text/html; charset=utf-8")
        elif path == "/api/state":
            self.send(200, json.dumps(state(), default=lambda o: None).encode(), "application/json")
        elif path == "/api/results.csv":
            self.send(200, results_csv().encode(), "text/csv; charset=utf-8",
                      {"Content-Disposition": "attachment; filename=disc_resid_results.csv"})
        elif path == "/api/manifest":
            q = dict(kv.split("=", 1) for kv in query.split("&") if "=" in kv)
            study = q.get("study", "fa")
            if study not in STUDIES:
                self.send(400, b"bad study", "text/plain")
            else:
                extra = ({"Content-Disposition": f"attachment; filename=experiment_manifest_{study}.json"}
                         if q.get("download") == "1" else None)
                self.send(200, json.dumps(study_manifest(study), indent=2, default=str).encode(), "application/json", extra)
        elif path == "/api/samples":
            q = dict(kv.split("=", 1) for kv in query.split("&") if "=" in kv)
            n = int(q["n"]) if q.get("n", "").isdigit() else 5
            res = prompt_samples(q.get("study", ""), q.get("arm", ""), max(1, min(5, n)))
            self.send(400 if "error" in res else 200, json.dumps(res, default=str).encode(), "application/json")
        elif path == "/api/log":
            m = re.search(r"(?:^|&)job=(dr_R[0-9]b?_s[0-4]|fa_F[0-2]_s[0-4])(?:&|$)", query)   # whitelist: training logs only
            if not m:
                self.send(400, b"bad job", "text/plain")
            else:
                folder = "flag_arch" if m.group(1).startswith("fa_") else "disc_resid"
                lines = tail_lines(P("logs", folder, f"train_{m.group(1)}.log"), 60)
                self.send(200, ("\n".join(lines) or "(log is empty or missing)").encode(), "text/plain; charset=utf-8")
        elif path.startswith("/files/") and path[7:] in SERVE:
            p = P(*SERVE[path[7:]])
            if os.path.exists(p):
                self.send(200, open(p, "rb").read(), "text/plain; charset=utf-8")
            else:
                self.send(404, b"not created yet", "text/plain")
        else:
            self.send(404, b"not found", "text/plain")

    def log_message(self, *a):
        pass


def main():
    global ARGS, R, HIST_FILE, INSIGHTS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", type=int, default=18781)
    ap.add_argument("--root", help="data root to read instead of the project (mock testing)")
    ap.add_argument("--no-insights", action="store_true", help="never launch text_insights.py")
    ap.add_argument("--manifest", choices=sorted(STUDIES), help="print that study's experiment manifest JSON and exit")
    ARGS = ap.parse_args()
    R = os.path.abspath(ARGS.root) if ARGS.root else os.path.abspath(os.path.join(HERE, "..", "..", ".."))
    if os.path.basename(HERE) != "v2":        # installed as the live dashboard (tools/dr-dashboard/)
        R = os.path.abspath(ARGS.root) if ARGS.root else os.path.abspath(os.path.join(HERE, "..", ".."))
    if ARGS.manifest:
        print(json.dumps(study_manifest(ARGS.manifest), indent=2, default=str))
        return
    state_dir = os.path.join(R, ".dashboard_state") if ARGS.root else HERE
    os.makedirs(os.path.join(state_dir, "cache"), exist_ok=True)
    HIST_FILE = os.path.join(state_dir, "history.json")
    INSIGHTS_DIR = os.path.join(state_dir, "cache")
    load_history()
    ThreadingHTTPServer(("127.0.0.1", ARGS.port), H).serve_forever()


INSIGHTS_DIR = HERE
if __name__ == "__main__":
    main()
