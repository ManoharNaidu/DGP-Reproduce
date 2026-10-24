#!/usr/bin/env python3
"""Starts server.py on 8799 against the real project data, checks every endpoint and the data rules, stops it.
Exit 0 = pass. Read-only: the server never writes, and this test only issues GET requests."""
import csv, importlib.util, io, json, os, subprocess, sys, time, urllib.error, urllib.request as u

HERE = os.path.dirname(os.path.abspath(__file__))
p = subprocess.Popen([sys.executable, os.path.join(HERE, "server.py"), "8799"])
call = lambda path: json.load(u.urlopen("http://127.0.0.1:8799" + path, timeout=30))
raw = lambda path: u.urlopen("http://127.0.0.1:8799" + path, timeout=30)


def status(path):
    try:
        return raw(path).status
    except urllib.error.HTTPError as e:
        return e.code


try:
    for _ in range(40):
        try: call("/health"); break
        except OSError: time.sleep(.25)
    s = call("/api/state")
    # v0.2 contract
    for k in ("jobs", "stores", "results", "gpus", "updated"): assert k in s, k
    assert any(r["exp"].startswith("dgp_amazonvideo") for r in s["results"]), "baseline row missing"
    assert raw("/").read().startswith(b"<!doctype html>")
    # v0.3: studies kept separate, baselines and paper target from files
    for k in ("studies", "paper", "overview", "insights", "diagnostics", "infra"): assert k in s and s[k] is not None, k
    keys = [x["key"] for x in s["studies"]]
    assert keys == ["main", "t2", "t3", "dr", "fa"], keys
    assert {x["kind"] for x in s["studies"] if x["key"] != "fa"} == {"historical"} and s["studies"][-1]["kind"] == "live"
    assert s["paper"]["source"].startswith("research/paper_specification.md:"), s["paper"]
    main = s["studies"][0]
    A = next(a for a in main["arms"] if a["arm"] == "A")
    assert A["n"] == 5 and A["is_baseline"] and A["paired"] is None
    for st in s["studies"]:
        for a in st["arms"]:
            for k, v in a["stats"].items():
                assert v is None or (v["n"] == a["n"] and v["n"] > 0), (st["key"], a["arm"], k)   # missing = null, never 0
            if a["n"] == 0:
                assert all(v is None for v in a["stats"].values()), (st["key"], a["arm"])
            for k, pr in (a.get("paired") or {}).items():
                if pr:
                    assert pr["n"] == len(pr["per_seed"]) and 0 <= pr["improved"] <= pr["n"]
                    assert pr["p"] is None or 0 <= pr["p"] <= 1
    t2 = s["studies"][1]
    B = next(a for a in t2["arms"] if a["arm"] == "B")
    ab = {sd: v["run"]["test"]["auroc"] for sd, v in A["seeds"].items() if v["run"]}
    bb = {sd: v["run"]["test"]["auroc"] for sd, v in B["seeds"].items() if v["run"]}
    want = sum(bb[k] - ab[k] for k in bb) / len(bb)
    assert abs(B["paired"]["auroc"]["mean"] - want) < 1e-9, "paired delta must be per-seed against baseline A"
    assert s["analysis_running"] is False                         # never inferred from a leftover log any more
    # endpoints
    rows = list(csv.reader(io.StringIO(raw("/api/results.csv").read().decode())))
    assert rows[0][:6] == ["study", "arm", "experiment", "seed", "run_id", "macro_f1"] and len(rows) > 50, rows[0]
    rid = next(v["run"]["run_id"] for v in A["seeds"].values() if v["run"])
    assert call("/api/run?id=" + rid)["run_id"] == rid
    for bad in ("/api/run?id=../../etc", "/api/run?id=nope", "/api/log?name=../x", "/files/../server.py"):
        assert status(bad) in (400, 404), bad
    assert status("/api/log?name=fa_queue") == 200 and status("/files/deviation.md") in (200, 404)
    assert raw("/palette.css").read().startswith(b":root{")
    # statistics helper against known t-distribution values
    spec = importlib.util.spec_from_file_location("srv", os.path.join(HERE, "server.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    for t, df, want in ((2.776445, 4, 0.05), (1.0, 9, 0.34343), (4.604095, 4, 0.01)):
        got = m.betainc(df / 2, .5, df / (df + t * t)); assert abs(got - want) < 2e-5, (t, df, got)
    assert m.paired_t([0.1]) is None and m.paired_t([0.2, 0.2]) is None
    print("OK")
finally:
    p.terminate(); p.wait()
