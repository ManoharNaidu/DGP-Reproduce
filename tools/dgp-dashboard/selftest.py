#!/usr/bin/env python3
"""Starts server.py on 8799, checks /health and that /api/state carries every section, stops it. Exit 0 = pass."""
import json, subprocess, sys, time, urllib.request as u
p = subprocess.Popen([sys.executable, "server.py", "8799"])
call = lambda path: json.load(u.urlopen("http://127.0.0.1:8799" + path, timeout=15))
try:
    for _ in range(20):
        try: call("/health"); break
        except OSError: time.sleep(.25)
    s = call("/api/state")
    for k in ("jobs", "stores", "results", "gpus", "updated"): assert k in s, k
    assert any(r["exp"].startswith("dgp_amazonvideo") for r in s["results"]), "baseline row missing"
    assert u.urlopen("http://127.0.0.1:8799/", timeout=5).read().startswith(b"<!doctype html>")
    print("OK")
finally:
    p.terminate(); p.wait()
