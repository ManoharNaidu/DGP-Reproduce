#!/usr/bin/env python3
"""usage: release.py release X.Y.Z "msg"  |  release.py rollback X.Y.Z   (needs git)"""
import datetime, os, subprocess, sys
g = lambda *a: subprocess.run(["git", *a], check=True)
def release(v, msg):
    open("VERSION", "w").write(v + "\n")
    old = open("CHANGELOG.md").read() if os.path.exists("CHANGELOG.md") else ""
    open("CHANGELOG.md", "w").write(f"## {v} ({datetime.date.today()})\n{msg}\n\n{old}")
    if "jobs.json" not in (open(".gitignore").read() if os.path.exists(".gitignore") else ""):
        open(".gitignore", "a").write("jobs.json\n")
    g("add", "-A"); g("commit", "-qm", f"v{v}: {msg}"); g("tag", f"v{v}"); print("released v" + v)
def rollback(v):  # restores files from the tag as a new commit, history kept
    subprocess.run(["git", "rev-parse", "-q", "--verify", f"v{v}"], check=True, stdout=subprocess.DEVNULL); g("checkout", f"v{v}", "--", "."); g("commit", "-qm", f"rollback to v{v}")
    print(f"rolled back to v{v}; restart the service")
a = sys.argv[1:]
if len(a) == 3 and a[0] == "release": release(a[1], a[2])
elif len(a) == 2 and a[0] == "rollback": rollback(a[1])
else: sys.exit(__doc__)
