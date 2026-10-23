"""Write a batch record at launch time: git state, the exact launch command and the run hypothesis.

    python scripts/record_batch.py --study flag_arch --hypothesis "D lines per subgraph help" \
        --command "nohup scripts/fa_queue.sh > logs/flag_arch/queue.log 2>&1 &" \
        --config configs/experiments/flagarch/F1_disc.yaml --config configs/experiments/flagarch/F2_disc_resid_finetune.yaml

Writes results/<study>/batch_<UTC>.json (the dashboard's /api/manifest reads every batch_*.json of a study). The
hypothesis may also come from $RUN_HYPOTHESIS. Call it right before launching the queue, so the git state is the one
the batch starts from; each run still records its own commit in run_manifest.json.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, help="results/<study>/ folder, e.g. flag_arch or disc_resid")
    ap.add_argument("--command", required=True, help="the exact command that launches the batch")
    ap.add_argument("--hypothesis", default=os.environ.get("RUN_HYPOTHESIS"), help="short text; default $RUN_HYPOTHESIS")
    ap.add_argument("--config", action="append", default=[], help="experiment config(s) the batch runs (repeatable)")
    a = ap.parse_args()
    if not a.hypothesis:
        sys.exit("record_batch.py: give --hypothesis or set RUN_HYPOTHESIS")
    status = git("status", "--porcelain")
    ts = time.gmtime()
    rec = {"schema": "batch-record-v1", "study": a.study, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", ts),
           "git_commit": git("rev-parse", "HEAD") or "unknown", "git_branch": git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
           "git_dirty": bool(status) if status is not None else "unknown",
           "git_dirty_files": status.splitlines() if status else [],
           "command": a.command, "configs": a.config, "run_hypothesis": a.hypothesis}
    out = os.path.join(ROOT, "results", a.study, f"batch_{time.strftime('%Y%m%dT%H%M%SZ', ts)}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        sys.exit(f"record_batch.py: {out} already exists")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2)
    print(out)


if __name__ == "__main__":
    main()
