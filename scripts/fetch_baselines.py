"""Fetch official baseline repositories at pinned commits into methods/<method>/upstream/.

    python scripts/fetch_baselines.py                 # all
    python scripts/fetch_baselines.py consisgad pmp   # some

Third-party code is never committed to this repository (methods/*/upstream/ is git-ignored) and is
never edited in place. Adapters live in src/dgp_repro/baselines/. Licences are kept as shipped;
PMP, GAAP and FLAG publish NO licence, so their code must not be redistributed at all.
Commits were verified against the GitHub API on 2026-09-13 (research/evidence_matrix.md, section E).
"""

import subprocess
import sys

from _common import ROOT

BASELINES = {
    "graphsage":   ("https://github.com/williamleif/GraphSAGE", "a0fdef95", "NOASSERTION"),
    "hgt":         ("https://github.com/acbull/pyHGT", "85eaccd4", "MIT"),
    "consisgad":   ("https://github.com/Xtra-Computing/ConsisGAD", "36811c5b", "MIT"),
    "pmp":         ("https://github.com/Xtra-Computing/PMP", "3f7629f6", "NONE"),
    "gaap":        ("https://github.com/AtwoodDuan/GAAP", "6a7dbb04", "NONE"),
    "tape":        ("https://github.com/XiaoxinHe/TAPE", "d9881f7e", "MIT"),
    "flag":        ("https://github.com/BUPT-GAMMA/FLAG", "cb83944e", "NONE"),
    "graphgpt":    ("https://github.com/HKUDS/GraphGPT", "db25a66f", "Apache-2.0"),
    "higpt":       ("https://github.com/HKUDS/HiGPT", "2b0793e7", "Apache-2.0"),
    "instructglm": ("https://github.com/agiresearch/InstructGLM", "dd2dd5ec", "Apache-2.0"),
}


def fetch(name: str) -> None:
    url, commit, license_id = BASELINES[name]
    dest = ROOT / "methods" / name / "upstream"
    if not dest.exists():
        subprocess.run(["git", "clone", "--quiet", url, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", commit], check=True)
    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if not head.startswith(commit):
        raise SystemExit(f"{name}: expected commit {commit}, got {head}")
    note = "  (NO LICENCE: do not redistribute)" if license_id == "NONE" else ""
    print(f"{name:12s} {head[:12]}  licence={license_id}{note}")


def main():
    names = sys.argv[1:] or list(BASELINES)
    for name in names:
        if name not in BASELINES:
            raise SystemExit(f"unknown baseline {name!r}; choose from {list(BASELINES)}")
        fetch(name)


if __name__ == "__main__":
    main()
