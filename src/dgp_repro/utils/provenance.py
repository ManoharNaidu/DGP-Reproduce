"""Run provenance: git commit, library versions, hardware, and the run manifest."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return "not_a_git_repository"
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True,
                               timeout=10).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "git_unavailable"


def _version(module: str) -> str | None:
    try:
        return getattr(__import__(module), "__version__", "unknown")
    except ImportError:
        return None


def environment_info() -> dict:
    info = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": _version("numpy"), "scipy": _version("scipy"), "sklearn": _version("sklearn"),
        "torch": _version("torch"), "transformers": _version("transformers"), "peft": _version("peft"),
        "dgl": _version("dgl"),
    }
    try:
        import torch
        info["cuda"] = torch.version.cuda
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    return info


def peak_memory_gb() -> dict:
    out = {}
    try:
        import torch
        if torch.cuda.is_available():
            out["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
    except ImportError:
        pass
    try:
        import resource  # POSIX only
        out["cpu_peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2)
    except ImportError:
        pass
    return out


def write_manifest(run_dir: str | Path, config: dict, extra: dict | None = None) -> Path:
    """results/raw/<run_id>/run_manifest.json"""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_commit(),
        "environment": environment_info(),
        "config": config,
        **(extra or {}),
    }
    path = run_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path
