"""YAML configuration with simple inheritance.

A config may list parent files under `defaults:` (paths relative to the repo root or to the
file). Parents are merged first, then the file itself; nested dicts merge key by key and
everything else is replaced.

    defaults:
      - configs/experiments/dgp_amazon.yaml
    components:
      mdk: false
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve(path: str | Path, relative_to: Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    for base in (relative_to, REPO_ROOT, Path.cwd()):
        if base is not None and (base / path).exists():
            return base / path
    raise FileNotFoundError(f"config not found: {path}")


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    path = _resolve(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    merged: dict = {}
    for parent in raw.pop("defaults", []) or []:
        merged = deep_merge(merged, load_config(_resolve(parent, path.parent)))
    merged = deep_merge(merged, raw)
    if overrides:
        merged = deep_merge(merged, overrides)
    return merged


def load_experiment(path: str | Path, overrides: dict | None = None) -> dict:
    """An experiment config names its dataset with `dataset_config:`; that file is loaded into `dataset`."""
    cfg = load_config(path, overrides)
    if "dataset_config" in cfg:
        cfg["dataset"] = deep_merge(load_config(cfg["dataset_config"]), cfg.get("dataset", {}))
    return cfg


def set_by_dotted_key(config: dict, dotted: str, value) -> None:
    """Command-line style override: set_by_dotted_key(cfg, 'dgp.M', 4)."""
    node = config
    *parents, leaf = dotted.split(".")
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value


def parse_overrides(items: list[str] | None) -> dict:
    """['dgp.M=4', 'components.mdk=false'] -> nested dict, values parsed as YAML scalars."""
    out: dict = {}
    for item in items or []:
        key, _, value = item.partition("=")
        set_by_dotted_key(out, key.strip(), yaml.safe_load(value))
    return out
