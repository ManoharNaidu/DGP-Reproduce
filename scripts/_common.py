"""Shared command-line plumbing for the scripts in this folder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dgp_repro.config import deep_merge, load_config, load_experiment, parse_overrides  # noqa: E402
from dgp_repro.utils.device import resolve_device  # noqa: E402

MAIN_CONFIGS = {
    ("amazonvideo", "gpu"): "configs/experiments/dgp_amazon.yaml",
    ("yelpchi", "gpu"): "configs/experiments/dgp_yelp.yaml",
    ("amazonvideo", "smoke"): "configs/experiments/cpu_debug_amazon.yaml",
    ("synthetic", "smoke"): "configs/experiments/smoke.yaml",
}


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="amazonvideo", choices=["amazonvideo", "yelpchi", "synthetic"])
    parser.add_argument("--mode", default="gpu", choices=["gpu", "smoke"],
                        help="gpu = GPU_REPRODUCTION (Qwen3-8B); smoke = CPU_DEBUG (mock models)")
    parser.add_argument("--config", help="experiment config; default chosen from --dataset and --mode")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda:0 ...")
    parser.add_argument("--seeds", type=int, nargs="+", help="default: the config's seeds (0-4)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="config overrides, e.g. --set dgp.M=8 training.learning_rate=3e-5")
    return parser


def experiment_config(args, overlays: list[str] | None = None) -> dict:
    path = args.config or MAIN_CONFIGS.get((args.dataset, args.mode))
    if path is None:
        raise SystemExit(f"no default config for dataset={args.dataset} mode={args.mode}; pass --config")
    cfg = load_experiment(path)
    for overlay in overlays or []:
        cfg = deep_merge(cfg, load_config(overlay))
    return deep_merge(cfg, parse_overrides(args.set))


def device_for(args, cfg: dict) -> str:
    device = resolve_device(args.device)
    if cfg.get("mode") == "GPU_REPRODUCTION" and not device.startswith("cuda"):
        print("WARNING: GPU_REPRODUCTION on CPU. Qwen3-8B summarization and LoRA training are not practical "
              "on CPU; use --mode smoke for a CPU check.", file=sys.stderr)
    return device


def seeds_for(args, cfg: dict) -> list[int]:
    return args.seeds if args.seeds is not None else list(cfg.get("seeds", [0, 1, 2, 3, 4]))
