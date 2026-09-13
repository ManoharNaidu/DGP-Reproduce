"""Train and evaluate DGP for each seed; aggregate mean/std.

    python scripts/train.py --dataset amazonvideo --device cuda:0 --seeds 0 1 2 3 4
    python scripts/train.py --dataset amazonvideo --mode smoke --seeds 0        # CPU_DEBUG
    python scripts/train.py --config configs/experiments/dgp_amazon.yaml --set classifier.lora.rank=16

Completed seeds are skipped (results/raw/<run_id>/metrics.json exists), so the command is resumable.
"""

import json

from _common import base_parser, device_for, experiment_config, seeds_for

from dgp_repro.experiment import run_experiment


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--experiment", help="name for results files (default: config 'experiment')")
    parser.add_argument("--overlay", nargs="*", default=[], help="extra config files merged on top")
    args = parser.parse_args()
    cfg = experiment_config(args, args.overlay)
    name = args.experiment or cfg.get("experiment", cfg["dataset"]["name"])
    result = run_experiment(cfg, name, seeds_for(args, cfg), device_for(args, cfg))
    print(json.dumps(result["test"], indent=2))


if __name__ == "__main__":
    main()
