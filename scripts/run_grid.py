"""Hyperparameter search over the paper's published grid (arXiv v1 §5.1), selected by mean validation AUROC.

The paper tuned on B x K x M x LoRA rank x dropout x lr = 1,728 configurations per dataset but reports none of
the chosen values. How much of that grid to search is a budget decision (research/compute_plan.md §6), so this
script runs ONLY the stage you name:

    python scripts/run_grid.py --dataset amazonvideo --device cuda:0 --stage prompt   # K x M at B=10      (12 configs)
    python scripts/run_grid.py --dataset amazonvideo --device cuda:0 --stage lora --set dgp.K=2 dgp.M=8
                                                                                     # rank x lr, dropout 0.05 (12 configs)
    python scripts/run_grid.py --dataset amazonvideo --device cuda:0 --stage full    # all 1,728 (very expensive)

Each configuration uses the given --seeds (default: seed 0 only). Test metrics are recorded but NEVER used for
selection. Output: results/tables/grid_<dataset>_<stage>.{csv,md}, sorted by validation AUROC.
"""

import itertools

from _common import ROOT, base_parser, device_for, experiment_config

from dgp_repro.config import deep_merge, parse_overrides
from dgp_repro.evaluation.reporting import write_table
from dgp_repro.experiment import run_experiment


def stage_points(stage: str, grid: dict) -> list[dict]:
    if stage == "prompt":
        return [{"dgp.K": K, "dgp.M": M, "dgp.B_node": 10, "dgp.B_meta": 10} for K in grid["K"] for M in grid["M"]]
    if stage == "lora":
        return [{"classifier.lora.rank": r, "classifier.lora.alpha": 2 * r, "classifier.lora.dropout": 0.05,
                 "training.learning_rate": lr} for r in grid["lora_rank"] for lr in grid["learning_rate"]]
    if stage == "full":
        return [{"dgp.B_node": B, "dgp.B_meta": B, "dgp.K": K, "dgp.M": M, "classifier.lora.rank": r,
                 "classifier.lora.alpha": 2 * r, "classifier.lora.dropout": d, "training.learning_rate": lr}
                for B, K, M, r, d, lr in itertools.product(grid["B"], grid["K"], grid["M"], grid["lora_rank"],
                                                             grid["lora_dropout"], grid["learning_rate"])]
    raise SystemExit(f"unknown stage {stage}")


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--stage", required=True, choices=["prompt", "lora", "full"])
    args = parser.parse_args()
    base = experiment_config(args)
    seeds = args.seeds if args.seeds is not None else [0]
    points = stage_points(args.stage, base["grid"])
    print(f"{len(points)} configurations x {len(seeds)} seed(s)")

    rows = []
    for point in points:
        cfg = deep_merge(base, parse_overrides([f"{k}={v}" for k, v in point.items()]))
        tag = "_".join(f"{k.split('.')[-1]}{v}" for k, v in point.items())
        agg = run_experiment(cfg, f"grid_{cfg['dataset']['name']}_{args.stage}_{tag}", seeds, device_for(args, cfg))
        rows.append({**point, "val_auroc": round(agg["val"]["auroc_mean"], 2), "val_auprc": round(agg["val"]["auprc_mean"], 2),
                     "val_macro_f1": round(agg["val"]["macro_f1_mean"], 2), "n_seeds": len(seeds)})
    rows.sort(key=lambda r: r["val_auroc"], reverse=True)
    out = ROOT / "results" / "tables" / f"grid_{base['dataset']['name']}_{args.stage}"
    write_table(rows, out, f"Grid stage '{args.stage}' — selection by validation AUROC only")
    print(f"best by validation AUROC: {rows[0]}")


if __name__ == "__main__":
    main()
