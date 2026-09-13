"""Figure 4: DGP component ablations (DGP, w/o MDK, w/o PathSumm, w/o TextSumm, w/o NumSumm).

    python scripts/run_ablation.py --dataset amazonvideo --device cuda:0
    python scripts/run_ablation.py --dataset amazonvideo --mode smoke --seeds 0    # CPU_DEBUG

Outputs: results/tables/ablation_<dataset>.{csv,md}, results/figures/ablation_<dataset>.png
The paper prints no numbers for Figure 4, so there is nothing to compare numerically; the check is
the ordering the paper states (every removal hurts; TextSumm matters more than NumSumm).
"""

from _common import ROOT, base_parser, device_for, experiment_config, seeds_for

from dgp_repro.evaluation.reporting import plot_bars, result_row, write_table
from dgp_repro.experiment import run_experiment
from dgp_repro.metrics import METRICS

VARIANTS = ["full", "without_mdk", "without_path_summarization", "without_text_summarization",
            "without_numeric_summarization"]


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--variants", nargs="*", default=VARIANTS)
    args = parser.parse_args()
    rows, groups, errors = [], {}, {}
    for variant in args.variants:
        overlay = f"configs/experiments/ablations/{variant}.yaml"
        cfg = experiment_config(args, [overlay])
        name = f"ablation_{cfg['dataset']['name']}_{variant}" + ("_debug" if args.mode == "smoke" else "")
        agg = run_experiment(cfg, name, seeds_for(args, cfg), device_for(args, cfg))
        row = result_row(agg, "DGP", cfg["variant"])
        rows.append(row)
        groups[cfg["variant"]] = {m: row[f"{m}_mean"] for m in METRICS}
        errors[cfg["variant"]] = {m: row[f"{m}_std"] for m in METRICS}
    dataset = experiment_config(args)["dataset"]["name"] + ("_debug" if args.mode == "smoke" else "")
    write_table(rows, ROOT / "results" / "tables" / f"ablation_{dataset}", f"Ablation ({dataset})")
    plot_bars(groups, errors, ROOT / "results" / "figures" / f"ablation_{dataset}.png", f"Figure 4 equivalent: {dataset}")
    print(f"wrote results/tables/ablation_{dataset}.csv and results/figures/ablation_{dataset}.png")


if __name__ == "__main__":
    main()
