"""Figure 5 (summarization budget B) and Table 3 (task-aware vs task-agnostic summarization).

    python scripts/run_sensitivity.py --study budget --dataset amazonvideo --device cuda:0
    python scripts/run_sensitivity.py --study task_aware --dataset amazonvideo --device cuda:0
    python scripts/run_sensitivity.py --study budget --budgets 10 20 --dataset amazonvideo --mode smoke --seeds 0

Outputs (per dataset; scripts/evaluate.py combines them into results/budget_sensitivity.csv and
results/task_aware_comparison.csv):
    budget      results/tables/budget_<dataset>.{csv,md}, results/figures/budget_sensitivity_<dataset>.png
    task_aware  results/tables/task_aware_<dataset>.{csv,md}
Each budget needs its own node and metapath summaries, so the budget study regenerates summaries per B
(cached, so a rerun is cheap).
"""

from _common import ROOT, base_parser, device_for, experiment_config, seeds_for

from dgp_repro.evaluation.reporting import plot_lines, result_row, write_table
from dgp_repro.experiment import run_experiment
from dgp_repro.metrics import METRICS


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--study", required=True, choices=["budget", "task_aware"])
    parser.add_argument("--budgets", type=int, nargs="*", default=[5, 10, 20, 40, 80])
    args = parser.parse_args()
    suffix = "_debug" if args.mode == "smoke" else ""

    if args.study == "budget":
        rows, series, errors = [], {m: [] for m in METRICS}, {m: [] for m in METRICS}
        for B in args.budgets:
            cfg = experiment_config(args, [f"configs/experiments/sensitivity/budget_B{B}.yaml"])
            agg = run_experiment(cfg, f"budget_{cfg['dataset']['name']}_B{B}{suffix}", seeds_for(args, cfg), device_for(args, cfg))
            row = {"B": B, **result_row(agg, "DGP", f"B={B}"), "average_prompt_length": round(agg["token_usage"]["average_prompt_length"], 1)}
            rows.append(row)
            for m in METRICS:
                series[m].append(row[f"{m}_mean"])
                errors[m].append(row[f"{m}_std"])
        dataset = rows[0]["dataset"]
        write_table(rows, ROOT / "results" / "tables" / f"budget_{dataset}{suffix}", "Figure 5 equivalent")
        plot_lines(args.budgets, series, errors, ROOT / "results" / "figures" / f"budget_sensitivity_{dataset}{suffix}.png",
                   f"Figure 5 equivalent: {dataset}", "summarization budget B (tokens)")
    else:
        rows = []
        for variant in ("task_agnostic", "task_aware"):
            cfg = experiment_config(args, [f"configs/experiments/task_aware/{variant}.yaml"])
            agg = run_experiment(cfg, f"taskaware_{cfg['dataset']['name']}_{variant}{suffix}", seeds_for(args, cfg), device_for(args, cfg))
            rows.append(result_row(agg, "DGP", cfg["variant"]))
        write_table(rows, ROOT / "results" / "tables" / f"task_aware_{rows[0]['dataset']}{suffix}", "Table 3 equivalent")
    print("done")


if __name__ == "__main__":
    main()
