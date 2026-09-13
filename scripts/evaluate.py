"""Collect finished runs into the final tables. Recomputes every metric from predictions.csv.

    python scripts/evaluate.py

Evaluation is kept separate from training: metrics are recomputed from the saved per-node
probabilities, so a bug or change in the training loop's bookkeeping cannot leak into the tables.

Writes (real runs only: GPU_REPRODUCTION / CPU_REPRODUCTION; CPU_DEBUG runs are listed separately):
    results/tables/main_results.{csv,md}
    results/budget_sensitivity.csv
    results/task_aware_comparison.csv
    results/tables/debug_runs.{csv,md}
"""

import csv
import json
from collections import defaultdict

from _common import ROOT

from dgp_repro.evaluation.reporting import DATASET_NAMES, REAL_MODES, write_table
from dgp_repro.metrics import METRICS, aggregate_seeds, evaluate_split

RAW = ROOT / "results" / "raw"


def recompute(run_dir):
    """Test metrics under the shared protocol: Macro-F1 threshold chosen on validation predictions."""
    data = {"val": ([], []), "test": ([], [])}
    with open(run_dir / "predictions.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data[row["split"]][0].append(int(row["label"]))
            data[row["split"]][1].append(float(row["p_fraud"]))
    return evaluate_split(*data["val"], *data["test"])["test"]


def classify(experiment):
    """(table, method, variant) from the experiment naming convention used by the scripts."""
    experiment = experiment.removesuffix("_debug")
    if experiment.startswith("dgp_"):
        return "main", "DGP", "main"
    if experiment.startswith("mlp_"):
        return "main", "MLP", "main"
    if experiment.startswith("consisgad_"):
        return "main", "ConsisGAD", "main"
    if experiment.startswith("llm_"):
        return "main", "LLM", "main"
    if experiment.startswith("ablation_"):
        return "ablation", "DGP", experiment.split("_", 2)[2]
    if experiment.startswith("budget_"):
        return "budget", "DGP", experiment.rsplit("_", 1)[1]
    if experiment.startswith("taskaware_"):
        return "task_aware", "DGP", experiment.split("_", 2)[2]
    return "other", experiment, "main"


def write_token_usage_overview():
    """results/token_usage.{csv,json}: one row per experiment, from results/token_usage/token_usage_<experiment>.json."""
    rows = []
    for path in sorted((ROOT / "results" / "token_usage").glob("token_usage_*.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        experiment = path.stem.removeprefix("token_usage_")
        aggregated = ROOT / "results" / "aggregated" / f"{experiment}.json"
        mode = json.loads(aggregated.read_text(encoding="utf-8"))["mode"] if aggregated.exists() else "UNKNOWN"
        if summary:
            rows.append({"experiment": experiment, "mode": mode, **summary})
    if not rows:
        return
    (ROOT / "results" / "token_usage.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(ROOT / "results" / "token_usage.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    groups = defaultdict(list)
    for manifest_path in sorted(RAW.glob("*/run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not (manifest_path.parent / "predictions.csv").exists():
            continue
        key = (manifest["experiment"], manifest["dataset_summary"]["name"], manifest["mode"])
        groups[key].append(recompute(manifest_path.parent))

    tables = defaultdict(list)
    for (experiment, dataset, mode), per_seed in sorted(groups.items()):
        agg = aggregate_seeds(per_seed)
        table, method, variant = classify(experiment)
        row = {"dataset": DATASET_NAMES.get(dataset, dataset), "method": method, "variant": variant,
               "experiment": experiment, "mode": mode, "n_seeds": agg["n_seeds"]}
        row.update({f"{m}_{s}": round(agg[f"{m}_{s}"], 2) for m in (*METRICS, "macro_f1_at_0.5") for s in ("mean", "std")})
        tables["debug" if mode not in REAL_MODES else table].append(row)

    columns = ["dataset", "method", "mode", "n_seeds", "macro_f1_mean", "macro_f1_std", "auroc_mean", "auroc_std", "auprc_mean", "auprc_std",
               "macro_f1_at_0.5_mean", "macro_f1_at_0.5_std"]  # Macro-F1 = validation-tuned threshold; at_0.5 = fixed threshold
    if tables["main"]:
        write_table([{k: r[k] for k in columns} for r in tables["main"]], ROOT / "results" / "tables" / "main_results",
                    "Main results (real runs only, mean and std over seeds, percent)")
    if tables["budget"]:
        write_table(tables["budget"], ROOT / "results" / "budget_sensitivity")
    if tables["task_aware"]:
        write_table(tables["task_aware"], ROOT / "results" / "task_aware_comparison")
    if tables["ablation"]:
        write_table(tables["ablation"], ROOT / "results" / "tables" / "ablation_all")
    if tables["debug"]:
        write_table(tables["debug"], ROOT / "results" / "tables" / "debug_runs", "CPU_DEBUG runs (DEBUG_ONLY, not results)")
    write_token_usage_overview()
    for name, rows in tables.items():
        print(f"{name:11s} {len(rows)} experiment(s)")
    if not tables["main"]:
        print("No real (non-debug) runs yet: results/tables/main_results is not written.")


if __name__ == "__main__":
    main()
