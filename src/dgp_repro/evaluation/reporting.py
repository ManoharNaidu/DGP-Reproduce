"""Tables, figures and the comparison against the paper's reported numbers.

Comparison status (our convention, documented here because the paper defines none):
    MATCH     |ours - paper| <= 2 * sqrt(std_ours^2 + std_paper^2)   (within seed noise of both)
    CLOSE     not MATCH, but |ours - paper| <= 2.0 percentage points
    DEVIATES  otherwise
    NO_RESULT we have no finished run for that row
Only real runs (GPU_REPRODUCTION, or CPU_REPRODUCTION for CPU-only baselines such as the MLP) are compared;
CPU_DEBUG results are refused.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from dgp_repro.config import REPO_ROOT
from dgp_repro.metrics import METRICS

RESULTS = REPO_ROOT / "results"
REPORTED = REPO_ROOT / "research" / "reported_results.csv"
DATASET_NAMES = {"amazonvideo": "AmazonVideo", "yelpchi": "YelpReviews"}
CLOSE_POINTS = 2.0
REAL_MODES = ("GPU_REPRODUCTION", "CPU_REPRODUCTION")  # CPU_DEBUG is never a result


def load_reported() -> list[dict]:
    with open(REPORTED, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_aggregated(experiment: str) -> dict | None:
    path = RESULTS / "aggregated" / f"{experiment}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def result_row(aggregated: dict, method: str, variant: str = "main") -> dict:
    test = aggregated["test"]
    row = {"dataset": DATASET_NAMES.get(aggregated["dataset"], aggregated["dataset"]), "method": method,
           "variant": variant, "mode": aggregated["mode"], "n_seeds": test["n_seeds"]}
    for m in METRICS:
        row[f"{m}_mean"] = round(test[f"{m}_mean"], 2)
        row[f"{m}_std"] = round(test[f"{m}_std"], 2)
    return row


def write_table(rows: list[dict], stem: str | Path, title: str = "") -> None:
    """Writes <stem>.csv and <stem>.md."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0])
    with open(stem.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"# {title}\n"] if title else []
    lines += ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    lines += ["| " + " | ".join(str(r.get(k, "")) for k in fields) + " |" for r in rows]
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare(ours: dict, paper: dict) -> list[dict]:
    """One comparison row per metric for a (dataset, method, variant)."""
    if ours.get("mode") not in REAL_MODES:
        raise ValueError(f"refusing to compare a {ours.get('mode')} result with the paper")
    rows = []
    for m in METRICS:
        p_mean, p_std = float(paper[f"{m}_mean"]), float(paper[f"{m}_std"])
        o_mean, o_std = float(ours[f"{m}_mean"]), float(ours[f"{m}_std"])
        diff = o_mean - p_mean
        if abs(diff) <= 2 * math.sqrt(o_std ** 2 + p_std ** 2):
            status = "MATCH"
        elif abs(diff) <= CLOSE_POINTS:
            status = "CLOSE"
        else:
            status = "DEVIATES"
        rows.append({"dataset": paper["dataset"], "method": paper["method"], "variant": paper["variant"], "mode": ours["mode"], "metric": m,
                     "paper": p_mean, "paper_std": p_std, "ours": round(o_mean, 2), "ours_std": round(o_std, 2),
                     "abs_diff": round(diff, 2), "rel_diff_pct": round(100 * diff / p_mean, 2), "status": status})
    return rows


def plot_bars(groups: dict[str, dict[str, float]], errors: dict[str, dict[str, float]], path: str | Path,
              title: str, ylabel: str = "%") -> None:
    """groups: variant -> metric -> value. One cluster per metric, one bar per variant."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = list(groups)
    metrics = list(next(iter(groups.values())))
    width = 0.8 / max(len(variants), 1)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, v in enumerate(variants):
        xs = [j + i * width for j in range(len(metrics))]
        ax.bar(xs, [groups[v][m] for m in metrics], width, yerr=[errors[v][m] for m in metrics], label=v, capsize=2)
    ax.set_xticks([j + width * (len(variants) - 1) / 2 for j in range(len(metrics))])
    ax.set_xticklabels(metrics)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=len(variants), loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_lines(x: list, series: dict[str, list[float]], errors: dict[str, list[float]], path: str | Path,
               title: str, xlabel: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.6))
    for name, ys in series.items():
        ax.errorbar(range(len(x)), ys, yerr=errors[name], marker="o", capsize=2, label=name)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels([str(v) for v in x])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("%")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
