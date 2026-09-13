"""Compare our real (non-debug) results with the paper (research/reported_results.csv).

    python scripts/compare_with_paper.py

Status rule (documented in src/dgp_repro/evaluation/reporting.py):
    MATCH     |diff| <= 2 * sqrt(std_ours^2 + std_paper^2)
    CLOSE     otherwise, |diff| <= 2.0 percentage points
    DEVIATES  otherwise
    NO_RESULT no finished real run for the row
The paper numbers are used only as comparison targets; they are never copied into our results.
Output: results/tables/comparison_with_paper.{csv,md}
"""

import csv

from _common import ROOT

from dgp_repro.evaluation.reporting import compare, load_reported, write_table

OURS = ROOT / "results" / "tables"


def load_rows(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ours = {}
    for row in load_rows(OURS / "main_results.csv"):
        ours[(row["dataset"], row["method"], "main")] = row  # carries its real mode; debug rows never reach this table
    for row in load_rows(ROOT / "results" / "task_aware_comparison.csv"):
        variant = "task_aware" if "aware" in row["variant"] and "agnostic" not in row["variant"] else "task_agnostic"
        ours[(row["dataset"], "DGP", variant)] = row

    out = []
    for paper in load_reported():
        if paper["reproducible"] != "REPRODUCIBLE":
            continue
        key = (paper["dataset"], paper["method"], paper["variant"])
        if key in ours:
            out.extend(compare(ours[key], paper))
        else:
            out.append({"dataset": paper["dataset"], "method": paper["method"], "variant": paper["variant"],
                        "mode": "", "metric": "all", "paper": "", "paper_std": "", "ours": "", "ours_std": "",
                        "abs_diff": "", "rel_diff_pct": "", "status": "NO_RESULT"})
    write_table(out, OURS / "comparison_with_paper", "Comparison with paper (public datasets)")
    counts = {}
    for row in out:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(counts)


if __name__ == "__main__":
    main()
