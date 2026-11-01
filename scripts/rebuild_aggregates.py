"""Rebuild results/aggregated/<experiment>.json from every finished seed on disk (2026-11-01).

    python scripts/rebuild_aggregates.py            # rewrite every file whose seed set changes
    python scripts/rebuild_aggregates.py --check    # report only

The queues ran one seed per call, and run_experiment used to aggregate only the seeds of its own call, so most files
held one seed (DGP report, discrepancy D3). Each file keeps its config_hash, token usage and other keys; seeds, test
and val are recomputed from results/raw/<experiment>_seed<k>_<config_hash>/metrics.json (mean, population std).
"""
import argparse
import json

from _common import ROOT

from dgp_repro.experiment import seed_metrics_on_disk
from dgp_repro.metrics import aggregate_seeds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    for path in sorted((ROOT / "results" / "aggregated").glob("*.json")):
        agg = json.loads(path.read_text(encoding="utf-8"))
        exp, h = agg.get("experiment", path.stem), agg.get("config_hash")
        if not h:
            print(f"{path.name}: no config_hash, left as is")
            continue
        runs = seed_metrics_on_disk(exp, h)
        if not runs:
            print(f"{path.name}: no finished runs for hash {h}, left as is")
            continue
        seeds = sorted(runs)
        new = {**agg, "seeds": seeds, "test": aggregate_seeds([runs[s]["test"] for s in seeds]),
               "val": aggregate_seeds([runs[s]["val"] for s in seeds])}
        t = new["test"]
        print(f"{path.name}: seeds {agg.get('seeds')} -> {seeds}; test F1 {t['macro_f1_mean']:.2f} AUROC {t['auroc_mean']:.2f} "
              f"AUPRC {t['auprc_mean']:.2f}")
        if not args.check and new != agg:
            path.write_text(json.dumps(new, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
