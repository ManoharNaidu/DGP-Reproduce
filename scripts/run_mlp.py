"""MLP baseline (paper Table 2). Runs on CPU once the DeBERTaV3 embeddings are cached.

    python scripts/build_mdk.py --dataset amazonvideo --device cpu     # computes and caches the embeddings
    python scripts/run_mlp.py --dataset amazonvideo --seeds 0 1 2 3 4

Provenance: PAPER_RECONSTRUCTION (no official MLP code; input features unstated by the paper).
Mode is CPU_REPRODUCTION: a real result on real features, not a debug run.
Writes results/raw/mlp_<dataset>_seed<k>_<hash>/ in the same layout as DGP runs, then scripts/evaluate.py
adds the MLP row to results/tables/main_results.
"""

import csv
import json
import time

import numpy as np
from _common import ROOT, base_parser, experiment_config, seeds_for

from dgp_repro.baselines.mlp import GRID, run_mlp
from dgp_repro.data import load_graph, load_split
from dgp_repro.embeddings import build_mdk_features
from dgp_repro.experiment import config_hash
from dgp_repro.metrics import aggregate_seeds
from dgp_repro.pipeline import text_embedding_cache_path
from dgp_repro.utils.provenance import write_manifest


def main():
    args = base_parser(__doc__).parse_args()
    cfg = experiment_config(args)
    graph = load_graph(cfg["dataset"])
    split = load_split(cfg["dataset"], graph)
    emb_path = text_embedding_cache_path(graph, cfg)
    if not emb_path.exists():
        raise SystemExit(f"no cached embeddings at {emb_path}; run scripts/build_mdk.py --dataset {graph.name} first")
    features = build_mdk_features(np.load(emb_path), graph.numeric, "deberta_concat", "none")
    seeds = seeds_for(args, cfg)
    started = time.time()
    result = run_mlp(features, graph.labels.astype(int), split, seeds)

    run_cfg = {"method": "MLP", "dataset": cfg["dataset"]["name"], "features": "deberta-v3-base mean-pooled + numeric",
               "embedder": cfg["embedder"], "grid": GRID, "selected_params": result["selected_params"],
               "selection": "mean validation AUROC", "seeds": seeds}
    experiment = f"mlp_{graph.name}"
    for run in result["per_seed"]:
        run_dir = ROOT / "results" / "raw" / f"{experiment}_seed{run['seed']}_{config_hash(run_cfg)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["node", "split", "label", "p_fraud"])
            for name, prob in (("val", run["val_prob"]), ("test", run["test_prob"])):
                for node, p in zip(getattr(split, name), prob):
                    writer.writerow([int(node), name, int(graph.labels[node]), f"{p:.6f}"])
        (run_dir / "metrics.json").write_text(json.dumps({"seed": run["seed"], "val": run["val"], "test": run["test"]}, indent=2),
                                              encoding="utf-8")
        write_manifest(run_dir, run_cfg, {"run_id": run_dir.name, "experiment": experiment, "seed": run["seed"],
                                          "mode": "CPU_REPRODUCTION", "device": "cpu",
                                          "runtime_seconds": round(time.time() - started, 1),
                                          "dataset_summary": graph.summary(), "split_sizes": split.sizes(),
                                          "split_seed": split.seed, "provenance": "PAPER_RECONSTRUCTION"})
    agg = aggregate_seeds([r["test"] for r in result["per_seed"]])
    print(json.dumps({"selected_params": result["selected_params"], "test": agg}, indent=2))


if __name__ == "__main__":
    main()
