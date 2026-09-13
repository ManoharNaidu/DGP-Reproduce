"""Export a prepared dataset for the official baselines: CARE-GNN .mat + split.json, and a neutral .npz edge bundle.

    python scripts/export_baseline_data.py --dataset amazonvideo            # DeBERTaV3 + numeric features (needs cached embeddings)
    python scripts/export_baseline_data.py --dataset amazonvideo --features hashing-debug   # DEBUG_ONLY features, plumbing checks

Node features for baselines are unstated by the paper; DeBERTaV3 (mean-pooled) concat numeric is used, the same
matrix as DGP's MDK and the MLP baseline (PAPER_RECONSTRUCTION).
"""

import numpy as np
from _common import base_parser, experiment_config

from dgp_repro.baselines.export import export_care_gnn_mat, export_edge_bundle
from dgp_repro.data import load_graph, load_split, processed_dir
from dgp_repro.embeddings import HashingEmbedder, build_mdk_features
from dgp_repro.pipeline import text_embedding_cache_path


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--features", default="deberta", choices=["deberta", "hashing-debug"])
    args = parser.parse_args()
    cfg = experiment_config(args)
    graph = load_graph(cfg["dataset"])
    split = load_split(cfg["dataset"], graph)
    if args.features == "deberta":
        path = text_embedding_cache_path(graph, cfg)
        if not path.exists():
            raise SystemExit(f"no cached DeBERTa embeddings at {path}; run scripts/build_mdk.py first")
        text = np.load(path)
        tag = "baseline_bundle"
    else:
        text = HashingEmbedder(64).embed(graph.texts)
        tag = "baseline_bundle_DEBUG"
    features = build_mdk_features(text, graph.numeric, "deberta_concat", "none")
    out = processed_dir(cfg["dataset"])
    print(export_edge_bundle(graph, features, split, out / f"{tag}.npz"))
    if args.features == "deberta":
        print(export_care_gnn_mat(graph, features, split, out / "care_gnn"))


if __name__ == "__main__":
    main()
