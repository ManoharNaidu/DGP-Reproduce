"""Token complexity (paper formulas) with MEASURED L, D, R, plus measured neighbourhood sizes and the
THEORETICAL_ANALYSIS attention-dilution curves.

    python scripts/analyze_complexity.py --dataset amazonvideo
    python scripts/analyze_complexity.py --dataset amazonvideo --tokenizer words   # no download, DEBUG counts

Needs only CPU and the Qwen3 tokenizer files (~11 MB); no model weights.
Outputs:
    results/complexity/token_complexity.csv            formulas evaluated at measured L, D, R over the K/M/B grid
    results/complexity/measured_neighborhoods_<ds>.csv real |N_P(v)| and raw neighbour tokens per metapath
    results/complexity/attention_dilution.csv          THEORETICAL_ANALYSIS
"""

import csv
import json

import numpy as np
from _common import ROOT, base_parser, experiment_config

from dgp_repro.data import load_graph, load_split
from dgp_repro.evaluation.attention_dilution import dilution_curves
from dgp_repro.evaluation.complexity import complexity_table, degree_from_graph, write_complexity_csv
from dgp_repro.mdk import enumerate_metapaths, metapath_name, metapath_neighbors

OUT = ROOT / "results" / "complexity"


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--tokenizer", default="qwen", choices=["qwen", "words"])
    parser.add_argument("--max-hops", type=int, default=2, help="measure neighbourhoods up to this K")
    args = parser.parse_args()
    cfg = experiment_config(args)
    graph = load_graph(cfg["dataset"])
    split = load_split(cfg["dataset"], graph)
    targets = split.all_targets()

    if args.tokenizer == "qwen":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["classifier"]["model_id"])
        counts = np.array([len(ids) for ids in tok(graph.texts, add_special_tokens=False)["input_ids"]])
        counter_name = cfg["classifier"]["model_id"] + " tokenizer"
    else:
        counts = np.array([len(t.split()) for t in graph.texts])
        counter_name = "whitespace words (DEBUG)"

    L = float(counts.mean())
    D = degree_from_graph(graph.num_edges // 2, graph.num_nodes)
    R = len(graph.relations)
    anchors = {"dataset": graph.name, "token_counter": counter_name, "L_mean_tokens": round(L, 1),
               "L_median_tokens": float(np.median(counts)), "D_undirected_pairs_per_node": round(D, 1), "R": R,
               "paper_anchor": {"amazonvideo": "D = 133", "yelpchi": "L = 170 (arXiv v1)"}.get(graph.name)}
    print(json.dumps(anchors, indent=2))

    settings = [{"dataset": graph.name, "L": round(L, 1), "D": round(D, 1), "R": R, "K": K, "M": M, "B": B}
                for K in (1, 2, 3) for M in (2, 4, 8, 16) for B in (5, 10, 20, 40, 80)]
    write_complexity_csv(complexity_table(settings), OUT / "token_complexity.csv")

    rows = []
    for path in enumerate_metapaths(list(graph.relations), args.max_hops):
        nbrs = metapath_neighbors(graph.relations, path, targets)
        sizes = np.array([len(n) for n in nbrs])
        raw = np.array([counts[n].sum() for n in nbrs])
        rows.append({"dataset": graph.name, "metapath": metapath_name(path), "hops": len(path), "targets": len(targets),
                     "mean_neighbors": round(sizes.mean(), 1), "median_neighbors": float(np.median(sizes)),
                     "max_neighbors": int(sizes.max()), "empty": int((sizes == 0).sum()),
                     "mean_raw_neighbor_tokens": round(raw.mean(), 1), "max_raw_neighbor_tokens": int(raw.max())})
        print(f"{rows[-1]['metapath']:10s} mean |N_P|={rows[-1]['mean_neighbors']:>8}  mean raw tokens={rows[-1]['mean_raw_neighbor_tokens']:>12}")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"measured_neighborhoods_{graph.name}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / f"anchors_{graph.name}.json").write_text(json.dumps(anchors, indent=2), encoding="utf-8")

    fraud_rate = graph.num_frauds / graph.num_nodes
    curves = dilution_curves(L=L, p=fraud_rate, D_values=[10, 50, round(D, 1)], m_values=[10, L], K_values=[1, 2, 3])
    with open(OUT / "attention_dilution.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
