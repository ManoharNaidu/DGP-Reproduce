"""Convert FLAG-Reproduce's benchmark graphs into DGP's processed format (2026-11-04, DGP on FLAG's datasets).

Run with FLAG-Reproduce's environment (needs torch_geometric for the GLBench files and flagbench for the benchmark
construction), from DGP-Reproduce/:

    ../FLAG-Reproduce/.venv-gpu/bin/python scripts/convert_flag_datasets.py            # all three
    ../FLAG-Reproduce/.venv-gpu/bin/python scripts/convert_flag_datasets.py reddit

Nodes, labels, text and splits are FLAG's EXACTLY, so DGP's numbers are comparable with FLAG's backbone results:
  reddit, instagram  GLBench graph via flagbench.datasets.glbench.load_raw -> flagbench.datasets.benchmark.build
                     (BenchmarkConfig(): 1:10 downsampling seed 0, self-loops dropped, stratified 10/10/80 split
                     seed 0). One relation: URU (reply) / UFU (follow).
  amazon_musical     FLAG's text-augmented CARE-GNN Amazon (datasets/processed/amazon/flag_dataset.pt, loaded with
                     weights_only=True): its train/val/test indices and its three relations UPU / USU / UVU (not
                     CARE-GNN's separate 'homo' view). The 3,305 unlabelled users (no text) stay as neighbours, label -1.
Numeric features: none (Reddit / Instagram have only a 4096-d text embedding; Amazon's 25 CARE-GNN features are
unnamed hand-crafted statistics), so the DGP configs switch the numeric summary off.
Output: datasets/processed/<name>/ (gitignored: Instagram and Reddit data must not be re-hosted).
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
FLAG = ROOT.parent / "FLAG-Reproduce"
sys.path.insert(0, str(FLAG / "src"))
sys.path.insert(0, str(ROOT / "src"))

from flagbench.datasets.benchmark import BenchmarkConfig, build  # noqa: E402
from flagbench.datasets.glbench import load_raw  # noqa: E402

from dgp_repro.data.graph import HeteroGraph  # noqa: E402
from dgp_repro.data.splits import Split  # noqa: E402


def sym_csr(edge_index, n):
    ei = edge_index.cpu().numpy()
    m = sp.coo_matrix((np.ones(ei.shape[1], dtype=np.float32), (ei[0], ei[1])), shape=(n, n)).tocsr()
    m = ((m + m.T) > 0).astype(np.float32).tocsr()
    m.setdiag(0)
    m.eliminate_zeros()
    return m


def save(name, relations, texts, labels, train, val, test, meta):
    n = len(texts)
    g = HeteroGraph(name=name, relations=relations, texts=texts, numeric=np.zeros((n, 0), dtype=np.float32),
                    numeric_names=[], labels=labels.astype(np.int64), node_ids=[str(i) for i in range(n)],
                    metadata={"provenance": "CONVERTED_FROM_FLAG_REPRODUCE", **meta})
    out = ROOT / "datasets" / "processed" / name
    g.save(out)
    Split(np.asarray(train), np.asarray(val), np.asarray(test), 0).save(out / "split.json")
    s = g.summary()
    print(f"{name}: {s['nodes']} nodes, edges {s['edges_per_relation']}, positives {s['frauds']}, "
          f"labelled {s['labeled']}, split {len(train)}/{len(val)}/{len(test)}, "
          f"test positives {int(labels[np.asarray(test)].sum())}, empty texts {sum(not t for t in texts)}")


def glbench(name, rel):
    # torch.load stays weights_only=True; only PyG's plain container classes are allow-listed.
    from torch_geometric.data.data import Data, DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage
    with torch.serialization.safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]):
        data = load_raw(FLAG / "data" / "raw" / name / f"{name}.pt")
    graph, record = build(data, BenchmarkConfig(), name)
    idx = lambda m: torch.nonzero(m).flatten().numpy()  # noqa: E731
    save(name, {rel: sym_csr(graph.edge_index, graph.num_nodes)}, list(graph.raw_texts), graph.y.numpy(),
         idx(graph.train_mask), idx(graph.val_mask), idx(graph.test_mask),
         {"source": f"FLAG-Reproduce data/raw/{name}/{name}.pt via flagbench benchmark.build (benchmark-v1)",
          "label_names": graph.label_names, "benchmark_record": json.loads(json.dumps(record, default=str))})


def amazon():
    d = torch.load(FLAG / "datasets" / "processed" / "amazon" / "flag_dataset.pt", weights_only=True)
    n = int(d["y"].shape[0])
    y = d["y"].numpy().astype(np.int64)
    y[d["unlabeled_mask"].numpy().astype(bool)] = -1
    rels = {k.upper(): sym_csr(v, n) for k, v in d["relation_edges"].items()}
    save("amazon_musical", rels, list(d["raw_texts"]), y, d["train_idx"].numpy(), d["val_idx"].numpy(),
         d["test_idx"].numpy(),
         {"source": "FLAG-Reproduce datasets/processed/amazon/flag_dataset.pt (CARE-GNN Amazon.mat + McAuley 2014 "
                    "Musical Instruments reviews)",
          "flag_sha256": "069e1d6a8d4f079665a8d0ab52ab532cb7230b25096735cb401c2f1f028b23d3"})


if __name__ == "__main__":
    jobs = {"reddit": lambda: glbench("reddit", "URU"), "instagram": lambda: glbench("instagram", "UFU"),
            "amazon_musical": amazon}
    for t in sys.argv[1:] or list(jobs):
        jobs[t]()
