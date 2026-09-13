"""Export our graphs for the official fraud-GNN baselines (ConsisGAD, PMP, GAAP, GraphSAGE, HGT).

Provenance: BASELINE_ADAPTED (data interface only). The baselines' algorithms are not modified.

These repositories load the CARE-GNN `.mat` layout (keys `net_<relation>`, `homo`, `features`, `label`),
the same layout as the group's own `fd_yelp_chi.mat` mirror (research/reproduction_gaps.md section 3.2).
We write our graph in that layout so each baseline trains on exactly the DGP graph, together with a
`split.json` holding our train/val/test node ids; each baseline's split code is then pointed at that
file (documented per method in research/baselines/<method>.md, "adaptation_required").

Features are X = DeBERTaV3(text) concat numeric. The paper does not say which node features its GNN
baselines used on these text graphs; this choice is a PAPER_RECONSTRUCTION shared with the MLP.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp

from dgp_repro.data import HeteroGraph, Split


def export_care_gnn_mat(graph: HeteroGraph, features: np.ndarray, split: Split, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    homo = sum(graph.relations.values())
    homo = sp.csr_matrix((homo > 0).astype(np.float64))
    payload = {f"net_{rel.lower()}": sp.csc_matrix(mat, dtype=np.float64) for rel, mat in graph.relations.items()}
    payload.update({"homo": sp.csc_matrix(homo), "features": sp.csc_matrix(features.astype(np.float64)),
                    "label": graph.labels.astype(np.int64).reshape(1, -1)})
    path = out_dir / f"{graph.name}.mat"
    scipy.io.savemat(path, payload, do_compression=True)
    (out_dir / "split.json").write_text(json.dumps({
        "train": split.train.tolist(), "val": split.val.tolist(), "test": split.test.tolist(), "seed": split.seed,
        "note": "DGP reproduction split; baselines must use these ids instead of their own random splits"}),
        encoding="utf-8")
    return path
