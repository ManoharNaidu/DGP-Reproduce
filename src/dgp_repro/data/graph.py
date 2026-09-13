"""Heterogeneous graph container: G = {V, E, R, X} with x_v = (x_v^text, x_v^num).

Relation types are kept separate (one sparse matrix per relation); they are never
collapsed into a homogeneous graph.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

UNLABELED = -1


@dataclass
class HeteroGraph:
    name: str
    relations: dict[str, sp.csr_matrix]      # relation name -> N x N binary symmetric matrix
    texts: list[str]                          # x_v^text
    numeric: np.ndarray                       # x_v^num, shape (N, d)
    numeric_names: list[str]
    labels: np.ndarray                        # 1 fraud, 0 benign, -1 unlabeled
    node_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.texts)
        for rel, mat in self.relations.items():
            if mat.shape != (n, n):
                raise ValueError(f"relation {rel} has shape {mat.shape}, expected ({n}, {n})")
        if self.numeric.shape[0] != n or len(self.labels) != n:
            raise ValueError("texts, numeric and labels must all have N rows")
        if self.numeric.shape[1] != len(self.numeric_names):
            raise ValueError("numeric_names must name every numeric column")

    @property
    def num_nodes(self) -> int:
        return len(self.texts)

    @property
    def edge_counts(self) -> dict[str, int]:
        """Directed nnz per relation. Paper Table 1 counts edges this way (2 x undirected pairs)."""
        return {rel: int(mat.nnz) for rel, mat in self.relations.items()}

    @property
    def num_edges(self) -> int:
        return sum(self.edge_counts.values())

    @property
    def num_frauds(self) -> int:
        return int((self.labels == 1).sum())

    @property
    def num_labeled(self) -> int:
        return int((self.labels != UNLABELED).sum())

    def summary(self) -> dict:
        return {
            "name": self.name, "nodes": self.num_nodes, "edges": self.num_edges,
            "edge_types": len(self.relations), "edges_per_relation": self.edge_counts,
            "frauds": self.num_frauds, "labeled": self.num_labeled,
            "numeric_features": self.numeric_names,
        }

    # ------------------------------------------------------------------ persistence

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for rel, mat in self.relations.items():
            sp.save_npz(directory / f"relation_{rel}.npz", mat)
        np.save(directory / "numeric.npy", self.numeric)
        np.save(directory / "labels.npy", self.labels)
        with open(directory / "texts.jsonl", "w", encoding="utf-8") as f:
            for node_id, text in zip(self.node_ids or [""] * self.num_nodes, self.texts):
                f.write(json.dumps({"id": node_id, "text": text}, ensure_ascii=False) + "\n")
        meta = {**self.metadata, "name": self.name, "relations": list(self.relations),
                "numeric_names": self.numeric_names, "summary": self.summary()}
        (directory / "graph.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "HeteroGraph":
        directory = Path(directory)
        meta = json.loads((directory / "graph.json").read_text(encoding="utf-8"))
        relations = {rel: sp.load_npz(directory / f"relation_{rel}.npz").tocsr() for rel in meta["relations"]}
        node_ids, texts = [], []
        with open(directory / "texts.jsonl", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                node_ids.append(row["id"])
                texts.append(row["text"])
        summary = meta.pop("summary", None)
        return cls(
            name=meta.pop("name"), relations=relations, texts=texts,
            numeric=np.load(directory / "numeric.npy"), numeric_names=meta.pop("numeric_names"),
            labels=np.load(directory / "labels.npy"), node_ids=node_ids,
            metadata={k: v for k, v in meta.items() if k != "relations"} | ({"saved_summary": summary} if summary else {}),
        )
