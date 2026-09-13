"""Train/validation/test splits.

What is known (paper Table 1): the exact split SIZES, e.g. YelpReviews 1,348 / 1,348 / 13,479.
These are used directly as the target sizes.

What is NOT known and is reconstructed (PAPER_RECONSTRUCTION, research/evidence_matrix.md, D):
  - nodes are drawn from the labeled pool. With the default labels every node is labeled on both
    public datasets, so this is all N nodes, which also explains why the paper's split sizes are
    exact fractions of N (test = 20% of N);
  - sampling is stratified by label;
  - the split is fixed (split_seed) and the five experiment seeds vary only training.
    Inference: DGP's reported std is as small as 0.11 AUROC, which is hard to reconcile
    with re-drawing a 1.3k-node training set per seed.

Splits never look at anything except labels, and test labels are only read here and in
evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dgp_repro.data.graph import UNLABELED


@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    seed: int

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def all_targets(self) -> np.ndarray:
        return np.concatenate([self.train, self.val, self.test])

    def save(self, path: str | Path) -> None:
        payload = {"seed": self.seed, "train": self.train.tolist(), "val": self.val.tolist(),
                   "test": self.test.tolist()}
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Split":
        p = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(np.array(p["train"]), np.array(p["val"]), np.array(p["test"]), p["seed"])


def _allocate(counts: dict[int, int], total: int) -> dict[int, int]:
    """Split `total` across classes proportionally, largest-remainder rounding."""
    pool = sum(counts.values())
    exact = {c: total * n / pool for c, n in counts.items()}
    alloc = {c: int(np.floor(x)) for c, x in exact.items()}
    for c in sorted(exact, key=lambda c: exact[c] - alloc[c], reverse=True)[: total - sum(alloc.values())]:
        alloc[c] += 1
    return alloc


def make_split(labels: np.ndarray, sizes: dict[str, int], seed: int = 0, stratify: bool = True) -> Split:
    labels = np.asarray(labels)
    labeled = np.flatnonzero(labels != UNLABELED)
    need = sizes["train"] + sizes["val"] + sizes["test"]
    if need > len(labeled):
        raise ValueError(f"split needs {need} labeled nodes but only {len(labeled)} exist")

    rng = np.random.default_rng(seed)
    classes = [0, 1] if stratify else [None]
    remaining = {c: rng.permutation(labeled if c is None else labeled[labels[labeled] == c]) for c in classes}

    parts = {}
    for name in ("test", "val", "train"):
        counts = {c: len(v) for c, v in remaining.items()}
        alloc = _allocate(counts, sizes[name]) if stratify else {None: sizes[name]}
        chosen = []
        for c, k in alloc.items():
            chosen.append(remaining[c][:k])
            remaining[c] = remaining[c][k:]
        parts[name] = np.sort(np.concatenate(chosen))
    return Split(train=parts["train"], val=parts["val"], test=parts["test"], seed=seed)
