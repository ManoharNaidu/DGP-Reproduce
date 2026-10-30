"""Resumable artifact cache for expensive stages (summaries, embeddings, MDK neighbours, prompts).

Every artifact lives under datasets/cache/<stage>/<key>/ where <key> is a hash of the
metadata that determines its content (dataset, model, prompt version, budgets, K, M, ...).
A metadata.json sidecar records those fields plus the git commit and creation time, so any
cached file can be traced back to the settings that produced it.

Text caches are append-only JSONL, so an interrupted summarization run resumes where it
stopped instead of starting over.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from dgp_repro.config import REPO_ROOT
from dgp_repro.utils.provenance import git_commit


def cache_key(metadata: dict) -> str:
    canonical = json.dumps(metadata, sort_keys=True, default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


class ArtifactCache:
    def __init__(self, stage: str, metadata: dict, root: str | Path | None = None):
        self.stage = stage
        self.metadata = dict(metadata)
        self.key = cache_key(self.metadata)
        self.directory = Path(root or REPO_ROOT / "datasets" / "cache") / stage / self.key
        self.directory.mkdir(parents=True, exist_ok=True)
        meta_path = self.directory / "metadata.json"
        if not meta_path.exists():
            record = {**self.metadata, "stage": stage, "cache_key": self.key,
                      "git_commit": git_commit(), "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
            meta_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")

    def path(self, filename: str) -> Path:
        return self.directory / filename


class TextStore:
    """Append-only key -> text store backed by JSONL."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: dict[str, str] = {}
        self.extra: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue  # a concurrent writer's half-written last line; the key is simply regenerated
                        self.data[row["key"]] = row["text"]
                        self.extra[row["key"]] = {k: v for k, v in row.items() if k not in ("key", "text")}

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __getitem__(self, key: str) -> str:
        return self.data[key]

    def __len__(self) -> int:
        return len(self.data)

    def add_many(self, rows: list[tuple[str, str, dict]]) -> None:
        # One O_APPEND write per batch, so several shard processes can append to the same file without
        # interleaving partial lines (a buffered writer may split a line across two write calls).
        payload = "".join(json.dumps({"key": key, "text": text, **extra}, ensure_ascii=False) + "\n"
                          for key, text, extra in rows).encode("utf-8")
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o664)
        try:
            view = memoryview(payload)
            while view:
                view = view[os.write(fd, view):]
        finally:
            os.close(fd)
        for key, text, extra in rows:
            self.data[key] = text
            self.extra[key] = extra
