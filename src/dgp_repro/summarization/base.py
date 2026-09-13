"""Summarizer interface: Summarize(text; B) from paper Eq. 5 and Eq. 10.

Implementations:
    MockSummarizer   DEBUG_ONLY - keeps the first B words. For CPU smoke tests only.
    QwenSummarizer   frozen Qwen3-8B (summarization/qwen.py), as in the paper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from dgp_repro.config import REPO_ROOT

PROMPT_DIR = REPO_ROOT / "prompts" / "dgp"


def prompt_version() -> str:
    return (PROMPT_DIR / "VERSION").read_text(encoding="utf-8").strip()


def load_template(name: str) -> str:
    return (PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8")


def fill_summary_template(template: str, text: str, budget: int) -> str:
    return template.replace("{budget}", str(budget)).replace("{text}", text)


class Summarizer(Protocol):
    name: str

    def summarize(self, prompts: list[str], budget: int) -> list[str]:
        """Summaries for already-filled instruction prompts. `budget` is B in tokens."""
        ...


class MockSummarizer:
    """DEBUG_ONLY. Returns the first `budget` words of the text part of each prompt."""

    name = "mock-first-words"

    def summarize(self, prompts: list[str], budget: int) -> list[str]:
        out = []
        for prompt in prompts:
            text = prompt.split("\n\n", 1)[-1]
            out.append(" ".join(text.split()[:budget]))
        return out


def make_summarizer(cfg: dict, device: str = "cpu") -> Summarizer:
    kind = cfg.get("backend", "qwen")
    if kind == "mock":
        return MockSummarizer()
    if kind == "qwen":
        from dgp_repro.summarization.qwen import QwenSummarizer
        return QwenSummarizer(cfg, device)
    raise ValueError(f"unknown summarizer backend {kind!r}")


def template_names(task_aware: bool, apply_to_metapath: bool) -> tuple[str, str]:
    node = "node_summary_task_aware" if task_aware else "node_summary"
    meta = "metapath_summary_task_aware" if task_aware and apply_to_metapath else "metapath_summary"
    return node, meta


__all__ = ["Summarizer", "MockSummarizer", "make_summarizer", "load_template", "fill_summary_template",
           "prompt_version", "template_names", "PROMPT_DIR", "Path"]
