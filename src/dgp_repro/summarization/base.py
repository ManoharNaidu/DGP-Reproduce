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

    def complete(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        """DEBUG_ONLY answer to a FLAG subgraph prompt (summarization/flag_subgraph.py): one numbered line per
        listed node, made of that node's first words; a prompt containing 'MOCK_FAIL' gets one line too few."""
        import re
        out = []
        for prompt in prompts:
            items = re.findall(r"^\d+\. \[(.*)\]$", prompt.split("Question:", 1)[-1], flags=re.M)
            lines = [f"{i}. {' '.join(t.split()[:5]) or 'empty'}" for i, t in enumerate(items, 1)]
            if "MOCK_FAIL" in prompt:
                lines = lines[:-1]
            out.append("\n".join(lines))
        return out


class CachedOnlySummarizer:
    """Reads summaries another run already wrote; never loads or calls a model.

    `name` is the writer's model id, so cache keys match the run that generated the text (cache keys hold the
    summarizer name, not its loading options). Any cache miss raises, so a run using it is guaranteed to see exactly
    the cached text. Used by the writer/classifier split (gs_X0: Gemma-written text, Qwen classifier)."""

    def __init__(self, cfg: dict):
        if not cfg.get("model_id"):
            raise ValueError("summarizer.backend=cached needs summarizer.model_id (the model that wrote the cache)")
        self.name = cfg["model_id"]

    def summarize(self, prompts: list[str], budget: int) -> list[str]:
        raise RuntimeError(f"summarizer.backend=cached: {len(prompts)} summaries by {self.name} are not in the "
                           "cache; generate them with the real summarizer first")

    def complete(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        raise RuntimeError(f"summarizer.backend=cached: {len(prompts)} completions by {self.name} are not cached")


def make_summarizer(cfg: dict, device: str = "cpu") -> Summarizer:
    kind = cfg.get("backend", "qwen")
    if kind == "mock":
        return MockSummarizer()
    if kind == "cached":
        return CachedOnlySummarizer(cfg)
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
