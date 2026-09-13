"""Bi-level textual summarization with resumable caching.

    node level      s_v    = Summarize(x_v^text ; B_node)                  (Eq. 5)
    metapath level  S_P(v) = Summarize(concat of s_u, u in N~_P(v) ; B_meta)  (Eq. 10)

Summaries are generated only for nodes that are actually used: every trimmed neighbour of
every target. Target nodes keep their raw text (fine granularity) and are never summarized.

Leakage guard: the summarizer only ever sees review text. Labels are never read here, and
the task-agnostic template contains no fraud/spam/helpfulness wording.
"""

from __future__ import annotations

import hashlib
from typing import Callable

from dgp_repro.summarization.base import Summarizer, fill_summary_template
from dgp_repro.utils.cache import TextStore

NEIGHBOR_SEPARATOR = "\n"


def _batched_fill(store: TextStore, items: list[tuple[str, str]], summarizer: Summarizer, template: str,
                  budget: int, batch_size: int, progress: Callable[[int, int], None] | None) -> None:
    """Summarize (key, text) items missing from the store, appending results as they finish."""
    todo = [(k, t) for k, t in dict(items).items() if k not in store]
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        prompts = [fill_summary_template(template, text, budget) for _, text in batch]
        summaries = summarizer.summarize(prompts, budget)
        lengths = getattr(summarizer, "last_generated_lengths", None) or [None] * len(batch)
        store.add_many([(k, s, {"generated_tokens": n}) for (k, _), s, n in zip(batch, summaries, lengths)])
        if progress:
            progress(min(start + batch_size, len(todo)), len(todo))


def node_summaries(texts: list[str], node_ids: list[int], summarizer: Summarizer, template: str, budget: int,
                   store: TextStore, batch_size: int = 64,
                   progress: Callable[[int, int], None] | None = None) -> dict[int, str]:
    items = [(str(u), texts[u]) for u in sorted(set(node_ids))]
    _batched_fill(store, items, summarizer, template, budget, batch_size, progress)
    return {u: store[str(u)] for u in set(node_ids)}


def metapath_key(target: int, metapath: str, neighbors: list[int]) -> str:
    digest = hashlib.sha1(",".join(map(str, neighbors)).encode()).hexdigest()[:12]
    return f"{target}|{metapath}|{digest}"


def join_neighbor_texts(neighbors: list[int], per_node_text: dict[int, str]) -> str:
    return NEIGHBOR_SEPARATOR.join(per_node_text[u] for u in neighbors)


def metapath_summaries(requests: list[tuple[int, str, list[int]]], per_node_text: dict[int, str],
                       summarizer: Summarizer, template: str, budget: int, store: TextStore,
                       batch_size: int = 32,
                       progress: Callable[[int, int], None] | None = None) -> dict[tuple[int, str], str]:
    """requests: (target, metapath name, trimmed neighbours). Empty neighbourhoods are skipped."""
    items = [(metapath_key(v, name, nbrs), join_neighbor_texts(nbrs, per_node_text))
             for v, name, nbrs in requests if nbrs]
    _batched_fill(store, items, summarizer, template, budget, batch_size, progress)
    return {(v, name): store[metapath_key(v, name, nbrs)] for v, name, nbrs in requests if nbrs}
