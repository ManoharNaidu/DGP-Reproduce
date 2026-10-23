"""FLAG-architecture discriminative / residual text for DGP (extension; NOT from the DGP paper).

FLAG generates its LLM text after sampling, once per sampled subgraph: one prompt lists every node of the subgraph,
and the answer must contain exactly one line per node (`FLAG-Reproduce/methods/flag/chat.py:60-90`). Here the
"sampled subgraph" of a target is the target plus its MDK-trimmed neighbours, and the prompt builder and parser are
line-for-line ports of `FLAG-Reproduce/src/flagbench/llm/enhance.py` (question_block, build_prompt, parse_response,
strip_numbering). See prompts/dgp/PROVENANCE_FA.md.

A neighbour's line depends on the target whose subgraph it was written in, so text is keyed by (target, node).
A subgraph whose answer has the wrong number of lines fails as a whole; its nodes keep their raw text (FLAG's
fallback), and the subgraph is excluded from the residual phase (FLAG `finetune_extra`).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from dgp_repro.summarization.base import PROMPT_DIR
from dgp_repro.utils.cache import TextStore

ANSWER_SEPARATOR = "Answer:"
TEXT_TRUNCATE_CHARS = 1200
NOUN = "reviews"          # FLAG's noun for review-level nodes (scripts/llm/generate_text.py DATASET_NOUN)
KINDS = ("discriminative", "residual")


@dataclass
class FlagPrompts:
    system_instruction: str
    global_prompt: str
    discriminative: str
    residual: str
    hashes: dict = field(default_factory=dict)

    @classmethod
    def load(cls, name: str = "fa") -> "FlagPrompts":
        directory = Path(PROMPT_DIR) / name
        parts, hashes = {}, {}
        for role in ("system_instruction", "global", "discriminative", "residual"):
            text = (directory / f"{role}.txt").read_text(encoding="utf-8")
            parts[role] = text
            hashes[role] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return cls(parts["system_instruction"], parts["global"], parts["discriminative"], parts["residual"], hashes)

    def prompt_for(self, kind: str) -> str:
        if kind not in KINDS:
            raise ValueError(f"unknown prompt kind {kind!r}")
        return self.discriminative if kind == "discriminative" else self.residual


def question_block(texts: list[str], truncate: int = TEXT_TRUNCATE_CHARS, noun: str = NOUN) -> str:
    """Numbered list of node texts, exactly as FLAG builds it (enhance.py:question_block)."""
    lines = [f"The {noun} of these users are as follows:"]
    for index, text in enumerate(texts, start=1):
        clipped = text[:truncate] if len(text) > truncate else text
        lines.append(f"{index}. [{clipped}]")
    return "\n".join(lines) + "\n"


def build_prompt(prompts: FlagPrompts, kind: str, texts: list[str], truncate: int = TEXT_TRUNCATE_CHARS,
                 noun: str = NOUN) -> str:
    """FLAG's prompt layout (enhance.py:build_prompt; upstream chat.py:67)."""
    question = question_block(texts, truncate, noun)
    return (f"{prompts.system_instruction}\n"
            f"Prompt: {prompts.global_prompt}\n"
            f"{prompts.prompt_for(kind)}\n"
            f"Question: {question}\n"
            f"{ANSWER_SEPARATOR}")


def parse_response(answer: str, expected: int) -> list[str] | None:
    """One line per node or None (enhance.py:parse_response)."""
    if ANSWER_SEPARATOR in answer:
        answer = answer.split(ANSWER_SEPARATOR, 1)[1]
    lines = [line.strip() for line in answer.split("\n") if line.strip()]
    if len(lines) != expected:
        return None
    return lines


def strip_numbering(lines: list[str]) -> list[str]:
    """Remove a leading '1. ' / '12) ' enumeration (enhance.py:strip_numbering)."""
    pattern = re.compile(r"^\s*\d+\s*[.)]\s*")
    return [pattern.sub("", line).strip() for line in lines]


def subgraph_nodes(target: int, trims: dict) -> list[int]:
    """Target first, then trimmed neighbours in metapath order and rank order, each node once (FLAG's subset order)."""
    order, seen = [int(target)], {int(target)}
    for t in trims.values():
        for u in t["neighbors"].get(int(target), []):
            u = int(u)
            if u not in seen:
                seen.add(u)
                order.append(u)
    return order


def generate(summarizer, prompts: list[str], max_new_tokens: int, batch_size: int, max_input_tokens: int):
    """Greedy generation with the frozen Qwen3-8B summarizer's model, FLAG's token budget and no input truncation.

    Prompts longer than `max_input_tokens` are not sent (counted as failures, as FLAG's vLLM path does for
    context overflows). Returns (answers, generated token counts, prompt token counts); None for skipped prompts.
    """
    if not hasattr(summarizer, "model"):          # DEBUG_ONLY mock summarizer (CPU smoke tests)
        answers = summarizer.complete(prompts, max_new_tokens)
        return answers, [len(a.split()) for a in answers], [len(p.split()) for p in prompts]
    import torch

    tok, model = summarizer.tokenizer, summarizer.model
    rendered = [summarizer._render(p) for p in prompts]
    lengths = [len(tok(r, add_special_tokens=False)["input_ids"]) for r in rendered]
    answers, generated = [None] * len(prompts), [None] * len(prompts)
    runnable = sorted((i for i, n in enumerate(lengths) if n <= max_input_tokens), key=lambda i: lengths[i])
    special = None
    for start in range(0, len(runnable), batch_size):
        idx = runnable[start:start + batch_size]
        enc = tok([rendered[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(summarizer.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        if special is None:
            special = torch.tensor(tok.all_special_ids, device=new.device)
        for i, row in zip(idx, new):
            row = row[~torch.isin(row, special)]
            generated[i] = int(len(row))
            answers[i] = tok.decode(row, skip_special_tokens=True)
    return answers, generated, lengths


@dataclass
class SubgraphText:
    nodes: list[int]
    lines: list[str] | None          # None = format failure (or context overflow)

    @property
    def ok(self) -> bool:
        return self.lines is not None

    def text_for(self, node: int, raw_texts: list[str]) -> str:
        """The node's line in this subgraph; its raw text when the subgraph failed (FLAG's fallback)."""
        if self.lines is None:
            return raw_texts[node]
        return self.lines[self.nodes.index(node)]


def subgraph_texts(raw_texts: list[str], targets: list[int], trims: dict, summarizer, prompts: FlagPrompts, kind: str,
                   store: TextStore, *, max_new_tokens: int = 550, truncate: int = TEXT_TRUNCATE_CHARS,
                   batch_size: int = 8, max_input_tokens: int = 32768, chunk: int = 64,
                   log=print) -> dict[int, SubgraphText]:
    """Generate (resumably) one FLAG prompt per target subgraph and parse it. Raw answers are stored, so parsing
    can be audited later; the store key is the target id."""
    plan = {int(v): subgraph_nodes(v, trims) for v in targets}
    todo = [v for v in plan if str(v) not in store]
    log(f"FA {kind}: {len(plan)} subgraphs, {len(plan) - len(todo)} cached, {len(todo)} to generate")
    for start in range(0, len(todo), chunk):
        batch = todo[start:start + chunk]
        texts = [build_prompt(prompts, kind, [raw_texts[u] for u in plan[v]], truncate) for v in batch]
        answers, gen, plen = generate(summarizer, texts, max_new_tokens, batch_size, max_input_tokens)
        rows = []
        for v, a, g, n in zip(batch, answers, gen, plen):
            lines = parse_response(a, len(plan[v])) if a is not None else None
            rows.append((str(v), a if a is not None else "", {
                "nodes": plan[v], "ok": lines is not None, "overflow": a is None, "prompt_tokens": n,
                "generated_tokens": g, "hit_cap": g is not None and g >= max_new_tokens}))
        store.add_many(rows)
        log(f"FA {kind}: {min(start + chunk, len(todo))}/{len(todo)} generated, "
            f"ok so far {sum(1 for k in store.extra.values() if k.get('ok'))}/{len(store)}")
    out = {}
    for v, nodes in plan.items():
        extra = store.extra.get(str(v), {})
        lines = None
        if extra.get("ok"):
            parsed = parse_response(store[str(v)], len(nodes))
            lines = strip_numbering(parsed) if parsed is not None else None
        out[v] = SubgraphText(nodes, lines)
    return out


def store_metadata(dataset: str, summarizer_name: str, prompts: FlagPrompts, kind: str, max_new_tokens: int,
                   truncate: int, mdk_cache: str, targets_hash: str, prompt_set: str = "fa") -> dict:
    return {"dataset": dataset, "summarizer": summarizer_name, "prompt_set": prompt_set,
            "prompt_hashes": prompts.hashes, "kind": kind, "max_new_tokens": max_new_tokens,
            "truncate_chars": truncate, "chat_template": "qwen3 enable_thinking=False", "mdk_cache": mdk_cache,
            "targets_hash": targets_hash, "unit": "target subgraph (target + MDK-trimmed neighbours)"}


def coverage(results: dict[int, SubgraphText]) -> dict:
    """FLAG-style generation statistics: subgraph success rate and node coverage."""
    total = len(results)
    ok = sum(r.ok for r in results.values())
    nodes = sum(len(r.nodes) for r in results.values())
    covered = sum(len(r.nodes) for r in results.values() if r.ok)
    return {"subgraphs": total, "succeeded": ok, "subgraph_success_rate": ok / total if total else None,
            "nodes": nodes, "node_coverage": covered / nodes if nodes else None}


__all__ = ["FlagPrompts", "question_block", "build_prompt", "parse_response", "strip_numbering", "subgraph_nodes",
           "generate", "SubgraphText", "subgraph_texts", "store_metadata", "coverage", "KINDS", "json"]
