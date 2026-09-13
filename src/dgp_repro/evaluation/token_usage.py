"""Token accounting for DGP prompts: measured, not estimated.

Per target node (rows produced by dgp_repro.pipeline):
    raw_target_tokens            tokens of x_v^text
    raw_neighbor_tokens          raw text tokens over the FULL metapath neighbourhoods, summed over
                                 metapaths - what a full-neighbour text-only prompt would carry
    trimmed_neighbor_raw_tokens  raw text tokens of the M trimmed neighbours only
    node_summary_tokens          tokens of the node summaries s_u that feed metapath summarization
    metapath_summary_tokens      tokens of the S_P(v) placed in the prompt
    final_prompt_tokens          tokens of the final prompt (before the chat template)

Aggregates:
    average_prompt_length = mean(final_prompt_tokens)
    compression_ratio     = mean(raw_target + raw_neighbor) / mean(final_prompt)
    token_reduction       = 1 - mean(final_prompt) / mean(raw_target + raw_neighbor)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

COUNT_COLUMNS = ("raw_target_tokens", "raw_neighbor_tokens", "trimmed_neighbor_raw_tokens", "node_summary_tokens",
                 "metapath_summary_tokens", "final_prompt_tokens", "num_neighbors_full", "num_neighbors_trimmed")


def summarize_token_usage(rows: list[dict]) -> dict:
    if not rows:
        return {}
    col = {c: np.array([r[c] for r in rows], dtype=np.float64) for c in COUNT_COLUMNS}
    full_neighbor_prompt = col["raw_target_tokens"] + col["raw_neighbor_tokens"]
    out = {f"mean_{c}": float(col[c].mean()) for c in COUNT_COLUMNS}
    out.update({
        "nodes": len(rows),
        "average_prompt_length": float(col["final_prompt_tokens"].mean()),
        "p95_prompt_length": float(np.percentile(col["final_prompt_tokens"], 95)),
        "max_prompt_length": float(col["final_prompt_tokens"].max()),
        "mean_full_neighbor_prompt_tokens": float(full_neighbor_prompt.mean()),
        "compression_ratio": float(full_neighbor_prompt.mean() / col["final_prompt_tokens"].mean()),
        "token_reduction": float(1 - col["final_prompt_tokens"].mean() / full_neighbor_prompt.mean()),
        "targets_truncated": int(sum(bool(r.get("target_truncated")) for r in rows)),
    })
    for key in ("num_metapaths", "K", "M", "B_node", "B_meta"):
        out[key] = rows[0][key]
    return out


def write_token_usage(rows: list[dict], summary: dict, out_dir: str | Path, tag: str = "") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    with open(out_dir / f"token_usage{suffix}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / f"token_usage{suffix}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
