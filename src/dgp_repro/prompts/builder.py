"""Final DGP prompt (paper Eq. 12).

    prompt(v) = x_v^text  concat  [ concat over P in P_K of ( S_P(v) concat a_P(v) ) ]

Provenance: PAPER_RECONSTRUCTION of the layout; field labels from Figure 3.
See prompts/dgp/PROVENANCE.md.

The target node always contributes its raw text (fine-grained). Each metapath contributes a
coarse block holding its text summary and/or its numeric mean, depending on the component
switches, so the paper's ablations are produced by simply leaving a field empty:
    w/o TextSumm  -> text=None for every block
    w/o NumSumm   -> numeric=None for every block
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dgp_repro.summarization.base import PROMPT_DIR

METAPATH_SECTION = "Metapaths:\n{metapath_blocks}\n"


@dataclass
class MetapathBlock:
    name: str                          # e.g. "RTR-RSR"
    num_neighbors: int                 # |N~_P(v)| after trimming
    text: str | None = None            # S_P(v), or the joined node summaries if PathSumm is off
    numeric: np.ndarray | None = None  # a_P(v)


def load_final_template() -> str:
    return (PROMPT_DIR / "final_prompt.txt").read_text(encoding="utf-8")


def format_numeric(values: np.ndarray, names: list[str], precision: int = 2) -> str:
    return ", ".join(f"{name}={value:.{precision}f}" for name, value in zip(names, values))


def format_block(block: MetapathBlock, numeric_names: list[str], precision: int = 2) -> str | None:
    """One line per metapath. Returns None when the block has nothing to show."""
    if block.num_neighbors == 0:
        return f"- {block.name}: no neighbors"
    parts = []
    if block.text is not None:
        parts.append(block.text.strip())
    if block.numeric is not None:
        numeric = f"neighbor mean: {format_numeric(block.numeric, numeric_names, precision)}"
        parts.append(f"({numeric})" if block.text is not None else numeric)
    if not parts:
        return None
    return f"- {block.name}: " + " ".join(parts)


def build_prompt(target_text: str, blocks: list[MetapathBlock], numeric_names: list[str], question: str,
                 template: str | None = None, precision: int = 2) -> str:
    template = template or load_final_template()
    lines = [line for line in (format_block(b, numeric_names, precision) for b in blocks) if line is not None]
    if lines:
        body = template.replace("{metapath_blocks}", "\n".join(lines))
    else:
        body = template.replace(METAPATH_SECTION, "")  # target-only prompt (the "LLM" baseline)
    return body.replace("{target_text}", target_text.strip()).replace("{question}", question)
