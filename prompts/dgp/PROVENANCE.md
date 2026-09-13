# DGP prompt templates — provenance

Version: see `VERSION` (currently `v1-reconstruction`). Any change to a template must bump
`VERSION`; the version is part of every summary/prompt cache key, so stale caches are never reused.

The official DGP code is unreleased, so **no canonical template exists**. The table marks
exactly which characters come from the paper.

| File | Verbatim from paper | Reconstructed |
|---|---|---|
| `node_summary.txt` | `Summarize the text within 10 tokens.` (both versions, "Impact of Task-Aware Summarization"), with `10` generalised to `{budget}` for the B sweep | Placing the review text after a blank line |
| `node_summary_task_aware.txt` | `Summarize the text within 10 tokens, focusing on signals indicative of fraudulent behavior.` | Same layout |
| `metapath_summary.txt` | Nothing. The paper publishes no metapath-level prompt | Reuses the task-agnostic instruction on the concatenated neighbour summaries, consistent with Eq. 10 using the same `Summarize(·; B)` operator as Eq. 5 |
| `metapath_summary_task_aware.txt` | Instruction text as above | Used only if `task_aware.apply_to_metapath: true`. Table 3 is captioned "Impact of **node-level** summarization prompts", so the default task-aware run changes only the node-level prompt |
| `final_prompt.txt` | The three field labels `Target:`, `Metapaths:`, `Question: Is this fraud?` from Figure 3 | Line layout, and the per-metapath block format built in `src/dgp_repro/prompts/builder.py` |

## Metapath block format (reconstruction)

```
- RTR-RSR: <metapath summary S_P(v)> (neighbor mean: rating=1.67)
```

- Metapath names use the Figure 3 acronyms (`RUR`, `RTR-RSR`).
- `a_P(v)` is written with fixed precision (`numeric_precision`, default 2). Figure 3 shows one
  decimal (`r_mean=1.7`); two decimals keep more of the "precise signal" the paper argues for.
- A metapath with no neighbours is rendered as `- RSR: no neighbors`.

## Chat formatting

Both the summarizer and the classifier wrap the filled template as a single user message and
render it with the Qwen3 chat template using `enable_thinking=False`. This is required, not
optional: with thinking enabled, Qwen3's first generated token opens a `<think>` block, so
the paper's first-token Yes/No readout could not work (`research/llm_provenance.md` §1.7).
No system message is used; the paper mentions none.
