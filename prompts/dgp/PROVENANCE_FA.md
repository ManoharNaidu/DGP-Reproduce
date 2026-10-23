# FLAG-architecture study (`fa/` templates) — NOT from the DGP paper

Started 2026-10-23 on branch `flag-arch`. Independent of the extension study (`PROVENANCE_EXT.md`) and the dr study
(`PROVENANCE_DR.md`); no template, cache or result from them is reused.

What is different from the dr study: D and R are generated the way FLAG generates them, **after sampling and once per
sampled subgraph**, not once per review.

- Subgraph of target v = v first, then its MDK-trimmed neighbours (Top-M per metapath) in metapath order and rank
  order, each review once. This mirrors FLAG's subgraph `subset` (centre first, then neighbours in selection order,
  no duplicates; `FLAG-Reproduce/src/flagbench/sampling/semantic.py:333-364`).
- One prompt per subgraph and per kind, in FLAG's layout (`FLAG-Reproduce/methods/flag/chat.py:67`,
  `src/flagbench/llm/enhance.py:build_prompt`):
  `{system_instruction}\nPrompt: {global}\n{discriminative|residual}\nQuestion: The reviews of these users are as follows:\n1. [text[:1200]]\n...\nAnswer:`
- Parsing as FLAG (`enhance.py:parse_response`, `strip_numbering`): split at `Answer:` if present, keep non-empty
  lines, require exactly one line per node; otherwise the whole subgraph fails and its nodes keep their raw text.
- Decoding as FLAG: greedy, `max_new_tokens` 550, review text cut to 1,200 characters, one attempt.

Files in `fa/`:
- `system_instruction.txt`, `discriminative.txt`, `residual.txt`: copied verbatim from
  `FLAG-Reproduce/prompts/yelpchi_text/` (FLAG's prompts adapted to review-level nodes), with the class names
  `fraudulent or legitimate` replaced by `unhelpful or helpful` ("fraud" = unhelpful review on AmazonVideo; the
  positive class stays first, as in the YelpChi wording).
- `global.txt`: FLAG's dataset sentence, rewritten for AmazonVideo.

Deviations forced by the setting (recorded, not hidden):
1. The LLM is Qwen3-8B (DGP's own summariser, user decision 2026-10-23), not Gemma-2-9b-it. Qwen3 needs its chat
   template with `enable_thinking=False` (otherwise it opens a `<think>` block), so the FLAG prompt string is sent as
   one user message instead of as raw text.
2. Labels are never shown to the LLM; the class names are in the instruction, as in FLAG.
