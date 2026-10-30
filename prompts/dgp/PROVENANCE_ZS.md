# Structured zero-shot classification prompts (`zs_*` templates) — NOT from the DGP paper

Added 2026-10-29 on branch `gemma-swap` for the zero-shot-prompt experiments (Exp 2: Qwen3-8B, Exp 3: Gemma-2-9B).

- Source of the layout: `experiments/prompt_testing/prompts/structured_zero_shot_prompt.txt`, the best Gemma-2-9B
  prompt in the 2026-10-20 prompt-testing run once the few-shot prompt is excluded (F1 0.338 vs baseline 0.203;
  not significant, McNemar p = 0.115; `experiments/prompt_testing/results/REPORT.md`). Sections kept: `<task>`,
  `<definition>`, the input, `<constraints>`, `<output_format>`.
- What replaces what: these templates replace DGP's FINAL classification prompt only (`final_prompt.txt`,
  `dr_final_disc.txt`) and the residual branch's prompt (`dr_residual_only.txt`). Every field they carry is filled by the
  same code with the same text as before (target review, `Key signals:` line, the metapath lines with neighbour mean
  ratings); only layout and instructions differ. Neighbour summaries and D/R lines are not regenerated.
- Forced deviation from the prototype: DGP reads the first answer token, so the output format asks for `Yes` / `No`
  instead of `<label>truthful</label>` / `<label>deceptive</label>`.
- Content change (deliberate, approved in the plan): the task and definition state that "fraud" means a review other
  shoppers judge unhelpful. The original prompt only asks "Is this fraud?". Labels are never shown.
- `zs_final.txt` (arm 0), `zs_final_disc.txt` (arms 1, 1b, 2, 2b: adds `<key_signals>`), `zs_residual_only.txt`
  (arms 2, 2b residual branch: the target's residual line in place of the review, no neighbourhood).
- `VERSION` is NOT bumped: it keys the summary caches, which these experiments reuse unchanged.
