# Discriminative / residual study (`dr_*` templates) — NOT from the DGP paper

Started 2026-10-15. Independent of the earlier extension study (`PROVENANCE_EXT.md`); no template, cache or result
from it is reused.

- `dr_discriminative.txt`, `dr_residual.txt`: FLAG's causal (`unique_prompt`) and non-causal (`common_prompt`) prompts
  from BUPT-GAMMA/FLAG `chat.py` @ cb83944e, kept in FLAG's layout
  (`{system_instruction}\nPrompt: {global_prompt}\n{task prompt}\nQuestion: ...\nAnswer:`).
  Adapted: Reddit users / popular-normal → one AmazonVideo review / helpful-unhelpful ("fraud" = unhelpful in this
  dataset); the numbered multi-user list → a single review (DGP generates per node); FLAG's typo "notindicative" fixed.
  The class names are in the prompt (as in FLAG); the label never is. The input review is cut to 1,200 characters
  (FLAG's `[:1200]`), via `target_text.<kind>.truncate_chars`.
- `dr_final_disc.txt`: the baseline final prompt plus `Key signals: {target_disc}`.
- `dr_residual_only.txt`: input of the residual branch (R2).
- `dr_discriminative_nv.txt` ("no verdict", runs R1b/R2b): `dr_discriminative.txt` plus one sentence, "Describe the
  cues only; do not state or predict whether the review is helpful or unhelpful." Added after the pilot showed that
  FLAG's prompt makes the causal line a class verdict in 188/200 reviews (65% accurate on a 50/50 sample). Both
  versions are run.
