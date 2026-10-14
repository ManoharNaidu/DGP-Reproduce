# Extension templates (discriminative / residual text) - NOT from the DGP paper
`node_discriminative.txt`, `node_residual.txt`: adapted from FLAG's causal / non-causal prompts (BUPT-GAMMA/FLAG chat.py)
to a per-node, single-line, B-token setting for the AmazonVideo helpful/unhelpful task. The label is never given to the LLM.
`final_prompt_disc*.txt`, `final_prompt_resid.txt`: baseline final prompt plus a `Key signals:` / `Background:` field.
`residual_only_prompt.txt`: input of the residual auxiliary-loss branch (arm E).
These do not change `VERSION`; template names already enter every cache key, so baseline caches stay valid.
