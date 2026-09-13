# Troubleshooting

Each entry here was hit during development or is a verified property of the model stack.

### `AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'`
An old `torch` imported a newer `transformers` (typically through a venv created with
`--system-site-packages`). Create a fresh isolated environment (`docs/installation.md`).

### Classification AUROC ≈ 50 and probabilities stuck near 0.5 with Qwen3
Almost certainly thinking mode: without `enable_thinking=False` the first generated token is a `<think>` block,
not Yes/No. The code always renders prompts with `render_chat_prompt`, which disables it. If you changed the
template or the chat call, rerun `python scripts/smoke_test.py --real-tokenizer`: it checks that the rendered
prompt ends in `<think>\n\n</think>\n\n` and that `Yes`/`No` are single tokens at that boundary.

### `LabelTokenizationError: label 'Yes' is 2 tokens ...`
The tokenizer splits the label, or the prompt ends with a space so the leading-space variant (`" Yes"`,
id 7414 for Qwen3) would be generated instead of `Yes` (9454). Do not hard-code ids; fix the template spacing.

### DeBERTa embedding is extremely slow, or produces NaNs
transformers ≥ 5 loads a checkpoint in its **stored** dtype unless told otherwise, and `microsoft/deberta-v3-base`
is stored in float16. Measured on CPU: 84.6 s per batch of 64 median-length reviews in float16 versus 9.1 s in
float32. On GPU, float16 DeBERTaV3 can overflow in its disentangled attention. The embedder now always loads
float32 (`embeddings/text.py`). If you load any model yourself, pass `dtype=` explicitly.

### A MDK-only command starts downloading 16 GB
Fixed: stages construct only the models they use (`tests/unit/test_stage_loading.py`). If it happens again,
stop it and delete `*.incomplete` files under `~/.cache/huggingface/hub/models--Qwen--Qwen3-8B`.

### `MemoryError` / `_ArrayMemoryError` while loading MDK caches
Fixed: `NpzFile` re-reads an array on every `z[key]` access, so indexing inside a loop multiplied memory.
Arrays are now read once (`pipeline._load_trim`).

### `prepare_data.py` refuses to save: "MISMATCH against paper Table 1"
The graph does not reproduce the paper's statistics. For AmazonVideo check the downloaded file's checksum and
that `rsr_product_scope: false`. For YelpReviews use `--probe` to compare relation definitions.
`--allow-mismatch` saves anyway; results from such a graph must be reported as deviating.

### CUDA out of memory during training
In order: keep `gradient_checkpointing: true`; lower `training.eval_batch_size`; lower `dgp.max_target_tokens`;
use `training.batch_size: 2` with `gradient_accumulation: 2` (same effective batch of 4). The 4-bit
`classifier.precision: memory_optimized` mode also works but is **not canonical**: it is recorded in the run
manifest and must be reported as a deviation.

### A rerun did nothing
Finished seeds are skipped when `results/raw/<run_id>/metrics.json` exists, and caches are reused. The run id
contains a hash of the whole config, so any config change produces a new run.

### Summaries longer than B
Expected: the budget is an instruction, as in the paper's prompt. Realised lengths are recorded in the
summary caches and `results/token_usage/`. `summarizer.enforce_budget: true` hard-caps generation at B tokens
(a deviation to report).
