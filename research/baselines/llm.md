# LLM (Qwen3-8B finetuned on target nodes only)

This is DGP's "graph-agnostic LLM" row: the base LLM is LoRA-finetuned on the **target node's own text only**, with no neighbourhood/graph context. It is the ablation that isolates how much of DGP's gain comes from the graph.

- **paper**: Qwen3 Technical Report (cited by DGP as "Team 2025")
- **authors**: Qwen Team, Alibaba Group
- **venue**: arXiv / technical report, 2025
- **official_repository**:
  - Model weights: **https://huggingface.co/Qwen/Qwen3-8B** (verified)
  - Code: https://github.com/QwenLM/Qwen3
  - Finetuning is done with standard tooling, not a Qwen-specific repo — HuggingFace `peft` + `transformers`, or LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory).
- **selected_commit**: Pin the HuggingFace revision of `Qwen/Qwen3-8B` (record the commit hash from the model card) rather than a git SHA.
- **framework**: PyTorch + HuggingFace `transformers` + `peft` (LoRA)
- **requirements** (verified against the Qwen3-8B model card):
  ```
  transformers>=4.51.0     # HARD REQUIREMENT: with transformers<4.51.0 you get `KeyError: 'qwen3'`
  torch>=2.1 (2.4+ recommended for the A100/bf16 path)
  peft, accelerate, datasets, bitsandbytes (optional), deepspeed (optional)
  ```
  Model facts: 8.2B total params (6.95B non-embedding), 36 layers, GQA 32 Q-heads / 8 KV-heads, **32,768 native context** (131,072 with YaRN).
- **dataset_support**: N/A — text-in/label-out. You supply a JSONL of `{target node text -> fraud/benign}` instruction pairs built from DGP's YelpReviews / AmazonVideo splits.
- **training_entrypoint**: Custom, e.g. a `peft` SFT script or
  `llamafactory-cli train --model_name_or_path Qwen/Qwen3-8B --finetuning_type lora --lora_target all ...`
- **evaluation_entrypoint**: Generate a label token per test node, parse to binary, then compute AUC / F1-macro / AUPRC. For AUC/AUPRC you need a **score**, not just a label — take the softmax probability of the "fraud" vs "benign" answer token. DGP does not describe this step; record whatever convention you adopt.
- **adaptation_required** (MEDIUM):
  1. Build the instruction dataset from DGP's own prompt template for the target node (see the DGP paper's prompt appendix / Subagent A's extraction).
  2. **Thinking mode**: Qwen3 is a hybrid reasoning model with `enable_thinking` in its chat template. DGP does not say whether thinking was on. Default `enable_thinking=True` will emit `<think>...</think>` blocks that break naive label parsing. Recommend `enable_thinking=False` for classification and document it.
  3. DGP states: "We apply LoRA (Hu et al. 2022) to all attention layers and use AdamW". Match that — LoRA on q/k/v/o projections, not just q/v.
  4. Score extraction convention (above) must be fixed and applied identically to DGP itself and to GraphGPT/HiGPT/InstructGLM, or the AUC column is not comparable.
- **deviations**: LoRA rank/alpha/dropout, LR, epochs, and max sequence length are not given in the DGP paper (only "grid search on the validation set"). Thinking-mode and score-extraction conventions are ours.
- **license**: Qwen3-8B weights are **Apache-2.0**.
- **reproducibility_status**: **GOOD** — weights are openly available and the recipe is standard. Risk is prompt/decoding convention drift, not availability.

## Critical cross-cutting note
DGP states: *"For all LLM-tuning methods, we use the Qwen3-8B LLM backbone (Team 2025) for fair comparison."* That claim is in direct tension with "All baselines are implemented using official code" for GraphGPT / HiGPT / InstructGLM, all three of which hard-code a LLaMA-family backbone. See `graphgpt.md`, `higpt.md`, `instructglm.md`.
