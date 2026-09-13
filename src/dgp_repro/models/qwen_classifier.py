"""Qwen3-8B + LoRA fraud classifier.

Paper: "we use the Qwen3-8B LLM backbone ... We apply LoRA to all attention layers and use
AdamW optimizer for finetuning." (camera-ready + arXiv v1)

Verified facts (research/llm_provenance.md):
    model id           Qwen/Qwen3-8B (post-trained; not gated; Apache-2.0)
    attention Linear   q_proj, k_proj, v_proj, o_proj  (q_norm/k_norm are RMSNorm, not LoRA targets)
Not specified by the paper, reconstructed and set in configs/models/dgp.yaml:
    LoRA alpha, max sequence length, precision.

Precision modes (config `precision`):
    canonical         bf16 weights, no quantisation (default)
    memory_optimized  4-bit NF4 base weights (QLoRA). A REPRODUCTION OPTIMIZATION, recorded in
                      the run manifest; never used silently for canonical results.
"""

from __future__ import annotations

from pathlib import Path

import torch

from dgp_repro.models.label_tokens import LabelTokens, resolve_label_token_ids


def render_chat_prompt(tokenizer, prompt: str) -> str:
    """Single user turn, generation prompt appended, thinking disabled (required for first-token readout)."""
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def truncate_to_tokens(tokenizer, text: str, max_tokens: int | None) -> tuple[str, bool]:
    """Keep the first `max_tokens` tokens of a text. Returns (text, was_truncated)."""
    if not max_tokens:
        return text, False
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text, False
    return tokenizer.decode(ids[:max_tokens]), True


class QwenFraudClassifier:
    def __init__(self, cfg: dict, device: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.device = device
        self.name = cfg.get("model_id", "Qwen/Qwen3-8B")
        self.tokenizer = AutoTokenizer.from_pretrained(self.name, revision=cfg.get("revision"))
        self.tokenizer.padding_side = "left"  # the answer position is then always index -1

        load_kwargs = {"revision": cfg.get("revision")}
        precision = cfg.get("precision", "canonical")
        if precision == "canonical":
            load_kwargs["dtype"] = torch.bfloat16 if device.startswith("cuda") else torch.float32
        elif precision == "memory_optimized":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
            load_kwargs["device_map"] = {"": device}
        else:
            raise ValueError(f"unknown precision mode {precision!r}")
        model = AutoModelForCausalLM.from_pretrained(self.name, **load_kwargs)
        if precision == "canonical":
            model.to(device)
        if cfg.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        self.model = self._add_lora(model, cfg["lora"]) if cfg["lora"].get("enabled", True) else model
        self.labels: LabelTokens = resolve_label_token_ids(
            self.tokenizer, rendered_prompt=render_chat_prompt(self.tokenizer, "Target: example\nQuestion: Is this fraud?"))

    @staticmethod
    def _add_lora(model, lora_cfg: dict):
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(r=lora_cfg["rank"], lora_alpha=lora_cfg["alpha"], lora_dropout=lora_cfg["dropout"],
                            target_modules=list(lora_cfg["target_modules"]), bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, config)
        wrapped = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules() if name.endswith(tuple(config.target_modules))}
        missing = set(config.target_modules) - wrapped
        if missing:
            raise RuntimeError(f"LoRA target modules not found in the model: {missing}")
        return model

    def last_token_logits(self, prompts: list[str]) -> torch.Tensor:
        rendered = [render_chat_prompt(self.tokenizer, p) for p in prompts]
        enc = self.tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False)
        max_len = self.cfg.get("max_seq_len")
        if max_len and enc["input_ids"].shape[1] > max_len:
            raise ValueError(f"prompt of {enc['input_ids'].shape[1]} tokens exceeds max_seq_len={max_len}; "
                             "lower dgp.max_target_tokens rather than cutting the question off")
        enc = enc.to(self.device)
        out = self.model(**enc, logits_to_keep=1)
        return out.logits[:, -1, :]

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def state_for_checkpoint(self) -> dict:
        return {k: v.detach().to("cpu", copy=True) for k, v in self.model.named_parameters() if v.requires_grad}

    def load_checkpoint_state(self, state: dict) -> None:
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            for k, v in state.items():
                params[k].copy_(v.to(params[k].device))

    def save_adapter(self, directory: str | Path) -> None:
        self.model.save_pretrained(str(directory))
