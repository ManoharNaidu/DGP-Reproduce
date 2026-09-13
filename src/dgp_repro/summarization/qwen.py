"""Frozen Qwen3-8B summarizer ("DGP additionally uses a frozen Qwen3-8B for summary generation").

Provenance: PAPER_RECONSTRUCTION for decoding settings, which the paper does not give.

Settings and why:
    enable_thinking=False   REQUIRED. Otherwise Qwen3 emits a <think> block first
                            (research/llm_provenance.md §1.7).
    greedy decoding         deterministic, cacheable summaries (paper: unspecified).
    max_new_tokens          `max_new_tokens_factor * B + max_new_tokens_extra`, a generous cap that
                            leaves the budget to the instruction, as the paper's prompt does.
                            Set `enforce_budget: true` to cap generation at exactly B tokens instead.
                            Realised lengths are recorded either way.
"""

from __future__ import annotations


class QwenSummarizer:
    def __init__(self, cfg: dict, device: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.name = cfg.get("model_id", "Qwen/Qwen3-8B")
        self.revision = cfg.get("revision")
        self.device = device
        self.batch_size = cfg.get("batch_size", 16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name, revision=self.revision)
        self.tokenizer.padding_side = "left"  # decoder-only batched generation
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(self.name, revision=self.revision, dtype=dtype)
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.last_generated_lengths: list[int] = []

    def _render(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)

    def _max_new_tokens(self, budget: int) -> int:
        if self.cfg.get("enforce_budget", False):
            return budget
        return int(self.cfg.get("max_new_tokens_factor", 4) * budget + self.cfg.get("max_new_tokens_extra", 16))

    def summarize(self, prompts: list[str], budget: int) -> list[str]:
        import torch

        outputs, lengths = [], []
        for start in range(0, len(prompts), self.batch_size):
            rendered = [self._render(p) for p in prompts[start:start + self.batch_size]]
            enc = self.tokenizer(rendered, return_tensors="pt", padding=True,
                                 truncation=True, max_length=self.cfg.get("max_input_tokens", 4096)).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=self._max_new_tokens(budget), do_sample=False,
                                          pad_token_id=self.tokenizer.pad_token_id)
            new_tokens = gen[:, enc["input_ids"].shape[1]:]
            special = torch.tensor(self.tokenizer.all_special_ids, device=new_tokens.device)
            for row in new_tokens:
                row = row[~torch.isin(row, special)]  # drop padding and <|im_end|>
                lengths.append(int(len(row)))
                outputs.append(self.tokenizer.decode(row, skip_special_tokens=True).strip())
        self.last_generated_lengths = lengths
        return outputs
