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
        extra = {"attn_implementation": cfg["attn_implementation"]} if cfg.get("attn_implementation") else {}
        self.model = AutoModelForCausalLM.from_pretrained(self.name, revision=self.revision, dtype=dtype, **extra)
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

    def _batches(self, rendered: list[str]) -> list[list[str]]:
        """Consecutive batches of at most `batch_size` prompts, in input order. With `max_batch_tokens_sq` set, a batch
        is also closed before (prompts x longest prompt length^2) would exceed it: eager attention (Gemma-2) needs
        memory proportional to that product, and one batch of 16 ~4k-token reviews asked for 15 GiB at once.
        Unset (the default, every Qwen config) gives exactly the old fixed-size batches."""
        cap = self.cfg.get("max_batch_tokens_sq")
        if not cap:
            return [rendered[i:i + self.batch_size] for i in range(0, len(rendered), self.batch_size)]
        lens = [len(ids) for ids in self.tokenizer(rendered, add_special_tokens=False)["input_ids"]]
        out, cur, longest = [], [], 0
        for text, n in zip(rendered, lens):
            if cur and (len(cur) >= self.batch_size or (len(cur) + 1) * max(longest, n) ** 2 > cap):
                out.append(cur)
                cur, longest = [], 0
            cur.append(text)
            longest = max(longest, n)
        if cur:
            out.append(cur)
        return out

    def summarize(self, prompts: list[str], budget: int) -> list[str]:
        import torch

        outputs, lengths = [], []
        for rendered in self._batches([self._render(p) for p in prompts]):
            enc = self.tokenizer(rendered, return_tensors="pt", padding=True).to(self.device)
            # Never truncate: cutting the right end would drop the chat template's assistant header and the
            # model would continue the input instead of answering. Inputs within the limit are unchanged.
            limit = self.cfg.get("max_input_tokens", 4096)
            longest = int(enc["attention_mask"].sum(dim=1).max())
            if limit and longest > limit:
                raise ValueError(f"summarizer input of {longest} tokens exceeds max_input_tokens={limit}")
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
