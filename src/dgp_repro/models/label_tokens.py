"""Resolve and validate the Yes/No label tokens.

DGP reads the logits of the tokens "Yes" and "No" at the first generated position (Eq. 13-14).
That only works if each label is exactly ONE token *in the position where it is generated*.

Known facts for Qwen3-8B (research/llm_provenance.md §2, verified against vocab.json):
    "Yes" = 9454, "No" = 2753, " Yes" = 7414, " No" = 2308 — four different single tokens.
With enable_thinking=False the rendered prompt ends in "\\n\\n", so the no-space pair is the
correct one. These ids are NOT hard-coded: they are resolved from the tokenizer at run time,
and `rendered_prompt` is used to prove that prompt + label tokenizes as prompt tokens plus
exactly one label token. Any mismatch stops the run instead of silently scoring the wrong token.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class LabelTokenizationError(RuntimeError):
    pass


@dataclass
class LabelTokens:
    yes_id: int
    no_id: int
    positive: str = "Yes"
    negative: str = "No"
    report: dict = field(default_factory=dict)


def _single_token(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        pieces = [tokenizer.decode([i]) for i in ids]
        raise LabelTokenizationError(
            f"label {text!r} is {len(ids)} tokens {ids} ({pieces}) under {type(tokenizer).__name__}; "
            "first-token classification needs a single token. Investigate before continuing.")
    if tokenizer.decode(ids).strip() != text.strip():
        raise LabelTokenizationError(f"label {text!r} does not round-trip: decodes to {tokenizer.decode(ids)!r}")
    return ids[0]


def _check_boundary(tokenizer, rendered_prompt: str, label: str, label_id: int) -> None:
    prompt_ids = tokenizer.encode(rendered_prompt, add_special_tokens=False)
    joined_ids = tokenizer.encode(rendered_prompt + label, add_special_tokens=False)
    if joined_ids[:-1] != prompt_ids or joined_ids[-1] != label_id:
        raise LabelTokenizationError(
            f"label {label!r} does not tokenize as a single appended token after the rendered prompt "
            f"(prompt ends with {rendered_prompt[-12:]!r}; tail tokens {joined_ids[-3:]}, expected id {label_id}). "
            "The prompt/label spacing is inconsistent with the logits being read.")


def resolve_label_token_ids(tokenizer, positive: str = "Yes", negative: str = "No",
                            rendered_prompt: str | None = None) -> LabelTokens:
    yes_id = _single_token(tokenizer, positive)
    no_id = _single_token(tokenizer, negative)
    if yes_id == no_id:
        raise LabelTokenizationError("positive and negative labels map to the same token")
    report = {"positive": positive, "negative": negative, "yes_id": yes_id, "no_id": no_id,
              "boundary_checked": rendered_prompt is not None}
    for variant in (" " + positive, " " + negative):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        report[f"leading_space_variant{variant!r}"] = ids
    if rendered_prompt is not None:
        _check_boundary(tokenizer, rendered_prompt, positive, yes_id)
        _check_boundary(tokenizer, rendered_prompt, negative, no_id)
    return LabelTokens(yes_id=yes_id, no_id=no_id, positive=positive, negative=negative, report=report)
