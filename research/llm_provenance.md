# DGP Reproduction — Model Stack Provenance (Subagent E)

Target paper: *DGP: A Dual-Granularity Prompting Framework for Fraud Detection with Graph-Enhanced LLMs* (AAAI-2026).

Paper's Implementation Details (verbatim): "For all LLM-tuning methods, we use the Qwen3-8B LLM backbone (Team 2025) for fair comparison. We apply LoRA (Hu et al. 2022) to all attention layers and use AdamW (Loshchilov and Hutter 2019) optimizer for finetuning. DGP additionally uses a frozen Qwen3-8B for summary generation." Text embeddings for the diffusion kernel use "DeBERTa (He, Gao, and Chen 2021)".

Every fact below was fetched from the cited URL during this investigation. Nothing is recalled from memory. Derived arithmetic is labelled **DERIVED**; anything not directly evidenced is labelled **ESTIMATE**.

---

## 0. EXECUTIVE SUMMARY — THE TWO CORRECTNESS-CRITICAL FINDINGS

### 0.1 Qwen3 thinking mode is ON by default and breaks first-token Yes/No readout

`Qwen/Qwen3-8B` is a post-trained (instruct/reasoning) model whose chat template enables **thinking mode by default**. With `add_generation_prompt=True` and `enable_thinking` left unset, the rendered prompt terminates at `<|im_start|>assistant\n` and the model's **first generated token is the opening of a `<think>` block, not `Yes`/`No`**. DGP's classification rule — read the Yes/No logits at the first generated position — is therefore **silently wrong** unless thinking is explicitly disabled.

Setting `enable_thinking=False` makes the template append a pre-closed empty think block, so the prompt ends with:

```
<|im_start|>assistant\n<think>\n\n</think>\n\n
```

and the **first generated token is the answer token**. This is the configuration DGP's method requires. The paper never mentions `enable_thinking`. See §1.7.

### 0.2 "Yes"/"No" and " Yes"/" No" are four DIFFERENT single tokens — all verified

Qwen3 uses a byte-level BPE (`Qwen2Tokenizer`). All four variants exist as single vocabulary entries with distinct IDs, verified by direct lookup in the repo's own `vocab.json`:

| string | byte-level form | token id | single token? |
|---|---|---|---|
| `"Yes"` | `Yes` | **9454** | yes |
| `" Yes"` | `ĠYes` | **7414** | yes |
| `"No"` | `No` | **2753** | yes |
| `" No"` | `ĠNo` | **2308** | yes |

Because `enable_thinking=False` leaves the prompt ending in `\n\n`, the next token has **no preceding space**, so the correct logit indices are the **no-leading-space** pair **9454 (`Yes`) / 2753 (`No`)**. Using 7414/2308 would read the logits of tokens that are essentially impossible in that position. The paper specifies neither. See §2.

---

## 1. VERIFIED FACTS — Qwen3-8B

### 1.1 Canonical repo id and base-vs-instruct distinction

**VERIFIED.** The canonical id is **`Qwen/Qwen3-8B`** — confirmed by a `200` from the HF model API.
Source: https://huggingface.co/api/models/Qwen/Qwen3-8B

**`Qwen/Qwen3-8B-Base` also exists** and is a different model. Source: https://huggingface.co/api/models/Qwen/Qwen3-8B-Base (config fetched successfully, below).

`Qwen/Qwen3-8B`'s own API tags include `base_model:Qwen/Qwen3-8B-Base` and `base_model:finetune:Qwen/Qwen3-8B-Base`, i.e. **`Qwen3-8B` is the post-trained finetune of `Qwen3-8B-Base`**.
Source: https://huggingface.co/api/models/Qwen/Qwen3-8B

Differences that matter, all read from the two `config.json` / `generation_config.json` files:

| field | `Qwen/Qwen3-8B` (post-trained) | `Qwen/Qwen3-8B-Base` |
|---|---|---|
| `eos_token_id` (config) | `151645` (`<\|im_end\|>`) | `151643` (`<\|endoftext\|>`) |
| `max_position_embeddings` | `40960` | `32768` |
| `generation_config.eos_token_id` | `[151645, 151643]` | `151643` |
| `generation_config.do_sample` | `true` | `false` |
| `generation_config.temperature/top_k/top_p` | `0.6 / 20 / 0.95` | not set |
| chat template with `enable_thinking` | yes | a `chat_template` key exists, but the model is not instruction-tuned |

Everything else in the two configs is identical (same 36 layers, 4096 hidden, 151936 vocab, etc.).

Sources:
- https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json
- https://huggingface.co/Qwen/Qwen3-8B-Base/raw/main/config.json
- https://huggingface.co/Qwen/Qwen3-8B/raw/main/generation_config.json
- https://huggingface.co/Qwen/Qwen3-8B-Base/raw/main/generation_config.json

**The DGP paper does not say which of the two it used.** See §6 (Unresolved).

### 1.2 `Qwen/Qwen3-8B` config.json — VERBATIM

Fetched from https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json

```json
{
  "architectures": [
    "Qwen3ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 12288,
  "max_position_embeddings": 40960,
  "max_window_layers": 36,
  "model_type": "qwen3",
  "num_attention_heads": 32,
  "num_hidden_layers": 36,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": null,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.0",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}
```

Requested fields, extracted:

| field | value |
|---|---|
| `architectures` | `["Qwen3ForCausalLM"]` |
| `hidden_size` | `4096` |
| `num_hidden_layers` | `36` |
| `num_attention_heads` | `32` |
| `num_key_value_heads` | `8` (GQA, 4:1 ratio) |
| `head_dim` | `128` |
| `intermediate_size` | `12288` |
| `vocab_size` | `151936` |
| `max_position_embeddings` | `40960` |
| `rope_theta` | `1000000` |
| `torch_dtype` | `"bfloat16"` |
| `tie_word_embeddings` | `false` |

Note `num_attention_heads * head_dim = 32 * 128 = 4096 = hidden_size`, and `num_key_value_heads * head_dim = 8 * 128 = 1024`. This asymmetry matters for LoRA shape accounting (§4.3).

### 1.3 Attention projection module names (for LoRA `target_modules`) — VERIFIED FROM SOURCE

Fetched from https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3/modeling_qwen3.py

`class Qwen3Attention(nn.Module)` `__init__`, verbatim:

```python
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
```

**CONFIRMED:** the four `nn.Linear` attributes inside the attention class are exactly
**`q_proj`, `k_proj`, `v_proj`, `o_proj`.**

**CONFIRMED:** Qwen3 additionally defines **`q_norm` and `k_norm`**, which are **`Qwen3RMSNorm`, not `nn.Linear`**. They are **NOT valid LoRA targets** — PEFT's LoRA only wraps `Linear`/`Conv`/`Embedding` layers, and these are per-head-dim RMSNorm modules (`Qwen3RMSNorm(self.head_dim)`, i.e. 128 parameters each). This is a Qwen3-specific addition versus Qwen2/Llama; do not list them in `target_modules`.

The forward pass confirms the norms wrap the projections:

```python
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
```

**MLP module names**, from `class Qwen3MLP(nn.Module)` in the same file, verbatim:

```python
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
```

So **`gate_proj`, `up_proj`, `down_proj`** are the MLP projections. The paper says LoRA is applied to **"all attention layers"**, which on a literal reading **excludes** these three. The faithful configuration is therefore:

```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
```

Note `attention_bias: false` in config, so all four projections are bias-free — consistent with `bias="none"` in `LoraConfig`.

### 1.4 Gating and license — VERIFIED

From https://huggingface.co/api/models/Qwen/Qwen3-8B :

- `gated`: **`False`** — the model is **NOT gated**; no license acceptance click-through is required.
- `private`: `False`
- `cardData.license`: **`apache-2.0`** — confirms the expected Apache-2.0.
- tag list includes `license:apache-2.0`.
- The repo also ships a `LICENSE` file (11,343 bytes).

**No HF token is required to download `Qwen/Qwen3-8B`.** (See §5.)

### 1.5 Parameter count — VERIFIED

From https://huggingface.co/api/models/Qwen/Qwen3-8B :

```
"safetensors": {"parameters": {"BF16": 8190735360}, "total": 8190735360}
```

**Total parameters: 8,190,735,360 (≈8.19 B), all in BF16.**

The model card additionally states **"8.2B"** total and **"6.95B"** non-embedding parameters, and **36** layers, context **"32,768 natively and 131,072 tokens with YaRN"**.
Source: https://huggingface.co/Qwen/Qwen3-8B

(Note the tension: `config.json` says `max_position_embeddings: 40960` while the card says 32,768 native. The 40960 figure = 32768 context + 8192 generation budget. Neither matters for DGP, whose prompts are far shorter.)

### 1.6 File list and download size — VERIFIED

From https://huggingface.co/api/models/Qwen/Qwen3-8B?blobs=true :

| bytes | file |
|---:|---|
| 1,570 | `.gitattributes` |
| 11,343 | `LICENSE` |
| 16,660 | `README.md` |
| 728 | `config.json` |
| 239 | `generation_config.json` |
| 1,671,853 | `merges.txt` |
| 3,996,250,744 | `model-00001-of-00005.safetensors` |
| 3,993,160,032 | `model-00002-of-00005.safetensors` |
| 3,959,604,768 | `model-00003-of-00005.safetensors` |
| 3,187,841,392 | `model-00004-of-00005.safetensors` |
| 1,244,659,840 | `model-00005-of-00005.safetensors` |
| 32,878 | `model.safetensors.index.json` |
| 11,422,654 | `tokenizer.json` |
| 9,732 | `tokenizer_config.json` |
| 2,776,833 | `vocab.json` |

- **safetensors shards only: 16,381,516,776 bytes = 16.38 GB (decimal) = 15.26 GiB**
- **whole repo: 16,397,461,266 bytes = 16.40 GB (decimal) = 15.27 GiB**

5 shards, no `.bin` duplicates — a clean safetensors-only repo.

### 1.7 THINKING MODE — FULL INVESTIGATION (CRITICAL FOR DGP)

#### 1.7.1 The chat template's thinking logic — VERBATIM

Fetched from https://huggingface.co/Qwen/Qwen3-8B/raw/main/tokenizer_config.json ; the tail of the `chat_template` field reads, **verbatim**:

```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
```

**This is the whole story, and it is unambiguous:**

- The guard is `enable_thinking is defined and enable_thinking is false`. If `enable_thinking` is **not passed at all**, the condition is false, the empty think block is **not** emitted, and the prompt ends at `<|im_start|>assistant\n`.
- **Therefore the template default is thinking ENABLED** — omission behaves identically to `enable_thinking=True`.
- With thinking enabled, the model, being post-trained to reason, emits a `<think>` block first. **The first generated token is the start of that block, not `Yes`/`No`.**
- With `enable_thinking=False`, the template pre-writes a *closed, empty* think block into the prompt. The prompt ends `...<|im_start|>assistant\n<think>\n\n</think>\n\n`. The model's first generated token is then the first token of the actual answer. **This is the only configuration in which DGP's first-token Yes/No logit readout is meaningful.**

Other tokenizer_config values (verbatim):

```
add_bos_token = False
add_prefix_space = False
bos_token = None
eos_token = '<|im_end|>'
pad_token = '<|endoftext|>'
tokenizer_class = 'Qwen2Tokenizer'
model_max_length = 131072
clean_up_tokenization_spaces = False
split_special_tokens = False
```

`add_bos_token=False` and `add_prefix_space=False` are both relevant to §2.

#### 1.7.2 The model card's own statements — VERBATIM

From https://huggingface.co/Qwen/Qwen3-8B (raw README.md), verbatim excerpts:

Quickstart snippet:

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
```

Section `### enable_thinking=True`:

> "By default, Qwen3 has thinking capabilities enabled, similar to QwQ-32B. This means the model will use its reasoning abilities to enhance the quality of generated responses. For example, when explicitly setting `enable_thinking=True` or leaving it as the default value in `tokenizer.apply_chat_template`, the model will engage its thinking mode."

```python
    enable_thinking=True  # True is the default value for enable_thinking
```

> "In this mode, the model will generate think content wrapped in a `<think>...</think>` block, followed by the final response."

Section `### enable_thinking=False`:

> "We provide a hard switch to strictly disable the model's thinking behavior, aligning its functionality with the previous Qwen2.5-Instruct models. This mode is particularly useful in scenarios where disabling thinking is essential for enhancing efficiency."

```python
    enable_thinking=False  # Setting enable_thinking=False disables thinking mode
```

> "In this mode, the model will not generate any think content and will not include a `<think>...</think>` block."

Soft switch section:

> "We provide a soft switch mechanism that allows users to dynamically control the model's behavior when `enable_thinking=True`. Specifically, you can add `/think` and `/no_think` to user prompts or system messages to switch the model's thinking mode from turn to turn. The model will follow the most recent instruction in multi-turn conversations."

Critical caveat, verbatim:

> "For API compatibility, when `enable_thinking=True`, regardless of whether the user uses `/think` or `/no_think`, the model will always output a block wrapped in `<think>...</think>`. However, the content inside this block may be empty if thinking is disabled.
> When `enable_thinking=False`, the soft switches are not valid. Regardless of any `/think` or `/no_think` tags input by the user, the model will not generate think content and will not include a `<think>...</think>` block."

**This caveat is decisive for DGP:** the `/no_think` soft switch does **NOT** fix the problem. With `enable_thinking=True` the model still emits a `<think>...</think>` wrapper even when `/no_think` is used — only the *contents* become empty. The first generated token is still `<think>`-related. **Only the hard switch `enable_thinking=False` works for first-token logit readout.**

Also relevant — the card's Best Practices:

> "For thinking mode (`enable_thinking=True`), use `Temperature=0.6`, `TopP=0.95`, `TopK=20`, and `MinP=0`. **DO NOT use greedy decoding**, as it can lead to performance degradation and endless repetitions."
> "For non-thinking mode (`enable_thinking=False`), we suggest using `Temperature=0.7`, `TopP=0.8`, `TopK=20`, and `MinP=0`."

Note DGP does not sample at all — it reads logits — so sampling params are moot for classification, but they matter for the **frozen Qwen3-8B summary generation** stage, where the paper likewise specifies nothing.

#### 1.7.3 Concrete impact on DGP and the required fix

DGP classifies by comparing the Yes/No logits at the first generated position. Failure mode if thinking is left at default:

1. `apply_chat_template(..., add_generation_prompt=True)` with no `enable_thinking` → prompt ends `<|im_start|>assistant\n`.
2. Logits at that final position are the distribution over the *first* generated token. For a reasoning-post-trained Qwen3, that distribution is dominated by the think-block opener.
3. `logits[yes_id]` and `logits[no_id]` are then both tiny, essentially arbitrary tail mass. The resulting "probability" `softmax([logit_yes, logit_no])` is a ratio of two noise values — it may still produce an AUC above chance by accident, but it is **not** the quantity the method describes, and it is not reproducible.

**Required configuration for a faithful reproduction:**

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,          # MANDATORY for first-token Yes/No readout
)
```

**Verification assertion to put in the code** (do not trust, check):

```python
assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), text[-80:]
```

If that assertion fails, the transformers version or the template revision has changed and the token-index logic must be re-derived.

**Additional consequence for finetuning.** The LoRA finetuning targets must be trained under the *same* template. If training renders prompts with `enable_thinking=False` but evaluation renders with the default, or vice versa, there is a train/test prompt mismatch of four tokens plus two newlines at exactly the position whose logits are read. The paper is silent on this. Keep `enable_thinking=False` on **both** paths.

**Consequence for the summary-generation stage.** The frozen Qwen3-8B that produces node summaries is a *generation* task, not a logit readout. Thinking mode there is a free choice (it would change summary quality and cost a great deal of tokens), but any `<think>...</think>` block **must be stripped** from the generated text before the summary is embedded by DeBERTa, or the diffusion kernel will embed the model's reasoning trace rather than the summary. The paper does not mention this at all.

---

## 2. VERIFIED FACTS — Tokenizer and Yes/No tokens

### 2.1 Tokenizer identity

- `tokenizer_class`: **`Qwen2Tokenizer`** (Qwen3 reuses the Qwen2 byte-level BPE).
- `vocab_size` in `config.json`: **151936**.
- Entries actually present in `vocab.json`: **151,643**.
- Added/special tokens in `tokenizer_config.json` `added_tokens_decoder`: **26**, ids **151643–151668** (`<|endoftext|>`=151643, `<|im_start|>`=151644, `<|im_end|>`=151645, `<|object_ref_start|>`=151646, ...).
- 151,643 base + 26 added = 151,669 real tokens; `vocab_size` 151936 is padded upward (for tensor-core-friendly embedding dims). Ids 151669–151935 are unused padding slots.

Sources:
- https://huggingface.co/Qwen/Qwen3-8B/raw/main/tokenizer_config.json
- https://huggingface.co/Qwen/Qwen3-8B/raw/main/vocab.json
- https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json

### 2.2 Yes/No token ids — VERIFIED BY DIRECT VOCAB LOOKUP

These ids were **not guessed**. `vocab.json` was downloaded from the `Qwen/Qwen3-8B` repo (2,776,833 bytes, 151,643 entries) and the keys were looked up directly. In byte-level BPE, a leading space is represented by the character **`Ġ` (U+0120)**.

Source: https://huggingface.co/Qwen/Qwen3-8B/raw/main/vocab.json

| string | vocab.json key | token id |
|---|---|---:|
| `"Yes"` | `Yes` | **9454** |
| `" Yes"` | `ĠYes` | **7414** |
| `"No"` | `No` | **2753** |
| `" No"` | `ĠNo` | **2308** |
| `"yes"` | `yes` | 9693 |
| `" yes"` | `Ġyes` | 9834 |
| `"no"` | `no` | 2152 |
| `" no"` | `Ġno` | 902 |
| `"YES"` | `YES` | 14004 |
| `" YES"` | `ĠYES` | 14080 |
| `"NO"` | `NO` | 8996 |
| `" NO"` | `ĠNO` | 5664 |

Reverse lookup confirms id 9454 maps back to `Yes` and 7414 to `ĠYes`.

**Are they single tokens? YES — and the BPE merges confirm it.** `merges.txt` (151,388 merge rules) was downloaded from the same repo and contains the exact merge rules:

- line 9200: `Y es`
- line 2499: `N o`

Source: https://huggingface.co/Qwen/Qwen3-8B/raw/main/merges.txt

So greedy BPE genuinely merges `Y`+`es` → `Yes` and `N`+`o` → `No`. **`Yes` and `No` are each exactly one token**, and so are their leading-space variants. No multi-token handling is needed.

### 2.3 Which pair does DGP need? — `9454` / `2753`

With `enable_thinking=False`, the prompt ends with `</think>\n\n`. The character immediately preceding the generation point is a **newline, not a space**. Byte-level BPE attaches a leading space to the *following* token only when a space is actually present. Since there is none, the model must emit the **no-leading-space** form.

**Use `Yes`=9454 and `No`=2753.**

Using 7414/2308 would score tokens that cannot legally follow `\n\n` in this template, producing near-constant garbage logits.

This distinction is invisible in the paper, and it is the single easiest way to silently destroy the reproduction. A reproduction that hardcodes the wrong pair will still run, still produce an AUC, and still look plausible.

### 2.4 Resolve at runtime anyway — exact snippet

Even though the ids above are verified against the live repo, **hardcoding is fragile** (a repo revision, a different Qwen variant, or a tokenizer bump would change them). Resolve at runtime and assert:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

def single_token_id(tokenizer, s: str) -> int:
    """Resolve `s` to exactly one token id, or fail loudly."""
    ids = tokenizer.encode(s, add_special_tokens=False)   # MUST be False
    assert len(ids) == 1, f"{s!r} -> {ids} ({tokenizer.convert_ids_to_tokens(ids)}) is not a single token"
    return ids[0]

YES_ID = single_token_id(tok, "Yes")    # expect 9454
NO_ID  = single_token_id(tok, "No")     # expect 2753

# The leading-space variants are DIFFERENT tokens. Resolve them too, and be
# explicit about which pair the prompt template actually requires.
YES_SP_ID = single_token_id(tok, " Yes")  # expect 7414
NO_SP_ID  = single_token_id(tok, " No")   # expect 2308

assert (YES_ID, NO_ID) == (9454, 2753), (YES_ID, NO_ID)          # pin the verified values
assert (YES_SP_ID, NO_SP_ID) == (7414, 2308), (YES_SP_ID, NO_SP_ID)
assert YES_ID != YES_SP_ID and NO_ID != NO_SP_ID
```

Notes on the flags:

- **`add_special_tokens=False` is mandatory.** With the default `True`, `Qwen2Tokenizer` may prepend/append template tokens and the length-1 assertion breaks. (For Qwen3 specifically `add_bos_token=False`, so in practice `add_special_tokens=True` adds nothing — but do not rely on that; state it explicitly.)
- **`add_prefix_space=False`** in `tokenizer_config.json` means the tokenizer will **not** silently insert a leading space. So `tok.encode("Yes", add_special_tokens=False)` really does give the no-space form. Had this been `True`, the ids would silently become the `Ġ` variants — another reason to assert.
- Do **not** use `tok.convert_tokens_to_ids("Yes")` as a substitute without checking; it bypasses the BPE merge path and can mask a multi-token string.

Classification readout, for completeness:

```python
out = model(**model_inputs)
next_token_logits = out.logits[0, -1, :]              # distribution over the FIRST generated token
pair = torch.stack([next_token_logits[NO_ID], next_token_logits[YES_ID]])
p_fraud = torch.softmax(pair.float(), dim=-1)[1].item()
```

---

## 3. VERIFIED FACTS — DeBERTa

### 3.1 Which paper is "He, Gao, and Chen 2021"? — DeBERTaV3. CONFIRMED.

| | DeBERTa (original) | DeBERTaV3 |
|---|---|---|
| arXiv | 2006.03654 | 2111.09543 |
| title | "DeBERTa: Decoding-enhanced BERT with Disentangled Attention" | "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing" |
| authors | Pengcheng **He**, Xiaodong **Liu**, Jianfeng **Gao**, Weizhu **Chen** (4 authors) | Pengcheng **He**, Jianfeng **Gao**, Weizhu **Chen** (3 authors) |
| submitted | 2020-06-05 | 2021-11-18 |
| venue | ICLR 2021 | ICLR 2023 |

Sources:
- https://arxiv.org/abs/2006.03654
- https://arxiv.org/abs/2111.09543

**CONCLUSION: the citation "He, Gao, and Chen 2021" matches DeBERTaV3 exactly** — three authors, exactly He/Gao/Chen, arXiv year 2021. The original DeBERTa has a fourth author (Xiaodong Liu) and would be cited "He et al. 2021" or "He, Liu, Gao, and Chen". The three-author list plus the year 2021 is a unique match.

Caveat worth recording: the year is ambiguous in a different way. DeBERTaV3's *arXiv* year is 2021 but its *published venue* year is ICLR **2023**; the original DeBERTa's *arXiv* year is 2020 but its *venue* year is ICLR **2021**. A citation reading "He, Gao, and Chen 2021" is only consistent with DeBERTaV3-by-arXiv-year. The author list is the decisive evidence, and it points to **DeBERTaV3**. Recommend **`microsoft/deberta-v3-base`** as the default reproduction choice.

### 3.2 Canonical HF ids, hidden sizes, max positions — VERIFIED FROM config.json

All five configs fetched from `https://huggingface.co/<id>/raw/main/config.json`:

| HF id | `model_type` | `hidden_size` | `max_position_embeddings` | layers | heads | `intermediate_size` | `vocab_size` |
|---|---|---:|---:|---:|---:|---:|---:|
| `microsoft/deberta-base` | `deberta` | **768** | **512** | 12 | 12 | 3072 | 50265 |
| `microsoft/deberta-v3-xsmall` | `deberta-v2` | **384** | **512** | 12 | 6 | 1536 | 128100 |
| `microsoft/deberta-v3-small` | `deberta-v2` | **768** | **512** | 6 | 12 | 3072 | 128100 |
| `microsoft/deberta-v3-base` | `deberta-v2` | **768** | **512** | 12 | 12 | 3072 | 128100 |
| `microsoft/deberta-v3-large` | `deberta-v2` | **1024** | **512** | 24 | 16 | 4096 | 128100 |

Sources (each fetched individually):
- https://huggingface.co/microsoft/deberta-base/raw/main/config.json
- https://huggingface.co/microsoft/deberta-v3-xsmall/raw/main/config.json
- https://huggingface.co/microsoft/deberta-v3-small/raw/main/config.json
- https://huggingface.co/microsoft/deberta-v3-base/raw/main/config.json
- https://huggingface.co/microsoft/deberta-v3-large/raw/main/config.json

Observations:
- **All five have `max_position_embeddings: 512`.** No DeBERTa variant offers a longer context.
- All v3 models share `vocab_size: 128100` and a SentencePiece tokenizer (`tokenizer_config.json` for v3-base is literally `{"do_lower_case": false, "vocab_type": "spm"}`, and the repo ships `spm.model`). The original `deberta-base` uses a GPT2-style BPE (`bpe_encoder.bin`, vocab 50265).
- v3 models are `model_type: deberta-v2` — i.e. loaded by `DebertaV2Model`/`DebertaV2Tokenizer`. DeBERTaV3 is an architecture-compatible retrain of DeBERTa-v2, not a new architecture.
- Note `deberta-v3-small` has **6** layers (not 12) — easy to misread.

All five repos: **`gated: False`**, **license `mit`**. No token required.
Source: `https://huggingface.co/api/models/<id>?blobs=true` for each.

Repo sizes (full repo includes TF/Rust weights you do not need):

| HF id | `pytorch_model.bin` | full repo | PyTorch-only download (bin + spm + config) |
|---|---:|---:|---:|
| `microsoft/deberta-base` | 558,614,189 B | 1,599 MiB | ~536 MiB |
| `microsoft/deberta-v3-xsmall` | 241,453,931 B | 640 MiB | ~232 MiB |
| `microsoft/deberta-v3-small` | 286,059,269 B | 814 MiB | ~275 MiB |
| `microsoft/deberta-v3-base` | 371,146,213 B | 1,766 MiB | ~357 MiB |
| `microsoft/deberta-v3-large` | 873,673,253 B | 3,037 MiB | ~836 MiB |

**Practical gotcha: the DeBERTa repos ship NO safetensors.** `microsoft/deberta-v3-base` siblings are exactly: `.gitattributes`, `README.md`, `config.json`, `pytorch_model.bin`, `rust_model.ot`, `spm.model`, `tf_model.h5`, `tokenizer_config.json`. There is no `model.safetensors`. Downloading the whole repo pulls 742 MB of Rust weights and 735 MB of TF weights you will never use — **use `--include` to fetch only what you need** (§5.2). Loading `pytorch_model.bin` also requires `torch.load`, which recent torch/transformers versions gate behind `weights_only` handling; if your stack refuses `.bin`, convert once locally.
Source: https://huggingface.co/api/models/microsoft/deberta-v3-base?blobs=true

Note also `deberta-v3-large` and `deberta-v3-xsmall` ship an extra `pytorch_model.generator.bin` (the ELECTRA-style generator from pretraining) — not needed for embedding.

### 3.3 No native sentence-embedding pooling — CONFIRMED FROM SOURCE

`DebertaV2Model.forward` returns **`BaseModelOutput`**, not `BaseModelOutputWithPooling`. Confirmed two ways:

1. transformers docs for DeBERTa-v2 state the return type is `BaseModelOutput`.
   Source: https://huggingface.co/docs/transformers/en/model_doc/deberta-v2
2. The source itself: `modeling_deberta_v2.py` imports `BaseModelOutput` and returns it (lines 678, 795); `DebertaV2Model.forward` is annotated `-> tuple | BaseModelOutput`.
   Source: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/deberta_v2/modeling_deberta_v2.py

**There is no `pooler_output`.** `DebertaV2Model` gives you only `last_hidden_state`. This is a real difference from BERT/RoBERTa, where `pooler_output` exists (even if discouraged).

There *is* a `ContextPooler`, but it lives on the **task heads**, not on the base model. Verbatim from the source:

```python
class ContextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)
        self.dropout = nn.Dropout(config.pooler_dropout)
        self.config = config

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        return pooled_output
```

and it is instantiated only inside heads such as:

```python
class DebertaV2ForSequenceClassification(DebertaV2PreTrainedModel):
    def __init__(self, config):
        ...
        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim
        self.classifier = nn.Linear(output_dim, num_labels)
```

So `ContextPooler` = **take position 0 (`[CLS]`), dropout, dense, activation**. Its `dense` weights are **randomly initialised** when you instantiate a head that was not finetuned on that task — using it untrained would inject random noise into DGP's diffusion kernel.

**Conventions for extracting a fixed-size embedding** (three real options):

1. **CLS / first-token**: `last_hidden_state[:, 0]`. Matches what `ContextPooler` pools, without the untrained dense layer. Cheap, but for a model pretrained with ELECTRA-style RTD (DeBERTaV3) the `[CLS]` vector was never trained with a sentence-level objective (no NSP), so it is a weak sentence representation.
2. **Mean pooling over the attention mask** (the sentence-transformers convention, and the usual recommendation for encoders without a sentence-level pretraining objective):
   ```python
   out = model(**enc).last_hidden_state              # (B, T, H)
   m = enc["attention_mask"].unsqueeze(-1).float()   # (B, T, 1)
   emb = (out * m).sum(1) / m.sum(1).clamp(min=1e-9) # (B, H)
   ```
   Masked mean is essential — an unmasked `.mean(1)` averages in padding and makes the embedding depend on batch padding length.
3. **Max pooling** — rarer; mentioned only for completeness.

**DGP specifies none of these — UNRESOLVED reproduction gap.** Flagged in §6. Empirically, masked mean pooling and CLS produce materially different embedding geometries, which will change the diffusion kernel's similarity structure and therefore the reported numbers. **Recommendation: masked mean pooling** (standard for a non-sentence-pretrained encoder), with CLS as the documented ablation.

### 3.4 Sequence length limit and truncation

- **Hard limit 512 positions** for every DeBERTa/DeBERTaV3 checkpoint listed (`max_position_embeddings: 512` in all five configs).
- `microsoft/deberta-v3-base`'s `tokenizer_config.json` is `{"do_lower_case": false, "vocab_type": "spm"}` — it **does not set `model_max_length`**, so the tokenizer falls back to a very large sentinel and will **NOT truncate automatically**. Passing a >512-token sequence produces a position-embedding index error or silently degraded output depending on version. **You must pass `truncation=True, max_length=512` explicitly.**
  Source: https://huggingface.co/microsoft/deberta-v3-base/raw/main/tokenizer_config.json
- Consequence for DGP: long reviews (and long generated node summaries) are **truncated to the first 512 subword tokens**; everything after is discarded. For fraud-detection corpora with verbose reviews this is lossy, and whether DGP truncated, chunked-and-averaged, or used only short summaries is **not stated**. Because DGP embeds LLM-*generated summaries* (which are short by construction) rather than raw reviews in the diffusion kernel, truncation may bite less there — but it certainly bites if raw review text is embedded anywhere.

Recommended call:

```python
enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
```

---

## 4. VERIFIED FACTS — LoRA / PEFT

### 4.1 `LoraConfig` defaults — VERIFIED FROM SOURCE

Fetched from https://raw.githubusercontent.com/huggingface/peft/main/src/peft/tuners/lora/config.py (peft `VERSION = "0.20.1.dev0"`, per https://raw.githubusercontent.com/huggingface/peft/main/setup.py).

Verbatim field declarations:

```python
    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[list[str], str]] = field(
        default=None, ...)
    exclude_modules: Optional[Union[list[str], str]] = field(default=None, ...)
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    fan_in_fan_out: bool = field(default=False, ...)
    bias: Literal["none", "all", "lora_only"] = field(
        default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"}
    )
    use_rslora: bool = field(default=False, ...)
    modules_to_save: Optional[list[str]] = field(default=None, ...)
    init_lora_weights: (...) = field(default=True, ...)
```

`task_type` is inherited from `PeftConfig` in https://raw.githubusercontent.com/huggingface/peft/main/src/peft/config.py :

```python
    task_type: Optional[TaskType] = field(default=None, metadata={"help": "The type of task."})
```

**Defaults table — what "unspecified" actually means:**

| parameter | default | consequence if left unspecified |
|---|---|---|
| `r` | **8** | low rank; smallest of the conventional settings |
| `lora_alpha` | **8** | **scaling = alpha/r = 8/8 = 1.0** |
| `lora_dropout` | **0.0** | no regularisation on the adapter |
| `bias` | **`"none"`** | no bias terms trained (correct for Qwen3, which has `attention_bias: false`) |
| `task_type` | **`None`** | `get_peft_model` returns a generic `PeftModel`, **not** `PeftModelForCausalLM` — label shifting / `generate()` wiring may not behave as expected. **Set it explicitly to `TaskType.CAUSAL_LM`.** |
| `target_modules` | **`None`** | PEFT falls back to an architecture lookup table; if the architecture is unknown it **raises**. Do not rely on it — name the four projections explicitly. |
| `use_rslora` | `False` | scaling is `alpha/r`, not `alpha/sqrt(r)` |
| `init_lora_weights` | `True` | B initialised to zero, so the adapter starts as a no-op |
| `fan_in_fan_out` | `False` | correct for `nn.Linear` |

**The `lora_alpha` default of 8 is a trap.** Much of the literature and many blog posts assume `alpha=16` or `32`; PEFT's own default is 8, giving scaling 1.0. Since DGP states no hyperparameters, an unspecified reproduction silently lands on `r=8, alpha=8, dropout=0.0`, which is **not** the community-conventional `r=16, alpha=32` (scaling 2.0). This alone can move results.

### 4.2 Literature conventions — LABELLED CONVENTION, NOT DGP'S VALUES

> **These are CONVENTION ONLY. The DGP paper states none of them. Do not present any of these as the paper's values.**

- **Rank `r`**: 8, 16, 32 are the common choices for 7–8B models; 64 appears in QLoRA-style setups. `r=16` is the most common single default in the instruction-tuning literature.
- **`lora_alpha`**: commonly set to `r` or `2*r`, i.e. 16 or 32 for `r=16`. `alpha=32, r=16` (scaling 2.0) and `alpha=16, r=16` (scaling 1.0) are both widespread.
- **`lora_dropout`**: 0.0–0.1; `0.05` and `0.1` are the usual non-zero picks.
- **`bias`**: `"none"` is near-universal.
- **`target_modules`**: the LoRA paper's own experiments adapt attention weight matrices (`Wq, Wk, Wv, Wo`); later practice (QLoRA and successors) found targeting **all** linear layers including the MLP generally works better. **DGP explicitly says "all attention layers", so the faithful choice is the four attention projections only** — this is one of the few hyperparameter facts the paper does pin down.

LoRA paper: "LoRA: Low-Rank Adaptation of Large Language Models", Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, 2021 (arXiv), ICLR 2022 — matching the paper's "Hu et al. 2022" citation. Abstract: LoRA "injects trainable rank decomposition matrices into each layer of the Transformer architecture" and "can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times."
Source: https://arxiv.org/abs/2106.09685

### 4.3 DERIVED: trainable parameter counts for Qwen3-8B attention-only LoRA

Computed from the verified config (`hidden_size=4096`, `num_attention_heads*head_dim=4096`, `num_key_value_heads*head_dim=1024`, `num_hidden_layers=36`). Per-module LoRA params = `r * (fan_in + fan_out)`.

Shapes: `q_proj: 4096→4096`, `k_proj: 4096→1024`, `v_proj: 4096→1024`, `o_proj: 4096→4096`.
Per layer = `r*(4096+4096) + 2*r*(4096+1024) + r*(4096+4096)` = `r * 26624`.

| `r` | per layer | total over 36 layers | % of 8.19B |
|---:|---:|---:|---:|
| 8 | 212,992 | **7,667,712** | 0.0936% |
| 16 | 425,984 | **15,335,424** | 0.1872% |
| 32 | 851,968 | **30,670,848** | 0.3745% |
| 64 | 1,703,936 | **61,341,696** | 0.7489% |

**DERIVED** — arithmetic from verified config values, not quoted from any source. Useful as a sanity check: after `get_peft_model`, `model.print_trainable_parameters()` must match the row for your chosen `r`. If it does not, your `target_modules` are wrong (e.g. you accidentally included the MLP, which would roughly triple the count).

Note GQA makes `k_proj`/`v_proj` much cheaper than `q_proj`/`o_proj` — a detail that trips up anyone assuming four equal-sized matrices.

---

## 5. Practical setup

### 5.1 HF token — NOT REQUIRED

- `Qwen/Qwen3-8B`: `gated: False`, `private: False`, license `apache-2.0`. **No token, no license click-through.**
- All `microsoft/deberta*` checkpoints: `gated: False`, license `mit`. **No token.**

Sources: `https://huggingface.co/api/models/Qwen/Qwen3-8B`, `https://huggingface.co/api/models/microsoft/deberta-v3-base`, etc.

A token is still worth setting if you hit anonymous rate limits on a shared IP, but it is not a requirement for this stack.

### 5.2 Download commands — CLI RENAMED

**The CLI was renamed from `huggingface-cli` to `hf`.** Hugging Face's changelog post (dated 2025-07-25) states verbatim:

> "We've renamed `huggingface-cli` to `hf` and overhauled the command structure for speed and clarity."

and

> "the old CLI still works and will gently point you to the new commands."

Source: https://huggingface.co/changelog/better-hf-cli

The rename shipped in `huggingface_hub` **v0.34.0**; `huggingface-cli` remains available but is officially deprecated. Source: https://newreleases.io/project/github/huggingface/huggingface_hub/release/v0.34.0

The current CLI guide documents only `hf` — `huggingface-cli` no longer appears in it. New syntax is `hf <resource> <action> [options]`.
Source: https://huggingface.co/docs/huggingface_hub/main/en/guides/cli

**Install:**

```bash
pip install -U huggingface_hub          # CLI ships with the core package
# or the standalone installer:
#   curl -LsSf https://hf.co/cli/install.sh | bash            (macOS/Linux)
#   powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"   (Windows)
```

**Qwen3-8B (~15.3 GiB):**

```bash
hf download Qwen/Qwen3-8B --local-dir ./models/Qwen3-8B
```

Dry-run first to see exactly what will be pulled:

```bash
hf download Qwen/Qwen3-8B --dry-run
```

**DeBERTaV3-base — fetch ONLY the PyTorch weights (~357 MiB instead of 1,766 MiB):**

```bash
hf download microsoft/deberta-v3-base \
  --include "config.json" "pytorch_model.bin" "spm.model" "tokenizer_config.json" \
  --local-dir ./models/deberta-v3-base
```

Without `--include` you also download `tf_model.h5` (735 MB) and `rust_model.ot` (742 MB), which are dead weight.

**Legacy equivalent** (still functional, prints a deprecation pointer):

```bash
huggingface-cli download Qwen/Qwen3-8B --local-dir ./models/Qwen3-8B
```

**Login (only if you choose to authenticate):**

```bash
hf auth login          # browser flow by default; or paste a token
hf auth login --force  # to switch tokens
```

Or pass `--token` per command. Source: https://huggingface.co/docs/huggingface_hub/main/en/guides/cli

**Python equivalent** (no CLI needed — models download on first use into `HF_HOME`):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype="bfloat16", device_map="auto")
```

### 5.3 Disk space

| item | size |
|---|---|
| `Qwen/Qwen3-8B` full repo | 16,397,461,266 B = **15.27 GiB** |
| `microsoft/deberta-v3-base` PyTorch-only | ~**357 MiB** |
| `microsoft/deberta-v3-base` full repo | 1,766 MiB |
| **minimum for the DGP stack** | **≈ 15.7 GiB** |

Budget **~35 GiB** in practice: `--local-dir` also writes a `.cache/huggingface/` metadata folder at the directory root, and if you do not use `--local-dir` the hub cache keeps its own copy — downloading both ways doubles the footprint. Add LoRA checkpoints (small: tens of MB each at these ranks) and any cached summary-generation outputs.

### 5.4 VRAM

Base arithmetic, **DERIVED** from the verified parameter count 8,190,735,360:

- bf16 weights: `8,190,735,360 x 2 B` = 16,381,470,720 B = **15.26 GiB**
- NF4 4-bit weights: `x 0.5 B` = **3.81 GiB**; with QLoRA double quantisation at 4.127 bits/param ≈ **3.94 GiB**

**(a) bf16 inference — ESTIMATE ~18–20 GiB**

15.26 GiB weights + KV cache + CUDA context. KV cache per token (GQA, 8 KV heads x 128 head_dim x 36 layers x 2 tensors x 2 bytes) = **147,456 B/token ≈ 0.144 MiB/token**, so even 4k tokens is only ~0.56 GiB. Add ~1–2 GiB of CUDA/cuBLAS/allocator overhead. **A 24 GB card (RTX 3090/4090, A10, L4-24G) is comfortable; 16 GB is not enough in bf16.**
Basis: verified parameter count and config dims; the overhead figure is an ESTIMATE.

**(b) bf16 LoRA finetuning + gradient checkpointing — ESTIMATE ~20–26 GiB**

- frozen bf16 weights: 15.26 GiB
- LoRA params (r=16, attention-only, 15,335,424 params): bf16 weights 30.7 MB + bf16 grads 30.7 MB + AdamW fp32 states (m, v) and fp32 master ≈ 184 MB → **< 0.25 GiB total**. This is the whole point of LoRA: optimizer state is negligible.
- activations with gradient checkpointing, batch 1: stored per-layer inputs = `36 x seq x 4096 x 2 B` → **0.141 GiB @ 512 tok, 0.281 GiB @ 1024, 0.562 GiB @ 2048**, plus an in-layer recompute peak.
- **the sleeper cost is the output logits**: vocab 151,936 x seq x 2 B = **0.14 GiB @ 512, 0.29 GiB @ 1024, 0.58 GiB @ 2048** in bf16, and cross-entropy commonly upcasts to fp32, doubling it, plus a gradient of the same shape. At seq 2048 batch 1 that is ~2–3 GiB just for logits. Scale by batch size. **Use a fused/chunked CE or keep sequences short.** For DGP this is mild (prompts are short), but it dominates if you pad to 2048+.
- **A 24 GB card works for batch 1 at short sequence length with gradient checkpointing; 40–48 GB (A100-40G / A6000) is the comfortable choice.**

**Labelled ESTIMATE** — arithmetic is derived from verified values; the overhead and activation-peak terms are estimates that depend on framework, attention kernel and batch size.

**(c) 4-bit QLoRA — ESTIMATE ~8–12 GiB**

- NF4 + double-quant weights: **~3.94 GiB**
- LoRA adapter + optimizer: < 0.25 GiB (as above)
- activations + logits + overhead: ~3–6 GiB depending on sequence length and batch
- **Fits on a 12 GB card at short sequence length; 16 GB is comfortable.**

Supporting citation — QLoRA: "QLoRA: Efficient Finetuning of Quantized LLMs", Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer. Abstract claim, verbatim: the approach "reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance", via "4-bit NormalFloat (NF4)", "double quantization", and "paged optimizers".
Source: https://arxiv.org/abs/2305.14314

An 8.19B model is ~8x smaller than 65B, so the ~10 GiB figure is consistent with the paper's 48 GB-for-65B result. **The specific 8–12 GiB number is an ESTIMATE.**

**Note:** DGP needs a *frozen* Qwen3-8B for summary generation **in addition to** the LoRA-tuned one. If both are resident simultaneously you need roughly double the weight memory (~30.5 GiB in bf16). **Run summary generation as a separate offline pass and cache the summaries to disk** — this is almost certainly what the authors did, and it makes the whole thing fit on one 24 GB card. The paper does not say.

---

## 6. UNRESOLVED / NOT SPECIFIED BY PAPER

Ordered roughly by impact on reproducing the reported numbers.

1. **`enable_thinking` is never mentioned.** The default (True) breaks the first-token Yes/No readout entirely. Must be set to `False`. **Highest-impact gap.** (§1.7)
2. **Which Yes/No token variant is scored** — `Yes`(9454)/`No`(2753) vs `ĠYes`(7414)/`ĠNo`(2308). The template's trailing `\n\n` implies the no-space pair, but the paper never says, and the wrong choice fails silently. (§2.3)
3. **`Qwen/Qwen3-8B` vs `Qwen/Qwen3-8B-Base`.** The paper says only "Qwen3-8B". The post-trained model is the natural reading of the plain name, and prompt-based Yes/No classification needs instruction-following, so `Qwen/Qwen3-8B` is almost certainly correct — but it is inferred, not stated. The two differ in eos token, max positions and generation defaults. (§1.1)
4. **All LoRA hyperparameters**: `r`, `lora_alpha`, `lora_dropout` are unstated. PEFT defaults (`r=8, alpha=8, dropout=0.0`, scaling 1.0) differ from the common convention (`r=16, alpha=32`, scaling 2.0). (§4.1, §4.2)
5. **AdamW hyperparameters**: learning rate, betas, weight decay, warmup, schedule, epochs, batch size, max sequence length, gradient accumulation — none stated. Learning rate in particular is decisive for LoRA.
6. **DeBERTa vs DeBERTaV3, and which size.** Citation analysis points to DeBERTaV3 (§3.1), but the paper writes "DeBERTa" in prose. Size (`xsmall`/`small`/`base`/`large`) is unstated; hidden size varies 384/768/768/1024, which changes the diffusion kernel's input dimensionality. **Recommend `microsoft/deberta-v3-base` (hidden 768).**
7. **DeBERTa pooling strategy**: CLS vs masked mean vs max. Not specified, and `DebertaV2Model` provides no `pooler_output` to fall back on. Materially changes embedding geometry and hence the kernel. **Recommend masked mean pooling.** (§3.3)
8. **Truncation policy for text longer than 512 tokens.** The tokenizer does not truncate automatically (no `model_max_length` in the v3 config), so this must be set explicitly; whether DGP truncated, chunked, or only ever embedded short summaries is unknown. (§3.4)
9. **Whether `<think>` blocks are stripped from generated summaries** before DeBERTa embedding, if the frozen summariser ran with thinking enabled. Unaddressed. (§1.7.3)
10. **Summary-generation decoding parameters** for the frozen Qwen3-8B (temperature/top_p/top_k/max_new_tokens, and whether greedy). The card explicitly warns "DO NOT use greedy decoding" in thinking mode. Unstated.
11. **Whether the LoRA-tuned and frozen models are the same checkpoint**, and whether summaries were generated once offline or on the fly.
12. **Train/eval prompt-template consistency** — whether both paths render with identical `enable_thinking` and identical system prompts. Not stated; a mismatch would be invisible and damaging.
13. **transformers / peft / torch versions.** `config.json` records `transformers_version: 4.51.0` as the version that *saved* the checkpoint, which is a hint at the era but not a statement of what DGP ran. Current peft at time of writing is `0.20.1.dev0`.
14. **Whether `bias` and `task_type` were set in `LoraConfig`.** `task_type=None` (the default) yields a generic `PeftModel` rather than `PeftModelForCausalLM` — set `TaskType.CAUSAL_LM` explicitly.

---

## 7. Recommended reproduction configuration (a defensible default)

Not the paper's stated values — the paper states almost none. This is the configuration most consistent with what the paper *does* say, plus conventional choices for everything else. Every deviation from the paper is a documented guess.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

MODEL_ID = "Qwen/Qwen3-8B"               # post-trained, NOT -Base  (inferred, §6.3)
EMB_ID   = "microsoft/deberta-v3-base"   # DeBERTaV3, hidden 768    (inferred, §3.1)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="bfloat16", device_map="auto")

# --- CRITICAL: disable thinking so the first generated token is the answer ---
text = tok.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=False,
)
assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), text[-80:]

# --- CRITICAL: no-leading-space Yes/No, resolved at runtime and pinned ---
YES_ID = tok.encode("Yes", add_special_tokens=False); assert YES_ID == [9454], YES_ID
NO_ID  = tok.encode("No",  add_special_tokens=False); assert NO_ID  == [2753], NO_ID
YES_ID, NO_ID = YES_ID[0], NO_ID[0]

# --- LoRA on attention projections only, per the paper's "all attention layers" ---
peft_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,                             # default None is wrong here
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # NOT q_norm/k_norm (RMSNorm)
                                                              # NOT gate/up/down_proj (MLP)
    r=16, lora_alpha=32, lora_dropout=0.05,   # CONVENTION — paper states none
    bias="none",                              # Qwen3 has attention_bias=false
)
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()   # must report 15,335,424 trainable  (see §4.3)
model.gradient_checkpointing_enable()
```

DeBERTa side:

```python
from transformers import AutoModel, AutoTokenizer
etok = AutoTokenizer.from_pretrained(EMB_ID)
emb_model = AutoModel.from_pretrained(EMB_ID)   # DebertaV2Model -> BaseModelOutput, NO pooler_output

enc = etok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
h = emb_model(**enc).last_hidden_state
m = enc["attention_mask"].unsqueeze(-1).float()
emb = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)   # masked mean pooling, 768-d (CONVENTION, §3.3)
```

---

## 8. Source index (every URL actually fetched)

**Qwen3**
- https://huggingface.co/api/models/Qwen/Qwen3-8B
- https://huggingface.co/api/models/Qwen/Qwen3-8B?blobs=true
- https://huggingface.co/Qwen/Qwen3-8B
- https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json
- https://huggingface.co/Qwen/Qwen3-8B/raw/main/generation_config.json
- https://huggingface.co/Qwen/Qwen3-8B/raw/main/tokenizer_config.json
- https://huggingface.co/Qwen/Qwen3-8B/raw/main/vocab.json
- https://huggingface.co/Qwen/Qwen3-8B/raw/main/merges.txt
- https://huggingface.co/Qwen/Qwen3-8B/raw/main/README.md
- https://huggingface.co/Qwen/Qwen3-8B-Base/raw/main/config.json
- https://huggingface.co/Qwen/Qwen3-8B-Base/raw/main/generation_config.json
- https://huggingface.co/Qwen/Qwen3-8B-Base/raw/main/tokenizer_config.json
- https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3/modeling_qwen3.py

**DeBERTa**
- https://huggingface.co/microsoft/deberta-base/raw/main/config.json
- https://huggingface.co/microsoft/deberta-v3-base/raw/main/config.json
- https://huggingface.co/microsoft/deberta-v3-large/raw/main/config.json
- https://huggingface.co/microsoft/deberta-v3-small/raw/main/config.json
- https://huggingface.co/microsoft/deberta-v3-xsmall/raw/main/config.json
- https://huggingface.co/microsoft/deberta-v3-base/raw/main/tokenizer_config.json
- https://huggingface.co/api/models/microsoft/deberta-base?blobs=true
- https://huggingface.co/api/models/microsoft/deberta-v3-base?blobs=true
- https://huggingface.co/api/models/microsoft/deberta-v3-large?blobs=true
- https://huggingface.co/api/models/microsoft/deberta-v3-small?blobs=true
- https://huggingface.co/api/models/microsoft/deberta-v3-xsmall?blobs=true
- https://huggingface.co/docs/transformers/en/model_doc/deberta-v2
- https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/deberta_v2/modeling_deberta_v2.py

**PEFT / LoRA**
- https://raw.githubusercontent.com/huggingface/peft/main/src/peft/tuners/lora/config.py
- https://raw.githubusercontent.com/huggingface/peft/main/src/peft/config.py
- https://raw.githubusercontent.com/huggingface/peft/main/setup.py

**Papers**
- https://arxiv.org/abs/2111.09543  (DeBERTaV3 — He, Gao, Chen)
- https://arxiv.org/abs/2006.03654  (DeBERTa — He, Liu, Gao, Chen)
- https://arxiv.org/abs/2106.09685  (LoRA — Hu et al.)
- https://arxiv.org/abs/2305.14314  (QLoRA — Dettmers et al.)

**Tooling**
- https://huggingface.co/docs/huggingface_hub/main/en/guides/cli
- https://huggingface.co/changelog/better-hf-cli
- https://newreleases.io/project/github/huggingface/huggingface_hub/release/v0.34.0
