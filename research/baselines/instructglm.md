# InstructGLM

- **paper**: Language is All a Graph Needs
- **authors**: Ruosong Ye, Caiqi Zhang, Runhui Wang, Shuyuan Xu, Yongfeng Zhang (Rutgers University)
- **venue**: EACL 2024 (Findings); arXiv:2308.07134
- **official_repository**: https://github.com/agiresearch/InstructGLM (verified, 274 stars, default branch `main`, Apache-2.0)
- **selected_commit**: `dd2dd5ecc48089a44402a4e0744abf891be0c3c8` (2024-02-01, "Update README.md") on `main`. (Repo `pushed_at` is 2025-03-13, i.e. activity on a non-default branch — pin `main` explicitly.)
- **framework**: PyTorch + HuggingFace `transformers` + `peft` (LoRA) + DDP (`torch.distributed.launch`)
- **requirements** (`requirements.txt` — short and clean, fetched):
  ```
  torch==1.13.1
  transformers==4.28.0       <-- HARD BLOCKER for Qwen3, see below
  tokenizers==0.13.3
  peft==0.3.0
  accelerate==0.19.0
  bitsandbytes==0.39.0
  sentencepiece==0.1.97
  torch-geometric==2.3.0
  dgl==1.1.0
  ogb==1.3.6
  numpy==1.23.5, pandas==1.5.2, PyYAML==6.0, protobuf==3.20.3,
  packaging==22.0, requests==2.28.1, tqdm==4.65.0, wheel==0.37.1
  ```
  `torch==1.13.1` predates proper A100-80GB/bf16 support in this stack; expect to upgrade.
- **dataset_support**: **NO fraud datasets.** Three citation benchmarks only: **ogbn-arxiv, Cora, PubMed**. The repo's structure is one hard-forked source tree per (backbone × dataset):
  `llama_arxiv_src/`, `llama_cora_src/`, `llama_pubmed_src/`, `flan_arxiv_src/`, `flan_cora_src/`, `flan_pubmed_src/` — **six near-duplicate copies**, each with its own `pretrain.py` (~50 KB), `all_graph_templates.py` (the natural-language graph-description prompt bank), and a giant per-dataset data module (`arxiv.py` 166 KB, `Cora.py` 161 KB, `PubMed.py` 163 KB).
  Preprocessed data and the LLaMA-7B checkpoint are Google Drive downloads; `data_preprocess/{Arxiv,Cora,PubMed}_preprocess/` holds the scripts.
- **training_entrypoint**:
  ```
  bash scripts/train_llama_arxiv.sh 8     # "8" = number of GPUs for DDP
  # expands to:
  PYTHONPATH=$PYTHONPATH:./llama_arxiv_src python -m torch.distributed.launch --nproc_per_node=8 \
      --master_port 12321 llama_arxiv_src/pretrain.py --distributed --multiGPU --seed 42 \
      --gradient_accumulation_steps 8 --train Arxiv --valid Arxiv --batch_size 4 --optim adamw \
      --warmup_ratio 0.05 --clip_grad_norm 1.0 --losses 'link,classification' \
      --backbone './7B' --output snap/arxiv-7b --epoch 2 --max_text_length 2048 \
      --gen_max_length 64 --lr 0.00008
  ```
  Also `train_llama_{cora,pubmed}.sh` and `train_flan_{arxiv,cora,pubmed}.sh`.
- **evaluation_entrypoint**:
  ```
  bash scripts/test_llama_arxiv.sh 8
  ```
  (+ `test_llama_{cora,pubmed}.sh`, `test_flan_*.sh`.) Checkpoints published on Google Drive.

## Base LLM: can it be swapped to Qwen3-8B?

**Two assumed backbones, both hard-forked into separate source trees:**
- **LLaMA-7B** — README step 2: *"Download Llama-7b pretrained checkpoint via this Google Drive link ... Please then put the ./7B folder under the same path with ./scripts folder."* Scripts pass `--backbone './7B'`. The repo ships an empty `7B/note.txt` placeholder.
- **Flan-T5** — the `flan_*_src/` trees (`modeling_flan.py`).

**LLaMA hard-coding** in `llama_*_src/modeling_llama.py` (verified):
```python
from transformers.models.llama.modeling_llama import *
...
class GLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, ...):
        outputs = self.model(input_ids=input_ids, ...)
```
`--backbone` is a *path*, not a model-class selector; `pretrain_model.py` instantiates `GLM`, which subclasses `LlamaForCausalLM`.

**However — InstructGLM is the EASIEST of the three to re-backbone.** Crucially, InstructGLM has **no graph projector and no graph tokens**: the graph is verbalised entirely into natural language by `all_graph_templates.py` and fed as ordinary text. The `GLM` subclass in `modeling_llama.py` is a thin ~3 KB wrapper (compare GraphGPT's 435-line `GraphLlama.py`) that only customises `forward()` for the multi-task `'link,classification'` loss and constrained generation. There is nothing architecture-specific about the method.

**Effort to reach Qwen3-8B: MEDIUM.** Realistic options, in order of preference:
1. **Reimplement the prompting scheme on Qwen3-8B directly.** Reuse `all_graph_templates.py` (the actual scientific content of the paper) and run standard LoRA SFT. Loses the repo's custom trainer but faithfully reproduces the method, and is what DGP most plausibly did.
2. Make a `modeling_qwen.py` that does `class GLM(Qwen3ForCausalLM)` with the same `forward()` — mechanically a near-copy, since the wrapper touches only `self.model(...)` outputs and `lm_head`. Then upgrade `transformers` 4.28 → >=4.51 (`transformers<4.51` raises `KeyError: 'qwen3'`), which will break `peft==0.3.0`, `bitsandbytes==0.39.0`, and `torch.distributed.launch` (deprecated → `torchrun`).
3. Keep LLaMA-7B and declare the backbone deviation.

**Conclusion: DGP's Qwen3-8B claim is achievable here, but not with the official code unmodified** — `transformers==4.28.0` cannot load Qwen3 at all.

- **adaptation_required** (HIGH):
  1. Re-backbone per above, and upgrade `transformers`/`peft`/`torch`.
  2. **Write a 7th source tree** (or generalise one) for the fraud datasets — the repo has no dataset abstraction, only six forked copies. Requires new `all_graph_templates.py` prompts describing a review graph, plus a new data module replacing the 160 KB `arxiv.py`/`Cora.py`.
  3. Restate node classification as binary fraud/benign, and extract a probability score for AUC/AUPRC from constrained generation.
  4. `torch.distributed.launch` → `torchrun`; scripts assume 8 GPUs.
  5. E-Commerce / LifeService: proprietary, cannot be run.
- **deviations**: Backbone (LLaMA-7B → Qwen3-8B), dependency stack, entirely new dataset tree and prompt templates, new metric extraction.
- **license**: Apache-2.0 (`LICENSE.txt` present)
- **reproducibility_status**: **MODERATE.** Of the three graph-enhanced LLMs this is the most tractable, because the method is pure text prompting with no graph encoder or projector to port. The cost is in writing a fraud dataset tree from scratch, since the repo has no dataset abstraction layer.
