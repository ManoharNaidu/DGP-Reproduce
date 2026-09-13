# DGP Baseline Repositories — Consolidated Overview

Reproduction study of **DGP: A Dual-Granularity Prompting Framework for Fraud Detection with Graph-Enhanced LLMs** (AAAI 2026).
Compiled by Subagent C. Every repository below was verified to exist by fetching `https://api.github.com/repos/<owner>/<name>`; commit SHAs come from `/commits?per_page=1`; requirements were fetched from the actual files, not recalled.

Per-method detail lives in `research/baselines/<method>.md`.

---

## What the DGP paper actually claims

Verified by extracting the AAAI camera-ready PDF (`https://ojs.aaai.org/index.php/AAAI/article/download/38541/42503`):

> "...(i) GNNs, including GraphSAGE (Hamilton, Ying, and Leskovec 2017), HGT (Hu et al. 2020), ConsisGAD (Chen et al. 2024), PMP (Zhuo et al. 2024), and GAAP (Duan et al. 2025); (ii) Graph-agnostic models, including MLP (Rosenblatt 1958) and a Qwen3-8B LLM (Team 2025) finetuned on target nodes alone; (iii) LLM-enhanced GNNs, represented by TAPE (He et al. 2024) and FLAG (Yang et al. 2025); and (iv) graph-enhanced LLMs, including GraphGPT (Tang et al. 2024a), HiGPT (Tang et al. 2024b), and InstructGLM (Ye et al. 2024). **All baselines are implemented using official code.**"

> "For all LLM-tuning methods, we use the **Qwen3-8B LLM backbone** (Team 2025) for fair comparison. We apply LoRA (Hu et al. 2022) to all attention layers and use AdamW..."

Hardware: 64× Xeon Gold 6346, 1 TB RAM, **4× NVIDIA A100 80GB**.
Datasets: YelpReviews, AmazonVideo, **E-Commerce (proprietary, ByteDance)**, **LifeService (proprietary, ByteDance)**.

**Two of the four datasets are proprietary and unavailable. Half the results table is not reproducible by anyone outside ByteDance.**

---

## Summary table

| # | Method | Official repo | Verified | Commit (date) | License | Native Yelp/Amazon | Difficulty |
|---|---|---|---|---|---|---|---|
| 1 | GraphSAGE | `williamleif/GraphSAGE` | ✅ 3727★ | `a0fdef95` (2018-09-19) | Other/NOASSERTION | ❌ | Method trivial; repo dead (TF 1.8) |
| 2 | HGT | `acbull/pyHGT` | ✅ 932★ | `85eaccd4` (2023-09-08) | MIT | ❌ (OAG only) | Use PyG/DGL `HGTConv` |
| 3 | ConsisGAD | `Xtra-Computing/ConsisGAD` | ✅ 25★ | `36811c5b` (2024-03-15) | MIT | ✅ **native (DGL built-ins)** | **EASY** |
| 4 | PMP | `JhuoW/PMP` | ✅ 6★ | `3f7629f6` (2024-03-15) | **none** | ✅ **native (auto-download)** | **EASY** |
| 5 | GAAP | `AtwoodDuan/GAAP` | ✅ 5★ | `6a7dbb04` (2025-05-09) | **none** | ⚠️ Yelp yes, **Amazon config missing** | MEDIUM |
| 6 | MLP | n/a | — | — | — | n/a | TRIVIAL |
| 7 | LLM (Qwen3-8B) | `Qwen/Qwen3-8B` (HF) | ✅ | pin HF revision | Apache-2.0 | n/a | GOOD |
| 8 | TAPE | `XiaoxinHe/TAPE` | ✅ 272★ | `d9881f7e` (2025-04-14) | MIT | ❌ (arxiv/cora/pubmed) | HARD |
| 9 | FLAG (fraud) | `BUPT-GAMMA/FLAG` | ✅ 4★ ⚠️ | `cb83944e` (2025-06-04) | **none** | ❌ (Reddit/Instagram) | **VERY HARD** |
| 10 | GraphGPT | `HKUDS/GraphGPT` | ✅ 832★ | `db25a66f` (2024-06-25) | Apache-2.0 | ❌ | **VERY HARD** |
| 11 | HiGPT | `HKUDS/HiGPT` | ✅ 146★ | `2b0793e7` (2024-06-05) | Apache-2.0 | ❌ (IMDB/DBLP/ACM) | **VERY HARD** |
| 12 | InstructGLM | `agiresearch/InstructGLM` | ✅ 274★ | `dd2dd5ec` (2024-02-01) | Apache-2.0 | ❌ (arxiv/cora/pubmed) | HARD |

**No baseline was left without a located repository.** The one that required real detective work — FLAG — is flagged below with a confidence caveat.

---

## FLAG disambiguation (requested explicitly)

Two unrelated papers are named FLAG. Do not confuse them.

| | **CORRECT (DGP's baseline)** | **WRONG** |
|---|---|---|
| Title | FLAG: Fraud Detection with **LLM-enhanced Graph Neural Network** | FLAG: **F**ree **L**arge-scale **A**dversarial Augmentation on **G**raphs |
| Authors | Chengdong Yang, Hongrui Liu, Daixin Wang, Zhiqiang Zhang, Cheng Yang, **Chuan Shi** | Kezhi Kong, Guohao Li, Mucong Ding, Zuxuan Wu, Chen Zhu, Bernard Ghanem, Gavin Taylor, Tom Goldstein |
| Venue | **KDD 2025**, DOI 10.1145/3711896.3737220 | CVPR 2022 / arXiv 2020 |
| Repo | **`BUPT-GAMMA/FLAG`** | `devnkong/FLAG` (143★, MIT, last commit `b507e628` 2022-04-02) — **DO NOT USE** |

**`BUPT-GAMMA/FLAG` attribution — HIGH confidence, but NOT formally declared.** The repo has **no README, no description, no license, and no paper link**, and the `safe-graph` curated list leaves its code column empty. I attribute it on this verified evidence:
- Owner `BUPT-GAMMA` is Chuan Shi's BUPT lab (paper's senior author); repo created 2025-06-04, right after KDD 2025 acceptance.
- I extracted the paper PDF and matched it against the source line by line:
  - Paper: *"we use two social network datasets: Reddit and Instagram"* → `train.py` loads `Reddit/reddit2.pt`; `encode.py` defaults `--path "Instagram/"`.
  - Paper: *"We use Gemma-9b-it as LLM model with LoRA used for fine-tuning and Sentence-BERT as LM"* → `train.py` sets `model_name = "gemma-2-9b-it"`, builds `LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj","v_proj"])`, and loads `SentenceTransformer("all-MiniLM-L6-v2")`.
  - Repo ships the paper's own GNN baselines as files: `bwgnn.py`, `caregnn.py`, `pmp.py`, `geniepath.py`, `dga.py`.

**Recommendation: email the authors to confirm before citing it as official.**

---

## 🚩 Five findings that matter for the reproduction

### 1. "Qwen3-8B backbone" and "official code" are mutually exclusive for GraphGPT and HiGPT

Both repos **subclass the LLaMA modelling classes** to splice graph tokens into `inputs_embeds`:

```python
# GraphGPT/graphgpt/model/GraphLlama.py       # HiGPT/higpt/model/HeteroLlama.py
class GraphLlamaConfig(LlamaConfig):          class HeteroLlamaConfig(LlamaConfig):
    model_type = "GraphLlama"                     model_type = "HeteroLlama"
class GraphLlamaModel(LlamaModel): ...        class HeteroLlamaModel(LlamaModel): ...
class GraphLlamaForCausalLM(LlamaForCausalLM) class HeteroLlamaForCausalLM(LlamaForCausalLM)
AutoConfig.register("GraphLlama", ...)        AutoConfig.register("HeteroLlama", ...)
```

Both README/scripts hard-code **Vicuna-7B-v1.5-16k** (GraphGPT README §3.1: *"Prepare our base model Vicuna ... We generally utilize v1.1 and v1.5 model with 7B parameters"*; HiGPT `higpt_stage_1.sh` line 5: `base_model=/path/to/vicuna-7b-v1.5-16k`).

Worse, **both pin `transformers==4.31.0`, and Qwen3 requires `transformers>=4.51.0`** (`transformers<4.51` raises `KeyError: 'qwen3'`). Even ignoring the model class, the pinned stack physically cannot load Qwen3-8B. Upgrading cascades into `flash-attn==1.0.4`, `pydantic==1.10.9` (v1→v2), `peft==0.4.0`, `deepspeed==0.10.0`, and (HiGPT) Python 3.8.

**InstructGLM is the exception**: no graph projector, no graph tokens — the graph is verbalised into plain text by `all_graph_templates.py`, and `modeling_llama.py` is a ~3 KB `class GLM(LlamaForCausalLM)` wrapper. Re-backboning to Qwen3 is genuinely feasible (though `transformers==4.28.0` still must be upgraded).

**Action: decide per method whether to keep the original backbone (and declare the deviation) or port to Qwen3 (and declare the code deviation). You cannot have both. Report this as a headline reproducibility finding about the DGP paper.**

### 2. FLAG's own paper says the DGP datasets are unsuitable for it

Direct quote from the FLAG PDF:

> "for the public fraud detection datasets, we note that most of them, such as **Yelp-Fraud, Amazon-Fraud, T-Finance and T-Social, lack textual information**, which poses a challenge for applying fraud detection methods that leverage rich text"

FLAG therefore evaluated on Reddit + Instagram, never on Yelp/Amazon. DGP reports FLAG numbers on YelpReviews/AmazonVideo. Either DGP built text-augmented variants of those datasets (plausible — DGP is a text-prompting paper) or it adapted FLAG substantially. **Neither is disclosed.** This is the clearest case where "official code" cannot be taken at face value.

### 3. Only 2 of 12 baselines natively support the fraud datasets

**ConsisGAD** (`dgl.data.FraudAmazonDataset` / `FraudYelpDataset`, README: *"For Amazon and YelpChi, we use the built-in datasets in the DGL package"*) and **PMP** (README: *"directly run the project and these datasets will be download automatically"*) run out of the box. **GAAP** ships a `yelp.yaml` but **no `amazon.yaml`** — you must author it.

Everything else (TAPE, GraphGPT, HiGPT, InstructGLM, FLAG, GraphSAGE, HGT) needs a new dataset loader, new instruction/prompt corpora, and binary-metric plumbing.

### 4. The node-feature question is the biggest silent risk

The classic DGL fraud datasets ship **handcrafted** features (32-dim Yelp, 25-dim Amazon). DGP is a text/LLM paper working on "YelpReviews"/"AmazonVideo" with review text. If DGP fed the GNN baselines **text embeddings** rather than the handcrafted features, then ConsisGAD/PMP/GAAP run with *different inputs than their published configs assume* — and GAAP is worst affected, since its whole method is adaptive **binning of numerical attributes** (`toad`, `rtdl_num_embeddings`), which is meaningless on dense embeddings. That would explain GAAP's mediocre numbers in DGP's table.

**Action: cross-check with Subagent A/B which feature matrix DGP actually used. Whatever it is, use the same one for MLP, or the MLP-vs-GNN comparison is void.**

### 5. Missing licenses and unpinned environments

**No license at all**: `JhuoW/PMP`, `Xtra-Computing/PMP`, `AtwoodDuan/GAAP`, `BUPT-GAMMA/FLAG`. Treat as all-rights-reserved: cite, use for research, do not redistribute or vendor into a public repo.

**Unusable/absent requirements**: ConsisGAD's `requirements.txt` is a raw ~500-line conda dump including `_libgcc_mutex`, `ca-certificates`, `ffmpeg`, `gdb`, `aws-sdk-cpp` — `pip install -r` will fail. TAPE has **no** requirements file (conda recipe in README only). GAAP has **no** requirements file (README dependency list only). FLAG has **no** requirements file at all.

---

## Recommended execution order

**Tier 1 — do first, cheap, high confidence** (CPU/1 GPU, days)
1. **MLP** — write it; fixes the feature pipeline for everything else.
2. **PMP** — cleanest repo in the set; `python main.py --dataset yelp --train_ratio 0.4 --multirun 5 --gpu_id 0`.
3. **ConsisGAD** — `python main.py --config 'config/yelp.yml' --runs 5`; hand-build the env.
4. **GraphSAGE / HGT** — via `dgl.nn.SAGEConv` / `PyG HGTConv`, not the dead original repos.

**Tier 2 — moderate** (1 GPU)
5. **GAAP** — needs `amazon.yaml` authored + Google-Drive data + `toad`/`rtdl`/Lightning/Hydra/wandb stack (Linux only, `dgl==2.3.0` has no Windows wheel).
6. **Qwen3-8B LLM (target-node-only)** — standard LoRA SFT; establishes the prompt format and the label→probability convention that DGP and all LLM baselines must share.

**Tier 3 — expensive, expect deviations** (4× A100)
7. **InstructGLM** — most tractable of the graph-enhanced LLMs; reuse `all_graph_templates.py`, run LoRA SFT on Qwen3-8B.
8. **TAPE** — must generate an LLM explanation corpus for every node with an unpublished fraud prompt.
9. **GraphGPT** — port graph-token injection to Qwen3 *or* accept Vicuna-7B; also needs a fraud-pretrained graph transformer.
10. **HiGPT** — hardest. Additionally requires inventing a heterogeneous schema for Yelp/Amazon and regenerating `node_type.pt`/`edge_type.pt` meta-dicts, plus fixing hard-coded Baidu-internal paths (`/root/paddlejob/workspace/...`).
11. **FLAG** — effectively a reimplementation; no README, no data, hard-coded paths, single commit.

**Not reproducible at all**: the **E-Commerce** and **LifeService** columns (proprietary ByteDance data). Say so plainly in the writeup.

---

## Unresolved questions to route back

1. Which node features did DGP feed the GNN baselines — handcrafted DGL features or text embeddings? (Determines ConsisGAD/PMP/GAAP validity.)
2. What train/val/test ratio? PMP's `--train_ratio` and ConsisGAD's limited-supervision ratio default differently from each other and DGP states neither.
3. Are "YelpReviews"/"AmazonVideo" the standard YelpChi/Amazon-Fraud graphs, or DGP-built text-attributed variants? (FLAG's inclusion implies the latter.)
4. For TAPE, was Qwen3-8B the explanation *generator*, the LM *interpreter* (normally DeBERTa-base), or both?
5. Was Qwen3's `enable_thinking` on or off? Default-on emits `<think>` blocks that break label parsing.
6. How was a continuous score extracted from free-text LLM output for AUC/AUPRC? Unstated, and it must be identical across DGP and all LLM baselines.
7. Did DGP actually port GraphGPT/HiGPT to Qwen3, or did it quietly use Vicuna-7B? (Contact the authors.)

---

## Verification log

All GitHub metadata retrieved from `api.github.com` on 2026-09-13. READMEs and requirements files fetched from `raw.githubusercontent.com`. Source files inspected directly for backbone hard-coding (`GraphLlama.py`, `HeteroLlama.py`, `modeling_llama.py`, FLAG's `train.py`/`chat.py`/`encode.py`). Two PDFs extracted with `pypdf`: the FLAG KDD 2025 paper (`shichuan.org/doc/200.pdf`) and the DGP AAAI camera-ready (`ojs.aaai.org/.../38541/42503`). Qwen3-8B specs confirmed against its HuggingFace model card.

Repos checked and **rejected**: `acheng1996/ConsisGAD` (404, does not exist), `UCLA-DM/pyHGT` (exists but 1★ lab mirror; `acbull/pyHGT` is canonical), `devnkong/FLAG` (exists but is the wrong FLAG paper). `Xtra-Computing/PMP` and `JhuoW/PMP` are at the identical commit `3f7629f6` — either is acceptable.
