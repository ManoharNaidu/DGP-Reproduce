# FLAG (fraud detection) — DISAMBIGUATION REQUIRED

> **Two unrelated methods are called "FLAG". Read this section before touching any code.**
>
> | | Correct one (DGP's baseline) | WRONG one |
> |---|---|---|
> | Title | **FLAG: Fraud Detection with LLM-enhanced Graph Neural Network** | FLAG: **F**ree **L**arge-scale **A**dversarial Augmentation on **G**raphs |
> | Authors | Chengdong Yang, Hongrui Liu, Daixin Wang, Zhiqiang Zhang, Cheng Yang, Chuan Shi | Kezhi Kong, Guohao Li, Mucong Ding, Zuxuan Wu, Chen Zhu, Bernard Ghanem, Gavin Taylor, Tom Goldstein |
> | Venue | KDD 2025 (Applied Data Science track), Toronto | CVPR 2022 / arXiv 2020 |
> | Topic | LLM-enhanced GNN for fraud detection | Adversarial data augmentation for GNN robustness |
> | Repo | `BUPT-GAMMA/FLAG` (see below) | `devnkong/FLAG` — **DO NOT USE** |
>
> `devnkong/FLAG` (143 stars, MIT, last commit `b507e6286797b8a726d594c38fd69c54cd1b19ef` 2022-04-02, description literally "Official implementation of our FLAG paper (CVPR2022)") is a different paper entirely. DGP cites "Yang et al. 2025" and describes "discriminative text extraction ... neighborhood camouflage", which matches only the KDD 2025 paper.

---

- **paper**: FLAG: Fraud Detection with LLM-enhanced Graph Neural Network
- **authors**: Chengdong Yang, Hongrui Liu, Daixin Wang, Zhiqiang Zhang, Cheng Yang, Chuan Shi (BUPT + Ant Group)
- **venue**: KDD 2025 (ACM SIGKDD, 31st, Toronto). DOI 10.1145/3711896.3737220. PDF: http://www.shichuan.org/doc/200.pdf
- **official_repository**: **https://github.com/BUPT-GAMMA/FLAG** (verified to exist — 4 stars, default branch `main`, created 2025-06-04, last push 2025-06-04, commit `cb83944ed8a8a9b070a3f5a167d363973369fc80` "Add files via upload")

  **Confidence: HIGH but not formally declared.** The repo carries **no README, no description, no license, and no link to the paper**, and it is not listed as `[Code]` in `safe-graph/graph-fraud-detection-papers` (that row's code column is empty). I attribute it as official on the following verified evidence:
  - Owner is `BUPT-GAMMA` — Chuan Shi's lab at BUPT, the paper's senior author.
  - Created 2025-06-04, immediately after KDD 2025 acceptance.
  - Source content matches the paper exactly: I read the PDF and the repo files side by side.
    - Paper: *"we use two social network datasets: Reddit [25] and Instagram [25]"* → repo `train.py` loads `Reddit/reddit2.pt`, `Reddit/0_10_0/{train,val,test}_sampler2.pt`; `encode.py` defaults `--path "Instagram/"`.
    - Paper: *"We use Gemma-9b-it [37] as LLM model with LoRA [18] used for fine-tuning and Sentence-BERT [34] as LM"* → repo `train.py` sets `model_name = "gemma-2-9b-it"`, builds a `peft.LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj","v_proj"])`, and instantiates `SentenceTransformer("all-MiniLM-L6-v2")`.
    - Repo ships exactly the paper's GNN baselines as standalone files: `bwgnn.py`, `caregnn.py`, `pmp.py`, `geniepath.py`, `dga.py`.
  - **If you need certainty, email the authors** (shichuan.org) before citing this as official.
- **selected_commit**: `cb83944ed8a8a9b070a3f5a167d363973369fc80` (2025-06-04) — the repo has exactly one content commit.
- **framework**: PyTorch + **PyTorch Geometric** + HuggingFace `transformers` + `peft` + `sentence-transformers`
- **requirements**: **No `requirements.txt`, no README, no setup file.** Inferred from imports:
  ```
  torch, torch_geometric (NeighborLoader, Data, utils.{subgraph,k_hop_subgraph,index_to_mask,mask_to_index})
  transformers (AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer)
  peft (LoraConfig, get_peft_model, PeftModel)
  sentence-transformers  ("all-MiniLM-L6-v2")
  scikit-learn (f1_score, roc_auc_score), numpy, matplotlib
  ```
  Needs a `transformers` recent enough for `gemma2` (>=4.42). Versions are entirely unpinned — you must resolve them yourself.
- **dataset_support**: **NO YelpChi / Amazon — and the FLAG paper explains why.** Quoting the paper directly:
  > *"for the public fraud detection datasets, we note that most of them, such as Yelp-Fraud, Amazon-Fraud, T-Finance and T-Social, lack textual information, which poses a challenge for applying fraud detection methods that leverage rich text"*

  FLAG therefore evaluates on **Reddit** and **Instagram** (text-attributed social graphs), plus an internal Alipay deployment. The repo hard-codes paths `Reddit/reddit2.pt`, `Reddit/0_10_0/...`, `Reddit/model_lora2/gnn.pth`, `Instagram/` — **and none of these data files are in the repo.**
  - This directly contradicts DGP's claim to have run FLAG on YelpReviews/AmazonVideo using official code, unless DGP used its own text-augmented variants of those datasets. **This is the single most important finding for the reproduction: DGP must have adapted FLAG substantially, and did not say so.**
- **training_entrypoint**: `python train.py [--outer_epochs 3 --inner_epochs 10 --lr 1e-4 --hidden 32 --dropout 0.5 --weight_decay 5e-4 --patience 10 --alpha 0.1 --beta 0.1]`. A variant `train1.py` also exists. Pre-step: `python encode.py --path Instagram/ ...` produces the Sentence-BERT node encodings; the GNN checkpoint it loads (`Reddit/model_lora2/gnn.pth`) must exist beforehand.
- **evaluation_entrypoint**: `python test.py --path Reddit/0_10_0/ ...` and `python test_dual.py`. Metrics: `f1_score`, `roc_auc_score` from sklearn.
- **adaptation_required** (VERY HIGH):
  1. Hard-coded absolute-ish dataset paths (`Reddit/...`, `Instagram/...`) and hard-coded checkpoint paths must all be parameterised.
  2. Hard-coded `model_name = "gemma-2-9b-it"` → swap to `Qwen/Qwen3-8B` to honour DGP's fair-comparison claim. Mechanically easy (it is `AutoTokenizer`/`AutoModelForCausalLM`, and the LoRA `target_modules=["q_proj","v_proj"]` names are identical in Qwen3), but it changes the method from what the FLAG paper published.
  3. Build the Reddit/Instagram-style preprocessed tensors (`*.pt` graphs, `NeighborLoader` sampler dumps) for YelpReviews/AmazonVideo from scratch — the preprocessing scripts that produced them are **not in the repo**.
  4. No README means the intended run order (`encode.py` → GNN pretrain → `train.py` → `test.py`) is inferred, not documented. Files `chat.py` / `chat1.py` (prompting) and `dga.py` have no usage instructions.
- **deviations**: Everything above. Realistically the FLAG row cannot be reproduced as "official code on DGP's datasets"; it will be a reimplementation guided by the repo.
- **license**: **NONE declared** on `BUPT-GAMMA/FLAG`. (The unrelated `devnkong/FLAG` is MIT — irrelevant.)
- **reproducibility_status**: **POOR / HIGH RISK.** Official code very probably located but unlabelled, undocumented, unlicensed, unpinned, single-commit, with no data and hard-coded paths — and its own paper states the DGP datasets are unsuitable for it. Treat any reproduced FLAG number as a reimplementation, and flag the discrepancy with DGP's "official code" claim in the writeup.
