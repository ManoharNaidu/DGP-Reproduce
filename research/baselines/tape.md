# TAPE

- **paper**: Harnessing Explanations: LLM-to-LM Interpreter for Enhanced Text-Attributed Graph Representation Learning
- **authors**: Xiaoxin He, Xavier Bresson, Thomas Laurent, Adam Perold, Yann LeCun, Bryan Hooi
- **venue**: ICLR 2024 (arXiv:2305.19523; OpenReview `RXFVcynVe1`)
- **official_repository**: https://github.com/XiaoxinHe/TAPE (verified, 272 stars, default branch `main`, MIT) — README states "Official Implementation of ICLR 2024 paper".
- **selected_commit**: `d9881f7e277823d0c3a56851c6ccd0412adc773f` (2025-04-14, "Update README.md")
- **framework**: PyTorch + PyG + DGL + HuggingFace `transformers`
- **requirements**: **No `requirements.txt` in the repo** (confirmed 404). The README gives a conda recipe:
  ```
  conda create --name TAPE python=3.8
  conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
  conda install -c pyg pytorch-sparse pytorch-scatter pytorch-cluster pyg
  pip install ogb
  conda install -c dglteam/label/cu113 dgl
  pip install yacs
  pip install transformers
  pip install --upgrade accelerate
  ```
  `torch 1.12.1 + cu113` does **not** support A100 80GB well and is incompatible with the `transformers>=4.51` needed for Qwen3. Expect to bump to torch 2.x + cu118/cu121 manually.
- **dataset_support**: **NO YelpChi / Amazon-Fraud.** TAPE is a citation/product **node-classification** codebase. Built-in datasets: `ogbn-arxiv`, `ogbn-products` (subset), `arxiv_2023`, `cora`, `pubmed`. Loaders live in `core/data_utils/load_{arxiv,arxiv_2023,cora,products,pubmed}.py`; raw text goes to `dataset/<name>_orig/`, LLM outputs to `gpt_responses/<name>/`, and precomputed GPT predictions ship in `gpt_preds/*.csv`.
- **training_entrypoint** (three stages):
  ```
  # 1. Finetune the LM on original text attributes
  WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.trainLM dataset ogbn-arxiv
  # 1b. ...and on the LLM explanations
  ... python -m core.trainLM dataset ogbn-arxiv lm.train.use_gpt True
  # 2. Train GNNs on the resulting features
  python -m core.trainEnsemble gnn.model.name SAGE
  python -m core.trainGNN gnn.train.feature_type TA_P_E
  ```
  `run.sh` reproduces the published numbers end to end. Config system is `yacs` via `core/config.py` (CLI overrides are `key value` pairs).
- **evaluation_entrypoint**: Integrated in `core/trainGNN.py` / `core/trainEnsemble.py`; also `core/GNNs/ensemble_trainer.py` for the TA+P+E ensemble. Checkpoints (`*.ckpt`) and TAPE features (`*.emb`) are downloadable from the authors' Google Drive.
- **adaptation_required** (HIGH):
  1. **Write a new loader** `core/data_utils/load_yelp.py` / `load_amazon.py` returning a PyG `Data` plus the raw review text list, and register it in `core/data_utils/load.py`.
  2. **Regenerate the explanations.** TAPE's whole premise is LLM-produced *explanation + ranked prediction* text per node. The shipped `gpt_preds/` and `gpt_responses/` cover only the 5 academic datasets. For YelpReviews/AmazonVideo you must run an LLM over every node with a fraud-detection-adapted prompt (TAPE's prompt asks for a ranked list of paper categories; you need "fraud/benign + why"). **This is the expensive step and it is a prompt DGP does not publish.**
  3. **Binary + heavy class imbalance.** TAPE's LM head and metric code assume multi-class balanced accuracy. You must swap in AUC / F1-macro / AUPRC and likely a weighted loss.
  4. **Backbone**: TAPE's LM interpreter defaults to DeBERTa-base (`core/LMs/model.py`). DGP says LLM-tuning methods use Qwen3-8B — but TAPE is an *LLM-enhanced GNN*, where the LLM is only the frozen explanation generator and the trained LM is small. Whether DGP used Qwen3-8B as the *explainer*, the *LM interpreter*, or both is unstated. **Flag as a genuine ambiguity.**
  5. E-Commerce / LifeService: proprietary, cannot be run.
- **deviations**: Custom dataset loaders, custom fraud explanation prompt, binary metrics, and a modernised torch stack — all ours. The reproduced TAPE number is therefore only loosely "official code".
- **license**: MIT
- **reproducibility_status**: **MODERATE-TO-DIFFICULT.** Repo is healthy and maintained, but it has zero fraud-dataset support and the required LLM explanation corpus for Yelp/Amazon does not exist and must be generated from an unpublished prompt.
