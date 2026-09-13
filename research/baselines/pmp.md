# PMP (Partitioning Message Passing)

- **paper**: Partitioning Message Passing for Graph Fraud Detection
- **authors**: Wei Zhuo, Zemin Liu, Bryan Hooi, Bingsheng He, Guang Tan, Rizal Fathony, Jia Chen
- **venue**: ICLR 2024
- **official_repository**: https://github.com/JhuoW/PMP (verified, 6 stars, default branch `master`, **no license file**)
  - Mirror by the same collaboration, identical HEAD: https://github.com/Xtra-Computing/PMP (verified, 20 stars, branch `master`, no license)
  - Both repos are at the **same commit** — either is fine. `JhuoW/PMP` is the first author's ("jhuow@proton.me" is the contact in the README).
  - OpenReview: https://openreview.net/forum?id=tEgrUrUuwA ; arXiv mirror: https://arxiv.org/pdf/2412.00020
- **selected_commit**: `3f7629f6c180891a0bc1bba3c66d94d288a1ddae` (2024-03-15, "upd readme") — identical SHA in both repos.
- **framework**: PyTorch + **DGL** + PyTorch Geometric (both)
- **requirements** (`requirements.txt`, fetched — clean and pip-installable):
  ```
  torch==2.0.1
  dgl==1.1.1+cu118
  torch_geometric==2.3.1
  torch_scatter==2.1.1+pt20cu118
  torch_sparse==0.6.17+pt20cu118
  torchmetrics==0.11.4
  ogb==1.3.6
  imbalanced_learn==0.10.1
  scikit_learn==1.2.1
  scipy==1.10.0
  numpy==1.23.5
  networkx==2.8.4
  matplotlib==3.7.0
  PyYAML==6.0 / 6.0.1
  tqdm==4.64.1
  ```
  Note `torch_scatter`/`torch_sparse` need the pyg wheel index (`-f https://data.pyg.org/whl/torch-2.0.1+cu118.html`).
- **dataset_support**: **YES — native, auto-downloading.** README:
  > "``Yelp`` and ``amazon`` dataset: directly run the project and these datasets will be download automatically."
  (i.e. DGL's `FraudYelpDataset` / `FraudAmazonDataset`.)
  T-Finance / T-Social must be downloaded from the Google Drive link provided by https://github.com/squareRoot3/Rethinking-Anomaly-Detection and placed under `datasets/`.
- **training_entrypoint**:
  ```
  python main.py --dataset yelp   --train_ratio 0.4 --multirun 5 --gpu_id 0
  python main.py --dataset amazon --train_ratio 0.4 --multirun 5 --gpu_id 0
  python main.py --dataset tfinance --train_ratio 0.4 --gpu_id 0
  ```
  Hyper-parameters in `config/`.
- **evaluation_entrypoint**: Integrated — `training_procedure/` holds train/valid/test; metrics printed per run and averaged over `--multirun`. A pretrained-model path is described in the README ("Pretrained Model" section) loading from `checkpoints/`.
- **Directory map**: `config/` (hparams), `checkpoints/` (weights), `DataHelper/` (dataset processing), `model/` (PMP), `training_procedure/`, `utils/`.
- **adaptation_required** (LOW for YelpChi/Amazon, MEDIUM for DGP's variants):
  1. Yelp/Amazon run with a single command — nothing to adapt structurally.
  2. `--train_ratio` must be set to whatever DGP used (DGP does not obviously state it; PMP's own paper uses 0.4 and 0.01). **Flag this as an unresolved parameter.**
  3. Same feature caveat as ConsisGAD: DGP's YelpReviews/AmazonVideo carry text-derived features, while PMP consumes the DGL built-in handcrafted features. Substituting features means editing `DataHelper/` and the input dim in `config/`.
  4. PMP partitions neighbours by *predicted class*, so it depends on label availability — verify DGP's label budget matches.
  5. E-Commerce / LifeService: proprietary, cannot be run.
- **deviations**: No LICENSE in either repo — usage terms are unstated (treat as "all rights reserved"; cite, don't redistribute). Train ratio and feature source must be chosen and documented.
- **license**: **NONE declared** (GitHub API returns `license: null` for both `JhuoW/PMP` and `Xtra-Computing/PMP`).
- **reproducibility_status**: **GOOD.** Clean pip-installable requirements, auto-downloading Yelp/Amazon, one-line entrypoint. Best-behaved repo in this baseline set.

---

## Verified by the lead engineer from the upstream source (2026-09-13, commit 3f7629f6)

| Fact (from source) | Consequence |
|---|---|
| `requirements.txt`: dgl 1.1.1+cu118, torch 2.0.1, torch_geometric 2.3.1, torch_scatter/torch_sparse +pt20cu118 | CUDA builds pinned |
| `training_procedure/evaluate.py` hard-codes `torch.cuda.current_device()` | **Cannot run on CPU without shimming CUDA calls → run on the GPU machine** |
| Data enters through `DataHelper.datasetHelper.DatasetHelper.load` (DGL `FraudDataset` + masks, `data.etypes`) | Same bundle as ConsisGAD; adapter patches `load` to build the graph from our bundle and set our masks |
| Default config `amazon.yml`: `homo: true` (converts to homogeneous), `norm_feat: true` (row-normalises features), `monitor: auc_gnn`, `patience: 20` | Keep upstream settings; row normalisation of DeBERTa features is part of upstream preprocessing |
| DGL loaders use `num_workers=8` | Linux/GPU run expected; set lower on Windows |
| **Macro-F1 at fixed threshold 0.5** on `sigmoid(logits)[:, 1]` (`thres: 0.5`, `threshold_moving: true`) | Differs from ConsisGAD's validation-tuned threshold; both protocols are recomputed for every method |
| No licence | Fetch only; never redistribute |

Status: NOT_STARTED (GPU-dependent). Adapter design identical to ConsisGAD's.
