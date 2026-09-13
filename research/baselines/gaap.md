# GAAP (Global Attribute-Association Pattern Aggregation)

- **paper**: Global Attribute-Association Pattern Aggregation for Graph Fraud Detection
- **authors**: Mingjiang Duan, Da He, Tongya Zheng, Lingxiang Jia, Mingli Song, Xinyu Wang, Zunlei Feng
- **venue**: AAAI 2025 (Main Track), Proceedings of AAAI vol. 39(11), pp. 11616-11624. DOI 10.1609/aaai.v39i11.33264
- **official_repository**: https://github.com/AtwoodDuan/GAAP (verified, 5 stars, default branch `master`, **no license file**)
  - Repo self-identifies: "This is the official implementation of the following paper: Global Attribute-Association Pattern Aggregation for Graph Fraud Detection ... AAAI 2025 Main Track".
  - Cross-confirmed as the code link in three independent curated lists: `AI4Risk/awesome-fraud-detection`, `safe-graph/graph-fraud-detection-papers`, `mala-lab/Awesome-Deep-Graph-Anomaly-Detection`.
  - README filename is lowercase `readme.md` (a `README.md` fetch 404s).
- **selected_commit**: `6a7dbb0447c4897504525de49e41a0526ee777f8` (2025-05-09, "增加dgraphfin参数配置")
- **framework**: PyTorch + **DGL** + **PyTorch Lightning** + Hydra (config) + Weights & Biases
- **requirements** (no `requirements.txt` in repo; "Mainly Dependencies" section of `readme.md`):
  ```
  torch==2.2.0
  dgl==2.3.0
  toad==0.1.5              # credit-scoring binning library — the unusual one
  pandas==2.2.2
  numpy==1.26.4
  scikit-learn==1.5.0
  lightning==2.3.0
  wandb==0.16.5
  hydra-core==1.3.2
  rtdl_num_embeddings==0.0.12   # numerical-feature embeddings (Gorishniy et al.)
  ```
  `dgl==2.3.0` has **no official Windows wheel** — plan on Linux/WSL. `toad` and `rtdl_num_embeddings` are niche pip packages required by the attribute-binning module.
- **dataset_support**: **YES for YelpChi.** Config files present under `config/SAGE_MiniF_DyPLE_MHA/`: `yelp.yaml`, `tfinance.yaml`, `tsocial.yaml`, `elliptic.yaml`, `tolokers.yaml`, `dgraphfin.yaml`.
  - **NOTE: there is NO `amazon.yaml`.** The paper claims 7 datasets but the released configs cover 6 and Amazon is not among them. Running GAAP on AmazonVideo requires writing a new Hydra config and verifying the binning hyper-parameters yourself.
  - Datasets are NOT auto-downloaded: a single Google Drive archive must be fetched and unzipped into a `datasets/` folder inside the repo:
    https://drive.google.com/file/d/1txzXrzwBBAOEATXmfKzMUUKaXh6PJeR1/view?usp=sharing
- **training_entrypoint**:
  ```
  python mycode/exp/101_retrain.py -cn yelp        # YelpChi
  python mycode/exp/101_retrain.py -cn tfinance
  python mycode/exp/101_retrain.py -cn elliptic
  python mycode/exp/101_retrain.py -cn tolokers
  python mycode/exp/102_retrain_reallinear_att.py -cn tsocial
  python mycode/exp/102_retrain_reallinear_att.py -cn dgraphfin
  ```
  (`-cn` is the Hydra `--config-name` short flag.)
- **evaluation_entrypoint**: None separate. PyTorch Lightning `Trainer` runs fit+test in the same script; metrics are logged to **wandb** (`wandb==0.16.5`). Set `WANDB_MODE=offline` to run without an account. Single-run helper: `mycode/run_one.py`.
- **Code map**: `mycode/nn/gnn.py` + `mycode/nn/litnn.py` (model + Lightning module), `mycode/utils/dataloader.py`, `mycode/ENV.py` (paths), `config/config.yaml` (root Hydra config).
- **adaptation_required** (MEDIUM):
  1. Write an `amazon.yaml` config — not shipped. Copy `yelp.yaml` and adjust input dim / bin counts.
  2. Point `mycode/ENV.py` / `config/config.yaml` at your dataset root and supply the DGP-preprocessed graphs in whatever format `mycode/utils/dataloader.py` expects (inspect it — it reads the Google Drive dumps, likely DGL `.bin` / pickled graphs).
  3. GAAP's core idea is **adaptive binning of numerical attributes** (`toad`, `rtdl_num_embeddings`). If DGP feeds GAAP dense LLM text embeddings instead of the original tabular/handcrafted attributes, the binning module is operating far outside its design regime — this is a real threat to matching DGP's reported GAAP numbers (which are notably mediocre in DGP's table).
  4. Needs a wandb account or `WANDB_MODE=offline`.
  5. E-Commerce / LifeService: proprietary, cannot be run.
- **deviations**: Amazon config must be authored by us. No LICENSE. No pinned `requirements.txt` (README list only). Feature representation mismatch risk (binning vs. dense embeddings) is significant and should be documented.
- **license**: **NONE declared** (GitHub API returns `license: null`).
- **reproducibility_status**: **MODERATE.** Official code confirmed and recent (last push 2025-05-09), but no Amazon config, no license, manual Google-Drive data step, and an exotic dependency stack (toad + rtdl + Lightning + Hydra + wandb).
