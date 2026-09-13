# ConsisGAD

- **paper**: Consistency Training with Learnable Data Augmentation for Graph Anomaly Detection with Limited Supervision
- **authors**: Nan Chen, Zemin Liu, Bryan Hooi, Bingsheng He, Rizal Fathony, Jun Hu, Jia Chen
- **venue**: ICLR 2024
- **official_repository**: https://github.com/Xtra-Computing/ConsisGAD (verified, 25 stars, default branch `main`, MIT)
  - OpenReview: https://openreview.net/forum?id=elMKXvhhQ9
  - Contact given in README: nanchansysu@gmail.com
- **selected_commit**: `36811c5bc79be49c9740f25a1f260496bb4736af` (2024-03-15, "Update README.md")
- **framework**: PyTorch + **DGL**
- **requirements** (`requirements.txt` is a full `conda list` export, ~500 lines — key pins, fetched):
  ```
  dgl==1.1.0.cu118
  cuda-toolkit==11.7.1 / cuda==11.7.1
  (torch pin is in the conda block; the CUDA 11.8 dgl wheel implies torch 2.0.x cu118)
  numpy, scipy, scikit-learn, pandas, pyyaml, wandb
  ```
  WARNING: the requirements file is a raw conda environment dump including OS-level packages (`_libgcc_mutex`, `ca-certificates`, `ffmpeg`, `gdb`, `aws-sdk-cpp`...). `pip install -r requirements.txt` **will fail on Windows and will fail on most Linux setups**. Hand-pick: `torch==2.0.1+cu118`, `dgl==1.1.0+cu118`, `numpy`, `scikit-learn`, `pyyaml`.
- **dataset_support**: **YES — native, first-class.** README:
  > "For Amazon and YelpChi, we use the built-in datasets in the DGL package"
  i.e. `dgl.data.FraudAmazonDataset()` and `dgl.data.FraudYelpDataset()` — downloaded automatically by DGL into `data/`.
  T-Finance / T-Social are downloaded manually from https://github.com/squareRoot3/Rethinking-Anomaly-Detection and unzipped into `data/`.
  Expected files: DGL cache under `data/` (`FraudAmazon/`, `FraudYelp/`), plus `tfinance`/`tsocial` DGL graph dumps for the other two.
- **training_entrypoint**:
  ```
  python main.py --config 'config/yelp.yml' --runs 5
  python main.py --config 'config/amazon.yml' --runs 5
  ```
  Hyper-parameters live entirely in `config/*.yml`.
- **evaluation_entrypoint**: No separate script — evaluation is inside the training loop via `modules/evaluation.py`; AUC / F1-macro / AUPRC are reported per run and averaged over `--runs`. Trained weights land in `model-weights/`.
- **Key code locations** (per README): GNN backbone `simpleGNN_MR` in `models.py`; consistency training `UDA_train_epoch` in `main.py`; learnable augmentation `SoftAttentionDrop` in `main.py`; loader `modules/data_loader.py`.
- **adaptation_required** (LOW for YelpChi/Amazon, MEDIUM for DGP's variants):
  1. YelpChi/Amazon run out of the box — this is the easiest baseline in the set.
  2. **BUT**: DGP uses "YelpReviews" and "AmazonVideo" with **LLM/text-derived node features**, not the classic 32-dim (Yelp) / 25-dim (Amazon) handcrafted feature vectors that `dgl.data.Fraud*Dataset` ships. To match DGP you must substitute your own feature matrix into the DGL graph's `ndata['feature']` and adjust `in_feats` in the config. Confirm with Subagent A/B which feature set DGP actually fed the GNN baselines — this is the single biggest source of number mismatch.
  3. DGP's train ratio / split protocol must be matched; ConsisGAD's configs default to the ICLR paper's limited-supervision ratios (e.g. 1%), which is NOT the same as DGP's setting.
  4. E-Commerce / LifeService: proprietary, cannot be run.
- **deviations**: `requirements.txt` unusable as-is (conda dump) — record the hand-built environment. Node features likely replaced with DGP's text embeddings.
- **license**: MIT
- **reproducibility_status**: **GOOD.** Official code exists, is clean, targets exactly YelpChi+Amazon, and runs with a one-line command. Main risk is feature/split mismatch with DGP, not the code.
