# HGT (Heterogeneous Graph Transformer)

- **paper**: Heterogeneous Graph Transformer
- **authors**: Ziniu Hu, Yuxiao Dong, Kuansan Wang, Yizhou Sun
- **venue**: WWW 2020
- **official_repository**: https://github.com/acbull/pyHGT (verified, 932 stars, default branch `master`, MIT)
  - NOTE: a lab mirror exists at https://github.com/UCLA-DM/pyHGT (verified but only 1 star, created 2023-10-29). `acbull/pyHGT` is the canonical one — it is the first author's account and the one linked from the paper. The lab mirror's latest commit is identical content (`85eaccd...` merge references `UCLA-DM`).
  - Official alternative reference implementation endorsed in the README: https://github.com/dmlc/dgl/tree/master/examples/pytorch/hgt
- **selected_commit**: `85eaccd482bc1d1af56c2de297b6e3a88b96d5cd` (2023-09-08, "Merge pull request #60 from UCLA-DM/dependabot/pip/torch-1.1")
- **framework**: PyTorch + PyTorch Geometric
- **requirements** (from README; repo also has `requirements.txt`):
  ```
  torch 1.3.0
  torch-geometric 1.3.2
  torch-cluster==1.4.5, torch-scatter==1.3.2, torch-sparse==0.4.3
  gensim, scikit-learn, tqdm, dill, pandas
  ```
  These pins are ~2019-era and are NOT installable against CUDA 11/12 + A100.
- **dataset_support**: NO YelpChi / Amazon. The repo is built entirely around the **Open Academic Graph (OAG)** (CS / ML / NN paper splits, 1.9-8.1 GB preprocessed `.pk` dumps hosted by the authors). Data structure is a custom `class Graph` with `Graph.node_feature` (pandas DataFrame) and `Graph.edge_list` (nested dict).
- **training_entrypoint**: `python train_paper_field.py --data_dir ... --conv_name hgt` (also `train_paper_venue.py`, `train_author_disambiguation.py`, `train_pf_reddit.py`). There is no single generic `main.py`.
- **evaluation_entrypoint**: Same scripts — validation/test NDCG/MRR is computed in-loop; `eval_*.py` variants exist per task.
- **adaptation_required** (HIGH for the official repo, LOW via DGL/PyG):
  1. The OAG-specific data pipeline (`data.py`, `sample_subgraph`) would have to be replaced wholesale to ingest a review graph.
  2. YelpChi/Amazon are **multi-relation homogeneous** graphs (one node type, 3 edge relations for Yelp: R-U-R, R-S-R, R-T-R; Amazon: U-P-U, U-S-U, U-V-U). To run HGT you must cast them to a heterogeneous graph — either treat the 3 relations as 3 edge types over a single node type, or materialise reviewer/product/star nodes. This is a modelling decision the DGP paper does not specify, so any reproduction here involves a judgement call.
  3. Practical recommendation: use `dgl.nn.HGTConv` or `torch_geometric.nn.HGTConv` (both are official-adjacent, actively maintained, and the pyHGT README itself points at the DGL example).
  4. DGP's E-Commerce / LifeService graphs are genuinely heterogeneous, which is presumably why HGT is in the table — but those datasets are proprietary (ByteDance) and unavailable.
- **deviations**: Expect to use `PyG HGTConv` / `DGL HGTConv` rather than `acbull/pyHGT`. Relation-to-node-type casting for YelpChi/Amazon is undocumented in DGP and must be chosen and recorded.
- **license**: MIT
- **reproducibility_status**: **Method reproducible via PyG/DGL; official repo not runnable on a modern stack and not applicable to fraud datasets without a rewrite.**
