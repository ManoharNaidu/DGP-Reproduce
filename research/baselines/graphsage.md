# GraphSAGE

- **paper**: Inductive Representation Learning on Large Graphs
- **authors**: William L. Hamilton, Rex Ying, Jure Leskovec
- **venue**: NeurIPS (NIPS) 2017
- **official_repository**: https://github.com/williamleif/GraphSAGE (verified, 3727 stars, default branch `master`)
  - Official PyTorch reference re-implementation by the same author: https://github.com/williamleif/graphsage-simple (verified, 1054 stars, branch `master`)
- **selected_commit**:
  - `williamleif/GraphSAGE` @ `a0fdef95dca7b456dab01cb35034717c8b6dd017` (2018-09-19)
  - `williamleif/graphsage-simple` @ `d3105e5223e9602ba3ef2dd27e343d044fe4bb5f` (2018-06-24)
- **framework**: TensorFlow 1.x (original repo). `graphsage-simple` is PyTorch 0.x-era.
- **requirements** (from `requirements.txt` of the official repo, fetched):
  ```
  tensorflow==1.8.0, tensorboard==1.8.0, networkx==1.11, numpy==1.14.5,
  scipy==1.1.0, scikit-learn==0.19.1, protobuf==3.6.0, six==1.11.0,
  absl-py==0.2.2, astor==0.6.2, gast==0.2.0, grpcio==1.12.1, Markdown==2.6.11,
  mock==2.0.0, Werkzeug==0.14.1, termcolor==1.1.0
  ```
  `graphsage-simple` ships no `requirements.txt`.
- **dataset_support**: NO native YelpChi / Amazon support. The repo ships a PPI example under `example_data/`; full Reddit and PPI are downloaded from the SNAP project site (http://snap.stanford.edu/graphsage/). Input format is a custom JSON bundle: `<prefix>-G.json` (networkx node-link), `<prefix>-id_map.json`, `<prefix>-class_map.json`, `<prefix>-feats.npy`, `<prefix>-walks.txt`.
- **training_entrypoint**:
  - Official TF: `python -m graphsage.supervised_train --train_prefix ./example_data/ppi --model graphsage_mean --sigmoid` (or `./example_supervised.sh`); unsupervised variant `python -m graphsage.unsupervised_train ...` / `./example_unsupervised.sh`.
  - PyTorch reference: `python -m graphsage.model` (runs Cora/Pubmed demos).
- **evaluation_entrypoint**: No separate script. Validation/test F1 is printed during `supervised_train`; for the unsupervised model, embeddings are dumped to `--base_log_dir` and `eval_scripts/` contains downstream classifiers.
- **adaptation_required** (HIGH):
  1. TF 1.8 / Python 2-era code will not run on a modern CUDA/A100 stack. In practice every recent fraud-detection paper re-implements GraphSAGE rather than using this repo.
  2. Recommended substitution: `dgl.nn.SAGEConv` (DGL) or `torch_geometric.nn.SAGEConv` / `torch_geometric.nn.models.GraphSAGE` (PyG). Both are single-line drop-ins and are what ConsisGAD/PMP/GAAP codebases already use internally.
  3. DGP datasets: YelpReviews and AmazonVideo can be loaded via `dgl.data.FraudYelpDataset` / `dgl.data.FraudAmazonDataset`; the two ByteDance proprietary sets (E-Commerce, LifeService) are NOT public, so those two columns of the DGP table are not reproducible at all.
  4. Node features in DGP are LLM/text-derived, so a text encoder must be run first to produce the `feat` matrix GraphSAGE consumes.
- **deviations**: Reproduction will almost certainly use the DGL/PyG `SAGEConv` implementation, NOT `williamleif/GraphSAGE`. This is a deviation from "official code" but matches universal community practice and matches how the other fraud baselines embed GraphSAGE.
- **license**: "Other" / NOASSERTION per GitHub API (repo contains a permissive-style custom LICENSE.md). `graphsage-simple` has NO license file.
- **reproducibility_status**: **Method reproducible, official code not usable.** The algorithm is trivial and available in DGL/PyG; the 2017 TF repo is effectively abandoned (last commit 2018).
