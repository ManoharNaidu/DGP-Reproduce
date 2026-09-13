# MLP

- **paper**: The perceptron: A probabilistic model for information storage and organization in the brain (cited by DGP as Rosenblatt 1958)
- **authors**: Frank Rosenblatt
- **venue**: Psychological Review, 1958
- **official_repository**: **N/A — no repository exists or is needed.** This is a plain feed-forward classifier over node features with the graph ignored.
- **selected_commit**: N/A
- **framework**: PyTorch (our own implementation)
- **requirements**: `torch`, `numpy`, `scikit-learn` — no version constraints beyond whatever the rest of the reproduction pins.
- **dataset_support**: Whatever feature matrix we build for the GNN baselines. MLP consumes `X` and `y` only; `edge_index` / adjacency is discarded.
- **training_entrypoint**: To be written in this repo, e.g. `python -m dgp_repro.baselines.mlp --dataset yelp`.
- **evaluation_entrypoint**: Same script; report AUC / F1-macro / AUPRC (the three metrics DGP tabulates).
- **adaptation_required**: None beyond writing the module. Two decisions to record:
  1. **Which features?** Must be the *same* node features fed to GraphSAGE/HGT/ConsisGAD/PMP/GAAP, otherwise the MLP-vs-GNN comparison is meaningless. If DGP feeds GNNs text-derived embeddings, MLP gets the same.
  2. **Architecture/size.** DGP does not state hidden dims or depth. Use a 2-layer MLP with the same hidden width as the GNN baselines (common practice, e.g. 64) and grid-search LR/dropout on validation as DGP says it did for all models.
- **deviations**: Entirely our own implementation; DGP's exact MLP width/depth/dropout are unspecified in the paper, so the reproduced MLP number is a best-effort reconstruction.
- **license**: N/A (our code).
- **reproducibility_status**: **TRIVIAL / fully reproducible**, modulo the unstated architecture.

## Note on the useful reference implementation
TAPE (`XiaoxinHe/TAPE`) ships `core/GNNs/MLP/model.py` and supports `python -m core.trainEnsemble gnn.model.name MLP`. If the reproduction already stands up TAPE, reusing that MLP keeps the feature pipeline identical for free.
