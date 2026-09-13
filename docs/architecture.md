# Architecture

## Data flow (paper Figure 3 as code)

```
 configs/datasets/*.yaml ─► data/  ──► HeteroGraph {relations (one sparse matrix per type), texts, numeric, labels}
                                  └─► Split (Table 1 sizes)
                                            │
             ┌──────────────────────────────┼─────────────────────────────────────────┐
             │ target raw text x_v^text     │ neighbours                               │
             │ (fine-grained, kept whole)   ▼                                          │
             │               mdk/adjacency.py   metapaths P_K = all relation sequences of length ≤ K
             │               embeddings/        X = DeBERTaV3(text) ⊕ numeric
             │               mdk/diffusion.py   H = Z_P(K) X,  Z_P(K) = (1/K) Σ_{k=0..K} T_P^k   (Eq. 6–7)
             │               mdk/trimming.py    Top-M by ‖h_u − h_v‖₂ inside N_P(v)             (Eq. 8–9)
             │                              │
             │              ┌───────────────┴───────────────┐
             │              ▼                               ▼
             │   summarization/pipeline.py         summarization/numeric.py
             │   s_u = Summarize(x_u; B_node)      a_P(v) = mean of x_u^num      (Eq. 11)
             │   S_P(v) = Summarize(⊕ s_u; B_meta)  (Eq. 5, 10)
             │              └───────────────┬───────────────┘
             ▼                              ▼
          prompts/builder.py   prompt(v) = x_v^text ⊕ [⊕_P S_P(v) ⊕ a_P(v)]       (Eq. 12)
                                            │
                                            ▼
          models/qwen_classifier.py   Qwen3-8B + LoRA(q,k,v,o_proj), thinking disabled
          models/readout.py           train: CE on first token over full vocab       (Eq. 13)
                                      infer: p_v = softmax(logit_Yes, logit_No)[Yes]  (Eq. 14)
                                            │
                                            ▼
          metrics/ (scikit-learn) ─► results/raw/<run_id>/ ─► scripts/evaluate.py ─► tables, figures
```

`src/dgp_repro/pipeline.py` wires the middle section; `src/dgp_repro/experiment.py` wires training,
evaluation and record keeping. Scripts in `scripts/` are thin command-line wrappers around these two.

## Package map

| Module | Responsibility |
|---|---|
| `config.py` | YAML loading with `defaults:` inheritance and `--set key=value` overrides |
| `data/` | graph container, dataset builders (AmazonVideo, YelpReviews, synthetic, proprietary stubs), splits |
| `mdk/` | metapath adjacency, diffusion operator, distance, Top-M trimming |
| `embeddings/` | DeBERTaV3 mean-pooled embeddings; hashing embedder (DEBUG_ONLY) |
| `summarization/` | summarizer protocol, Qwen3-8B summarizer, mock (DEBUG_ONLY), cached bi-level pipeline, numeric means |
| `prompts/` | final prompt assembly; templates live in `prompts/dgp/` |
| `models/` | label-token resolution, first-token loss and readout, Qwen3 + LoRA classifier, mock LLM (DEBUG_ONLY) |
| `training/` | model-agnostic training loop with early stopping on validation loss |
| `metrics/` | Macro-F1, AUROC, AUPRC; seed aggregation |
| `evaluation/` | token accounting, complexity formulas, attention-dilution curves, tables/figures/comparison |
| `baselines/` | MLP baseline; export of our graphs for the official GNN baselines |
| `utils/` | seeding, device selection, artifact cache, provenance/manifest |

## Design rules

- **One code path.** The mock models and Qwen3-8B go through the same pipeline, loss, readout and metrics.
- **Relations stay typed** until the metapath product; nothing collapses the graph to homogeneous.
- **Never materialise N × N.** Diffusion applies `T_P` as a chain of sparse matrix-vector products.
- **Everything expensive is cached** under a key hashing the settings that determine it, with a
  `metadata.json` sidecar (dataset, model, prompt version, B_node, B_meta, K, M, target hash, git commit).
- **Models are built only by stages that use them.** An MDK-only run never loads Qwen3-8B; the summarizer
  is released before the classifier is loaded.
- **Leakage guards.** Labels are read only by the split builder and evaluation. MDK, summaries and prompts
  see text, numeric features and structure only; task-agnostic templates are tested to contain no task words;
  AmazonVideo helpfulness votes (which define the label) are never features.
- **Provenance on every choice.** Reconstructed decisions are tagged in code and configs and traced to
  `research/evidence_matrix.md`.
