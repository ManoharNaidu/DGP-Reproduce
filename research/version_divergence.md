# Version Divergence: arXiv v1 vs AAAI-26 Camera-Ready

**Status:** VERIFIED by the lead engineer against both primary documents (not delegated).

| | Document A | Document B |
|---|---|---|
| Version | arXiv:2507.21653**v1** | AAAI-26 camera-ready |
| Date | 29 Jul 2025 (only version; no v2 exists) | AAAI-26 proceedings, pp. 15171–15177 |
| Local copy | `research/arxiv_v1_fulltext.txt` | `research/paper_fulltext.txt` (from the local PDF) |
| Retrieved from | `https://arxiv.org/html/2507.21653v1` (HTTP 200, 224,692 bytes, LaTeXML) | `11. 01512-AAAI26.LiY-DM.pdf` |
| License | arXiv.org perpetual non-exclusive | AAAI copyright |

> **WHY THIS MATTERS.** The official code repository contains only a README (see `repository_audit.md`). The two paper versions are therefore the *entire* evidence base for this reproduction, and **they are not the same document**. Each contains implementation facts the other omits. A reproduction built from either one alone would be missing critical information. **Neither version alone is sufficient; both must be used jointly.**

---

## 1. Facts present ONLY in arXiv v1 (absent from the camera-ready)

These were cut from the camera-ready, almost certainly for AAAI's page limit. They are the single most valuable recovered material in this project.

### 1.1 The hyperparameter search grid — RECOVERED

Verbatim from arXiv v1, section 5.1 "Parameter Settings":

> "For all evaluated models, we tune hyperparameters using grid search based on validation performance. For DGP, we tune the bi-level summarization budgets `B_node, B_meta ∈ {10, 20, 40, 80}`, the the number of hops `K ∈ {1, 2, 3}`, and the neighbor truncation size `M ∈ {2, 4, 8, 16}` for each dataset. For LoRA-based finetuning of LLM methods, we tune the LoRA rank `r ∈ {4, 8, 16, 32}`, the LoRA dropout rate `∈ {0.0, 0.05, 0.1}`, and the learning rate `∈ {1e−5, 3e−5, 1e−4}`. We set the batch size to 4 and finetune for up to 10 epochs with early stopping based on validation loss. For all baseline methods, we tune hyperparameters within the recommended ranges reported in their original papers to ensure fair and optimized comparisons. All hyperparameters are selected to optimize the average AUROC on the validation set."

(The duplicated "the the" is in the original.)

The camera-ready reduces this entire paragraph to one sentence: *"For all evaluated models, we tune hyperparameters using grid search on the validation set."*

**Recovered — now firm:**

| Quantity | Value |
|---|---|
| `B_node`, `B_meta` search grid | `{10, 20, 40, 80}` |
| `K` search grid | `{1, 2, 3}` |
| `M` search grid | `{2, 4, 8, 16}` |
| LoRA rank `r` search grid | `{4, 8, 16, 32}` |
| LoRA dropout search grid | `{0.0, 0.05, 0.1}` |
| Learning rate search grid | `{1e-5, 3e-5, 1e-4}` |
| Batch size | **4** (fixed, not searched) |
| Max epochs | **10**, early stopping on **validation loss** |
| Model selection criterion | **average AUROC on the validation set** |
| Baseline tuning policy | "recommended ranges reported in their original papers" |

**Still NOT specified even in arXiv:** the *selected* value of every searched hyperparameter; LoRA **alpha**; max sequence length; gradient accumulation; LR schedule/warmup/weight decay; precision; early-stopping patience; the five seed values.

> So `GRID_NOT_SPECIFIED` is now **resolved** — we must run the paper's actual grid rather than inventing one. But `SELECTED_VALUES_NOT_SPECIFIED` remains open, and the grid is large (see §4).

### 1.2 Graph construction and relation definitions — RECOVERED

Verbatim from arXiv v1, section 5.1 "Datasets":

> "**YelpReviews** (Rayana and Akoglu 2015) is a review-level spam detection dataset, where each node represents a review labeled as spam or non-spam. Following prior work (**Dou et al. 2020**), we construct a heterogeneous graph with three types of edges: reviews written by the same user (**R-U-R**), reviews on the same product with the same star rating (**R-S-R**), and reviews posted in the same month for the same product (**R-T-R**). Instead of using the handcrafted features introduced in the original work (Rayana and Akoglu 2015), **we directly utilize the original texts** for LLM-based methods."

> "**Amazon** (McAuley and Leskovec 2013) is a product review dataset from the **Amazon Video category**. We follow a similar graph construction, where each node is a review labeled as **helpful or unhelpful**. The graph contains three types of edges: reviews posted by the same user (**R-U-R**), reviews posted on the same product (**R-P-R**), and same-product reviews posted with the same rating and **within the same week** (**R-S-R**)."

The camera-ready contains **none** of this. It says only that YelpChi is used "and the associated raw texts", and describes AmazonVideo in a single clause as "a product review dataset for unhelpful review detection". The relation acronyms appear in the camera-ready *only* as rendered text inside the Figure 3 image.

**Recovered relation definitions:**

| Dataset | Relation | Definition |
|---|---|---|
| YelpReviews | `R-U-R` | reviews written by the same user |
| YelpReviews | `R-S-R` | reviews on the same product with the same star rating |
| YelpReviews | `R-T-R` | reviews posted in the same month for the same product |
| AmazonVideo | `R-U-R` | reviews posted by the same user |
| AmazonVideo | `R-P-R` | reviews posted on the same product |
| AmazonVideo | `R-S-R` | same-product reviews, same rating, within the same week |

**Note the trap:** `R-S-R` means *different things* in the two datasets, and AmazonVideo's third relation is `R-P-R` (same product), which has no Yelp counterpart. The Figure 3 legend in the camera-ready ("Same-User / Same-Time / Same-Star") describes the **Yelp** relations only. A naive implementation that reuses one relation set for both datasets would be wrong.

**Also recovered:** the "Dou et al. 2020" citation is **CARE-GNN**, which fixes the lineage of the relation design. And "we directly utilize the original texts" instead of "the handcrafted features" explains the node-count discrepancy: DGP rebuilds the graph from the **original Rayana & Akoglu release** (which carries review text), not from the 45,954-node feature-only `YelpChi.mat` used by CARE-GNN/PC-GNN/ConsisGAD.

### 1.3 Concrete token-accounting anchors — RECOVERED

arXiv v1: *"the average node text length `L` continues to grow in modern web-scale datasets (e.g., **`L = 170` on the Yelp dataset**)"*

The camera-ready replaced this with *"(e.g., `L > 1,500` on industry datasets)"*.

Both versions state `D = 133` on the Amazon dataset.

> `L = 170` for Yelp is directly usable to validate our complexity/token-accounting module against the paper (§52 of the project brief). It is the only per-dataset `L` value published anywhere.

### 1.4 Metrics implementation — RECOVERED

arXiv v1: *"All evaluation metrics are computed using the **scikit-learn** library (Pedregosa et al. 2011)."*

Absent from the camera-ready. Fixes the metric implementation — we must use `sklearn.metrics` (`f1_score(average='macro')`, `roc_auc_score`, `average_precision_score`) rather than a hand-rolled or torchmetrics implementation, since AUPRC in particular differs between implementations (`average_precision_score` vs trapezoidal PR-AUC).

### 1.5 Section structure

arXiv v1 has a numbered section **4.6 "Attention Dilution under Class Imbalance"**. The camera-ready retitles it "Attention Dilution in Fraud Detection". Content is equivalent.

---

## 2. Facts present ONLY in the camera-ready (absent from arXiv v1)

### 2.1 DeBERTa text embeddings for the diffusion kernel — a METHOD CHANGE

This is the most consequential divergence, and it runs the *other* way.

**arXiv v1, Eq. 7 context (verbatim):**
> "Let `X ∈ R^{n×d}` denote the **raw node features**. We propagate these features via `Z_P(K)` to obtain structure-aware semantic embeddings"

**Camera-ready, Eq. 7 context (verbatim):**
> "Let `X = emb(X^text) ⊕ X^num ∈ R^{N×(d_LM+d)}` denote the node embeddings, where **text is embedded using DeBERTa (He, Gao, and Chen 2021)**. We propagate `X` via `Z_P(K)` ..."

The string "DeBERTa" does **not occur anywhere in arXiv v1** (verified: 0 matches).

So between July 2025 and the camera-ready, the authors changed the MDK input from *raw node features* to *a concatenation of DeBERTa text embeddings and numerical features*. This changes what the diffusion distance actually measures and therefore which neighbours Top-M selects — it is a substantive method change, not an editorial one.

**Decision for this reproduction:** follow the **camera-ready** (the later, peer-reviewed, archival version) — i.e. implement `X = emb_DeBERTa(X^text) ⊕ X^num`. The arXiv "raw node features" formulation will be retained as a documented, switchable variant (`mdk.features: raw | deberta_concat`) so the divergence can be tested empirically rather than assumed away. Recorded as a config flag, not a silent choice.

### 2.2 "YelpChi subset"

Only the camera-ready says "We use its **YelpChi** subset". arXiv v1 never uses the string "YelpChi". Combined with arXiv's construction description, this confirms: original Rayana & Akoglu release → YelpChi subset → rebuilt review-level graph with CARE-GNN-style relations → 67,395 nodes.

### 2.3 Summarizer identity

Camera-ready: *"DGP additionally uses a **frozen Qwen3-8B** for summary generation."*
arXiv v1 says only *"two frozen-LLM summarization passes"* — it never names the summarizing model.

The camera-ready therefore **resolves** the "which model summarizes?" question: a frozen Qwen3-8B.

### 2.4 Hardware detail

Camera-ready: "64 Intel Xeon Gold 6346 CPUs, 1TB of RAM, and four NVIDIA A100 GPUs (80GB)".
arXiv v1: "a machine with 4× NVIDIA A100 GPUs (80GB)" only.

### 2.5 Eq. 12 metapath index

arXiv: `⊕_{P ∈ 𝒫}`. Camera-ready: `⊕_{P ∈ 𝒫_K}` with "`P_K` denotes the set of metapath types up to K hops". The camera-ready is the more precise statement and is what we implement.

---

## 3. Contradiction between versions: is `B` measured in TOKENS or WORDS?

| Source | Wording |
|---|---|
| arXiv v1, Figure 5 caption | "Impact of summarization length (**words**)" |
| Camera-ready, Figure 5 caption | "Impact of summarization budget (**tokens**)" |
| Both versions, body text | "the token budget per node", "budgets `B ∈ {5,10,20,40,80}`" |
| Both versions, the actual prompt string | `Summarize the text within 10 **tokens**.` |

**Assessment:** the arXiv figure caption is the lone outlier against body text, the camera-ready caption, and the literal prompt string. We treat `B` as **tokens** and treat the arXiv caption as a stale caption corrected before publication.

**However** — this exposes a deeper, genuinely unresolved issue: the budget is expressed *inside a natural-language instruction* ("Summarize the text within 10 tokens"). It is not stated whether the budget is additionally **enforced** by `max_new_tokens` at generation time, or left entirely to the model's (unreliable) instruction-following. These give materially different summary lengths, and the paper's whole token-efficiency claim depends on which. Tracked as an open gap; our implementation will make enforcement an explicit config option and will *measure* realised summary lengths rather than assume them.

---

## 4. What remains unresolved after merging BOTH versions

Merging both documents resolves gaps #4, #7, #15 (partly), #16 (partly) and #19 (partly) from `paper_specification.md` §9. Still open:

| Gap | Status after merge |
|---|---|
| Selected `K`, `M`, `B_node`, `B_meta` per dataset | **OPEN** — only the grid is known |
| Selected LoRA `r`, dropout, learning rate | **OPEN** — only the grid is known |
| LoRA **alpha** | **OPEN** — never mentioned in either version, not even a grid |
| Max sequence length | **OPEN** |
| Gradient accumulation, LR schedule, warmup, weight decay, precision | **OPEN** |
| Early-stopping patience | **OPEN** (only "up to 10 epochs, early stopping on val loss") |
| The five seed values | **OPEN** |
| Split construction rule (train+val+test < N) | **OPEN** in both versions |
| Exact final classification prompt template | **OPEN** — only the Figure 3 sketch exists |
| Exact metapath-summarization prompt | **OPEN** — only the node-level instruction is published |
| How `a_P(v)` (a float vector) is verbalized into the text prompt | **OPEN** |
| Whether Qwen3 thinking mode is disabled | **OPEN** — critical, see `llm_provenance.md` |
| Whether `A_P` is binarised before forming `T_P` | **OPEN** |
| Feature scaling before concatenation in `X` | **OPEN** |
| Which Qwen3-8B variant (base vs instruct) | **OPEN** |
| DeBERTa variant and pooling | **OPEN** |

### Cost implication of the recovered grid

The recovered grid is **not** free to run. For DGP alone, a full grid is
`|B_node × B_meta| × |K| × |M| × |r| × |dropout| × |lr|`. Even treating `B_node = B_meta` (4 values) it is `4 × 3 × 4 × 4 × 3 × 3 = 1,728` configurations per dataset, each an 8B LoRA finetune — and the summarization/MDK caches must be rebuilt whenever `B`, `K` or `M` changes. This is far beyond a realistic budget and must be reduced to a documented, transparent subset. See `reproduction_gaps.md` for the proposed staged search.

---

## 5. Provenance rule adopted for this project

1. Where the two versions **agree**, the fact is firm.
2. Where only **one** version states something, that version is cited explicitly as the source in code comments and configs.
3. Where they **conflict** (§2.1 DeBERTa, §3 tokens/words), the **camera-ready wins** as the archival peer-reviewed version, and the arXiv variant is preserved as a switchable, documented option.
4. Nothing is taken from a version without recording which one it came from.
