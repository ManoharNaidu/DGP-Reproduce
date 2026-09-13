# DGP — Paper Specification (primary source extraction)

**Status:** COMPLETE for the AAAI-26 camera-ready.
**Source document:** `11. 01512-AAAI26.LiY-DM.pdf` (local copy), 9 pages, AAAI-26 pp. 15171–15177 + references.
**Full extracted text:** `research/paper_fulltext.txt`
**Paper:** Yuan Li, Jun Hu, Bryan Hooi, Bingsheng He, Cheng Chen. *DGP: A Dual-Granularity Prompting Framework for Fraud Detection with Graph-Enhanced LLMs.* AAAI-26.
**Affiliations:** 1 National University of Singapore, 2 ByteDance Inc. ("Work done during internship at ByteDance.")
**Code URL stated in paper:** https://github.com/Xtra-Computing/DGP

> **CRITICAL STRUCTURAL FACT:** The AAAI camera-ready contains **no appendix and no supplementary section**. Pages 8–9 are references only. There is **no hyperparameter table anywhere in the paper**. Every hyperparameter listed as UNSPECIFIED below is genuinely absent from *this* document — it is not an extraction failure.
>
> **⚠ READ `version_divergence.md` BEFORE USING THIS FILE.** arXiv:2507.21653**v1** is *not* the same document as the camera-ready. It has no appendix either, but its body text contains implementation facts that the camera-ready **cut for space** — including the **full hyperparameter search grid**, the **exact relation definitions for both public datasets**, the token anchor `L = 170` for Yelp, and the fact that metrics are computed with **scikit-learn**. Conversely the camera-ready adds facts arXiv lacks (DeBERTa embeddings in Eq. 7; the summarizer being a frozen Qwen3-8B). Section 9 below is annotated with which gaps the merge closes.
>
> **Neither version alone is sufficient. Both are required.** The official repository is README-only (see `repository_audit.md`), so these two documents are the entire evidence base.

---

## 1. Problem formulation

Heterogeneous graph `G = {V, E, R, X}`:

- `V` — set of `N` nodes.
- `E` is a subset of `V x V x R` — typed edges.
- `R = {r_1, ..., r_|R|}` — edge relation types.
- `X = {x_v}` for v in V — node features of mixed types.

Each node carries a feature tuple:

```
x_v = (x_v^text , x_v^num)
```

- `x_v^text` — raw textual content (e.g. user-written reviews).
- `x_v^num` in `R^d` — "stacks all numeric or one-hot categorical features (e.g. a rating ranging from 1 to 5 stars)".

Binary label `y_v` in `{0,1}` (1 = fraudulent, 0 = benign). Objective (Eq. 1):

```
L = (1/|V_train|) * SUM_{v in V_train}  loss( f(v), y_v )
```

with `loss` = **binary cross-entropy** and `V_train` a subset of `V`, the labelled training nodes. At inference `f` is applied to each unseen node.

## 2. Metapaths (Eq. 2-4)

For each relation `r` in `R`, the typed adjacency `A_r` in `{0,1}^(N x N)`, with `(A_r)_{uv} = 1` iff `(u,v,r)` is in `E`.

A metapath is a finite sequence of relations (Eq. 2):

```
P = r_1 o r_2 o ... o r_L
```

(paper's example composite semantic: `Review -> User -> Review`)

Metapath-specific adjacency (Eq. 3):

```
A_P = A_{r_1} A_{r_2} ... A_{r_L}
```

Metapath-specific neighbourhood (Eq. 4):

```
N_P(v) = { u in V : (A_P)_{vu} > 0 }
```

> **NOTE (ambiguity, see `mdk_research.md`):** `A_P` as defined by Eq. 3 is a matrix **product**, so its entries are *path counts*, not 0/1. Eq. 4 thresholds at `> 0` for set membership, but Eq. 6 uses `A_P` itself to build the transition matrix. The paper never states whether `A_P` is re-binarised before forming `T_P`. UNRESOLVED.

## 3. Architecture — three core modules

Quoted from the paper: "the DGP framework is composed of three core modules: (i) node-level summarization to distill the essence of each node's raw text, (ii) diffusion-based metapath trimming to preserve the most structurally and semantically relevant neighbors along each metapath, and (iii) metapath-level summarization to further aggregate both textual and numerical features."

Motivation cited: RpHGNN (Hu, Hooi, and He 2024) — "retaining fine-grained target node features while abstracting neighborhood context".

### 3.1 Module 1 — Node-level summarization (Eq. 5)

```
s_v = Summarize( x_v^text ; B_node )
```

`B_node` = token budget per node.

Paper's design statement (verbatim): *"To avoid hand-crafting domain-specific prompts, we adopt a task-agnostic summarization approach that condenses text within a fixed token budget. Although some detail may be lost, the resulting summaries remain effective for downstream metapath extraction and reasoning. In contrast, we empirically observe that task-specific prompts may underperform in the absence of dataset-specific expertise, as they can misguide the model and degrade summarization quality."*

The only summarization prompt text given anywhere in the paper appears in the task-aware study (see 7.6):

- **task-agnostic:** `Summarize the text within 10 tokens.`
- **task-aware:** `Summarize the text within 10 tokens, focusing on signals indicative of fraudulent behavior.`

These two strings are the **only verbatim prompt text in the entire paper**. The final classification prompt is shown only as a figure sketch (Figure 3): `Target: <v0>  Metapaths: <RUR> <RTR-RSR>  Question: Is this fraud?`

### 3.2 Module 2 — Diffusion-based metapath trimming (Eq. 6-9)

Guided by the **Markov Diffusion Kernel (MDK)**, citing Fouss et al. 2012 and Zhu & Koniusz 2021.

Transition matrix (row-stochastic):

```
T_P = D_P^{-1} A_P ,     D_P = diag(A_P * 1)
```

Markov diffusion operator (Eq. 6) — **reproduced exactly as printed**:

```
Z_P(K) = (1/K) * SUM_{k=0}^{K} T_P^k
```

> **INDEXING ANOMALY — DO NOT SILENTLY "FIX".** The sum has `K+1` terms (`k = 0 ... K`) but divides by `K`. The paper's prose says "Averaging the first K random-walk powers, i.e., K-hops". The classical Fouss/SSGC operator sums `k = 1 ... K` and divides by `K`. Under investigation in `mdk_research.md`. Our implementation must reproduce Eq. 6 **literally** as the default and expose alternative normalisations as explicitly-flagged variants.
>
> **Key consequence:** the scalar `1/K` is a *constant* applied identically to every node, so it rescales all L2 distances by the same factor and **cannot change the Top-M ranking**. The only ranking-relevant question is whether the `k=0` (identity) term is included — that term adds each node's own raw features to its diffused embedding.

Node embedding matrix:

```
X = emb(X^text) (concat) X^num   in R^(N x (d_LM + d))
```

where "text is embedded using **DeBERTa** (He, Gao, and Chen 2021)".

Diffused embeddings (Eq. 7):

```
h_i^(P)(K) = [ Z_P(K) X ]_{i,:}
```

Joint diffusion distance (Eq. 8):

```
delta_K^(P)(u,v) = || h_u^(P)(K) - h_v^(P)(K) ||_2
```

Top-M trimming (Eq. 9):

```
Ntilde_P(v) = TopM over u in N_P(v) of ( - delta_K^(P)(u,v) )
```

i.e. the `M` neighbours **within the metapath neighbourhood `N_P(v)`** with smallest diffusion distance to `v`.

### 3.3 Module 3 — Metapath-level summarization (Eq. 10)

```
S_P(v) = Summarize( concat of s_u for u in Ntilde_P(v) ;  B_meta )
```

`B_meta` = token budget per metapath summary.

### 3.4 Numerical summarization (Eq. 11)

```
a_P(v) = (1 / |Ntilde_P(v)|) * SUM_{u in Ntilde_P(v)} x_u^num
```

Plain **mean aggregation** over the *trimmed* neighbour set. "`x_u^num` denotes either a real-valued numerical feature or a categorical vector encoded as one-hot or multi-hot."

Figure 3 illustrates this concretely: ratings `r0=5.0, r1=2.0, r2=2.0, r3=1.0` produce `r_mean=1.7`; categories `c0=<1,0>, c1=<0,1>, c2=<1,0>, c3=<1,0>` produce `c_mean=<0.7,0.3>`.

> Observe: `r_mean=1.7` and `c_mean=<0.7,0.3>` are consistent with a mean over **3 of the 4** displayed nodes: mean(2.0, 2.0, 1.0) = 1.67 ≈ 1.7 and mean(<0,1>, <1,0>, <1,0>) = <0.67, 0.33> ≈ <0.7, 0.3> — i.e. neighbours v1, v2, v3 with the target v0 (rating 5.0) **excluded**. This corroborates that aggregation runs over trimmed neighbours only and excludes the target node. (INFERENCE from figure arithmetic, flagged as such — not a paper statement.)

### 3.5 Final prompt (Eq. 12)

```
prompt(v) = x_v^text (concat) [ concat over P in P_K of ( S_P(v) (concat) a_P(v) ) ]
```

`P_K` = "the set of metapath types up to K hops".
The target node contributes its **raw full text** `x_v^text` (fine-grained), never its summary — this is the "dual granularity".

### 3.6 Training objective (Eq. 13)

```
L = - (1/|V_train|) * SUM_{v in V_train} log p_theta( y_v | prompt(v) )
```

"cross-entropy loss over the **first generated token**", where `y_v` is in `{Yes, No}` and `p_theta(y_v | prompt(v))` is the token probability output by the LLM.

### 3.7 Inference (Eq. 14)

```
p_v = exp(logit_Yes) / ( exp(logit_Yes) + exp(logit_No) )
```

"During inference, we apply softmax over the logits of the first generated token"; `logit_Yes`, `logit_No` are "the pre-softmax scores assigned by the LLM to the tokens Yes and No". `p_v` = model confidence that `v` is fraudulent.

> Therefore `Yes` = fraudulent (label 1), `No` = benign (label 0).

## 4. Complexity analysis (to be reproduced in code)

Symbols: `L` = average token length of a node's text; `D` = average out-degree; `R` = number of relation types; `B` = summarization budget; `K` = hops; `M` = metapath neighbour truncation.

| Quantity | Formula (verbatim from paper) |
|---|---|
| Full-neighbour prompt length | `(D^(K+1) - 1)/(D - 1) * L` |
| Fully-vectorized prompt | `L + (D^(K+1) - D)/(D - 1)` |
| DGP bi-level summarization prompts | `L + (R^(K+1) - R)/(R - 1) * M * B` |
| DGP final prompt | `L + (R^(K+1) - R)/(R - 1) * B` |
| Node + metapath summarization cost (all N nodes) | `O((L+B)^2 N)` and `O( ((R^(K+1)-R)/(R-1) * M * B)^2 N )` |
| Finetuning, E epochs | `O( (L + (R^(K+1)-R)/(R-1) * B)^2 E N )` |
| Inference (summaries cached) | `O( (L + (R^(K+1)-R)/(R-1) * B)^2 N )` |

Stated empirical anchors: `D = 133` on the Amazon dataset; `L > 1,500` on the industry datasets; "each neighboring node may be associated with over 1,500 tokens, resulting in a 2-hop neighborhood containing up to 2 million tokens" (industrial scenario).

## 5. Attention-dilution analysis (THEORETICAL_ANALYSIS)

Sequence length `T_K = L + m * n_K`; `m` tokens per neighbour; `K`-hop neighbourhood size `n_K = (D^(K+1) - D)/(D - 1)`. Global fraud ratio `p << 1`. Fraud-related token fraction (Eq. 15):

```
r = (L + p*m*n_K)/(L + m*n_K) = p + L(1-p)/(L + m*n_K)  <=  p + L(1-p)/(m*n_K)
```

"which gradually decreases from 1 to p." Softmax attention (Eq. 16) is the standard formulation. "Under the assumption that token similarities are roughly uniform, the expected attention mass assigned to fraud-related tokens is `r`."

> The paper itself frames this as an assumption-laden bound, not an empirical measurement. Must be labelled THEORETICAL_ANALYSIS and must not be presented as evidence about actual attention behaviour.

## 6. Datasets (Table 1, verbatim)

| Dataset | Node Type | Textual | Numerical | # Nodes | # Edges | # Edge Types | # Frauds | # Train / Val / Test |
|---|---|---|---|---|---|---|---|---|
| YelpReviews | Service Review | yes | yes | 67,395 | 17,486,608 | 3 | 8,919 | 1,348 / 1,348 / 13,479 |
| AmazonVideo | Product Review | yes | yes | 37,126 | 9,883,406 | 3 | 4,379 | 1,299 / 1,299 / 7,425 |
| E-Commerce | Shop Profile | yes | yes | 182,043 | 27,196,608 | 9 | 3,256 | 1,309 / 1,309 / 3,928 |
| LifeService | Shop Profile | yes | yes | 12,868 | 82,912 | 5 | 2,868 | 1,287 / 1,287 / 2,574 |

Descriptions (verbatim):

- **YelpReviews** (Rayana and Akoglu 2015): "a spam detection dataset in which each node represents a review labeled as spam or non-spam. We use its **YelpChi subset and the associated raw texts** for model evaluation."
- **AmazonVideo** (McAuley and Leskovec 2013): "a product review dataset for **unhelpful review detection**."
- **LifeService** and **E-Commerce**: "two proprietary industry datasets ... real-world graphs sampled from our industry partner, **ByteDance**." → NOT_REPRODUCIBLE.

Split rationale (verbatim): *"Given the high cost of manual annotation in industry settings, we construct training sets with limited labeled samples, simulating realistic constraints where high-quality fraud labels are costly and difficult to obtain. We also note that the sum of the dataset split sizes, including the training, validation, and test sets, can be smaller than the total number of nodes. This aligns with real-world scenarios in which the majority of nodes are unlabeled, leaving them outside the regular data splits."*

> The paper gives **no construction rule** for the splits — no percentages, no seed, no stratification statement. UNRESOLVED (see section 9).

Relation types shown in Figure 3 for the public review datasets: **Same-User (RUR)**, **Same-Time (RTR)**, **Same-Star (RSR)**. Metapaths illustrated: **`RUR`** and **`RTR-RSR`** ("Neighbors via metapaths up to K-hops (e.g., RUR and RTR-RSR)"). `RTR-RSR` being a 2-relation composition implies `K >= 2` in that illustration, but the paper never states the `K` actually used in experiments.

## 7. Experimental protocol

### 7.1 Baselines (verbatim list)

- (i) **GNNs**: GraphSAGE (Hamilton, Ying, Leskovec 2017), HGT (Hu et al. 2020), ConsisGAD (Chen et al. 2024), PMP (Zhuo et al. 2024), GAAP (Duan et al. 2025).
- (ii) **Graph-agnostic**: MLP (Rosenblatt 1958), and "a Qwen3-8B LLM (Team 2025) finetuned on target nodes alone" (reported in tables as `LLM`).
- (iii) **LLM-enhanced GNNs**: TAPE (He et al. 2024), FLAG (Yang et al. 2025).
- (iv) **Graph-enhanced LLMs**: GraphGPT (Tang et al. 2024a), HiGPT (Tang et al. 2024b), InstructGLM (Ye et al. 2024).

"All baselines are implemented using official code."

### 7.2 Parameter settings (the paper's entire statement)

> *"For all evaluated models, we tune hyperparameters using grid search on the validation set."*

No grid is given **in the camera-ready**.

> **RESOLVED via arXiv v1.** The camera-ready compressed a much longer paragraph. arXiv v1 publishes the complete grid: `B_node, B_meta ∈ {10,20,40,80}`, `K ∈ {1,2,3}`, `M ∈ {2,4,8,16}`, LoRA `r ∈ {4,8,16,32}`, LoRA dropout `∈ {0.0,0.05,0.1}`, learning rate `∈ {1e-5,3e-5,1e-4}`, **batch size 4**, **up to 10 epochs with early stopping on validation loss**, and **selection by average validation AUROC**. Metrics via **scikit-learn**. Full verbatim quote and analysis in `version_divergence.md` §1.1.
>
> `GRID_NOT_SPECIFIED` is therefore **withdrawn**. What remains open is `SELECTED_VALUES_NOT_SPECIFIED` — the grid is published, the chosen points are not.

### 7.3 Implementation details (verbatim)

- Hardware: "a Linux system with **64 Intel(R) Xeon(R) Gold 6346 CPUs, 1TB of RAM, and four NVIDIA A100 GPUs (80GB)**."
- "For all LLM-tuning methods, we use the **Qwen3-8B** LLM backbone (Team 2025) for fair comparison."
- "We apply **LoRA** (Hu et al. 2022) to **all attention layers** and use **AdamW** (Loshchilov and Hutter 2019) optimizer for finetuning."
- "DGP additionally uses a **frozen Qwen3-8B** for summary generation."
- Metrics: "**Macro-F1, AUROC, and AUPRC**", "mean and standard deviation over **five random seeds**".
- Frameworks: "**PyTorch** (Paszke et al. 2019), **Transformers** (Wolf et al. 2019), and **DGL** (Wang et al. 2019)."

### 7.4 Main results

Table 2 is transcribed in full into `research/reported_results.csv`.

### 7.5 Ablations (Figure 4, on YelpReviews and AmazonVideo)

Variants: `DGP`, `w/o MDK`, `w/o PathSumm`, `w/o TextSumm`, `w/o NumSumm`.
Reported qualitative findings: removing TextSumm or NumSumm causes "a clear performance drop"; "textual summarization exhibits higher importance than numerical summarization"; removing MDK or PathSumm each "leads to a performance decline".

> **Figure 4 is a bar chart. The paper prints NO numeric values for the ablations.** Exact reported ablation numbers are `NOT_AVAILABLE_NUMERICALLY`. We must not transcribe invented values into `reported_results.csv`. Only the ordering claims above are citable.

### 7.6 Summarization budget (Figure 5)

`B` in `{5, 10, 20, 40, 80}` with the simplifying assumption `B = B_node = B_meta`, on YelpReviews and AmazonVideo.
Findings: 5 tokens gives "insufficient context"; 80 tokens causes "token dilution"; "the best performance is generally achieved with a relatively small summarization budget (**10 tokens**)".

> **Figure 5 carries no printed numbers** → `NOT_AVAILABLE_NUMERICALLY`. The one anchor is that B=10 is stated to be generally best, and the B=10 point must coincide with the Table 2 DGP row.

### 7.7 Task-aware vs task-agnostic (Table 3) — numeric values ARE given

| Dataset | Task-Aware | Macro-F1 | AUROC | AUPRC |
|---|---|---|---|---|
| YelpReviews | No (agnostic) | 69.07 ± 0.23 | 84.28 ± 0.11 | 48.87 ± 0.82 |
| YelpReviews | Yes (aware) | 58.65 ± 0.05 | 70.65 ± 1.05 | 29.02 ± 0.36 |
| AmazonVideo | No (agnostic) | 66.91 ± 0.13 | 77.32 ± 0.11 | 34.63 ± 0.24 |
| AmazonVideo | Yes (aware) | 65.55 ± 0.21 | 73.73 ± 0.17 | 31.82 ± 0.27 |

The task-agnostic rows are **identical** to the DGP rows of Table 2, confirming Table 2's DGP result is the task-agnostic, B=10 configuration.

### 7.8 Figure 2 (token usage vs AUROC)

Scatter of AUROC (%) against `#Tokens per Prompt`, for DGP, InstructGLM, GraphGPT, HiGPT, TAPE, on Yelp (x-axis spans 0–4000+) and Amazon (0–3000). **No printed numeric coordinates** → `NOT_AVAILABLE_NUMERICALLY`; only the qualitative claim "DGP achieves top performance with moderate token consumption".

## 8. Headline claims to be tested

- "improving fraud detection performance by up to **6.8% (AUPRC)** over state-of-the-art methods."
  - Check: Yelp AUPRC DGP 48.87 vs second-best ConsisGAD 42.11 gives **+6.76 points** ≈ 6.8. This confirms the claim refers to **absolute percentage points on Yelp AUPRC**, not a relative improvement. (INFERENCE, arithmetic on Table 2.)
- `*` in Table 2 marks improvement over second-best significant at `p < 0.05`. The significance test used is **not named**.

## 9. Complete register of quantities the paper does NOT specify

Each row is annotated with its status **after merging arXiv v1 with the camera-ready** (see `version_divergence.md`). The official repo is README-only, so anything still OPEN must be marked `PAPER_RECONSTRUCTION` in code.

| # | Missing quantity | Impact | Status after version merge |
|---|---|---|---|
| 1 | `K` (number of hops) used in experiments | HIGH | **PARTIAL** — grid `{1,2,3}` known; selected value OPEN |
| 2 | `M` (Top-M neighbours) | HIGH | **PARTIAL** — grid `{2,4,8,16}` known; selected value OPEN |
| 3 | `B_node`, `B_meta` | MEDIUM | **PARTIAL** — grid `{10,20,40,80}` known; B=10 stated "generally best" |
| 4 | The **exact relation/metapath definitions** per dataset | HIGH | **RESOLVED** — arXiv v1 gives all six relation definitions verbatim |
| 5 | LoRA rank, alpha, dropout, target module names | HIGH | **PARTIAL** — `r ∈ {4,8,16,32}`, dropout `∈ {0,0.05,0.1}`; **alpha never stated in either version**; target modules = "all attention layers" (names must come from the Qwen3 architecture) |
| 6 | Learning rate, batch size, epochs, grad accumulation, warmup, weight decay, max seq length | HIGH | **PARTIAL** — lr grid `{1e-5,3e-5,1e-4}`, **batch size 4**, **≤10 epochs + early stopping on val loss** now known; accumulation / warmup / weight decay / max seq length OPEN |
| 7 | The grid-search ranges | MEDIUM | **RESOLVED** — full grid published in arXiv v1 |
| 8 | The five seed values | LOW | OPEN (any 5 fixed seeds acceptable if documented) |
| 9 | Qwen3-8B **base vs instruct** variant | HIGH | OPEN |
| 10 | Whether Qwen3 thinking mode is disabled ("first generated token" = Yes/No) | BLOCKER-level correctness issue | OPEN |
| 11 | DeBERTa variant and pooling (CLS vs mean) | MEDIUM | OPEN — and note arXiv v1 uses **raw features**, not DeBERTa, at all (`version_divergence.md` §2.1) |
| 12 | Whether `A_P` is binarised before forming `T_P` | MEDIUM | OPEN |
| 13 | Feature scaling before concatenation in `X` | MEDIUM — L2 distance is scale-sensitive | OPEN |
| 14 | Split construction rule (train+val+test < N) | HIGH | OPEN in **both** versions |
| 15 | The YelpChi variant giving 67,395 nodes **with raw text** | HIGH | **MOSTLY RESOLVED** — arXiv: built from the original Rayana release using CARE-GNN (Dou et al. 2020) relation definitions, "directly utilize the original texts" rather than handcrafted features |
| 16 | AmazonVideo construction and labelling | HIGH | **MOSTLY RESOLVED** — arXiv: Amazon **Video** category, node = review, label **helpful/unhelpful**, relations R-U-R / R-P-R / R-S-R; exact helpful-ratio thresholds still OPEN |
| 17 | Exact final classification prompt template | MEDIUM | OPEN — only the Figure 3 sketch |
| 18 | Exact metapath-summarization prompt | MEDIUM | OPEN |
| 19 | Summarizer decoding settings; is `B` enforced via `max_new_tokens` or only instructed? | MEDIUM | **PARTIAL** — summarizer identity resolved (frozen Qwen3-8B, camera-ready); decoding + enforcement OPEN |
| 20 | Handling when `|N_P(v)| < M` or `N_P(v)` is empty | LOW | OPEN |
| 21 | Statistical significance test behind the `*` markers | LOW | OPEN |
| 22 | Class-imbalance handling in the CE loss | MEDIUM | OPEN |
| 23 | How GraphGPT / HiGPT / InstructGLM were switched to Qwen3-8B | MEDIUM | **PARTIAL** — arXiv: baselines tuned "within the recommended ranges reported in their original papers" |
| 24 | Metric implementation | — | **RESOLVED** — scikit-learn (arXiv v1) |

---

*Generated by direct extraction from the camera-ready PDF. All quoted strings are verbatim. Items marked INFERENCE are arithmetic or figure-based deductions by the reproduction team, not paper statements.*
