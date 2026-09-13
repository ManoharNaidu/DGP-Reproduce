# DGP (AAAI-2026) — External Evidence & Reproducibility Risk Register

Subagent G. Compiled 2026-09-13. **Rule applied: every claim below is backed by a URL that was actually fetched. Anything not found is marked NOT FOUND.**

Paper: *DGP: A Dual-Granularity Prompting Framework for Fraud Detection with Graph-Enhanced LLMs* — Yuan Li, Jun Hu, Bryan Hooi, Bingsheng He (NUS), Cheng Chen (ByteDance).

---

## TL;DR — the two answers that matter

1. **The arXiv version does NOT contain an appendix or a hyperparameter table.** ~~arXiv v1 is substantively identical to the AAAI camera-ready.~~ **[CORRECTED by lead engineer]** arXiv v1 is **not** identical to the camera-ready: arXiv carries the search grid, batch size, epochs and all six relation definitions (cut from the camera-ready), while the camera-ready adds DeBERTa embeddings as the MDK input (arXiv uses raw features) and names the frozen Qwen3-8B summarizer. Verified line-by-line — see `version_divergence.md`. Grep of the full arXiv HTML for `Appendix|appendix|Supplement` returns **zero hits**. The paper gives **grid-search ranges only** — it never reports the selected K, M, B_node, B_meta, LoRA rank, dropout, or learning rate for any dataset.

2. **Dataset preprocessing was recovered — but not from a prior repo.** The authors' public repos do *not* contain the DGP data. However, a **2026 follow-up paper by the same authors (FraudCoT, arXiv:2601.22949) has the appendices DGP lacks**, and it reuses the identical Amazon graph. Combining that with the source datasets, I **exactly reproduced both public datasets' node and label counts** (see §3). The AmazonVideo construction is now fully solved; YelpReviews is solved up to an email-gated download.

---

## 1. The arXiv version

**URL fetched:** https://arxiv.org/abs/2507.21653 and https://arxiv.org/html/2507.21653v1

| Field | Value |
|---|---|
| Title | DGP: A Dual-Granularity Prompting Framework for Fraud Detection with Graph-Enhanced LLMs |
| Authors | Yuan Li, Jun Hu, Bryan Hooi, Bingsheng He, Cheng Chen |
| Versions | **v1 only** — Tue, 29 Jul 2025 10:10:47 UTC (331 KB). No v2. |
| Subjects | cs.LG; cs.AI |
| Comments field | **Empty / not present** (no "AAAI 2026" note, no page count) |
| License | **arXiv Nonexclusive-Distribution License 1.0** — http://arxiv.org/licenses/nonexclusive-distrib/1.0/ |
| Code/data links in abstract page | NOT FOUND |

### License implication (important)
The `nonexclusive-distrib/1.0` license is **not** a CC license. It grants arXiv the right to distribute; it does **not** grant third parties reuse rights. **Do not copy text, tables, or figures from the arXiv PDF into a report.** Paraphrase, or quote short passages under fair-use/academic-citation with attribution.

### Appendix / supplementary material
**NOT FOUND.** Verified by full-text grep of the arXiv HTML rendering. Section list is: Abstract, 1 Introduction, 2 Related Work, 3 Preliminaries, 4 Methodology, 5 Experiments, 6 Conclusion, References. Nothing after References.

### Everything the arXiv version actually states about hyperparameters (verbatim, §5.1)

> "For all evaluated models, we tune hyperparameters using grid search based on validation performance. For DGP, we tune the bi-level summarization budgets B_node, B_meta ∈ {10, 20, 40, 80}, the the number of hops K ∈ {1,2,3}, and the neighbor truncation size M ∈ {2,4,8,16} for each dataset."

> "For LoRA-based finetuning of LLM methods, we tune the LoRA rank r ∈ {4,8,16,32}, the LoRA dropout rate ∈ {0.0, 0.05, 0.1}, and the learning rate ∈ {1e−5, 3e−5, 1e−4}."

> "We set the batch size to 4 and finetune for up to 10 epochs with early stopping based on validation loss."

> "All hyperparameters are selected to optimize the average AUROC on the validation set."

> "We conduct all experiments on a machine with 4×NVIDIA A100 GPUs (80GB)."

> "For all LLM-tuning methods, we use Qwen3-8B LLM backbone (Team 2025) for fair comparison."

> "We insert LoRA (Hu et al. 2022) adapters into all attention layers and use the AdamW (Loshchilov and Hutter 2019) optimizer for finetuning."

> "We adopt classification metrics including Macro-F1, AUROC, and AUPRC, and report the mean and standard deviation over 5 random seeds. All evaluation metrics are computed using the scikit-learn library."

**Note the typo "the the number of hops" is in the original** — a useful marker that the arXiv and camera-ready text are the same draft.

**Not stated anywhere:** selected values of K/M/B, warmup, weight decay, LR schedule, max sequence length, the 5 seed values, gradient accumulation, precision (bf16/fp16), or which Qwen3-8B checkpoint.

### Metapath definitions (verbatim, §5.1) — this is the full list, K is separate
> "YelpReviews (Rayana and Akoglu 2015) ... we construct a heterogeneous graph with three types of edges: reviews written by the same user (**R-U-R**), reviews on the same product with the same star rating (**R-S-R**), and reviews posted in the same month for the same product (**R-T-R**). Instead of using the handcrafted features introduced in the original work, we directly utilize the original texts for LLM-based methods."

> "Amazon (McAuley and Leskovec 2013) is a product review dataset from the Amazon Video category. ... The graph contains three types of edges: reviews posted by the same user (**R-U-R**), reviews posted on the same product (**R-P-R**), and same-product reviews posted with the same rating and within the same week (**R-S-R**)."

So **R = 3 relation types per public dataset**, and the "metapaths" are these three 2-step relations. **K is the number of hops of those relations (1–3), tuned, and the chosen value is never reported.** Metapaths for E-Commerce (9 edge types) and LifeService (5 edge types) are **NOT FOUND** — never enumerated.

### Prompt templates
The paper gives **only two fragments**, no full template:
- Task-agnostic (the one used): *"Summarize the text within 10 tokens"*
- Task-aware (ablation, underperformed): *"Summarize the text within 10 tokens, focusing on signals indicative of fraudulent behavior"*

Classification is scored as a **binary logit comparison over the tokens `Yes` and `No`**: p_v = softmax over (logit_Yes, logit_No). The final classification prompt itself is **NOT FOUND** in DGP. (A close proxy from the same group's follow-up paper is reproduced in §3.3.)

### Other details found only by reading the full text
- Complexity section gives corpus stats used nowhere else: **L = 170** (avg node text tokens, Yelp), **D = 133** (avg out-degree, Amazon).
- Numerical summarization = plain **mean aggregation** over the trimmed metapath neighborhood; categorical features one-hot/multi-hot encoded first. Which concrete numerical fields are used per dataset is **NOT FOUND**.
- Metapath trimming uses the **Markov Diffusion Kernel** of Fouss et al. 2012, *Neural Networks* 31:53–72.
- **Inconsistency worth flagging:** §5.3 text says budget B is in **tokens** ("Summarize the text within 10 tokens"), but the **Figure 5 caption reads "Impact of summarization length (words)"**. Tokens vs words is a real ~1.3x discrepancy for the single most important hyperparameter. Treat B=10 as approximate.
- **DeBERTa is NOT mentioned anywhere in the paper.** Grep for `deberta|bert|encoder` returns only generic prose. DGP is text-only prompting with no auxiliary text encoder. If the reproduction plan assumes a DeBERTa component, **that assumption is unfounded** — drop it.

---

## 2. Other venues, code, and reviews

### 2.1 Official GitHub repo — EXISTS, but is EMPTY
**https://github.com/Xtra-Computing/DGP** — "DGP: A Dual-Granularity Prompting Framework for Fraud Detection with Graph-Enhanced LLMs (AAAI 2026)"

- README states the code is **"Pending approval from our industry partner before public release."**
- **Repo root contains only `README.md`.** No code, no data, no config, no license file.
- 3 commits, all by `yuanlics` (Yuan Li): Initial commit (2025-11-11), Update README.md (2025-11-12), Update README with citation details (2025-11-13). **No commits since.**
- 4 stars, 0 forks.
- **Open issue #1**, "会更新源码吗" ("Will the source code be updated?") by `starizyj`, opened **2026-07-23**, body "wait for code" — **no maintainer response in ~7 weeks.**

**Assessment: treat the code as unavailable indefinitely.** The blocker is a ByteDance legal/IP review, not an authoring backlog, and the ~10-month gap since the repo was created with no code is a bad sign.

### 2.2 AAAI OJS
**https://ojs.aaai.org/index.php/AAAI/article/view/38541**
- DOI 10.1609/aaai.v40i18.38541; Vol. 40, No. 18, **pp. 15171–15179**; published 2026-03-14.
- Affiliations confirmed: Li/Hu/Hooi/He = NUS; Cheng Chen = ByteDance Inc.
- **Supplementary material: NOT FOUND.** Only the article PDF is downloadable. No appendix file, no data supplement.

### 2.3 OpenReview
**NOT FOUND.** Queried the OpenReview API directly (`api.openreview.net/notes/search`) for "Dual Granularity Prompting Fraud Detection" and "Dual-Granularity Prompting" — returns only unrelated papers (a 2022 NeurIPS fraud benchmark, a 2023 ACL ARR MOOC paper). **AAAI 2026 reviews are not public.** No reviewer-author discussion of hyperparameters or code is obtainable.

### 2.4 Semantic Scholar
**https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.21653**
```
paperId:      eaaf5e6a7544e7a8c03c93c68e1744ed5a53c0c3
DBLP:         conf/aaai/LiHHHC26
CorpusId:     280338094
venue:        AAAI Conference on Artificial Intelligence
citationCount: 12    influentialCitationCount: 0
openAccessPdf: url "" (none)
```
Note: 12 citations but **0 influential** — no one has built on it closely enough to reveal implementation details.

### 2.5 Papers With Code
`paperswithcode.co/paper/2507.21653` returned **HTTP 403**. The canonical paperswithcode.com is defunct. **No code/dataset linkage obtainable.** NOT FOUND.

### 2.6 Slides / poster / video / author blog
**NOT FOUND.** The only third-party writeup located is a non-author blog post:
**https://cognaptus.com/blog/2025-07-30-fraud-trimmed-and-tagged-how-dualgranularity-prompts-sharpen-llms-for-graph-detection/** by "Zelina", 2025-07-30. It is a summary of the arXiv paper and **contains no information beyond it** (it restates ~10 tokens best, 4×A100 80GB, Qwen3-8B + LoRA, Yelp L=170 / Amazon D=133, and headline numbers). **No new hyperparameters.** Not by the authors — do not cite it as authoritative.

---

## 3. Authors' prior and follow-up work — where the real answers were

### 3.1 RpHGNN (the paper DGP cites for "selective granularity")
- Paper: **"Efficient Heterogeneous Graph Learning via Random Projection"**, Jun Hu, Bryan Hooi, Bingsheng He — **IEEE TKDE 2024**. arXiv: https://arxiv.org/abs/2310.14481 ; IEEE: https://ieeexplore.ieee.org/document/10643347/
- **Official repo: https://github.com/CrawlScript/RpHGNN** (98 stars, GPLv3). `CrawlScript` is **Jun Hu's personal GitHub account** — this is the group's real code home, not the Xtra-Computing org.
- **What it reveals about group conventions:** datasets are **ACM, DBLP, Freebase, IMDB (HGB), OGBN-MAG, OAG-Venue, OAG-L1-Field** — all *academic* heterogeneous graphs. Stack is **DGL + OGB + PyTorch**, graphs loaded via standard HGB/OGB/NARS loaders, datasets fetched by a `download_hgb_datasets.sh` script.
- **It contains no Yelp or Amazon fraud data.** RpHGNN is methodologically related to DGP but shares **zero dataset lineage**. Its value to the reproduction is only the confirmation that this group builds on **DGL**.

### 3.2 THE KEY FIND — the group hosts fraud datasets, and they are the WRONG ones
**https://github.com/CrawlScript/gnn_datasets** (Jun Hu's dataset mirror, GPLv3) contains a folder `Abnormal/` with:
- `fd_yelp_chi.zip` → `fd_yelp_chi.mat`
- `fd_amazon.zip` → `fd_amazon.mat`

I **downloaded and inspected both**. Contents:

| File | Shape | Relations | Features | Labels |
|---|---|---|---|---|
| `fd_yelp_chi.mat` | **45,954 × 45,954** | `net_rur` (98,630 nnz), `net_rtr` (1,147,232), `net_rsr` (6,805,486) | **32 handcrafted, dense** | 6,677 pos / 45,954 = **14.53%** |
| `fd_amazon.mat` | **11,944 × 11,944** | `net_upu`, `net_usu`, `net_uvu` | 25 handcrafted | 821 pos |

**This is decisive and negative:** the group's own hosted fraud data is the **standard CARE-GNN `.mat` pair** — 45,954 Yelp review nodes with **no raw text**, and the 11,944-*user* Amazon (Musical Instruments) graph. **Neither matches DGP.** DGP's 67,395 / 37,126 graphs were **built fresh for the paper and have not been released.** So: no prior-work repo reveals the DGP preprocessing directly. The reconstruction below had to be derived independently.

### 3.3 THE OTHER KEY FIND — the follow-up paper HAS the appendices DGP lacks
**"Autonomous Chain-of-Thought Distillation for Graph-Based Fraud Detection" (FraudCoT)** — Yuan Li, Jun Hu, + 3 others, **arXiv:2601.22949, 2026-01-30**. https://arxiv.org/abs/2601.22949 (HTML: https://arxiv.org/html/2601.22949v1)

Same first two authors, same ByteDance industry partner, same baseline suite (GraphSAGE, HGT, ConsisGAD, PMP, GAAP, MLP, Qwen3-8B, TAPE, GraphGPT, HiGPT, InstructGLM), same metrics, same "5 random seeds". **It has Appendices A–E.**

**Its Table 2 contains the exact DGP fingerprint:**
```
InstantVideo    37,126 nodes   9,883,406 edges   3 edge types   1,098 / 549 / 1,098
DigitalMusic    64,706 nodes   7,732,420 edges   3 edge types   6,444 / 3,222 / 6,444
PromotionAbuse 371,464 nodes   1,388,598 edges   3 edge types 143,592 / 17,949 / 17,949
```
**37,126 nodes and 9,883,406 edges are byte-identical to DGP's "AmazonVideo" row.** The same graph is simply renamed **InstantVideo**. (The splits differ — FraudCoT uses 1,098/549/1,098, DGP uses 1,299/1,299/7,425 — so the *graph* is shared but the *split* is not.)

**FraudCoT's dataset description (verbatim) confirms the construction rule:**
> "InstantVideo and DigitalMusic are category-specific datasets, where each labeled node represents a user review associated with raw text and labeled as either **helpful or unhelpful**."
> "Following prior work Dou et al. (2020), we construct heterogeneous graphs with three types of edges for InstantVideo and DigitalMusic: reviews posted by the same user (R-U-R), reviews posted on the same product (R-P-R), and same-product reviews posted with the same rating and within the same week (R-S-R)."

**FraudCoT Implementation Details (verbatim) — reveals the software stack DGP omits:**
> "We conduct experiments using a Linux system with 64 Intel(R) Xeon(R) Gold 6346 CPUs, 1TB of RAM, and **one NVIDIA A100 GPU (80GB)**."
> "The model is implemented via **PyTorch** (Paszke et al. 2019) and **DGL** (Wang et al. 2019)."

**Note: the follow-up runs on ONE A100 80GB, not four.** DGP's "4×A100" is therefore likely a throughput/convenience figure, not a hard memory requirement — this materially downgrades the hardware risk.

**FraudCoT Appendix C (Parameter Settings), verbatim** — the LoRA block is *character-identical* to DGP's, so it is the group's house recipe:
> "For LoRA-based finetuning of LLM methods, we tune the LoRA rank r ∈ {4,8,16,32}, the LoRA dropout rate ∈ {0.0, 0.05, 0.1}, and the learning rate ∈ {1e−5, 3e−5, 1e−4}. We set the batch size to 128 and finetune for up to 300 epochs with early stopping based on validation loss."

(FraudCoT uses batch 128 / 300 epochs because it co-trains a GNN; DGP uses batch 4 / 10 epochs. The **LoRA grid is shared verbatim**, which raises confidence that the selected LoRA values are also shared.)

**FraudCoT Appendix E gives a full prompt template on the same Amazon data** — the closest available proxy for DGP's unreleased classification prompt:
> `<System Message>` You are provided with a list of Amazon customers' reviews. Each review is classified as either helpful or unhelpful based on their interactions and content.
> `<User Message>` Target Node: [TARGET_TEXT] / Neighbors: [NEIGHBOR_TEXTS] / 1. Give 1–2 short reasoning points for the unhelpfulness of the target node itself (each within 20 words). If none, just say "Looks normal". 2. Give 1–2 short reasoning points for connections inferring unhelpfulness (each within 20 words). If none, just say "Looks normal". 3. Give a prediction (Helpful / Unhelpful).

And its ego-graph serialization format:
> `<Target Node>` Rating: 5.0; Text: ...
> `<Neighbor Nodes>` Relation: Same Rating; Rating: 5.0; Text: ... / Relation: Same Product; Rating: 4.0; Text: ... / Relation: Same User; Rating: 1.0; Text: ...

**This tells you the numerical feature DGP mean-aggregates on Amazon is the star `Rating`,** and gives the exact relation-label strings ("Same Rating" / "Same Product" / "Same User") the group serializes into prompts.

### 3.4 AmazonVideo construction — **SOLVED, exactly reproduced**
Hypothesis: "AmazonVideo"/"InstantVideo" = the **Amazon Instant Video 5-core** file from McAuley's Amazon Product Data (2014 edition).

**Verified against https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html** — listed 5-core counts: **Amazon Instant Video = 37,126** and **Digital Music = 64,706**. Both match the two papers' node counts exactly.

**I then downloaded the file and confirmed it computationally:**
Source: `https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Amazon_Instant_Video_5.json.gz`
```
reviews 37126   users 5130   items 1685
fields: reviewerID, asin, reviewerName, helpful, reviewText, overall, summary, unixReviewTime, reviewTime
```
- **37,126 reviews — exact match to DGP's node count.**
- Label rule search against DGP's stated **4,379 frauds**:

| Candidate rule | Count | % |
|---|---|---|
| **helpful[1] > 0 AND helpful[0]/helpful[1] < 0.5** | **4,379** | **11.795%** |
| helpful[1] ≥ 3 and ratio < 0.5 | 1,777 | 4.79% |
| helpful[1] > 0 and ratio ≤ 0.4 | 4,261 | 11.48% |
| helpful[1] ≥ 2 and ratio < 0.5 | 2,220 | 5.98% |
| helpful[0] == 0 and helpful[1] > 0 | 3,027 | 8.15% |

**Exact hit, first try, unique among plausible rules.** The "fraud" (unhelpful) label is: *a review that received at least one helpfulness vote and whose helpful-vote ratio is below 50%.* 23,993 of the 37,126 reviews have zero votes and are therefore **unlabeled** — which is exactly why the paper says "the sum of the dataset split sizes ... can be smaller than the total number of nodes." Labeled pool = **13,133**.

### 3.5 YelpReviews (67,395) — **SOLVED in identity, gated in access**
**https://shebuti.com/yelpchi-dataset/** (Shebuti Rayana's own page) states: **67,395 reviews, 38,063 reviewers, 201 hotels and restaurants, 13.23% filtered (fake) reviews**, and confirms plaintext reviews are included.

DGP reports **8,919 frauds of 67,395 = 13.2339%** — matches the stated 13.23% exactly.

**Conclusion: DGP's "YelpReviews" is the FULL, UNFILTERED original YelpChi from Rayana & Akoglu (KDD 2015), with raw review text.** It is *not* the 45,954-node CARE-GNN `.mat` (which drops products with >800 reviews — 21,441 reviews removed — and carries only 32 handcrafted features, no text).

**Access:** the page says **"To get the datasets with ground truth please email: srayana@cs.stonybrook.edu"**. There is **no direct download link.** The Stony Brook ODDS mirror (`odds.cs.stonybrook.edu/yelpchi-dataset/`) is **currently unreachable — TLS handshake failure** on both http:// and https://. I searched for a GitHub/HuggingFace mirror carrying the raw text version: **NOT FOUND.**

### 3.6 Split construction — partially reconstructed
Test set size is **exactly 20% of total nodes** for three of four datasets:

| Dataset | N | Train | Val | Test | Train %N | Test %N |
|---|---|---|---|---|---|---|
| YelpReviews | 67,395 | 1,348 | 1,348 | 13,479 | 2.000% | **20.000%** |
| AmazonVideo | 37,126 | 1,299 | 1,299 | 7,425 | 3.499% | **19.9995%** |
| E-Commerce | 182,043 | 1,309 | 1,309 | 3,928 | 0.719% | 2.158% |
| LifeService | 12,868 | 1,287 | 1,287 | 2,574 | 10.002% | **20.003%** |

**Inferred rule:** test = round(0.20 × N); train = val = a per-dataset percentage chosen so the training set lands at **≈1,300 labeled nodes** (2% / 3.5% / 10%). This is consistent with the paper's stated motive ("we construct training sets with a limited number of labeled samples, simulating realistic constraints"). E-Commerce deviates on the test fraction. **This is an inference from the numbers, not a documented rule — the sampling procedure (stratified? seeded? drawn from labeled pool or all nodes?) is NOT FOUND.** Note for Amazon the labeled pool is 13,133 and 1,299 ≈ 9.89% of it — so whether the percentage applies to N or to the labeled pool is ambiguous.

### 3.7 GAAP (Duan et al. 2025) — shares nothing
Full citation from DGP's reference list: *Duan, M.; He, D.; Zheng, T.; Jia, L.; Song, M.; Wang, X.; and Feng, Z. 2025. Global Attribute-Association Pattern Aggregation for Graph Fraud Detection. AAAI, vol. 39, 11616–11624.* **Entirely disjoint author set** from the NUS/ByteDance group — no shared affiliation, no co-authorship. It is a cited baseline only. **No shared dataset splits.** Its code repo was **NOT FOUND** via search (it appears listed in https://github.com/mala-lab/Awesome-Deep-Graph-Anomaly-Detection but no direct repo link surfaced).

### 3.8 Other same-group repos checked (all negative for DGP data)
| Repo | Content | Relevant? |
|---|---|---|
| https://github.com/CrawlScript/RpHGNN | TKDE'24 HGNN, HGB/OGB academic graphs, DGL | Stack only |
| https://github.com/CrawlScript/Echoless-LP | AAAI'26 label-based precomputation (arXiv 2511.11081) | No fraud data |
| https://github.com/CrawlScript/Torch-MGDCF | TKDE'24 "Markov Graph Diffusion" collaborative filtering — **same Markov-diffusion machinery DGP uses for metapath trimming** | Possible reference implementation for the MDK |
| https://github.com/CrawlScript/gnn_datasets | Hosts `Abnormal/fd_yelp_chi.mat`, `fd_amazon.mat` | **Wrong variants** — see §3.2 |
| https://github.com/PyRGL/rgl | RGL, Yuan Li's RAG-on-graphs library (arXiv 2503.19314), DGL/PyG integration, C++ retrieval | Possible retrieval utilities |
| https://github.com/yuanlics (8 public repos) | None DGP-related | No |
| Xtra-Computing org (12 graph repos) | Only the empty `DGP` repo | No |

`CrawlScript/Torch-MGDCF` is worth a look if you need to implement the Markov Diffusion Kernel — same author, same kernel family.

---

## 4. Reproducibility Risk Register

Severity key: **BLOCKER** = cannot proceed / cannot compare numbers · **HIGH** = results will diverge materially · **MEDIUM** = results shift but conclusions survive · **LOW** = cosmetic.

| # | Item needed | Provided by paper? | External source? | Severity | Recommended action |
|---|---|---|---|---|---|
| 1 | **E-Commerce + LifeService datasets** (ByteDance) | Stats only | **Confirmed unavailable.** Proprietary; repo blocked pending "approval from our industry partner" | **BLOCKER** (for those 2 datasets) | **Drop them. Scope the reproduction to YelpReviews + AmazonVideo only** and say so explicitly. Half of Table 2 is permanently unverifiable. |
| 2 | **Official code** | No | Repo exists but is **empty**; issue #1 unanswered 7 weeks; 10 months with no code | **BLOCKER** for exact reproduction | Reimplement from the paper. Use `CrawlScript/Torch-MGDCF` for the MDK and `PyRGL/rgl` for neighbor retrieval. Budget for full reimplementation. |
| 3 | **YelpChi with raw text (67,395)** | Identified as Rayana & Akoglu full set | shebuti.com confirms 67,395 / 13.23% (= 8,919, exact match). **Email-gated**: srayana@cs.stonybrook.edu. ODDS mirror TLS-broken. No public mirror found | **HIGH** | **Email srayana@cs.stonybrook.edu TODAY** — this is the longest-lead item. Fallback: run on Amazon only, or use the 45,954 `.mat` and report it as a *different* dataset (no text → DGP inapplicable; would need a text join). |
| 4 | **AmazonVideo construction (37,126)** | Rule stated, source not named | **SOLVED.** `reviews_Amazon_Instant_Video_5.json.gz` (McAuley 5-core), 37,126 reviews verified; label = `helpful[1]>0 and helpful[0]/helpful[1] < 0.5` → **4,379, exact match** | **LOW** (resolved) | Build this dataset first. It is fully reproducible and unblocks all method development. Use it as the primary reproduction target. |
| 5 | **Metapath definitions** | **Yes for both public datasets** (R-U-R, R-S-R, R-T-R for Yelp; R-U-R, R-P-R, R-S-R for Amazon) | Corroborated by FraudCoT Appendix | **LOW** for public data; **BLOCKER** for E-Commerce/LifeService (never enumerated) | Implement as stated. Note R-S-R for Yelp = *same product AND same star rating*; R-T-R = *same product AND same month*; Amazon R-S-R = *same product AND same rating AND same week*. |
| 6 | **K (number of hops)** | Range {1,2,3} only. **Selected value NOT REPORTED** | NOT FOUND | **HIGH** | Re-run the grid {1,2,3}. Cost is 3x. Report the whole curve rather than one number — this is honest and more informative than guessing. Prior: K=2 (standard, and complexity section treats K>2 as prohibitive). |
| 7 | **M (top-M truncation)** | Range {2,4,8,16} only. **Selected NOT REPORTED** | NOT FOUND | **HIGH** | Grid over {2,4,8,16}. Combined with #6 this is a 12-cell grid before LoRA — the dominant compute cost. Consider fixing K=2 and sweeping M. |
| 8 | **B_node / B_meta** | Range {10,20,40,80}; §5.3 says **"best performance is generally achieved with ~10 tokens"** and Fig. 5 sweeps {5,10,20,40,80} with B_node=B_meta | NOT FOUND | **MEDIUM** (partially answered) | Use **B_node = B_meta = 10**. This is the one hyperparameter the paper effectively discloses. **Caveat: Fig. 5 caption says "words", text says "tokens"** — pick tokens, document the ambiguity. |
| 9 | **LoRA rank / alpha / dropout** | rank ∈{4,8,16,32}, dropout ∈{0,0.05,0.1}. **alpha NEVER MENTIONED.** Selected values NOT REPORTED | FraudCoT Appendix C has the **identical grid** — house recipe, but also no selected values | **HIGH** | Grid rank {8,16}, dropout {0.0,0.05}; set **alpha = 2×rank** (the PEFT-community default) and document the assumption loudly. Adapters go on **all attention layers** (stated). |
| 10 | **Learning rate / batch / epochs** | LR ∈{1e-5,3e-5,1e-4}; **batch = 4**; **≤10 epochs, early stop on val loss**; AdamW | FraudCoT confirms AdamW + same LR grid | **MEDIUM** | batch/epochs are given — use them. Grid the 3 LRs. **Warmup, weight decay, and LR schedule are NOT FOUND** → use AdamW defaults (wd=0.01, linear warmup 3%) and document. |
| 11 | **Max sequence length** | **NOT FOUND** | NOT FOUND | **MEDIUM** | Derive it: L≈170 target tokens + (R^(K+1)−R)/(R−1) × B metapath summaries. For R=3, K=2, B=10 → 170 + 12×10 ≈ 290 tokens. Set 1024 or 2048 for headroom; log actual token usage to compare against the paper's Figure 2 token-budget claims. |
| 12 | **Qwen3-8B: base vs instruct** | Says only "Qwen3-8B (Team 2025)" | HF shows both **`Qwen/Qwen3-8B`** (12.9M downloads) and **`Qwen/Qwen3-8B-Base`** (419K) | **MEDIUM** | Use **`Qwen/Qwen3-8B`** — the bare name maps to it, and it's 30x more downloaded. **Extra unaddressed risk: Qwen3-8B is a hybrid-thinking model.** The paper never says whether thinking mode was on. For a Yes/No logit readout, **disable thinking** (`enable_thinking=False`). Document this. |
| 13 | **DeBERTa variant + pooling** | **Does not exist in this paper.** Zero mentions of DeBERTa/BERT/any text encoder | N/A | **N/A — remove from plan** | DGP is text-only prompting with **no auxiliary encoder**. If a reproduction plan includes DeBERTa, it is based on a misreading. Delete that component. |
| 14 | **Hardware: 4×A100 80GB** | Stated | **FraudCoT (same group, same backbone, harder co-training) used ONE A100 80GB** | **MEDIUM → LOW** | A single 80GB A100 is sufficient for LoRA on an 8B model at batch 4, seq ≈300. On smaller GPUs use QLoRA/4-bit + grad accumulation. **Note this changes numerics** — document. Not a blocker. |
| 15 | **The 5 random seed values** | "5 random seeds" — **values NOT REPORTED** | NOT FOUND | **LOW** | Use 0–4 and report mean±std. Std comparison against the paper is still meaningful; exact per-seed reproduction is not. |
| 16 | **Train/val/test split construction** | Sizes given; **procedure NOT FOUND**. Sum < N (unlabeled majority — confirmed on Amazon: 23,993 of 37,126 have no votes) | Partially reconstructed: test = exactly 20% of N (Yelp/Amazon/LifeService) | **HIGH** | Reconstruct as: labeled pool → random split to hit 1,299/1,299/7,425 (Amazon), 1,348/1,348/13,479 (Yelp). **Whether it is stratified by label is unknown** — with ~12–13% positives, stratification materially affects AUPRC. Run both, report both. |
| 17 | **How baselines were adapted to Qwen3-8B** | "All baselines are implemented using official code" — nothing more. GraphGPT/HiGPT/InstructGLM ship with **Vicuna/LLaMA**, not Qwen3 | NOT FOUND | **HIGH** | Swapping the backbone in GraphGPT/HiGPT is non-trivial (graph-token projector dims, chat template, tokenizer). **Do not attempt to reproduce the full baseline table.** Reproduce 2–3 cheap, well-specified baselines (MLP, GraphSAGE, plain Qwen3-8B LLM-SFT on target text) and cite the paper's numbers for the rest, flagged as unverified. |
| 18 | **Numerical features per dataset** | Method = mean aggregation; **which fields NOT FOUND** | FraudCoT ego-graph shows `Rating:` serialized for Amazon | **MEDIUM** | Amazon: use star `overall` rating (and optionally helpful-vote counts — **but beware leakage, since the label is derived from those**). **Do NOT feed helpful-vote counts as features.** Yelp: rating + timestamp-derived features. |
| 19 | **Full prompt templates** | Two summarization fragments + Yes/No logit readout only | FraudCoT Appendix E gives a full analogous template + ego-graph format (§3.3) | **MEDIUM** | Adapt FraudCoT's format: system message naming the domain, `Target Node:` / `Neighbors:` with `Relation: Same User/Same Product/Same Rating` labels, ending in a forced binary. Score via Yes/No logits per DGP §4.4. |
| 20 | **arXiv text reuse** | License = arXiv nonexclusive-distrib 1.0 | Verified on abs page | **LOW (compliance)** | **Not a CC license.** Paraphrase; short quotes with citation only. Don't lift tables/figures wholesale. |

### Severity rollup
- **BLOCKER: 2** (proprietary datasets; no code) — both are structural and unresolvable. Scope around them.
- **HIGH: 6** (#3 Yelp access, #6 K, #7 M, #9 LoRA, #16 splits, #17 baselines)
- **MEDIUM: 7** · **LOW: 4** · **Removed: 1** (DeBERTa)

### Recommended sequencing
1. **Today:** email srayana@cs.stonybrook.edu for YelpChi with ground truth (longest lead time, gates ~half the reproduction).
2. **Immediately, in parallel:** build AmazonVideo from `reviews_Amazon_Instant_Video_5.json.gz` using the verified label rule. It is 100% reproducible and unblocks everything else.
3. Implement DGP with **B=10, K=2, M=8** as the center point; sweep K and M around it.
4. Report a **hyperparameter sensitivity surface instead of a single number.** Given that the selected K/M/LoRA values were never published, a point estimate cannot be claimed to reproduce the paper — a curve that brackets the reported value is the defensible result.
5. Cap the baseline table at cheap, well-specified baselines; cite the rest as unverified.

### Honest framing for the write-up
This paper is **partially reproducible at best**. Two of four datasets are permanently unavailable, the code has never been released, and the selected values of every tuned hyperparameter (K, M, B, LoRA rank/dropout, LR) are absent from both the camera-ready and the arXiv version. What *is* now fully pinned down — thanks to the follow-up paper and direct verification against source data — is the **exact identity and labelling of both public datasets**, which is the single most common failure point in fraud-detection reproductions and is usually the hardest thing to recover.

---

## Appendix: every URL fetched for this report

**Primary paper**
- https://arxiv.org/abs/2507.21653
- https://arxiv.org/html/2507.21653v1 (downloaded, converted, grepped)
- https://ojs.aaai.org/index.php/AAAI/article/view/38541
- https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.21653

**Code / repos**
- https://github.com/Xtra-Computing/DGP
- https://github.com/Xtra-Computing/DGP/issues
- https://github.com/Xtra-Computing/DGP/issues/1
- https://github.com/Xtra-Computing/DGP/commits/main
- https://github.com/CrawlScript/RpHGNN
- https://github.com/CrawlScript/gnn_datasets (+ GitHub contents API for `Abnormal/`)
- https://raw.githubusercontent.com/CrawlScript/gnn_datasets/master/Abnormal/fd_yelp_chi.zip (downloaded, inspected)
- https://raw.githubusercontent.com/CrawlScript/gnn_datasets/master/Abnormal/fd_amazon.zip (downloaded, inspected)
- https://github.com/CrawlScript/Torch-MGDCF
- https://github.com/yuanlics?tab=repositories
- https://github.com/Xtra-Computing (org listing)
- https://api.github.com/users/CrawlScript/repos
- https://api.github.com/search/repositories (several queries)

**Prior / follow-up work**
- https://arxiv.org/abs/2601.22949 (FraudCoT)
- https://arxiv.org/html/2601.22949v1 (downloaded, converted, grepped — Appendices A–E)
- https://arxiv.org/abs/2310.14481 (RpHGNN)
- https://ieeexplore.ieee.org/document/10643347/
- https://arxiv.org/abs/2511.11081 (Echoless-LP)
- https://arxiv.org/abs/2503.19314 (RGL) / https://github.com/PyRGL/rgl
- Semantic Scholar author endpoints: authorId 2258900169 (Yuan Li), 2353551808 (Jun Hu)

**Datasets**
- https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html
- https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Amazon_Instant_Video_5.json.gz (downloaded, verified)
- https://shebuti.com/yelpchi-dataset/
- http://odds.cs.stonybrook.edu/yelpchi-dataset/ — **UNREACHABLE (TLS handshake failure)**
- https://huggingface.co/api/models?search=Qwen3-8B&author=Qwen

**Other**
- https://api.openreview.net/notes/search (2 queries) — no DGP record
- https://cognaptus.com/blog/2025-07-30-fraud-trimmed-and-tagged-how-dualgranularity-prompts-sharpen-llms-for-graph-detection/
- https://paperswithcode.co/paper/2507.21653 — **HTTP 403**
