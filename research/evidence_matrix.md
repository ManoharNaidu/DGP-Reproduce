# Evidence Matrix — DGP Reproduction

**Built after Loop 1 + lead-engineer verification.** Every row names its source and whether the lead engineer independently re-checked it against the primary source (`VERIFIED`) or it rests on a subagent report that was spot-checked only (`REPORTED`).

**Confidence scale:** `CONFIRMED` (primary source, verified) · `HIGH` (strong converging evidence) · `MEDIUM` (inference with support) · `LOW` (convention / weak) · `NONE` (no evidence).

**Evidence base, in authority order:**

1. AAAI-26 camera-ready (`paper_fulltext.txt`) — archival; wins on conflict.
2. arXiv:2507.21653v1 (`arxiv_v1_fulltext.txt`) — carries details cut from the camera-ready.
3. FraudCoT, arXiv:2601.22949 — **same five authors**, same Amazon graph; used only as same-group corroboration, never as DGP's own specification.
4. Upstream data and model artefacts (SNAP file, Qwen3 tokenizer, official baseline repos).
5. Official DGP repo — **README only; contributes nothing technical.**

---

## A. Code availability

| Question | Evidence | Source | Check | Confidence | Action |
|---|---|---|---|---|---|
| Is official DGP code released? | **No.** Git tree = 1 blob, `README.md`, 557 B, `truncated: false`. README: pending industry-partner approval. Issue #1 (2026-07-23, "wait for code") has 0 replies. | GitHub API | VERIFIED | CONFIRMED | All DGP code is `PAPER_RECONSTRUCTION`. Re-check repo before final report. |
| Hidden appendix anywhere? | None in camera-ready; arXiv has v1 only, no appendix. | both texts, arXiv abs | VERIFIED | CONFIRMED | — |

## B. Method

| Question | Evidence | Source | Check | Confidence | Action / default |
|---|---|---|---|---|---|
| Eq. 6 operator | `(1/K)·Σ_{k=0}^{K} T^k`, identical in both versions | both texts | VERIFIED | CONFIRMED | Implement literally (`paper_eq6`) |
| Is `k=0` intentional? | Matches SSGC official code structure (`alpha·X + avg_{k=1..K}`) and group's MGDCF (`h0` retained) | SSGC `utils.py`; `Torch-MGDCF` | VERIFIED | HIGH | Include identity term |
| Does `1/K` vs `1/(K+1)` matter? | Positive scalar → uniform distance scaling → Top-M invariant | mathematical | VERIFIED (argument) | CONFIRMED | Enforce via unit test |
| MDK input `X` | arXiv: raw features. Camera-ready: `DeBERTa(text) ⊕ num` | both texts | VERIFIED | CONFIRMED divergence | Camera-ready default; `raw` as switch |
| DeBERTa variant | Bibliography: "He, Gao, Chen 2021. DeBERTaV3 … arXiv:2111.09543" | camera-ready refs | VERIFIED | CONFIRMED (V3) | `microsoft/deberta-v3-base` (size unstated → LOW on *base*) |
| DeBERTa pooling | Not stated | — | — | NONE | `mean` pooling, `PAPER_RECONSTRUCTION` |
| `A_P` binarised before `T_P`? | Not stated; Eq. 3 defines `A_P` as the (count-valued) product, Eq. 6 uses it directly | both texts | VERIFIED absent | MEDIUM | `weighted` (literal, exact via matvec chain); `binary` switch |
| Which metapaths form `P_K`? | Paper's own count of summaries per prompt is `(R^{K+1}−R)/(R−1) = R + R² + … + R^K`, i.e. **every relation sequence of length 1…K** | complexity formula, both versions | VERIFIED (derivation) | HIGH | Enumerate all sequences of length ≤ K (3 / 12 / 39 for K = 1 / 2 / 3) |
| Scaling of `X` blocks | Not stated | — | — | NONE | `none` (literal); sensitivity check |
| Target excluded from `N_P(v)` | Figure 3 arithmetic: `r_mean = 1.7` = mean of v1–v3 only | camera-ready Fig. 3 | VERIFIED (arithmetic) | MEDIUM | Exclude `v` |
| Numeric summary | Mean over trimmed neighbours (Eq. 11) | both | VERIFIED | CONFIRMED | — |
| Target text granularity | Raw full `x_v^text` in prompt (Eq. 12) | both | VERIFIED | CONFIRMED | — |
| Summarizer model | Frozen Qwen3-8B | camera-ready only | VERIFIED | CONFIRMED | — |
| Task-agnostic instruction | `Summarize the text within 10 tokens.` | both | VERIFIED | CONFIRMED | Template with `{B}` |
| Task-aware instruction | `…, focusing on signals indicative of fraudulent behavior.` | both | VERIFIED | CONFIRMED | Separate versioned template |
| Metapath-summary prompt | Not published | — | — | NONE | Reuse node instruction on concatenated summaries, `PAPER_RECONSTRUCTION` |
| Final classification prompt | Fig. 3 sketch only: `Target / Metapaths / Question: Is this fraud?` | camera-ready Fig. 3 | VERIFIED | LOW | Reconstruct from sketch; version it |
| How `a_P(v)` is verbalised | Not stated | — | — | NONE | Fixed-precision numbers, `PAPER_RECONSTRUCTION` |
| Budget `B` unit | Tokens (body, camera-ready caption, prompt string) vs "words" (arXiv Fig. 5 caption only) | both | VERIFIED | HIGH (tokens) | Tokens |
| Budget enforcement (`max_new_tokens`?) | Not stated | — | — | NONE | Instruction + measured length; enforcement is a switch |
| Classification readout | First generated token, softmax over {Yes, No} logits | both | VERIFIED | CONFIRMED | — |
| `Yes`/`No` token ids (Qwen3) | `Yes`=9454, `No`=2753; `ĠYes`=7414, `ĠNo`=2308; all single tokens | Qwen3-8B `vocab.json` | VERIFIED | CONFIRMED | Resolve at runtime and assert |
| Thinking mode | Template inserts empty `<think></think>` **only if** `enable_thinking is false`; default leaves first token = think block | Qwen3-8B `tokenizer_config.json` | VERIFIED | CONFIRMED | `enable_thinking=False` for summarizer **and** classifier (method is impossible otherwise) |
| LoRA target modules | "all attention layers"; Qwen3 attention Linear = `q_proj,k_proj,v_proj,o_proj` (`q_norm/k_norm` are RMSNorm) | both texts; `modeling_qwen3.py` | REPORTED (source quoted) | HIGH | `[q_proj,k_proj,v_proj,o_proj]` |
| Qwen3-8B variant | HF id `Qwen/Qwen3-8B` = post-trained; `-Base` separate. Summarization needs instruction following | llm_provenance | REPORTED | HIGH | `Qwen/Qwen3-8B` |

## C. Training protocol

| Question | Evidence | Source | Check | Confidence | Action |
|---|---|---|---|---|---|
| `B_node, B_meta` grid | `{10,20,40,80}` | arXiv v1 §5.1 | VERIFIED | CONFIRMED | — |
| `K` grid | `{1,2,3}` | arXiv v1 | VERIFIED | CONFIRMED | — |
| `M` grid | `{2,4,8,16}` | arXiv v1 | VERIFIED | CONFIRMED | — |
| LoRA `r` grid | `{4,8,16,32}` | arXiv v1; identical in FraudCoT | VERIFIED | CONFIRMED | — |
| LoRA dropout grid | `{0.0,0.05,0.1}` | arXiv v1; FraudCoT | VERIFIED | CONFIRMED | — |
| LR grid | `{1e-5,3e-5,1e-4}` | arXiv v1; FraudCoT | VERIFIED | CONFIRMED | — |
| Batch size | 4 | arXiv v1 | VERIFIED | CONFIRMED | — |
| Epochs | ≤10, early stopping on **val loss** | arXiv v1 | VERIFIED | CONFIRMED | Patience unstated |
| Model selection | Mean **val AUROC** | arXiv v1 | VERIFIED | CONFIRMED | — |
| Optimizer | AdamW | both | VERIFIED | CONFIRMED | — |
| Selected values of all the above | **Not published** | — | — | NONE | **Needs a user decision on search budget** |
| LoRA alpha | Not in either version, nor FraudCoT | — | VERIFIED absent | NONE | `alpha = 2r`, `PAPER_RECONSTRUCTION` |
| Max seq len, grad accumulation, warmup, weight decay, precision, patience | Not stated | — | — | NONE | Documented defaults; bf16 on A100 |
| Seeds | "5 random seeds", values unstated | both | VERIFIED | — | Seeds `0–4` |
| Metrics | Macro-F1, AUROC, AUPRC via **scikit-learn** | arXiv v1 | VERIFIED | CONFIRMED | `f1_score(macro)`, `roc_auc_score`, `average_precision_score` |
| Macro-F1 threshold | Not stated. Official ConsisGAD: best F1 over 19 validation thresholds; official PMP: fixed 0.5 | ConsisGAD `modules/evaluation.py`; PMP `training_procedure/evaluate.py` | VERIFIED | LOW (sources disagree) | Report both; val-tuned primary |
| Class reweighting | Not stated | — | — | NONE | None (plain CE, as Eq. 13) |
| Significance test for `*` | Not named | — | — | NONE | Report Welch t-test, labelled ours |

## D. Datasets

| Question | Evidence | Source | Check | Confidence | Action |
|---|---|---|---|---|---|
| AmazonVideo source file | `reviews_Amazon_Instant_Video_5.json.gz`, 2014 McAuley 5-core, 9,517,526 B, SHA-256 `7816bf30c235c526…2b48c4f54`, live at SNAP, no registration | download | VERIFIED | CONFIRMED | Auto-download + checksum |
| AmazonVideo nodes | 37,126 = line count | computed | VERIFIED | CONFIRMED | — |
| AmazonVideo label | unhelpful (fraud) ⇔ `total_votes ≥ 1` and `helpful/total < 0.5` → **4,379** exact | computed | VERIFIED | CONFIRMED | — |
| AmazonVideo zero-vote reviews (23,993) | **benign**, not unlabelled — see Loop 2 addendum (paper AUROC/AUPRC pairs fit 11.8% prevalence within 1.6 pts; 33.3% misses by 27.6) | Table 2 + binormal analysis | VERIFIED (analysis) | HIGH | `zero_vote_label: benign`; `unlabeled` switch |
| AmazonVideo R-U-R | same reviewer → 179,449 pairs | computed | VERIFIED | CONFIRMED | — |
| AmazonVideo R-P-R | same `asin` → 1,649,749 pairs | computed | VERIFIED | CONFIRMED | — |
| AmazonVideo R-S-R | **same rating + same week, NO product constraint** → 3,112,505. Paper prose says "same-product" (→ only 43,157) | computed vs arXiv/FraudCoT prose | VERIFIED | CONFIRMED (data over prose) | Follow data; document the prose contradiction |
| Week bucketing | `(unixReviewTime + 5·86400) // 604800` | computed | VERIFIED (matches) | HIGH | — |
| Edge count convention | `2 × Σ undirected pairs` = 9,883,406 exact; per-relation (no cross-relation dedup), no self-loops | computed | VERIFIED | CONFIRMED | — |
| AmazonVideo `x_num` | Rating only. **`helpful` votes excluded — they define the label (leakage).** FraudCoT serializes only `Rating` | FraudCoT App. E; leakage analysis | VERIFIED | HIGH | `x_num = [overall]` (+ one-hot variant as option) |
| YelpReviews source | Original Rayana & Akoglu YelpChi: 67,395 reviews, 13.23% filtered, plaintext included. 8,919/67,395 = 13.234% | shebuti.com | VERIFIED | CONFIRMED | — |
| YelpReviews access | **Email-gated**: `srayana@cs.stonybrook.edu`. ODDS mirror TLS-fails. No public mirror with text found | shebuti.com | VERIFIED | CONFIRMED | **USER ACTION REQUIRED** |
| Yelp ≠ `YelpChi.mat` | `.mat` = 45,954 nodes, 32 handcrafted feats, no text, 14.53% | group's `gnn_datasets` mirror | REPORTED | HIGH | Do not use `.mat` for DGP |
| Yelp relations | R-U-R same user; R-S-R same product + same star; R-T-R same product + same month | arXiv v1; CARE-GNN §4.1.2 | VERIFIED (arXiv) | CONFIRMED | Test product-scope variants against 17,486,608 (Amazon precedent) |
| Yelp edge count | 17,486,608 — untestable until data arrives | — | — | — | Validation target |
| Yelp `x_num` | Not enumerated | — | — | NONE | Rating (+ any non-label metadata), no filter flags |
| Split sizes | test = round(0.20·N) exact for Yelp, Amazon, LifeService; train = val = round(p·N), p = 2% / 3.5% / 10%; E-Commerce is an outlier | arithmetic | VERIFIED (arithmetic) | MEDIUM (INFERENCE) | Implement rule; sample from **labelled** pool, **stratified**, split fixed per seed |
| Stratification / pool / seeding | Not stated | — | — | NONE | Stratified; pool = all labelled nodes = **all N** under the default labels; `PAPER_RECONSTRUCTION` |
| E-Commerce, LifeService | ByteDance proprietary | both | VERIFIED | CONFIRMED | Interface only → `NOT_REPRODUCIBLE` |

## E. Baselines

| Method | Official repo @ head | License | Check | Notes |
|---|---|---|---|---|
| GraphSAGE | `williamleif/GraphSAGE` @ `a0fdef95` (2018) | NOASSERTION | VERIFIED | TF1 repo; DGL `SAGEConv` adapter likely required |
| HGT | `acbull/pyHGT` @ `85eaccd4` | MIT | VERIFIED | OAG-specific; DGL/PyG `HGTConv` |
| ConsisGAD | `Xtra-Computing/ConsisGAD` @ `36811c5b` | MIT | VERIFIED | Same lab; native Yelp/Amazon loaders (the `.mat` variants) |
| PMP | `Xtra-Computing/PMP` = `JhuoW/PMP` @ `3f7629f6` | **none** | VERIFIED | No license → fetch at commit, do not vendor |
| GAAP | `AtwoodDuan/GAAP` @ `6a7dbb04` | **none** | VERIFIED | Fetch, do not vendor |
| TAPE | `XiaoxinHe/TAPE` @ `d9881f7e` | MIT | VERIFIED | Needs LLM explanations |
| FLAG (KDD'25 fraud) | `BUPT-GAMMA/FLAG` @ `cb83944e` | **none** | VERIFIED exists; attribution REPORTED (no README) | Not `devnkong/FLAG` |
| GraphGPT | `HKUDS/GraphGPT` @ `db25a66f` | Apache-2.0 | VERIFIED | Vicuna backbone; Qwen3 swap is heavy |
| HiGPT | `HKUDS/HiGPT` @ `2b0793e7` | Apache-2.0 | VERIFIED | Same |
| InstructGLM | `agiresearch/InstructGLM` @ `dd2dd5ec` | Apache-2.0 | VERIFIED | — |
| MLP, LLM-only | n/a / `Qwen/Qwen3-8B` | — / Apache-2.0 | — | In-repo |

## F. Hardware

| Question | Evidence | Confidence | Action |
|---|---|---|---|
| Paper hardware | DGP: 4× A100-80GB. FraudCoT (same group, same Qwen3-8B LoRA recipe): **1× A100-80GB** | CONFIRMED | 4× is not a hard memory requirement |
| This machine | Windows 11, **no NVIDIA driver / GPU detected** (`nvidia-smi` absent) | CONFIRMED | CPU smoke path locally; Qwen3-8B stages need cloud GPU → **USER DECISION** |

---

## Loop 2 — gap resolution outcome

Loop 2 was run **inline by the lead engineer**: three of the Loop 1 agents (plus the partial fourth) terminated on the account's API spend limit, and re-spawning parallel agents would have hit the same limit. Each unresolved Loop 1 item was pursued against primary sources:

| Loop-1 gap | Loop-2 result |
|---|---|
| Hyperparameter grid | **Resolved** (arXiv v1) |
| Relation definitions | **Resolved** (arXiv v1) + Amazon edge counts reproduced exactly |
| Amazon source/labels | **Resolved**, byte-exact |
| Yelp identity | **Resolved**; access still gated |
| DeBERTa variant | **Resolved** (bibliography) |
| MDK `k=0` / normalisation | **Resolved for ranking** (SSGC code + invariance) |
| Yes/No tokens, thinking mode | **Resolved** (tokenizer artefacts) |
| Selected hyperparameter values | **Unresolvable from sources** — escalate to user (budget) |
| Split procedure details | **Unresolvable** — documented default |
| Prompt templates beyond instruction | **Unresolvable** — reconstruct, version, disclose |
| Yelp raw data | **Unresolvable without user action** |

No two primary sources disagree on an item without the disagreement being recorded above (DeBERTa vs raw; tokens vs words; R-S-R prose vs data).

---

## Loop 2 addendum — AmazonVideo label prevalence (resolved by quantitative analysis)

**Question.** Are the 23,993 AmazonVideo reviews with zero helpfulness votes *unlabelled* (the dataset subagent's inference, which makes every split 33.3% fraud), or *benign* (11.8% fraud)?

**Why it matters.** It changes the class balance of train, validation and test, and therefore every reported metric, especially AUPRC, whose random baseline equals the prevalence.

**Evidence.**

1. At 33.3% prevalence, 12 of the 13 methods in Table 2 (e.g. MLP 26.55, LLM 27.03, GraphGPT 27.82) would score an AUPRC *below random guessing* while their AUROC is 70–77, well above chance. Consistently across 12 independent methods, that is implausible.
2. Under an equal-variance binormal score model, each method's reported AUROC implies an expected AUPRC. Across all 13 methods, the mean absolute gap to the reported AUPRC is **1.6 points at 11.8% prevalence** and **27.6 points at 33.3%**. DGP: expected 34.8 vs reported 34.63.
3. Control: on YelpReviews (all reviews labelled, prevalence 13.23% known) the same model gives gaps of 2–4 points, so it is a reasonable instrument.
4. The paper's split sizes are exact fractions of **N** (test = 20% of 37,126 = 7,425), which is natural when splits are drawn from all nodes.
5. The paper's "majority of nodes are unlabeled" sentence is generic (it is also false for Yelp, where every review carries a label) and is read as "most nodes lie outside the train/val/test splits".

**Resolution.** Zero-vote reviews are **benign**; splits are drawn from all 37,126 nodes. Confidence HIGH (strong quantitative inference, not a paper statement). The other reading remains available as `graph.label.zero_vote_label: unlabeled`.

Analysis script: reproduced inline in the session log; the method is the binormal simulation described above (400k samples per point).

## Loop 2 addendum — cache-key correctness

The MDK trimming cache was keyed on split seed and target *count*. A changed split with the same size would have silently reused stale neighbour sets. The key now includes a hash of the sorted target ids.
