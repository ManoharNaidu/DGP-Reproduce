# DGP Official Repository Audit (Subagent A)

**Target:** https://github.com/Xtra-Computing/DGP
**Paper:** Li, Hu, Hooi, He, Chen — "DGP: A Dual-Granularity Prompting Framework for Fraud Detection with Graph-Enhanced LLMs", AAAI-2026
**Audit date:** 2026-09-13
**Method:** direct HTTP fetches (curl) against the GitHub REST API, raw.githubusercontent.com, arxiv.org, ojs.aaai.org; plus WebSearch/WebFetch. Every claim below is backed by a URL that was actually fetched.

---

## 1. SUMMARY VERDICT

# `README_ONLY`

The repository **exists** and is **public**, but contains **exactly one file: `README.md` (557 bytes)**. There is **zero source code**, zero data, zero configs, zero scripts, zero notebooks. The README explicitly states the code is withheld:

> `> Pending approval from our industry partner before public release.`

**There is NO arXiv appendix.** arXiv v1 (the only version) is the same 9-section paper with no appendix; the AAAI camera-ready PDF is 9 pages (pp. 15171–15179) and likewise has no appendix. There is therefore **no supplementary hyperparameter table anywhere**.

No mirror, fork, HuggingFace release, Zenodo/Drive link, or community reimplementation was found.

---

## 2. REPOSITORY EXISTENCE & STATE

Source: `https://api.github.com/repos/Xtra-Computing/DGP` (HTTP 200)

| Field | Value |
|---|---|
| `full_name` | `Xtra-Computing/DGP` |
| `id` | 1094051047 |
| `private` | `false` (public) |
| `visibility` | `public` |
| `archived` | `false` |
| `description` | "DGP: A Dual-Granularity Prompting Framework for Fraud Detection with Graph-Enhanced LLMs (AAAI 2026)" |
| `default_branch` | `main` |
| `size` | **3** (KB) |
| `created_at` | **2025-11-11T07:27:40Z** |
| `pushed_at` | **2025-11-13T01:59:39Z** (last code push — ~10 months before this audit) |
| `updated_at` | 2026-08-23T23:51:28Z (metadata only, e.g. a star) |
| `stargazers_count` | 4 |
| `forks_count` | **0** |
| `open_issues_count` | 1 |
| `license` | **null** (no license file) |
| `homepage` | `null` |
| `topics` | `[]` |
| `language` | **`null`** (GitHub detected no programming language — consistent with no source files) |
| `has_wiki` | `false` |
| `has_discussions` | `false` |
| `has_pages` | `false` |

---

## 3. FULL RECURSIVE FILE TREE

Source: `https://api.github.com/repos/Xtra-Computing/DGP/git/trees/main?recursive=1` (HTTP 200)

```json
{
  "sha": "3bfe436c289d419f7d75acb5c01e12a9b0a6c663",
  "tree": [
    {
      "path": "README.md",
      "mode": "100644",
      "type": "blob",
      "sha": "a5525147fd7e7078d2d8c001bfaf1edb2a3d3f0f",
      "size": 557
    }
  ],
  "truncated": false
}
```

**Complete file list (1 path, `truncated: false` so this is exhaustive):**

1. `README.md` — 557 bytes

That is the entire repository.

### Branches
Source: `https://api.github.com/repos/Xtra-Computing/DGP/branches`
```json
[{"name":"main","commit":{"sha":"3bfe436c289d419f7d75acb5c01e12a9b0a6c663"},"protected":false}]
```
**Only one branch (`main`). No hidden `dev`/`code`/`release` branch.**

### Tags
Source: `https://api.github.com/repos/Xtra-Computing/DGP/tags` → `[]` (empty)

### Releases
Source: `https://api.github.com/repos/Xtra-Computing/DGP/releases` → `[]` (empty). No release assets, no attached zip.

### Forks
Source: `https://api.github.com/repos/Xtra-Computing/DGP/forks` → `[]` (empty). **No fork exists that could contain code.**

### Pull requests
Source: `https://api.github.com/repos/Xtra-Computing/DGP/pulls?state=all&per_page=100` → `[]` (empty)

---

## 4. README — VERBATIM, COMPLETE

Source: `https://raw.githubusercontent.com/Xtra-Computing/DGP/main/README.md` (HTTP 200, 557 bytes — this is the file in full)

````markdown
# DGP
DGP: A Dual-Granularity Prompting Framework for Fraud Detection with Graph-Enhanced LLMs (AAAI 2026)

> Pending approval from our industry partner before public release.

## Cite

If you use DGP in a scientific publication, we would appreciate citations to the following paper:

```
@article{li2025dgp,
  title={DGP: A Dual-Granularity Prompting Framework for Fraud Detection with Graph-Enhanced LLMs},
  author={Li, Yuan and Hu, Jun and Hooi, Bryan and He, Bingsheng and Chen, Cheng},
  journal={arXiv preprint arXiv:2507.21653},
  year={2025}
}
```
````

**Installation instructions:** NOT FOUND (no such section).
**Dataset instructions:** NOT FOUND.
**Run commands:** NOT FOUND.
**Stated hyperparameters:** NOT FOUND.
**Data/checkpoint links (Drive/HF/Zenodo/OneDrive):** NOT FOUND — the README contains no URL of any kind other than the arXiv id in the BibTeX.

---

## 5. COMMIT HISTORY (complete — 3 commits)

Source: `https://api.github.com/repos/Xtra-Computing/DGP/commits?per_page=100`

| SHA (12) | Date (UTC) | Author | Message |
|---|---|---|---|
| `3bfe436c289d` | 2025-11-13T01:59:39Z | yuanlics | `Update README with citation details` / `Added citation information for the DGP framework.` |
| `8ab8b3836a6a` | 2025-11-12T07:25:49Z | yuanlics | `Update README.md` |
| `2cbf9439dee3` | 2025-11-11T07:27:41Z | yuanlics | `Initial commit` |

All three commits touch only the README. **No code was ever committed and then deleted** — the initial commit is the repo creation and the history is 3 README edits. There is no orphan/dangling code commit reachable via the commits API.

---

## 6. ISSUES (complete — 1 issue)

Source: `https://api.github.com/repos/Xtra-Computing/DGP/issues?state=all&per_page=100` — returns exactly 1 item.

**Issue #1** — https://github.com/Xtra-Computing/DGP/issues/1
- State: **open**
- Author: `starizyj`
- Created: **2026-07-23T11:37:38Z**
- Comments: **0**
- Title (verbatim): `会更新源码吗` ("Will the source code be updated/released?")
- Body (verbatim): `wait for code`

**This is highly informative:** a third party asked for the code on 2026-07-23 and, as of 2026-09-13, **the authors have not replied** (0 comments) and have pushed nothing since 2025-11-13.

---

## 7. PYTHON SOURCE / IMPLEMENTATION DETAILS

**NOT FOUND — there is no Python source of any kind in the repository.**

Consequently, all of the following are **NOT FOUND in the repo**:

- MDK / Markov diffusion kernel implementation — **NOT FOUND**
- Metapath construction code — **NOT FOUND**
- Node summarization prompt strings — **NOT FOUND**
- Metapath summarization prompt strings — **NOT FOUND**
- Final classification prompt template — **NOT FOUND**
- LoRA config (`peft` `LoraConfig`) — **NOT FOUND**
- Training loop / trainer script — **NOT FOUND**
- Yes/No logit extraction code — **NOT FOUND**
- Dataset preprocessing / split construction — **NOT FOUND**
- `requirements.txt` / `environment.yml` / `pyproject.toml` / `setup.py` / Dockerfile — **NOT FOUND** (the tree contains only `README.md`)

**What I tried:** full recursive git tree on `main` (`truncated:false`, 1 blob); branch listing (1 branch); tag listing (empty); release listing (empty); fork listing (empty); PR listing (empty); GitHub code search for a mirror. Every avenue confirms a single README blob.

### 7.13 Eq. 6 diffusion operator normalization

**Cannot be answered from code — NOT FOUND (no code exists).**

The **paper text** (arXiv v1 HTML, §4.2 "Diffusion-based Metapath Trimming") states, verbatim:

> "For each metapath `P`, we form the row-stochastic transition matrix `T_P = D_P^{-1} A_P` from the metapath-specific adjacency matrix `A_P` and degree matrix `D_P = diag(A_P 1)`. Averaging the first `K` random-walk powers, i.e., `K`-hops, yields the Markov diffusion operator:"
>
> `Z_P(K) = \frac{1}{K}\sum_{k=0}^{K}\mathbf{T}_{P}^{k}`  — Eq. (6)

Source: https://arxiv.org/html/2507.21653v1

**Note for the reproduction:** as written, the sum runs `k = 0 … K` (i.e. **K+1 terms**, including the identity `T^0`) but is divided by **`K`**, not `K+1`. This is an inconsistency/typo in the paper: the operator is not an average and is not row-stochastic (its rows sum to `(K+1)/K`). Since ranking by Euclidean diffusion distance (Eq. 8) is invariant to a positive global scale factor, `1/K` vs `1/(K+1)` does **not** change the Top-M neighbor selection (Eq. 9) — the choice is immaterial for reproduction, but there is **no code to confirm which the authors actually used**, and whether `k` starts at 0 or 1 **does** matter (including `T^0` adds the node's own features to every node's embedding, which is a constant shift per node and does affect distances).

---

## 8. METAPATH DEFINITIONS

**NOT FOUND in the repo.** Available only from the paper (arXiv v1 §5.1 "Datasets"), verbatim:

**YelpReviews** (Rayana and Akoglu 2015; graph construction following Dou et al. 2020 / CARE-GNN):
> "we construct a heterogeneous graph with three types of edges: reviews written by the same user (**R-U-R**), reviews on the same product with the same star rating (**R-S-R**), and reviews posted in the same month for the same product (**R-T-R**)."

**AmazonVideo** (McAuley and Leskovec 2013):
> "The graph contains three types of edges: reviews posted by the same user (**R-U-R**), reviews posted on the same product (**R-P-R**), and same-product reviews posted with the same rating and within the same week (**R-S-R**)."

> ⚠️ Note the Amazon relation set here (**R-U-R / R-P-R / R-S-R**) is **not** the standard Amazon benchmark relation set from Dou et al. 2020, which is **U-P-U / U-S-U / U-V-U** over *user* nodes. DGP redefines Amazon as a **review-level** graph (node = review, label = helpful/unhelpful), which is a non-standard construction. No code is given to reproduce it.

**LifeService** and **E-Commerce** (ByteDance proprietary): metapaths **NOT DEFINED** in the paper. Table 1 only says they have **9** and **5** edge types respectively.

---

## 9. HYPERPARAMETERS

**NOT FOUND in the repo (no code, no config).** The paper gives only **search grids, never the selected values**. Verbatim from arXiv v1 §5.1 "Parameter Settings" and "Implementation Details":

> "For all evaluated models, we tune hyperparameters using grid search based on validation performance. For DGP, we tune the bi-level summarization budgets `B_node, B_meta ∈ {10, 20, 40, 80}`, the the number of hops `K ∈ {1, 2, 3}`, and the neighbor truncation size `M ∈ {2, 4, 8, 16}` for each dataset."

> "For LoRA-based finetuning of LLM methods, we tune the LoRA rank `r ∈ {4, 8, 16, 32}`, the LoRA dropout rate `∈ {0.0, 0.05, 0.1}`, and the learning rate `∈ {1e−5, 3e−5, 1e−4}`. We set the batch size to 4 and finetune for up to 10 epochs with early stopping based on validation loss."

> "All hyperparameters are selected to optimize the average AUROC on the validation set."

> "We conduct all experiments on a machine with 4 × NVIDIA A100 GPUs (80GB). For all LLM-tuning methods, we use **Qwen3-8B** LLM backbone (Team 2025) for fair comparison. We insert **LoRA adapters into all attention layers** and use the **AdamW** optimizer for finetuning. We adopt classification metrics including Macro-F1, AUROC, and AUPRC, and report the mean and standard deviation over **5 random seeds**. All evaluation metrics are computed using the scikit-learn library."

From §5.3 "Impact of Summarization Length":
> "Figure 5 presents fraud detection metrics across a range of budgets `B ∈ {5, 10, 20, 40, 80}` for the YelpReviews and AmazonVideo datasets. For simplicity, we assume a unified budget, i.e., `B_node = B_meta`." (Figure 5 caption: "Impact of summarization length (**words**)" — note the caption says *words* while the text says *tokens*; an unresolved ambiguity.)

**Not specified anywhere:** LoRA **alpha**, LoRA **target_modules** (only "all attention layers" prose), **max sequence length**, **gradient accumulation steps**, **warmup / LR schedule**, **weight decay**, the **5 seed values**, and the **selected** values of `K`, `M`, `B_node`, `B_meta`, `r`, dropout, and LR per dataset.

---

## 10. PROMPT TEMPLATES

**NOT FOUND in the repo. NOT printed in full in the paper.** The only verbatim prompt fragments that exist anywhere are in arXiv v1 §5.3 "Impact of Task-Aware Summarization":

> "In the task-agnostic setting, we use a generic instruction such as **`Summarize the text within 10 tokens`**. In contrast, the task-aware setting introduces domain-specific cues, e.g., **`Summarize the text within 10 tokens, focusing on signals indicative of fraudulent behavior`**."

DGP uses the **task-agnostic** variant (the paper finds task-aware degrades performance).

The final classification prompt is given **only as a formula**, not as text (Eq. 12):
> `prompt(v) = x^text_v ⊕ [ ⊕_{P∈𝒫} ( S_P(v) ⊕ a_P(v) ) ]`

The metapath summarization is likewise only a formula (Eq. 10):
> `S_P(v) = Summarize( ⊕_{u ∈ Ñ_P(v)} s_u ; B_meta )`

and node-level summarization (Eq. 5):
> `s_v = Summarize( x^text_v ; B_node )`

**There is no verbatim system prompt, no instruction wrapper, no question phrasing ("Is this review fraudulent? Answer Yes or No."), no chat template, and no example prompt printed anywhere.** Figure 3 ("Overview of the proposed DGP framework") is a schematic; there is no prompt-listing figure.

### Classification head / Yes-No logit extraction (paper only, §4.4, verbatim)

> "We finetune the LLM on labeled nodes by minimizing the cross-entropy loss over the first generated token:
> `L = -\frac{1}{|V_train|} \sum_{v∈V_train} log p_θ(y_v | prompt(v))`
> where `y_v ∈ {Yes, No}` denotes the correct answer, and `p_θ(y_v | prompt(v))` represents the token probability output by the LLM.
> During inference, we apply a softmax function over the logits of the first generated token for the fraud probability:
> `p_v = \frac{exp(logit_Yes)}{exp(logit_Yes) + exp(logit_No)}`
> where `logit_Yes` and `logit_No` are the pre-softmax scores assigned by the model to the tokens `Yes` and `No`, respectively."

### Numerical summarization (paper only, §4.3, verbatim)
> "we perform mean aggregation along each metapath ... `a_P(v) = \frac{1}{|Ñ_P(v)|} \sum_{u∈Ñ_P(v)} x_u^num` where `x_u^num` denotes either a real-valued numerical feature or a categorical vector encoded as one-hot or multi-hot."

**How `a_P(v)` (a float vector) is verbalized into the text prompt is NEVER specified.** This is a material reproduction gap.

---

## 11. DATASETS, SPLITS, DOWNLOAD LINKS

**No download link, no preprocessing script, no split file in the repo — NOT FOUND.**

Paper Table 1 (arXiv v1), transcribed verbatim:

| Dataset | Node Type | Textual | Numerical | # Nodes | # Edges | # Edge Types | # Frauds | # Train / Val / Test |
|---|---|---|---|---|---|---|---|---|
| YelpReviews | Service Review | ✓ | ✓ | 67,395 | 17,486,608 | 3 | 8,919 | 1,348 / 1,348 / 13,479 |
| AmazonVideo | Product Review | ✓ | ✓ | 37,126 | 9,883,406 | 3 | 4,379 | 1,299 / 1,299 / 7,425 |
| E-Commerce | Shop Profile | ✓ | ✓ | 182,043 | 27,196,608 | 9 | 3,256 | 1,309 / 1,309 / 3,928 |
| LifeService | Shop Profile | ✓ | ✓ | 12,868 | 82,912 | 5 | 2,868 | 1,287 / 1,287 / 2,574 |

Paper notes (verbatim):
> "Instead of using the handcrafted features introduced in the original work (Rayana and Akoglu 2015), we directly utilize the **original texts** for LLM-based methods."
> "We also perform evaluation on two **proprietary industry datasets: LifeService and E-Commerce**, which are real-world graphs sampled from our industry partner, **ByteDance**."
> "we construct training sets with a limited number of labeled samples ... the sum of the dataset split sizes, including the training, validation, and test sets, can be **smaller than the total number of nodes**. This aligns with a real-world scenario in which the majority of nodes are unlabeled."

Observations relevant to reproduction:
- Yelp: 67,395 nodes but only 16,175 are in any split (2% train / 2% val / 20% test).
- **YelpReviews nodes = 67,395, but the standard YelpChi benchmark has 45,954 nodes.** DGP's graph is a different/larger construction. No script is given.
- AmazonVideo (37,126 review nodes) is **not** the standard Amazon-Instruments benchmark (11,944 user nodes). No script given.
- The two ByteDance datasets are **permanently unavailable** (proprietary; this is the exact reason the code is withheld per the README).
- **No random seed, no split-generation code, no split index files** — the exact train/val/test partitions are unrecoverable.

---

## 12. EXTERNAL DATA / CHECKPOINT LINKS (Drive, HF, Zenodo, OneDrive)

**NOT FOUND.** Checked:
- README: contains no URLs at all (only an arXiv id inside BibTeX). Source: raw README fetch.
- arXiv abs page: "No supplementary materials or code repositories are explicitly linked." Source: https://arxiv.org/abs/2507.21653
- arXiv full text: no code-availability statement, no URL. Source: https://arxiv.org/html/2507.21653v1 (full text downloaded and grepped locally).
- AAAI OJS article page: only the article PDF is downloadable; **no supplementary file, no appendix file, no code/data link**. Source: https://ojs.aaai.org/index.php/AAAI/article/view/38541
- GitHub releases/tags: empty.

---

## 13. ARXIV VERSION & APPENDIX CHECK (explicitly requested)

Source: https://arxiv.org/abs/2507.21653 and https://arxiv.org/html/2507.21653v1 (both fetched, HTTP 200; HTML downloaded locally, 224,692 bytes)

- **Versions: exactly ONE.** Submission history: `[v1] Tue, 29 Jul 2025 10:10:47 UTC (331 KB)`. There is **no v2**.
- **Comments field:** empty / not present (no "N pages, M figures" note).
- **Categories:** cs.LG (primary); cs.AI.
- **DOI:** https://doi.org/10.48550/arXiv.2507.21653
- **Full table of contents of arXiv v1** (extracted verbatim from the HTML nav):
  Abstract · 1 Introduction · 2 Related Work (2.1 GNNs for Fraud Detection, 2.2 Integrating LLMs with Graphs) · 3 Preliminaries (3.1 Graph-based Fraud Detection, 3.2 Metapaths on Heterogeneous Graphs) · 4 Methodology (4.1 Dual Granularity Prompting, 4.2 Textual Summarization [Node-level Summarization, Diffusion-based Metapath Trimming, Metapath Summarization], 4.3 Numerical Summarization, 4.4 Fraud Detection with DGP, 4.5 Complexity Analysis of DGP, 4.6 Attention Dilution under Class Imbalance) · 5 Experiments (5.1 Experimental Setup, 5.2 Performance Evaluation, 5.3 Detailed Analysis) · 6 Conclusion · **References**
- **The ToC terminates at References. There is NO Appendix, no supplementary section, no "Reproducibility Checklist" section.** A local grep of the full extracted text for `appendix|Reproducib` returned **zero** matches outside the two "Parameter Settings"/"Implementation Details" headings.

**AAAI camera-ready cross-check:** https://ojs.aaai.org/index.php/AAAI/article/download/38541/42503 downloaded (406,277 bytes, valid PDF). **9 page objects**, matching the OJS page range **15171–15179**. No appendix. So the arXiv and AAAI versions carry the **same** information content — **the hoped-for "arXiv appendix with hyperparameters" does not exist.**

---

## 14. MIRROR / REIMPLEMENTATION SEARCH

| Where I looked | URL fetched | Result |
|---|---|---|
| GitHub repo search | `https://api.github.com/search/repositories?q=dual+granularity+prompting+fraud` | `total_count: 1` — only `Xtra-Computing/DGP` itself |
| Forks of DGP | `https://api.github.com/repos/Xtra-Computing/DGP/forks` | `[]` |
| Whole Xtra-Computing org (68 repos) | `https://api.github.com/orgs/Xtra-Computing/repos?per_page=100` | Only one DGP-related repo (`DGP`). Nearest-neighbour repos are `ConsisGAD` (a DGP **baseline**) and `PMP` (another DGP baseline) — both are baselines, not DGP code. |
| First author's personal GitHub | `https://api.github.com/users/yuanlics/repos?per_page=100` | 8 repos, **none** DGP-related (MMClaw, crypto_trade_backend, iperf, onlchallenge, group1_graph_infomax, team_p_cas, DeepFM_with_PyTorch, NUS-Slides-Latex-Template) |
| Web search (2 queries) | WebSearch | Only points back to the same empty GitHub repo, arXiv, and the AAAI OJS page. No HuggingFace model/dataset, no Zenodo, no Drive, no OpenReview supplementary. |
| PapersWithCode mirror | `https://paperswithcode.co/paper/2507.21653` | HTTP **403 Forbidden** — could not read (stated for honesty; a PwC entry likely just links the same empty GitHub repo) |

**Useful side-finding:** the baselines `ConsisGAD` (https://github.com/Xtra-Computing/ConsisGAD) and `PMP` (https://github.com/Xtra-Computing/PMP) ARE released by the same lab and contain real code — these are the standard sources for the Yelp/Amazon fraud graph loaders and can serve as a starting point for dataset construction.

---

## 15. DETAIL TABLE

| Detail | Found? | Value | Source URL |
|---|---|---|---|
| Repo exists / public | **YES** | public, not archived, 3 KB, 4 stars, 0 forks | https://api.github.com/repos/Xtra-Computing/DGP |
| Repo completeness | **README_ONLY** | 1 blob: `README.md` (557 B); `language: null` | https://api.github.com/repos/Xtra-Computing/DGP/git/trees/main?recursive=1 |
| Reason code withheld | **YES** | "Pending approval from our industry partner before public release." | https://raw.githubusercontent.com/Xtra-Computing/DGP/main/README.md |
| Branches | **YES** | 1 (`main`) | https://api.github.com/repos/Xtra-Computing/DGP/branches |
| Tags / Releases / Forks / PRs | **YES (all empty)** | `[]`, `[]`, `[]`, `[]` | `.../tags`, `.../releases`, `.../forks`, `.../pulls?state=all` |
| Commits | **YES** | 3, all README-only, 2025-11-11 → 2025-11-13 | https://api.github.com/repos/Xtra-Computing/DGP/commits?per_page=100 |
| Issues | **YES** | 1 open, unanswered code request (2026-07-23) | https://github.com/Xtra-Computing/DGP/issues/1 |
| License | **NO** | `license: null`, no LICENSE file | https://api.github.com/repos/Xtra-Computing/DGP |
| Any Python source | **NO** | NOT FOUND | tree API (truncated:false) |
| requirements.txt / env.yml / pyproject | **NO** | NOT FOUND | tree API |
| MDK / diffusion implementation | **NO (code)** | Paper Eq. 6: `Z_P(K) = (1/K) Σ_{k=0}^{K} T_P^k`, `T_P = D_P^{-1} A_P` | https://arxiv.org/html/2507.21653v1 |
| Eq. 6 normalization `1/K` vs `1/(K+1)` | **PAPER ONLY** | Paper literally writes `1/K` with sum `k=0..K` (K+1 terms) — internally inconsistent; no code to disambiguate | https://arxiv.org/html/2507.21653v1 |
| Diffusion distance | **PAPER ONLY** | Eq. 8: `δ_K^(P)(u,v) = ‖h_u^(P)(K) − h_v^(P)(K)‖_2` on `h_i = [Z_P(K) X]_{i:}` | https://arxiv.org/html/2507.21653v1 |
| Top-M trimming | **PAPER ONLY** | Eq. 9: `Ñ_P(v) = TopM_{u∈N_P(v)} (−δ_K^(P)(u,v))` | https://arxiv.org/html/2507.21653v1 |
| Metapaths — Yelp | **PAPER ONLY** | **R-U-R**, **R-S-R** (same product + same star rating), **R-T-R** (same month + same product) | https://arxiv.org/html/2507.21653v1 |
| Metapaths — Amazon | **PAPER ONLY** | **R-U-R**, **R-P-R** (same product), **R-S-R** (same product + same rating + same week) | https://arxiv.org/html/2507.21653v1 |
| Metapaths — LifeService / E-Commerce | **NO** | NOT DEFINED (only "5" and "9" edge types) | https://arxiv.org/html/2507.21653v1 |
| K (hops) | **GRID ONLY** | searched over `{1,2,3}`; **selected value never stated** | https://arxiv.org/html/2507.21653v1 |
| M (top-M neighbors) | **GRID ONLY** | searched over `{2,4,8,16}`; **selected value never stated** | same |
| B_node, B_meta | **GRID ONLY** | searched over `{10,20,40,80}`; ablation sweeps `B∈{5,10,20,40,80}` with `B_node=B_meta`; **selected values never stated**; text says "tokens", Fig. 5 caption says "words" | same |
| LoRA rank r | **GRID ONLY** | `{4,8,16,32}`; selected never stated | same |
| LoRA alpha | **NO** | NOT FOUND anywhere | — |
| LoRA dropout | **GRID ONLY** | `{0.0, 0.05, 0.1}`; selected never stated | same |
| LoRA target_modules | **PARTIAL** | prose only: "We insert LoRA adapters into **all attention layers**" (no module-name list) | same |
| Learning rate | **GRID ONLY** | `{1e−5, 3e−5, 1e−4}`; selected never stated | same |
| Batch size | **YES** | **4** | same |
| Epochs | **YES** | "up to **10** epochs with **early stopping based on validation loss**" | same |
| Optimizer | **YES** | **AdamW** (Loshchilov & Hutter 2019) | same |
| Gradient accumulation | **NO** | NOT FOUND | — |
| Max sequence length | **NO** | NOT FOUND | — |
| LR schedule / warmup / weight decay | **NO** | NOT FOUND | — |
| Seeds | **PARTIAL** | "mean and standard deviation over **5 random seeds**"; **actual seed values NOT stated** | same |
| Backbone LLM | **YES** | **Qwen3-8B** (Team 2025), same for all LLM-tuning methods | same |
| Hardware | **YES** | 4 × NVIDIA A100 80GB | same |
| Model selection criterion | **YES** | "selected to optimize the average **AUROC** on the validation set" | same |
| Metrics | **YES** | Macro-F1, AUROC, AUPRC via scikit-learn | same |
| Node summarization prompt | **PARTIAL/NO** | only the generic example `Summarize the text within 10 tokens`; full template NOT FOUND | same |
| Task-aware variant prompt | **YES (example)** | `Summarize the text within 10 tokens, focusing on signals indicative of fraudulent behavior` (found to *hurt*; DGP uses task-agnostic) | same |
| Metapath summarization prompt | **NO** | formula only (Eq. 10); no text template | same |
| Final classification prompt | **NO** | formula only (Eq. 12): `prompt(v)=x^text_v ⊕ [⊕_P (S_P(v) ⊕ a_P(v))]`; no instruction/question wording | same |
| Which LLM does the summarizing | **NO** | never stated (Qwen3-8B? a separate summarizer? an API model?) | — |
| Yes/No logit extraction | **PAPER ONLY** | Eq. 14: `p_v = exp(logit_Yes)/(exp(logit_Yes)+exp(logit_No))` over **first generated token**; train loss = CE on first token (Eq. 13), `y_v ∈ {Yes, No}` | same |
| Numerical summarization | **PAPER ONLY** | Eq. 11: mean-pool `x^num` over `Ñ_P(v)`; categorical as one-hot/multi-hot. **Verbalization into prompt NOT specified** | same |
| Dataset stats | **YES (paper Table 1)** | see §11 above | same |
| Dataset download links | **NO** | NOT FOUND | — |
| Preprocessing / split code | **NO** | NOT FOUND | — |
| ByteDance datasets | **UNAVAILABLE** | proprietary; the stated blocker for the whole code release | README + paper |
| arXiv versions | **YES** | **v1 only**, 29 Jul 2025, 331 KB | https://arxiv.org/abs/2507.21653 |
| arXiv appendix | **NO** | ToC ends at References; zero `appendix` matches in full text | https://arxiv.org/html/2507.21653v1 |
| AAAI supplementary | **NO** | 9-page PDF (pp. 15171–15179), no supplementary files on OJS page | https://ojs.aaai.org/index.php/AAAI/article/view/38541 |
| Any mirror / reimplementation | **NO** | GitHub search total_count=1; 0 forks; author's repos unrelated | see §14 |
| PapersWithCode entry | **INACCESSIBLE** | HTTP 403 on paperswithcode.co (could not verify content) | https://paperswithcode.co/paper/2507.21653 |

---

## 16. UNRESOLVED ITEMS — what the paper needs but the repo does not provide

Every item below must be **guessed, re-derived, or swept** in the reproduction; none is recoverable from any released artifact.

**A. Prompt engineering (highest risk — DGP *is* a prompting method)**
1. The **exact node-level summarization prompt** (only "Summarize the text within 10 tokens" is exemplified, and `B` varies).
2. The **exact metapath-level summarization prompt** — never shown in any form.
3. The **exact final classification prompt**: system message, task instruction, the Yes/No question wording, field labels/delimiters, whether metapath names ("R-U-R") appear literally in the prompt, ordering of target text vs. summaries.
4. **Chat template** handling for Qwen3-8B (whether `<|im_start|>`-style chat formatting or raw completion; whether Qwen3 *thinking mode* is enabled/disabled — critical, since the method reads the **first generated token** and thinking mode would emit `<think>` first).
5. **Which model performs the summarization** — the same Qwen3-8B, a different/larger model, or an API model. Unstated. This alone can move results substantially.
6. **How the numerical vector `a_P(v)` is rendered as text** (rounding, units, feature names, ordering). Unstated.
7. How truncation/overflow is handled when the concatenated prompt exceeds context.

**B. Hyperparameters**
8. **Selected** `K`, `M`, `B_node`, `B_meta` per dataset (only the 3×4×4 grid is given → 48 combos/dataset).
9. **Selected** LoRA `r`, dropout, LR per dataset (4×3×3 = 36 combos).
10. **LoRA alpha** — completely absent.
11. **LoRA `target_modules`** explicit list ("all attention layers" ⇒ presumably `q_proj,k_proj,v_proj,o_proj`, but unconfirmed; MLP modules explicitly excluded?).
12. **max sequence length** / context window used.
13. **Gradient accumulation**, LR schedule, warmup ratio, weight decay, max grad norm, precision (bf16?), and whether early stopping had a patience value.
14. The **5 seed values** (needed to reproduce the reported std devs).
15. Whether `B` is measured in **tokens** or **words** (text and Figure 5 caption disagree).

**C. Data**
16. **No download links** for YelpReviews / AmazonVideo in the DGP node/edge construction. The reported counts (Yelp 67,395 nodes / 17.5M edges; Amazon 37,126 nodes / 9.9M edges) do **not** match the standard YelpChi (45,954) / Amazon-Instruments (11,944) benchmarks — DGP builds its own **review-level** graphs. No script is provided.
17. **Raw text fields**: the paper says it uses "the original texts" rather than the handcrafted 32-D/25-D features — but which text fields (review body only? title? shop profile fields?) is unstated, and the standard `.mat` benchmark files do **not** ship raw text, so an independent join back to the original Yelp/Amazon corpora is required and unspecified.
18. **Split construction**: exact train/val/test index sets, the sampling procedure, the stratification (fraud ratio per split), and the seed. Only the sizes are given.
19. **LifeService and E-Commerce are permanently unobtainable** (ByteDance proprietary) — 2 of the 4 datasets, and half of the headline results, are unreproducible in principle.
20. Metapath definitions for the two proprietary datasets (5 and 9 edge types) are undocumented.

**D. Algorithm**
21. **Eq. 6 normalization**: `1/K` with `Σ_{k=0}^{K}` is inconsistent (K+1 terms). Whether the implementation used `1/K` or `1/(K+1)`, and whether `k` starts at 0 or 1, is unverifiable. (Scale is immaterial for Top-M ranking; the `k=0` term is not.)
22. **Which features `X` feed the diffusion** (Eq. 7): raw numerical features, text embeddings, or a concatenation? Unstated — the paper says only "raw node features" and this determines the entire trimming behaviour.
23. Whether diffusion distance is computed **globally** (dense `Z_P X` over all n nodes) or locally per target node; at 17.5M edges the dense `T^k` construction matters for feasibility and is unaddressed.
24. Tie-breaking in Top-M, and behaviour when `|N_P(v)| < M` or `= 0` (empty metapath neighborhood → what goes into the prompt?).
25. Whether the K-hop expansion (the `R^{K+1}` term in the complexity analysis) means metapath **sequences** of length K or K applications of the same metapath.
26. Whether node summaries `s_u` are computed **once globally and cached** or per-target (affects both cost and results).

**E. Reproduction logistics**
27. **No license** on the repo — even when released, redistribution terms are undefined.
28. **No baseline configurations** — "All baselines are implemented using official code" with hyperparameters tuned "within the recommended ranges", but no configs are given for GraphSAGE/HGT/ConsisGAD/PMP/GAAP/MLP/TAPE/GraphGPT/HiGPT/InstructGLM.
29. **No evaluation script** — Macro-F1 threshold selection (0.5? tuned on val?) is unstated, which materially changes Macro-F1 on imbalanced data.

**Outlook on release:** last push **2025-11-13**; the only issue (a code request, 2026-07-23) is **unanswered after ~7 weeks**; release is gated on a ByteDance approval that the README flags and that has not moved in ~10 months. **A reproduction should be planned assuming the code will never appear.**
