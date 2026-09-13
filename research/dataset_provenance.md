# DGP Dataset Provenance — YelpReviews & AmazonVideo

**Subagent D report.** Target paper: *DGP: A Dual-Granularity Prompting Framework for Fraud
Detection with Graph-Enhanced LLMs* (AAAI-2026), Yuan Li, Jun Hu, Bryan Hooi, Bingsheng He,
Cheng Chen.

Every factual claim below is tied to a URL that was actually fetched, or to a computation run
locally on a file that was actually downloaded. Claims that could not be verified are marked
**UNVERIFIED**. Arithmetic observations are marked **INFERENCE**.

---

## 0. HEADLINE RESULT

**AmazonVideo is fully solved and byte-level reproducible.** Every number in DGP's Table 1 for
AmazonVideo was reproduced exactly from a single public file, including the 7-digit edge count.
See §2.6.

**YelpReviews is identified but access-gated.** The source is confirmed as the *original*
Rayana & Akoglu YelpChi release (67,395 reviews, with raw text), **not** the `YelpChi.mat` used
by CARE-GNN/PC-GNN/DGL (45,954 nodes, 32 numeric features, no text). The original release is
obtainable only by emailing the author. See §1.

**The official DGP code repo is empty.** <https://github.com/Xtra-Computing/DGP> contains only a
README stating *"Pending approval from our industry partner before public release."* No data, no
preprocessing scripts. (Fetched; repo shows 3 commits, README-only.)

**Critical: the arXiv v1 preprint contains a much more detailed Datasets section than the AAAI
camera-ready.** arXiv:2507.21653v1 spells out all six edge-type definitions; the AAAI version
deletes them. Use the arXiv version for reproduction. Both are cited below.

---

## 1. DATASET 1 — YelpReviews (YelpChi)

### 1.1 What DGP reports (Table 1, verbatim from the local PDF and arXiv v1)

```
Dataset      Node Type       Textual Numerical #Nodes  #Edges      #EdgeTypes #Frauds  #Train/Val/Test
YelpReviews  Service Review     ✓        ✓      67,395  17,486,608      3       8,919  1,348 / 1,348 / 13,479
```

### 1.2 Canonical source (a)

| Item | Value | Source |
|---|---|---|
| Author page | <https://shebuti.com/yelpchi-dataset/> | fetched |
| ODDS mirror | <https://odds.cs.stonybrook.edu/yelpchi-dataset/> | **UNVERIFIED by direct fetch** — TLS handshake failure (`SSLV3_ALERT_HANDSHAKE_FAILURE`, curl exit 35) from this environment on 3 attempts. The page exists and is indexed by search. |
| Reviews | **67,395** | shebuti.com |
| Businesses | 201 hotels and restaurants (Chicago area) | shebuti.com |
| Reviewers | 38,063 | shebuti.com |
| Spam rate | **13.23%** of reviews, by 20.33% of reviewers | shebuti.com |
| Content | "product and user information, timestamp, ratings, and a plaintext review" | shebuti.com |
| **Download** | **No direct URL. Email request required:** `srayana@cs.stonybrook.edu` | shebuti.com (identical wording on the YelpNYC page, fetched) |
| License | None stated beyond a generic copyright notice | shebuti.com |
| Checksums | None published | — |
| File names / column names | **UNVERIFIED** — not published on either the author page or the ODDS page. Searched for `metadata` / `reviewContent` file naming; no authoritative listing found. | — |

Ground truth is Yelp's own filter: *recommended* reviews = genuine, *filtered/unrecommended* =
fake. Originally collected by Mukherjee et al., ICWSM 2013, *What Yelp Fake Review Filter Might
Be Doing?* (<https://ojs.aaai.org/index.php/ICWSM/article/view/14389>); reused and released by
Rayana & Akoglu, KDD 2015.

### 1.3 The GNN-community `YelpChi.mat` (b) — **this is NOT what DGP uses**

From the DGL source `python/dgl/data/fraud.py` (fetched) and the CARE-GNN paper PDF
(arXiv:2008.08692, downloaded and text-extracted locally):

| | CARE-GNN paper Table 2 (directed pairs) | DGL `FraudYelpDataset` (both directions) |
|---|---|---|
| Nodes | 45,954 (14.5% fraud) | 45,954 |
| R-U-R | 49,315 | 98,630 |
| R-T-R | 573,616 | 1,147,232 |
| R-S-R | 3,402,743 | 6,805,486 |
| ALL (merged) | 3,846,979 | — |
| Features | 32 handcrafted | 32 |
| Labels | — | spam 6,677 / legit 39,277 |

DGL's counts are exactly 2× CARE-GNN's, confirming DGL stores each undirected edge twice.

Download URLs (both verified live, HTTP 200, this session):

- `https://data.dgl.ai/dataset/FraudYelp.zip` — 17,980,652 bytes, `application/zip`, contains `YelpChi.mat`
- `https://data.dgl.ai/dataset/FraudAmazon.zip` — 26,122,297 bytes, contains `Amazon.mat`
- `https://raw.githubusercontent.com/YingtongDou/CARE-GNN/master/data/YelpChi.zip` — HTTP 200
- `https://raw.githubusercontent.com/YingtongDou/CARE-GNN/master/data/Amazon.zip` — HTTP 200
- PC-GNN (`PonderLY/PC-GNN`) `data/*.zip` on branch `main` → **HTTP 404**. Branch/path differs; use CARE-GNN or DGL instead.

No checksums are published for any of these. No registration required.

### 1.4 Resolving the 67,395 vs 45,954 discrepancy (c) — **RESOLVED**

DGP does **not** use `YelpChi.mat`. It rebuilds the graph from the original Rayana release.
The arXiv v1 says so explicitly (verbatim, arXiv:2507.21653v1 §5.1):

> "Following prior work (Dou et al. 2020), we construct a heterogeneous graph with three types of
> edges: reviews written by the same user (R-U-R), reviews on the same product with the same star
> rating (R-S-R), and reviews posted in the same month for the same product (R-T-R). **Instead of
> using the handcrafted features introduced in the original work (Rayana and Akoglu 2015), we
> directly utilize the original texts for LLM-based methods.**"

So: *graph construction recipe* from CARE-GNN (Dou et al. 2020), but applied to the **full**
67,395-review original release rather than the 45,954-review `.mat` subset, and with raw text
replacing the 32 handcrafted features.

**Corroborating evidence (INFERENCE, but very strong):** DGP reports 8,919 frauds out of 67,395
= **13.234%**. Rayana's page states the YelpChi spam rate is **13.23%**. The `YelpChi.mat` subset
has a *different* rate (14.5% / 6,677 frauds). DGP's fraud count therefore matches the original
release, not the `.mat`.

There is no separate "67,395 file". 67,395 is the total for the whole YelpChi release; the
hotel and restaurant reviews are **both** included in that single figure (201 hotels *and*
restaurants). The per-subset hotel/restaurant breakdown from Mukherjee et al. 2013 could **not**
be verified — **UNVERIFIED**; neither shebuti.com nor ODDS publishes the split, and the ICWSM
paper's own numbers were not retrievable in this session.

Why the `.mat` has only 45,954: **UNVERIFIED**. No source found that explains the reduction from
67,395. (Plausibly a filtering step in Dou et al.'s preprocessing, but no evidence was located.)

### 1.5 Raw review texts (d)

Confirmed present in the original release. shebuti.com states the data includes "a plaintext
review" alongside product/user info, timestamp and ratings. The exact file names and column
headers are **UNVERIFIED** (see §1.2).

A third-party repo, <https://github.com/zyni2001/Anomaly-detection-LLM> (fetched), ships
preprocessing scripts for exactly this (`helper.py`, `yelp_preprocess.py`, producing
`UNPRUNED_DATA_prod-ID_usr-ID_rating_label_review.json`, `PRUNED_DATA_...json`,
`rid_mapping.pkl`) and documents the label convention: review_label `1` = non-spam, `-1` = spam,
"13.23% spam reviews in total"; user_label `1` = normal, `-1` = fraud, "20.33% spam users".
**It does not redistribute the raw data** — it expects you to supply `data/YelpChi.zip` yourself.
This is useful as a preprocessing reference, not as a data source.

### 1.6 The three edge types (e) — reconciled

CARE-GNN paper §4.1.2, **verbatim** (extracted from the downloaded PDF):

> "we take reviews as nodes in the graph and design three relations: 1) **R-U-R**: it connects
> reviews posted by the same user; 2) **R-S-R**: it connects reviews under the same product with
> the same star rating (1-5 stars); 3) **R-T-R**: it connects reviews under the same product
> posted in the same month."

DGP Figure 3 labels these "Same-User (RUR)", "Same-Time (RTR)", "Same-Star (RSR)".
**These are consistent with CARE-GNN — no conflict.** The task brief's worry that R-T-R might be
"same star rating & same month" is incorrect: R-T-R is *same product + same month* (no rating
constraint); R-S-R is *same product + same star rating* (no time constraint). DGP's arXiv text
restates both correctly.

Minor note: the CARE-GNN *paper body* writes the Amazon relation as `U-S-V` while its own Table 2
and the DGL implementation both use `U-S-U`. The paper text is a typo.

### 1.7 Edge count 17,486,608 (f) — **UNVERIFIED**

No published source reports this number other than DGP itself. It cannot be checked without the
raw 67,395-review file, which is email-gated. It is not derivable from the `.mat` counts
(DGL's Yelp total is 8,051,348 double-counted edges over 45,954 nodes).

**INFERENCE:** given the Amazon result in §2.6 — where DGP's reported total is exactly
`2 × (sum of undirected pairs over the three relations)` — the Yelp figure is almost certainly
the same double-counted convention, i.e. 8,743,304 undirected edges. This is a prediction to
test once the raw file is obtained, not an established fact.

---

## 2. DATASET 2 — AmazonVideo

### 2.1 What DGP reports

```
AmazonVideo  Product Review  ✓  ✓  37,126  9,883,406  3  4,379  1,299 / 1,299 / 7,425
```

### 2.2 It is NOT the CARE-GNN "Amazon" dataset (a) — confirmed

CARE-GNN's Amazon is **Musical Instruments**, nodes are **users** (11,944, 9.5% fraud), relations
U-P-U / U-S-U / U-V-U, 25 handcrafted features. CARE-GNN §4.1.1 **verbatim**:

> "The Amazon dataset includes product reviews under the **Musical Instruments** category.
> Similar to [47], we label **users** with more than 80% helpful votes as benign entities and
> users with less than 20% helpful votes as fraudulent entities."

CARE-GNN §4.1.2 **verbatim** on relations:

> "we take **users** as nodes in the graph and design three relations: 1) **U-P-U**: it connects
> users reviewing at least one same product; 2) **U-S-V** [sic, = U-S-U]: it connects users having
> at least one same star rating within one week; 3) **U-V-U**: it connects users with top 5%
> mutual review text similarities (measured by TF-IDF) among all users."

DGP's AmazonVideo differs on **every** axis. arXiv:2507.21653v1 §5.1, **verbatim**:

> "Amazon (McAuley and Leskovec 2013) is a product review dataset from the **Amazon Video**
> category. We follow a similar graph construction, where each node is a **review** labeled as
> **helpful or unhelpful**. The graph contains three types of edges: reviews posted by the same
> user (**R-U-R**), reviews posted on the same product (**R-P-R**), and same-product reviews
> posted with the same rating and within the same week (**R-S-R**)."

Nodes are reviews, not users. Relations are R-U-R / R-P-R / R-S-R, not U-P-U / U-S-U / U-V-U.

### 2.3 Canonical McAuley source (b) — identified exactly

The category "Amazon Instant Video" exists **only in the 2014 version** of the McAuley Amazon
data. Verified:

- **2014 version** (<https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html>, fetched):
  has "Amazon Instant Video" — full 583,933 reviews; **5-core 37,126 reviews**.
  Also Musical Instruments: full 500,176; 5-core 10,261. Page total: "142.8 million reviews
  spanning May 1996 - July 2014". Page notes it is the older version, kept "for the sake of
  reproducing past results".
- **2018 version** (<https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/>, fetched): 31
  categories, **no "Amazon Instant Video"**. Video content is folded into "Movies and TV"
  (8,765,568 reviews full; 3,410,019 5-core). URL pattern
  `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/[CATEGORY]_5.json.gz`.
- **2023 version** (<https://amazon-reviews-2023.github.io/>, fetched): 571.54M reviews, 33
  categories, **no "Amazon Instant Video"** — `Movies_and_TV` instead. Hosted on
  `mcauleylab.ucsd.edu`. Includes helpful votes and review text. No explicit license stated.

**Therefore DGP must be using the 2014 release.** No registration needed; downloads are open.

Record schema (2014), confirmed by inspecting the actual file:
`reviewerID`, `asin`, `reviewerName`, `helpful` (= `[helpful_votes, total_votes]`), `reviewText`,
`overall`, `summary`, `unixReviewTime`, `reviewTime`. Both helpful votes and raw text present.

### 2.4 The file, verified by download (d) — **EXACT MATCH**

```
URL      https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Amazon_Instant_Video_5.json.gz
HTTP     200, Content-Type: application/x-gzip
Size     9,517,526 bytes (9.1 MB)
MD5      10812e43e99c345f63333d8ee10aef6a
SHA-256  7816bf30c235c5261560e6ebd9ff6875c154ed15fb9827f0eded74b2c48c4f54
Lines    37,126   <-- exactly DGP's node count
```

(Checksums computed locally this session; **not** published by McAuley — no official checksums
exist for these files.)

The full (non-5-core) file is also live:
`reviews_Amazon_Instant_Video.json.gz`, 95,665,783 bytes, 583,933 reviews.

Derived counts from the 5-core file: 5,130 unique reviewers, 1,685 unique products,
13,133 reviews with at least one helpfulness vote.

> **[CORRECTION by lead engineer, Loop 2]** Section 2.5 below treats the 23,993 zero-vote reviews as
> *unlabeled*. Quantitative analysis of the paper's own AUROC/AUPRC pairs shows they must be **benign**
> (11.8% prevalence fits all 13 Table 2 methods within 1.6 AUPRC points on average; 33.3% misses by 27.6).
> The fraud rule itself (ratio < 0.5 among voted reviews, 4,379 frauds) is unchanged and verified.
> See `evidence_matrix.md`, "Loop 2 addendum".

### 2.5 Labeling rule (c) — **SOLVED, and it differs from CARE-GNN**

CARE-GNN's rule is >80% benign / <20% fraud on *users*. DGP's is different. Tested empirically
against DGP's reported 4,379 frauds:

| rule (among 13,133 reviews with ≥1 vote) | count |
|---|---|
| ratio < 0.1 | 3,091 |
| ratio < 0.2 (CARE-GNN threshold) | 3,283 |
| ratio < 0.3 | 3,662 |
| ratio < 0.4 | 4,158 |
| **ratio < 0.5** | **4,379** ← exact match |
| ratio < 0.6 | 6,239 |
| ratio < 0.8 | 8,002 |

**DGP's AmazonVideo label = `helpful_votes / total_votes < 0.5`**, computed over the 13,133
reviews that have at least one vote. The remaining 8,754 voted reviews are "helpful"; the 23,993
reviews with zero votes are unlabeled. This is a simple majority-helpfulness rule on *reviews*,
not CARE-GNN's 80/20 rule on *users*.

### 2.6 Edge count 9,883,406 — **REPRODUCED EXACTLY**

Reconstructed the three relations directly from `reviews_Amazon_Instant_Video_5.json.gz`:

| relation | undirected pairs | ×2 |
|---|---|---|
| R-U-R — same `reviewerID` | 179,449 | 358,898 |
| R-P-R — same `asin` | 1,649,749 | 3,299,498 |
| R-S-R — same `overall` **+ same week** | 3,112,505 | 6,225,010 |
| **total** | **4,941,703** | **9,883,406** ✅ |

`2 × (179,449 + 1,649,749 + 3,112,505) = 9,883,406` — exactly DGP's reported figure.

Two findings from this:

1. **DGP counts each undirected edge twice** (same convention as DGL). Confirmed.
2. **The paper's stated R-S-R definition is wrong / imprecise.** The arXiv text says
   "**same-product** reviews posted with the same rating and within the same week". Scoping by
   product gives only 48,805 pairs and a total of ~3.7M — nowhere near 9.88M. The exact match
   requires **dropping the product constraint**: same star rating + same week, across all
   products. That is CARE-GNN's `U-S-U` definition ("same star rating within one week") applied
   to review nodes. **Implement the code, not the prose.**

Week bucketing — only one formulation matches exactly:

| week definition | R-S-R pairs | |
|---|---|---|
| `strftime('%Y-%U')` (Sun-start) | 3,077,542 | ✗ |
| `strftime('%Y-%W')` (Mon-start) | 3,088,384 | ✗ |
| ISO calendar week | 3,104,995 | ✗ |
| `unixReviewTime // 604800` (Thu-start) | 3,166,013 | ✗ |
| **`(unixReviewTime + 5*86400) // 604800`** (Sat-start) | **3,112,505** | ✅ |
| equivalently `(date - date(1970,1,3)).days // 7` | **3,112,505** | ✅ |

i.e. weeks are Saturday-aligned. Probably an artifact of a naive epoch-division in the original
code rather than a deliberate choice, but it must be replicated to hit the reported numbers.

---

## 3. SPLIT SIZES

The paper's own explanation (arXiv v1 and AAAI, verbatim):

> "We also note that the sum of the dataset split sizes, including the training, validation, and
> test sets, can be smaller than the total number of nodes. This aligns with a real-world scenario
> in which the majority of nodes are unlabeled, leaving them outside the regular data splits."

Yelp: 1,348 + 1,348 + 13,479 = 16,175 of 67,395.
Amazon: 1,299 + 1,299 + 7,425 = 10,023 of 37,126.

**⚠️ The stated explanation does not hold for Yelp.** All 67,395 YelpChi reviews carry a
Yelp-filter label — there is no unlabeled majority. It does hold for Amazon: only 13,133 reviews
have votes, and 10,023 ≤ 13,133, so the splits fit inside the labeled pool. For Yelp the small
splits appear to be a deliberate low-label-budget design ("we construct training sets with a
limited number of labeled samples"), not a consequence of missing labels.

### Arithmetic — all **INFERENCE**

**Test set = exactly 20% of total nodes, both datasets.** This is exact, not approximate:

- Yelp: 67,395 / 5 = **13,479** exactly ✅
- Amazon: floor(0.20 × 37,126) = floor(7,425.2) = **7,425** ✅
- (also holds for LifeService: floor(0.20 × 12,868) = 2,574 ✅; but **not** for E-Commerce:
  3,928 / 182,043 = 2.16%)

**Train = Val, but the ratio differs per dataset:**

- Yelp: 1,348 = round(0.020 × 67,395) = round(1,347.9) → **2.0%**
- Amazon: 1,299 = floor(0.035 × 37,126) = floor(1,299.41) → **3.5%**
- LifeService: 1,287 = round(0.10 × 12,868) → 10%
- E-Commerce: 1,309 / 182,043 = 0.719% (no round number)

**More likely reading:** train sizes across all four datasets are 1,287 / 1,299 / 1,309 / 1,348 —
all clustered within 5% of ~1,300. This looks like a **fixed labeling budget of roughly 1,300
training samples** per dataset, with the percentages being coincidental consequences. The paper's
framing ("high cost of manual annotation", "limited number of labeled samples") supports this.

No convention involving the fraud counts (8,919 / 4,379) matched any split size.

---

## 4. DOWNLOAD PLAN

### 4.1 AmazonVideo — fully actionable now

```bash
mkdir -p data/amazon_video && cd data/amazon_video

# 37,126 reviews, 9.1 MB, no registration
curl -L -O https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Amazon_Instant_Video_5.json.gz

# integrity check (checksums computed by this study; NOT published upstream)
sha256sum reviews_Amazon_Instant_Video_5.json.gz
# expect 7816bf30c235c5261560e6ebd9ff6875c154ed15fb9827f0eded74b2c48c4f54
md5sum reviews_Amazon_Instant_Video_5.json.gz
# expect 10812e43e99c345f63333d8ee10aef6a

gzip -dc reviews_Amazon_Instant_Video_5.json.gz | wc -l   # expect 37126
```

Reconstruction recipe (verified to reproduce Table 1 exactly):

```python
import gzip, json
from collections import defaultdict

rows = [json.loads(l) for l in
        gzip.open('reviews_Amazon_Instant_Video_5.json.gz', 'rt', encoding='utf-8')]
assert len(rows) == 37126

# labels: unhelpful (fraud) = helpful ratio < 0.5, only for reviews with >=1 vote
label = {}
for i, r in enumerate(rows):
    h, t = r['helpful']
    if t > 0:
        label[i] = 1 if h / t < 0.5 else 0
assert sum(label.values()) == 4379          # DGP's # Frauds
assert len(label) == 13133                  # labeled pool

# edges: 2 * sum of undirected pairs over three relations
WEEK = 7 * 86400
def groups(keyfn):
    g = defaultdict(list)
    for i, r in enumerate(rows):
        g[keyfn(r)].append(i)
    return g
def npairs(g):
    return sum(len(v) * (len(v) - 1) // 2 for v in g.values())

rur = npairs(groups(lambda r: r['reviewerID']))                      # 179,449
rpr = npairs(groups(lambda r: r['asin']))                            # 1,649,749
rsr = npairs(groups(lambda r: (r['overall'],
      (r['unixReviewTime'] + 5 * 86400) // WEEK)))                   # 3,112,505
assert 2 * (rur + rpr + rsr) == 9883406                              # DGP's # Edges

# splits (INFERENCE — not stated in the paper)
# test  = floor(0.20 * 37126)  = 7425
# train = val = floor(0.035 * 37126) = 1299, drawn from the 13,133 labeled reviews
```

Node text for the LLM = `reviewText` (optionally with `summary`). Numerical fields available for
DGP's numerical-summarization branch: `overall` (1–5), `helpful[0]`, `helpful[1]`,
`unixReviewTime`.

### 4.2 YelpReviews — requires an email request

**Step 1 (blocking, do this first — expect days of latency):**

Email `srayana@cs.stonybrook.edu` requesting the YelpChi dataset with ground truth, per the
instruction on <https://shebuti.com/yelpchi-dataset/>. State academic/reproduction use and that
you need the **raw review text** (`reviewContent`), not just metadata. Expect 67,395 reviews,
201 businesses, 38,063 reviewers, 13.23% spam.

Also try the ODDS mirror <https://odds.cs.stonybrook.edu/yelpchi-dataset/> from a browser — it
was unreachable from this environment due to a TLS handshake failure, but may work elsewhere and
could carry a direct link.

**Step 2 (do in parallel — unblocks all GNN baselines immediately):**

```bash
mkdir -p data/yelpchi_mat && cd data/yelpchi_mat

# YelpChi.mat + Amazon.mat, 45,954 / 11,944 nodes, 32 / 25 features, NO raw text
curl -L -O https://data.dgl.ai/dataset/FraudYelp.zip      # 17,980,652 bytes
curl -L -O https://data.dgl.ai/dataset/FraudAmazon.zip    # 26,122,297 bytes

# or via CARE-GNN (both verified HTTP 200)
curl -L -O https://raw.githubusercontent.com/YingtongDou/CARE-GNN/master/data/YelpChi.zip
curl -L -O https://raw.githubusercontent.com/YingtongDou/CARE-GNN/master/data/Amazon.zip
```

Or in one line via DGL:

```python
from dgl.data import FraudYelpDataset, FraudAmazonDataset
g = FraudYelpDataset()[0]   # 45,954 nodes; net_rur / net_rtr / net_rsr
```

No checksums are published for any of these archives.

**Step 3 — graph construction once the raw file arrives** (recipe from arXiv:2507.21653v1 §5.1,
following CARE-GNN):

- R-U-R: reviews by the same user
- R-S-R: reviews on the same product with the same star rating
- R-T-R: reviews on the same product posted in the same month
- Use `reviewContent` as node text; **do not** use the 32 handcrafted features
- Count edges in both directions (see §1.7) and check against 17,486,608
- Labels: Yelp-filter spam flag; expect 8,919 frauds (13.23%)
- Splits (INFERENCE): test = 67,395/5 = 13,479; train = val = round(0.02 × 67,395) = 1,348

**Step 4 — preprocessing reference:** <https://github.com/zyni2001/Anomaly-detection-LLM>
(`Data/helper.py`, `Data/yelp_preprocess.py`). Expects `data/YelpChi.zip` and `data/Amazon.zip`;
does not redistribute data.

### 4.3 Reference summary

| | download | size | registration | license | checksums |
|---|---|---|---|---|---|
| YelpChi original (67,395, w/ text) | email `srayana@cs.stonybrook.edu` | unknown | **yes, by email** | none stated | none |
| YelpChi.mat (45,954, no text) | `data.dgl.ai/dataset/FraudYelp.zip` | 17.98 MB | no | none stated | none |
| Amazon.mat (11,944, Musical Instruments) | `data.dgl.ai/dataset/FraudAmazon.zip` | 26.12 MB | no | none stated | none |
| Amazon Instant Video 5-core (37,126) | SNAP URL in §4.1 | 9.52 MB | no | none stated | none published (ours in §2.4) |
| Amazon Instant Video full (583,933) | SNAP, same dir | 95.67 MB | no | none stated | none |

---

## 5. UNRESOLVED

1. **YelpChi raw data not in hand.** Email-gated; no direct URL exists. This is the single
   blocking dependency for reproducing YelpReviews. Everything downstream of it is unverified.
2. **YelpChi file names and column headers — UNVERIFIED.** Neither shebuti.com nor ODDS publishes
   a file manifest. Whether the text column is literally named `reviewContent`, and whether the
   release ships as `metadata` + `reviewContent` text files, could not be confirmed.
3. **Edge count 17,486,608 — UNVERIFIED.** Untestable without the raw file. Predicted to be
   `2 × 8,743,304` undirected pairs by analogy with the Amazon result (§1.7).
4. **Why `YelpChi.mat` has 45,954 of 67,395 reviews — UNVERIFIED.** No source located that
   documents the filtering step.
5. **YelpChi hotel vs restaurant per-subset counts — UNVERIFIED.** Not published on either
   dataset page; the Mukherjee et al. ICWSM 2013 numbers were not retrievable in this session.
6. **ODDS page never fetched.** Persistent `SSLV3_ALERT_HANDSHAKE_FAILURE` from this environment.
   Retry from a browser; it may host a direct download link that would remove blocker #1.
7. **Whether DGP's Yelp R-S-R also silently drops the product constraint.** On Amazon, the exact
   edge count required ignoring the product scope that the prose asserts (§2.6). The same
   discrepancy may exist on Yelp. Test both variants against 17,486,608 once the data arrives.
8. **Yelp week/month bucketing convention — UNVERIFIED.** Amazon needed a Saturday-aligned epoch
   week; Yelp's R-T-R "same month" may have an analogous quirk.
9. **Exact split construction — INFERENCE only.** The paper gives no ratios. The 20%-test rule is
   an exact arithmetic match on both public datasets, but stratification, seeding, and whether
   the pool is all nodes or only labeled nodes are all unspecified. Five seeds are used; the seed
   values are not given.
10. **Official code unavailable.** <https://github.com/Xtra-Computing/DGP> is a README-only
    placeholder pending industry-partner approval. Worth re-checking periodically.
11. **Node feature set for AmazonVideo numerical branch — UNVERIFIED.** DGP marks AmazonVideo as
    having numerical features but never enumerates them. `overall`, `helpful[0]`, `helpful[1]`,
    `unixReviewTime` are the only candidates in the source file; degree-based features may also
    be constructed.

---

## 6. SOURCES FETCHED

- <https://shebuti.com/yelpchi-dataset/>
- <https://shebuti.com/yelpnyc-dataset/>
- <https://docs.dgl.ai/generated/dgl.data.FraudYelpDataset.html> (via `dgl/python/dgl/data/fraud.py` raw source)
- <https://github.com/YingtongDou/CARE-GNN> + arXiv:2008.08692 PDF (text-extracted locally)
- <https://github.com/Xtra-Computing/DGP>
- <https://github.com/zyni2001/Anomaly-detection-LLM>
- <https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html>
- <https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/>
- <https://amazon-reviews-2023.github.io/>
- <https://arxiv.org/abs/2507.21653> and the v1 PDF (text-extracted locally)
- <https://ojs.aaai.org/index.php/AAAI/article/download/38541/42503> (DGP camera-ready)
- <https://ojs.aaai.org/index.php/ICWSM/article/view/14389> (Mukherjee et al. 2013, metadata only)
- `reviews_Amazon_Instant_Video_5.json.gz` — downloaded and analyzed locally
- Failed: <https://odds.cs.stonybrook.edu/yelpchi-dataset/> (TLS), `PonderLY/PC-GNN` data zips (404)
