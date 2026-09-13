# Markov Diffusion Kernel (MDK) — Research Notes

**Author:** lead engineer (the MDK subagent terminated on an API spend limit before writing; this was researched and verified directly).
**Status:** Implementation decision RESOLVED for ranking purposes. Two secondary ambiguities remain OPEN and are exposed as config switches.

---

## 1. What DGP specifies (both versions agree on Eq. 6)

Camera-ready and arXiv v1 print the **same** operator (verified in both text files):

```
T_P    = D_P^{-1} A_P ,      D_P = diag(A_P 1)                 (row-stochastic)
Z_P(K) = (1/K) * SUM_{k=0}^{K} T_P^k                            (Eq. 6)
h_i    = [Z_P(K) X]_i                                           (Eq. 7)
delta  = || h_u - h_v ||_2                                      (Eq. 8)
Ñ_P(v) = TopM_{u in N_P(v)} ( -delta(u,v) )                     (Eq. 9)
```

Because both versions carry the identical `k=0 ... K` with `1/K`, this is **not a one-off typesetting slip** in the camera-ready.

Where the versions **differ** is the input `X` (see `version_divergence.md` §2.1):

| Version | `X` |
|---|---|
| arXiv v1 | "raw node features", `X ∈ R^{n×d}` |
| camera-ready | `emb_DeBERTa(X^text) ⊕ X^num ∈ R^{N×(d_LM+d)}` |

## 2. Reference operator 1 — SSGC (Zhu & Koniusz, ICLR 2021) — VERIFIED FROM OFFICIAL CODE

Repository: `https://github.com/allenhaozhu/SSGC` ("Implementation for Simple Spectral Graph Convolution in ICLR 2021"), default branch `main`.
File: `utils.py`, function `sgc_precompute`, verbatim:

```python
def sgc_precompute(features, adj, degree, alpha):
    t = perf_counter()
    ori_features = features
    emb = alpha * features
    for i in range(degree):
        features = torch.spmm(adj, features)
        emb = emb + (1-alpha)*features/degree
    precompute_time = perf_counter()-t
    return emb, precompute_time
```

In closed form (with `K = degree`):

```
emb = alpha * X  +  ((1 - alpha) / K) * SUM_{k=1}^{K} T^k X
```

- The power sum runs **k = 1 … K** and is divided by **K** — a proper average of the `K` random-walk powers.
- The **identity (k=0) contribution is kept separately**, weighted by `alpha`.
- `adj` in SSGC is the symmetric-normalised adjacency; DGP instead specifies the **row-stochastic** `D^{-1} A`. This is a stated DGP choice, not an ambiguity.

## 3. Reference operator 2 — Fouss et al. 2012 (Markov diffusion kernel)

The classical Markov diffusion kernel is usually written as `Z(t) = (1/t) SUM_{τ=1}^{t} P^τ`, with kernel `K_MD = Z(t) Z(t)^T`.

> **UNVERIFIED_SECONDARY.** The Fouss et al. 2012 paper (Neural Networks) is paywalled and was not read directly in this session. The statement above is standard in the secondary literature but has **not** been checked against the original text. It is used here only for context; no implementation decision depends on it. The SSGC operator in §2 (verified from code) carries the same `k=1…K, /K` averaging.

Note DGP does **not** use the kernel `Z Z^T`. It uses the diffusion *operator* `Z` to propagate features, then takes L2 distances — which is exactly the SSGC style of use, not the kernel-matrix style. This fits DGP citing SSGC.

## 4. Reconciling DGP's Eq. 6

Expand DGP's operator:

```
Z_P(K) X = (1/K) * X  +  (1/K) * SUM_{k=1}^{K} T^k X
```

Compare SSGC:

```
emb      = alpha * X  +  ((1-alpha)/K) * SUM_{k=1}^{K} T^k X
```

DGP's Eq. 6 has **exactly SSGC's structure**: an average of `K` random-walk powers plus a retained identity term. It corresponds to an identity weight of `1/K` on the same footing as each walk power, without SSGC's `(1-alpha)` rescaling.

**Verdict:**

1. **The `k = 0` term is intentional and meaningful.** It keeps each node's own features in its diffused representation, matching both SSGC (§2) and the group's own diffusion code in `CrawlScript/Torch-MGDCF` (`torch_mgdcf/layers/mgdcf.py`: `h = h * self.beta + h0 * self.alpha`, retaining `h0` at every step). Dropping it would silently change which neighbours are selected.
2. **The `1/K` vs `1/(K+1)` question is irrelevant to the method's output.** A positive scalar multiplies every `h_i` by the same constant, which scales every distance `delta(u,v)` by the same constant. **The Top-M ordering in Eq. 9 is therefore invariant to it.** It would matter only if the diffused embeddings were reused as model inputs, which DGP does not do.

**Implementation decision (`PAPER_RECONSTRUCTION`, high confidence):**

- Default `mdk.operator: paper_eq6` → literal `(1/K) * SUM_{k=0}^{K} T^k`.
- Variant `mdk.operator: walk_only` → `(1/K) * SUM_{k=1}^{K} T^k` (the classical form without the identity) — **ablation only**, clearly labelled.
- A unit test asserts that `paper_eq6` and a `1/(K+1)`-normalised version produce **identical** Top-M sets. This turns the invariance argument into a checked property instead of an assumption.

## 5. Efficient computation — never materialise `Z` or `T^k`

`Z_P(K)` is `N × N`. For YelpReviews `N = 67,395`:

- dense `float32` `N × N` = 67,395² × 4 bytes ≈ **18.2 GB for one metapath** (≈36 GB at float64). Powers of `T` fill in rapidly, so sparsity is lost after a hop or two on these graphs.
- There are several metapaths (3 relations; up to `(3^{K+1}-3)/2` = 39 metapaths at `K=3`).

Only `Z_P(K) X` (size `N × d`) is ever needed. Horner-free iterative form:

```
cur = X                      # k = 0 term
acc = X.copy()
for k in 1..K:
    cur = T_P @ cur          # sparse @ dense, O(nnz(T_P) * d)
    acc += cur
H = acc / K
```

Memory is `O(N·d)` plus the sparse `T_P`. With DeBERTa-v3-base (`d_LM = 768`), `N·d ≈ 67,395 × 769 × 4 B ≈ 207 MB` per metapath at float32.

### Composite metapath adjacency

`A_P = A_{r1} A_{r2} … A_{rL}` must be formed as a **sparse** product. But the product itself can be dense for the public graphs: Yelp's same-product relations make R-S-R and R-T-R contain large cliques (FraudCoT and CARE-GNN report millions of edges), and `A_{RSR} A_{RTR}` compositions approach dense. Mitigation, in order of preference:

1. **Never form `A_P` for propagation (weighted/literal variant — exact).** Since `A_P = A_{r1} … A_{rL}` is a product, both pieces of `T_P x = D_P^{-1} A_P x` are matrix-vector chains:
   - `A_P x = A_{r1}(A_{r2}( … (A_{rL} x)))`
   - `diag(D_P) = A_P 1 = A_{r1}(A_{r2}( … (A_{rL} 1)))`

   So `T_P x = (A_{r1}(…(A_{rL} x))) / (A_{r1}(…(A_{rL} 1)))`, computed row-wise **exactly**, with memory `O(nnz(A_r) + N·d)`. *(Correction: an earlier draft of this note wrongly said this factorisation was inexact. It is exact for the literal, weighted `A_P`; what does **not** factorise is a normalisation of each single relation separately, which we never use.)*
2. **Binary variant requires materialising `A_P`.** Re-binarising the product (`A_P > 0`) cannot be expressed as a matvec chain, so `binary` must build `A_P` sparsely. Measure `nnz` and fail loudly above a configurable cap rather than silently truncating.
3. Neighbourhood membership `N_P(v)` for Top-M is only needed for **target nodes** (train/val/test, ~16k on Yelp), not for all `N` — compute the support of those rows only, in chunks (`A_{r1}[targets] @ A_{r2} @ …`).

Exact memory for the composite paths depends on the real graph and will be **measured** in `build_mdk.py` and recorded in the run manifest rather than estimated here.

## 6. Open ambiguities (exposed as config, never silently chosen)

| # | Ambiguity | Options | Consequence | Default |
|---|---|---|---|---|
| M1 | Is `A_P` re-binarised before `T_P`? Eq. 3 is a matrix product (entries are **path counts**); Eq. 4 thresholds `>0` only for set membership. | `weighted` (use counts) / `binary` | Weighted favours neighbours reachable by many paths; binary treats all path-neighbours equally | **`weighted`** — Eq. 3 *defines* `A_P` as the product and Eq. 6 builds `T_P` from `A_P` with no binarisation step; Eq. 4's `>0` is used only for set membership. This literal reading is also the only one computable without materialising `A_P` (§5). Medium confidence; `binary` is a one-flag switch for small graphs |
| M2 | Scaling before concatenating DeBERTa embeddings with numeric features | `none` / `standardize` (per-column z-score) / `block_l2` | L2 distance is dominated by the larger-magnitude block; ratings 1–5 vs embeddings ≈ unit scale | **`none`** (literal paper). Recorded as a sensitivity check, not a silent fix |
| M3 | DeBERTa pooling | `mean` / `cls` | Different embeddings → different neighbours | **`mean`** over non-padding tokens. DeBERTaV3 is not trained with a CLS-sentence objective. Low confidence |
| M4 | Zero-degree rows in `A_P` | zero row / self-loop | Isolated node: `N_P(v)` is empty anyway | zero row (`D^{-1}` treated as 0); such nodes get no neighbours for `P` |
| M5 | `|N_P(v)| < M` | keep all / pad | — | keep all available neighbours |
| M6 | Does `N_P(v)` exclude `v` itself? Composite paths like R-U-R return to `v` | exclude / include | Including `v` puts the target into its own neighbour summary | **exclude** — the Figure 3 arithmetic (`r_mean = 1.7`) averages only v1–v3, not v0 (see `paper_specification.md` §3.4) |
| M7 | `X` input: raw features (arXiv) vs DeBERTa ⊕ numeric (camera-ready) | `raw` / `deberta_concat` | Different neighbours | **`deberta_concat`** (camera-ready wins) |

## 7. Sources verified in this session

- `https://github.com/allenhaozhu/SSGC` → `utils.py` `sgc_precompute` (raw file fetched)
- `https://github.com/CrawlScript/Torch-MGDCF` → `torch_mgdcf/layers/mgdcf.py` (raw file fetched)
- `research/paper_fulltext.txt` and `research/arxiv_v1_fulltext.txt` (Eq. 6 identical in both)
- Fouss et al. 2012 — **not read** (paywalled); marked UNVERIFIED_SECONDARY
