# Markov diffusion distance: DGP vs FLAG-MD

Verified by `analysis/mdk_comparison.py` (results in `analysis/out/mdk_comparison.json`).
Code read: `src/dgp_repro/mdk/{adjacency,diffusion,distance,trimming}.py` and
`FLAG-Reproduce/src/flagbench/sampling/markov_diffusion.py`.

## The computation (identical in both)

1. Row-stochastic transition matrix `T = D^-1 A`, `D = diag(A 1)`; zero-degree rows stay all-zero.
2. Diffusion operator `Z(K) = (1/K) * sum_{k=0..K} T^k` (DGP Eq. 6 as printed: K+1 terms, divided by K).
3. Diffused embeddings `H = Z(K) X`, computed as K sparse-times-dense products; `Z` and `T^k` are never formed.
4. Distance `delta(w, u) = ||H_w - H_u||_2`; smaller means more relevant. Ties break on lower node id.

The 1/K factor is a constant, so it cannot change a ranking (`mean_k0`, with 1/(K+1), selects identical sets).
Dropping the identity term (`walk_only`) does change the ranking.

## What differs

| | DGP | FLAG-MD |
|---|---|---|
| Graph `A` | metapath adjacency `A_P = A_r1 ... A_rL`; entries are **path counts** (`adjacency=weighted`) | one **binary** adjacency, self-loops removed, duplicate edges collapsed |
| Applied how | `T x = D^-1 (A_r1 (A_r2 (... x)))`, degree from `A_P 1` | `T = D^-1 A` as a CSR matrix |
| Features `X` | DeBERTa-v3-base mean-pooled (768) concatenated with rating, unscaled | L2-normalised MiniLM Sentence-BERT (384), the same matrix the cosine sampler compares |
| Candidates | the metapath neighbourhood `N_P(v)` (self excluded) | the direct neighbours of each frontier node during 2-hop expansion |
| How many are kept | Top-`M` (default 4) per metapath | `matched_cosine`: as many as cosine sampling would keep; or `top_n` |
| Role of `K` | sets metapath length (<=K) **and** diffusion steps | diffusion steps only (default 2) |
| Computed when | once per metapath, for all nodes | once per graph |

## Verification results

- DGP iterative `H` and FLAG `H` both match a dense `NumPy` reference to ~1e-16 for K = 1, 2, 3.
- On a single-relation graph with identical `X`, DGP and FLAG give the same `H`, the same distances and
  the same top-4 sets for every node.
- On a two-relation metapath, count-weighted and binarised `A_P` differ (max |dH| = 0.175 on a random test
  graph). Binarising changed the top-4 set for 9 of 38 targets that had more than 4 candidates. DGP defaults to weighted.
- `walk_only` changed the top-4 set for 13 of 40 nodes against `paper_eq6`. `mean_k0` changed none.

## Implication for the experiments

The two implementations are the same operator. Differences in behaviour come from the **graph** (metapath
product vs single relation), the **features**, and the **candidate set**, not from the diffusion formula.
