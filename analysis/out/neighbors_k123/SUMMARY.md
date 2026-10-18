# Neighbours at K = 1, 2, 3 after Markov diffusion (AmazonVideo, Top-4 per metapath)

Scripts: `analysis/neighbors_k123/{common,01_build,02_analyze}.py` (CPU, parallel). Outputs here: `phase0_verify.json`,
`per_metapath.csv`, `neighbors_long.csv.gz` (every kept neighbour of every target, all K), `neighbors_k123_summary.json`, `work/`.

- **Integrity:** 54 caches (3 + 12 + 39 metapaths) recomputed live on a 600-target audit sample each: 32,400 Top-4 selections, 0 mismatches.
- **Counts:** kept per target 11.9 / 47.7 / 155.4, unique 11.8 / 32.6 / 51.0 (repeats 1.1% / 31.7% / 67.2%) for K = 1 / 2 / 3.
- **1-hop paths:** RUR, RPR, RSR are unions of disjoint cliques (100% of nodes). Diffusion distance = c_K(n) x raw distance
  (max relative error 7.6e-15), so the diffusion Top-4 equals the raw-distance Top-4 for 100% of audited targets at every K.
- **Composite paths:** diffusion Top-4 equals raw Top-4 for 47% (2-hop) and 35% (3-hop) of audited selections (Jaccard .71 / .65);
  against text cosine .56 / .48.
- **Union growth:** +20.8 new neighbours K=1 to 2, +19.1 K=2 to 3; K=1 set is inside K=2 for 100% of targets, K=2 inside K=3 for 53%.
- **Label signal (train+val targets):** neighbour fraud-rate gap 0.118 / 0.097 / 0.095; on the 400-target audit sample diffusion and raw selection keep
  ~0.12 on composite paths, text cosine ~0.07, random-from-candidates 0.05 (2-hop) and 0.03 (3-hop).
- **Distance:** rating is 2.1% of the squared distance (text embedding mean norm 27.3, rating std 1.1).
Labels are read only for train and validation targets; test targets contribute structure, distance, rating and text statistics only.
