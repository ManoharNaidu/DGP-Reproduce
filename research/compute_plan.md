# Compute Plan and Cost Estimate

Tags: **MEASURED** (from our real AmazonVideo runs) · **DERIVED** (arithmetic on verified numbers) · **ESTIMATE** (assumption; could be off by ~2×).

> **Update (2026-09-17): the DGP main result (5 seeds, AmazonVideo) has been run for real** on 2× A100 80GB.
> §4's ESTIMATE row for "DGP main, 5 seeds" is now replaced with MEASURED figures. Ablations, the budget sweep,
> the task-aware study and the hyperparameter grid search have not been run — their rows below are still
> ESTIMATE/DERIVED from the same measured per-seed and per-summary costs.
>
> **First step on any GPU: run the calibration job (§5).** It measures real summarization and training throughput in under an hour and replaces every ESTIMATE below with a measurement before real money is committed.

## 1. What needs a GPU, and why

| Stage | Needs GPU? | Why |
|---|---|---|
| Graph building, splits, metapaths, MDK diffusion + Top-M | **No** | Sparse linear algebra. MEASURED: all 12 AmazonVideo metapaths at K=2 in ~1 min on a 12-thread laptop CPU |
| DeBERTaV3-base embeddings (37k reviews) | No (faster on GPU) | MEASURED: ~2.5 h on the laptop CPU in float32; done and cached for AmazonVideo (copy `datasets/cache/embeddings/` and `datasets/cache/mdk/` to the GPU machine to skip it) |
| **Node + metapath summaries (frozen Qwen3-8B)** | **Yes** | 8.19B-parameter generation over ~150k prompts per configuration |
| **LoRA fine-tuning + inference (Qwen3-8B)** | **Yes** | 8B forward/backward passes |
| Metrics, tables, figures, complexity analysis | No | — |
| MLP baseline | No | MEASURED: 5 seeds incl. grid search in 3 min on CPU |
| ConsisGAD baseline | Strongly preferred | MEASURED: ~4 min/epoch on CPU, 100 epochs × 5 seeds ≈ 35 h |

Development happened on a machine with **no CUDA GPU** (AMD integrated graphics), so the two Qwen3-8B stages
ran on a rented **2× A100-SXM4-80GB** cloud instance instead (§4 update below).

## 2. Memory

| Item | Value | Tag |
|---|---|---|
| Qwen3-8B parameters | 8,190,735,360 | verified (HF API) |
| bf16 weights | 15.26 GiB | DERIVED |
| bf16 inference (summarizer) | ~18–20 GiB | ESTIMATE (`llm_provenance.md` §5.4) |
| bf16 LoRA training + gradient checkpointing, short prompts | ~20–26 GiB | ESTIMATE |
| Disk: Qwen3-8B | 16.4 GB | verified |
| Disk: DeBERTaV3-base + caches + checkpoints | ~5 GB | ESTIMATE |

The pipeline releases the frozen summarizer before loading the classifier (the two are never resident together).

**Evidence that one GPU suffices:** FraudCoT (arXiv:2601.22949, the same five authors, identical Qwen3-8B LoRA grid) ran on **one A100 80GB**. DGP's 4×A100 is throughput, not a memory requirement. **MEASURED**: real LoRA training runs (gradient checkpointing, bf16, batch 4) peaked at **19.68–19.69 GiB**, confirming the ~20–26 GiB ESTIMATE above and leaving huge headroom on an 80 GB card (`run_manifest.json → peak_memory.gpu_peak_gb`, 3 of 5 seeds; the other 2 ran on `cuda:1` where the peak-memory query itself reads the wrong device — a tooling bug, not evidence memory was actually near zero).

**Recommended GPU:** 1× **A100 80GB** (comfortable headroom, and matches the authors' hardware class). A 48 GB card (A6000/L40S) should also work for the canonical bf16 path; 24 GB cards need batch 1 or the flagged `memory_optimized` (4-bit) mode, which would not be canonical.

## 3. Workload for one AmazonVideo configuration (K=2, M=4, B=10)

| Quantity | Value | Tag |
|---|---|---|
| Target nodes (train+val+test) | 10,023 | MEASURED |
| Metapaths | 12 | DERIVED |
| Node summaries needed | 33,410 | MEASURED |
| Metapath summaries needed | 119,216 | MEASURED |
| Mean review length L | 115.7 Qwen3 tokens | MEASURED |
| DGP final prompt (paper formula at this L) | ~236 tokens | DERIVED |
| Training examples / epoch | 1,299 × up to 10 epochs, batch 4 | PAPER |

YelpReviews scales these by ≈1.6× (16,175 targets, 67,395 nodes; test set 13,479).

## 4. Runtime and cost

**MEASURED** throughput, replacing the assumptions this section used before any GPU run: on one A100-80GB
(bf16, batch 64), Qwen3-8B `generate` for node/metapath summaries (greedy, ≤56 new tokens at B=10) sustained
**~10.5 summaries/s** once warm — about 3× slower than the original 30/s guess. One trained seed (LoRA
fine-tune, gradient checkpointing, batch 4, up to 10 epochs with early stopping, plus val + test inference)
took **35–80 minutes** wall-clock (2,263–4,757 s measured across the 5 AmazonVideo seeds, stopping at epoch
1–3 of 10) — close to the original ~30 min guess. AmazonVideo needed 155,576 node + metapath summary requests
in total (35,747 + 119,829); generating them from a cold cache costs **~4.1 GPU-hours** on one A100, not the
~1.5 h originally assumed.

| Experiment (per dataset) | New summaries | Training runs | GPU-hours AmazonVideo | GPU-hours YelpReviews |
|---|---|---|---|---|
| **DGP main, 5 seeds** | 1 set (**MEASURED ~4.1 h**) | 5 | **MEASURED ~8.4 h** (4.1 h summaries + 5×~0.86 h train/eval, run partly in parallel on 2 GPUs → ~6.3 h wall-clock) | ~13 (ESTIMATE, scaled ×1.6 for YelpReviews' larger graph) |
| LLM target-only baseline, 5 seeds | none | 5 | ~4.3 (5×~0.86 h, MEASURED per-seed cost) | ~7 |
| Ablations (4 variants × 5 seeds) | w/o MDK needs 1 set | 20 | ~21 (ESTIMATE, scaled from measured per-seed cost) | ~34 |
| Budget B ∈ {5,20,40,80} (B=10 reused) × 5 seeds | 4 sets | 20 | ~33 (ESTIMATE) | ~53 |
| Task-aware, 5 seeds | 1 set | 5 | ~8.4 (ESTIMATE, same shape as main) | ~13 |
| **Subtotal, paper experiments** | | | **~75** | **~120** |

Revised upward from the original ~41/~61 h: real summary generation and training both ran slower than the
pre-GPU guess (3× and ~1.7× respectively). **Both public datasets, no hyperparameter search: ~195 GPU-hours ≈
$130–$400** at the listed single-A100-80GB rates ($0.67/h Vast.ai marketplace low · $1.49/h RunPod · $2.06/h
Lambda). Budget ~1.5× for failed runs and reruns: **~$195–$600**. Running independent seeds two-at-a-time on a
2-GPU box (as the main result did) roughly halves wall-clock time without changing GPU-hour cost.

API cost: **none**. Both models are open weights and not gated; no Hugging Face token is required.

## 5. Calibration job (do this first, ~1 GPU-hour, ~$1–2)

Already done for AmazonVideo (§4's MEASURED figures came from the real main run, which supersedes a separate
calibration pass). Still the right first step before spending GPU-hours on **YelpReviews** once its data
arrives, since its ~1.6× larger graph may shift throughput.

```bash
python scripts/prepare_data.py --dataset amazonvideo
python scripts/build_mdk.py --dataset amazonvideo --device cuda:0            # or copy the CPU-built cache
python scripts/generate_summaries.py --dataset amazonvideo --device cuda:0 --until node_summaries \
       --set summarizer.batch_size=64
python scripts/train.py --dataset amazonvideo --device cuda:0 --seeds 0 --set training.max_epochs=1
```

Summaries/second and seconds/epoch are then read from the logs and `results/raw/*/run_manifest.json`, and §4 is recomputed from measurements.

## 6. The hyperparameter grid: a decision for the project owner

The paper tuned on a published grid (arXiv v1 §5.1), selecting by mean validation AUROC, but **never reports the selected values**. The full DGP grid is

`B (4) × K (3) × M (4) × LoRA rank (4) × dropout (3) × lr (3) = 1,728 configurations per dataset`,

and every change of B, K or M requires new summaries. At ~0.5 GPU-h per training run plus summary generation, that is **>1,000 GPU-hours per dataset (>$1,500–$4,000)**, far beyond the rest of the study combined.

Options:

| Option | What runs | Extra GPU-h (both datasets) | Fidelity |
|---|---|---|---|
| **A. Staged search (recommended)** | Stage 1: K × M = 12 prompt configs at B=10 with mid LoRA settings, 1 seed. Stage 2: rank × lr = 12 LoRA configs at the best K, M, dropout fixed at 0.05, 1 seed. Winner → 5 seeds. B is covered by the Figure 5 study | ~60–90 | Searches the paper's own grid along its most important axes; documented as a reduction |
| B. Full grid | All 1,728 configs × 2 datasets | >2,000 | Exactly the paper's protocol; cost prohibitive |
| C. No search | Reconstructed defaults (K=2, M=4, B=10, r=8, dropout=0.05, lr=1e-4) | 0 | Cheapest; results depend on unverified defaults and are harder to interpret against the paper |

No grid choice is hard-coded until the owner decides.
