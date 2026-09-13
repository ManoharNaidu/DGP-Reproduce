# Compute Plan and Cost Estimate

Tags: **MEASURED** (from our real AmazonVideo runs) · **DERIVED** (arithmetic on verified numbers) · **ESTIMATE** (assumption; could be off by ~2×).

> **First step on any GPU: run the calibration job (§5).** It measures real summarization and training throughput in under an hour and replaces every ESTIMATE below with a measurement before real money is committed.

## 1. What needs a GPU, and why

| Stage | Needs GPU? | Why |
|---|---|---|
| Graph building, splits, metapaths, MDK diffusion + Top-M | **No** | Sparse linear algebra. MEASURED: all 12 AmazonVideo metapaths at K=2 in ~1 min on a 12-thread laptop CPU |
| DeBERTaV3-base embeddings (37k reviews) | No (faster on GPU) | 184M-parameter encoder; running now on the local CPU |
| **Node + metapath summaries (frozen Qwen3-8B)** | **Yes** | 8.19B-parameter generation over ~150k prompts per configuration |
| **LoRA fine-tuning + inference (Qwen3-8B)** | **Yes** | 8B forward/backward passes |
| Metrics, tables, figures, complexity analysis | No | — |

This machine has **no CUDA GPU** (AMD integrated graphics), so the two Qwen3-8B stages must run in the cloud.

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

**Evidence that one GPU suffices:** FraudCoT (arXiv:2601.22949, the same five authors, identical Qwen3-8B LoRA grid) ran on **one A100 80GB**. DGP's 4×A100 is throughput, not a memory requirement.

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

## 4. Runtime and cost (ESTIMATE)

Throughput assumptions, to be replaced by the calibration run: HF `generate` on A100-80GB at ~30 summaries/s; LoRA training at ~2–3 min per epoch; ~30 min per trained seed including validation and test inference.

| Experiment (per dataset) | New summaries | Training runs | GPU-hours AmazonVideo | GPU-hours YelpReviews |
|---|---|---|---|---|
| DGP main, 5 seeds | 1 set (~1.5 h) | 5 | ~4 | ~6 |
| LLM target-only baseline, 5 seeds | none | 5 | ~2 | ~3 |
| Ablations (4 variants × 5 seeds) | w/o MDK needs 1 set | 20 | ~12 | ~18 |
| Budget B ∈ {5,20,40,80} (B=10 reused) × 5 seeds | 4 sets | 20 | ~19 | ~28 |
| Task-aware, 5 seeds | 1 set | 5 | ~4 | ~6 |
| **Subtotal, paper experiments** | | | **~41** | **~61** |

**Both public datasets, no hyperparameter search: ~100 GPU-hours ≈ $70–$210** at the listed single-A100-80GB rates ($0.67/h Vast.ai marketplace low · $1.49/h RunPod · $2.06/h Lambda). Budget ~1.5× for failed runs and reruns: **~$100–$320**.

API cost: **none**. Both models are open weights and not gated; no Hugging Face token is required.

## 5. Calibration job (do this first, ~1 GPU-hour, ~$1–2)

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
