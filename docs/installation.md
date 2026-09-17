# Installation

Two supported set-ups. Use a **fresh, isolated** environment: borrowing a system Python's packages broke
this project once (an old system `torch` imported a newer system `transformers` and crashed).

## CPU (development, smoke tests, graph work)

Works on Windows, macOS and Linux. No GPU needed.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]" "transformers>=4.56" sentencepiece protobuf
python scripts/smoke_test.py                    # 14 checks, ~30 s
python scripts/smoke_test.py --real-tokenizer   # + real Qwen3 tokenizer check (downloads ~11 MB)
python -m pytest
```

Or with conda: `conda env create -f environment/base.yml && conda activate dgp-base && pip install -e .`

What CPU mode can do: prepare datasets, run MDK trimming, compute DeBERTaV3 embeddings (slowly),
build prompts with the mock summarizer, run all unit tests and the complexity analysis.
What it cannot practically do: Qwen3-8B summarization or LoRA fine-tuning.

## GPU (the reproduction)

Linux, one NVIDIA GPU with ≥48 GB (A100 80GB recommended; see `research/compute_plan.md`), CUDA 12.x.

```bash
conda env create -f environment/gpu.yml
conda activate dgp-gpu
pip install -e .
python scripts/smoke_test.py --real-tokenizer
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

**Verified working environment** (the 5-seed AmazonVideo run in README §16 used this): a Vast.ai instance image
with `torch==2.11.0+cu128` already installed in `/venv/main`; the project's own deps layered on top with
`pip install -e ".[llm,dev]"` (equivalent to the conda path above — no conda needed if a working torch+CUDA is
already present). Actual resolved versions: `transformers==5.17.0`, `peft==0.21.0`, `accelerate==1.15.0`,
`scipy==1.18.1`, `scikit-learn==1.9.1`, CUDA 12.8, 2× **NVIDIA A100-SXM4-80GB**. Only one GPU is required —
LoRA training peaked at **19.7 GB** (`run_manifest.json → peak_memory.gpu_peak_gb`), matching the ~20–26 GB
ESTIMATE in `research/compute_plan.md` §2. A second GPU is only useful to parallelize independent seeds
(`--device cuda:0` / `--device cuda:1` in separate processes — `train.py` itself does not do multi-GPU).

**Measured throughput** (replaces the ESTIMATEs in `research/compute_plan.md` §4, batch size 64):
summary generation (Qwen3-8B, greedy decode, B=10 → ≤56 new tokens) sustained **~10.5 summaries/s** once warm;
one full training seed (LoRA fine-tune + val + test inference, up to 10 epochs with early stopping) took
**35–80 minutes** wall-clock depending on the stopped epoch (2,263–4,757 s across 5 seeds, `best_epoch` 1–3).
Generating the AmazonVideo node + metapath summary set from scratch (119,829 + 35,747 requests) is therefore
**~4 GPU-hours** on one A100, not the ~1.5 h in the original ESTIMATE — the paper's own workload is heavier than
the initial guess assumed. Summaries and the MDK cache are shared across all 5 training seeds (keyed on
dataset + K/M/B, not on the training seed), so only the first seed pays the summarization cost.

## Models

Neither model is gated and no Hugging Face token is required (a token only raises download rate limits).

| Model | Hugging Face id | Download | Used for |
|---|---|---|---|
| Qwen3-8B | `Qwen/Qwen3-8B` | 16.4 GB | frozen summarizer and LoRA classifier |
| DeBERTaV3-base | `microsoft/deberta-v3-base` | ~0.7 GB | text embeddings for MDK |

They download automatically on first use. To pre-download (for example onto a cloud volume):

```bash
pip install -U huggingface_hub
hf download Qwen/Qwen3-8B
hf download microsoft/deberta-v3-base
# optional: export HF_HOME=/path/with/space  before downloading
```

For a final reproduction run, pin the exact model commit: set `summarizer.revision` and
`classifier.revision` in `configs/models/dgp.yaml` to the hash shown on the model page.

## Baselines

Each baseline uses its own environment, created from its upstream requirements:

```bash
python scripts/fetch_baselines.py consisgad
conda create -n bl-consisgad python=3.9 && conda activate bl-consisgad
pip install -r methods/consisgad/upstream/requirements.txt
```

See `methods/README.md` and `environment/baselines.yml`.
