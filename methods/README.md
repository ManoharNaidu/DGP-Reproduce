# Methods and provenance

`methods/<method>/upstream/` holds each official repository, fetched at a pinned commit by
`python scripts/fetch_baselines.py`. Third-party code is **never committed here and never edited in place**;
our adapters live in `src/dgp_repro/baselines/`. Licences stay exactly as shipped upstream.

Detailed per-method records (paper, requirements, entry points, adaptation, deviations, status):
`research/baselines/<method>.md`.

| Method | Family | Official repository | Pinned commit | Licence | Provenance | Status |
|---|---|---|---|---|---|---|
| **DGP** | Graph-enhanced LLM | `Xtra-Computing/DGP` — **README only, code unreleased** | — | — | PAPER_RECONSTRUCTION (in `src/dgp_repro/`) | SMOKE_TESTED |
| MLP | Graph-agnostic | none exists | — | — | PAPER_RECONSTRUCTION (`src/dgp_repro/baselines/mlp.py`) | IMPLEMENTED |
| LLM (Qwen3-8B, target only) | Graph-agnostic | n/a | — | Apache-2.0 (model) | PAPER_RECONSTRUCTION (DGP with neighbours off) | SMOKE_TESTED |
| GraphSAGE | GNN | [williamleif/GraphSAGE](https://github.com/williamleif/GraphSAGE) | `a0fdef95` | NOASSERTION | BASELINE_ADAPTED | NOT_STARTED |
| HGT | GNN | [acbull/pyHGT](https://github.com/acbull/pyHGT) | `85eaccd4` | MIT | BASELINE_ADAPTED | NOT_STARTED |
| ConsisGAD | GNN | [Xtra-Computing/ConsisGAD](https://github.com/Xtra-Computing/ConsisGAD) | `36811c5b` | MIT | BASELINE_ADAPTED | NOT_STARTED |
| PMP | GNN | [Xtra-Computing/PMP](https://github.com/Xtra-Computing/PMP) (= `JhuoW/PMP`) | `3f7629f6` | **none** | BASELINE_ADAPTED | NOT_STARTED |
| GAAP | GNN | [AtwoodDuan/GAAP](https://github.com/AtwoodDuan/GAAP) | `6a7dbb04` | **none** | BASELINE_ADAPTED | NOT_STARTED |
| TAPE | LLM-enhanced GNN | [XiaoxinHe/TAPE](https://github.com/XiaoxinHe/TAPE) | `d9881f7e` | MIT | BASELINE_ADAPTED | NOT_STARTED |
| FLAG (KDD'25) | LLM-enhanced GNN | [BUPT-GAMMA/FLAG](https://github.com/BUPT-GAMMA/FLAG) | `cb83944e` | **none** | BASELINE_ADAPTED | NOT_STARTED |
| GraphGPT | Graph-enhanced LLM | [HKUDS/GraphGPT](https://github.com/HKUDS/GraphGPT) | `db25a66f` | Apache-2.0 | BASELINE_ADAPTED | NOT_STARTED |
| HiGPT | Graph-enhanced LLM | [HKUDS/HiGPT](https://github.com/HKUDS/HiGPT) | `2b0793e7` | Apache-2.0 | BASELINE_ADAPTED | NOT_STARTED |
| InstructGLM | Graph-enhanced LLM | [agiresearch/InstructGLM](https://github.com/agiresearch/InstructGLM) | `dd2dd5ec` | Apache-2.0 | BASELINE_ADAPTED | NOT_STARTED |

All repositories and commits were verified against the GitHub API on 2026-09-13.

**Licence warning.** PMP, GAAP and FLAG publish no licence. Without one, no rights to copy or redistribute
are granted: fetch them locally for research use, but never commit or share their code.

**FLAG disambiguation.** DGP's FLAG is *"FLAG: Fraud Detection with LLM-enhanced Graph Neural Network"*
(Yang et al., KDD 2025). `devnkong/FLAG` is an unrelated 2022 method (adversarial augmentation) and must not be used.
`BUPT-GAMMA/FLAG` has no README; the attribution rests on author/lab and code-to-paper matching
(`research/baselines/flag.md`).

**Fairness adaptations shared by all GNN baselines** (`src/dgp_repro/baselines/export.py`): the DGP graph is
exported in the CARE-GNN `.mat` layout these loaders read, together with our `split.json`, so every method uses
the same graph, split, labels and scikit-learn metrics. Algorithms are not modified. Node features for
baselines on these text graphs are not specified by the paper; DeBERTaV3 ⊕ numeric is used and recorded as a
reconstruction.

**Qwen3-8B backbone for LLM baselines.** The paper states all LLM-tuning methods used Qwen3-8B. GraphGPT and
HiGPT hard-code a Vicuna/LLaMA backbone, so that swap is substantial adaptation work and a documented deviation
risk (`research/baselines/graphgpt.md`, `higpt.md`).
