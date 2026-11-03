# Hyperparameter search: grid and cost (proposal, 2026-11-03)

**Status (2026-11-03T07:13Z): APPROVED IN PART and RUNNING.** Owner decision of 2026-11-03:

1. Scope: **S1 only** of the core stages (S2 rank × lr on AmazonVideo and S3 confirmation are not run).
2. Seeds per grid point: **2**.
3. YelpChi: **add its own learning-rate sweep (S2-Y)**. Without S2 there is no AmazonVideo winner to transfer, so S2-Y
   runs lr ∈ {1e-5, 3e-5} at the default rank 8 (lr 1e-4 is S1), seeds 0 and 1, with the same all-epochs protocol as S1
   so every selection rule can be compared across learning rates (about 24 GPU-h instead of the 14 below).

Running: `scripts/sel_queue.sh` (configs `configs/experiments/sel/`), README §23. Approved total about 67 GPU-h, about
52 h wall-clock on one A100 with two lanes; expected finish about 2026-11-05.

The proposal as written for the decision follows. It takes option A of `compute_plan.md` §6 (the staged search) and
rebuilds it from the measured costs and the findings so far.

## 0. What is already known (no GPU needed)

**The paper's grid** (arXiv v1 §5.1), selected by mean validation AUROC:

| Axis | Values |
|---|---|
| B | {10, 20, 40, 80} |
| K | {1, 2, 3} |
| M | {2, 4, 8, 16} |
| LoRA r | {4, 8, 16, 32} |
| dropout | {0, 0.05, 0.1} |
| lr | {1e-5, 3e-5, 1e-4} |

That makes 1,728 configurations. The paper trains for at most 10 epochs, early-stops on validation loss, and reports
none of the values it selected. Our defaults are K=2, M=4, B=10, r=8 (alpha 16), dropout 0.05, lr 1e-4 and patience 2.

**Axes already covered:**
- K: §17 swept it on AmazonVideo. K=1 and K=3 are within ±0.2 AUROC of K=2.
- B: this is the Figure 5 budget study (§15). It stays a separate item and is not part of this search.

**Epoch selection, mined from all 178 existing training logs:**

| Runs | Kept epoch (lowest val loss) vs best val-AUROC epoch | Val AUROC / AUPRC given up by keeping the val-loss epoch |
|---|---|---|
| AmazonVideo, all 168 runs | different in about half the runs | about +0.4 / +0.4 on average, small |
| **YelpChi DGP, 5 runs** | **different in 5/5. The best val-AUROC epoch is the last epoch run in every seed (5, 3, 4, 5, 4), so training was cut off while AUROC was still rising.** | **+4.02 / +6.67** |

These are validation gains measured on the same set that picks the epoch, so they overstate what the test set would
show. They are still the strongest lead we have on the YelpChi gap (−5.9 AUROC, −17.8 AUPRC against the paper).

## 1. Measured unit costs (one A100-80GB)

| | AmazonVideo (train/val/test 1,299/1,299/7,425) | YelpChi (1,348/1,348/13,479) |
|---|---|---|
| One epoch (train + val pass) | 0.21 h (dr_R0: 765 s) | 0.44 h on its own, 0.68 h with two lanes sharing the GPU |
| Fixed cost per run (model load + one test pass) | 0.27 h | 0.78 h |
| Typical run under the current rule (3–4 epochs) | 1.1 h | 2.1–3.4 h |
| One full 10-epoch run | 2.4 h | 5.2 h |
| New metapath and node summaries for one M value | about 3.5–6 h (scales from the 4.1 h main set) | about 11–14 h |

**Units:** "GPU-h" below means time on a single lane. Two lanes on the 80 GB card gave about 1.3× the throughput
(YelpChi: 2 × 1,590 s against 2,440 s per epoch), so wall-clock time ≈ GPU-h / 1.3.

## 2. Stages

### S1 · Epoch-selection rule (E19). Recommended, and it goes first.

- **Setup:** default configuration, 5 seeds, both datasets.
- **Training:** always 10 epochs, with the LoRA adapter saved after every epoch (about 15 MB at r=8).
- **Test passes:** the test set is scored only at the epoch each rule picks:
  - (a) lowest val loss: the paper's rule, which reproduces today's numbers.
  - (b) highest val AUROC.
  - (c) also recorded: early stopping on val AUROC with patience 2 or 3, simulated from the same trajectory.
- **Why one run is enough:** every rule is compared on the same seed and the same training trajectory, so one run gives a
  clean paired comparison.
- **Free check:** epochs 1 to k should reproduce today's logs exactly.
- **Cost:**
  - AmazonVideo: 5 × about 2.6 h = **about 13 GPU-h**.
  - YelpChi: 5 × about 6 h = **about 30 GPU-h**.
- **Answers:** does selecting on val AUROC close part of the YelpChi gap? Does it matter on AmazonVideo?

### S2 · LoRA rank × learning rate on AmazonVideo (E20). Recommended.

- **Points:** 12 (r ∈ {4, 8, 16, 32} × lr ∈ {1e-5, 3e-5, 1e-4}), with dropout fixed at 0.05 and alpha = 2r.
  - The default point (r=8, lr 1e-4) is already covered by S1, so **11 new configurations**.
- **Seeds:** 0 and 1, about 0.5 AUROC seed noise, so a single seed cannot rank the configurations.
- **Selection rule:** whichever S1 chooses; until then, the default is early stopping on val AUROC with patience 3 and at
  most 10 epochs.
- **Choosing the winner:** **mean validation AUROC only.** Test metrics are recorded but never used.
  - The script already exists: `scripts/run_grid.py --stage lora`.
- **Cost:** 22 runs × about 1.5 h = **about 33 GPU-h**.
  - Worst case is about 52 GPU-h, if lr 1e-5 runs the full 10 epochs everywhere.

### S3 · Confirm the winner (E21). Recommended.

- **Setup:** the winner gets 5 seeds and is compared seed by seed with the default (paired t, bootstrap CI, Holm).
- **AmazonVideo:** seeds 2–4 are new, **about 4.5 GPU-h**.
- **YelpChi:** the AmazonVideo winner moved over, 5 seeds, **about 17 GPU-h**.
  - This is 0 if the winner is the default, because S1 already holds those runs.

### Optional stages

| Stage | What runs | Extra GPU-h | Value |
|---|---|---|---|
| S2-Y | YelpChi's own lr sweep: the 2 other lr values at the winning rank, 2 seeds | about 14 | The paper tunes per dataset, and YelpChi's training curve differs. **Worth it if S1 shows YelpChi is under-trained.** |
| S4 | M ∈ {2, 8, 16} at K=2 on AmazonVideo, 2 seeds: new summaries, then training | about 23 | The only prompt axis not yet searched. K was flat, so a gain is unlikely. |
| S5 | Dropout {0, 0.1} at the winner on AmazonVideo, 2 seeds | about 6 | Low |
| — | The full 1,728-point grid | > 2,000 | Not proposed |

## 3. Totals

| Plan | Stages | GPU-h | Wall-clock (2 lanes) | Cost at $0.67–$2.06 per GPU-h |
|---|---|---|---|---|
| Minimal | S1 only | about 43 | about 1.4 days | $29–$89 |
| **Core (recommended)** | S1 + S2 + S3 | **about 80–98** | **about 2.6–3.1 days** | **$54–$202** |
| Core + S2-Y | | about 95–112 | about 3–3.6 days | $64–$231 |
| Everything optional | Core + S2-Y + S4 + S5 | about 125–140 | about 4–4.5 days | $84–$288 |

- The ranges cover early stopping ending sooner or later than expected, and whether S3 needs new YelpChi seeds.
- Budget about 1.2× extra for reruns.
- Disk use is small (adapters are 15–60 MB per epoch, and the ones not selected are deleted after the test pass). There is
  43 GB free.

## 4. How the results will be reported

- **The paper's rule stays the primary result.** The paper-rule numbers (early stopping on val loss, default
  hyperparameters) remain the reproduction result. The rest is reported as recorded deviations.
  - Selecting the epoch on val AUROC goes against the paper's stated "early stopping on validation loss".
- **No selection uses the test set.** Every choice of epoch, configuration or stage winner is made on validation metrics
  alone.
- **Statistics:**
  - New families go into `scripts/stats_standard.py`:
    - E19: rule (b) − rule (a) per dataset.
    - E21: winner − default per dataset.
  - Holm is applied within each family and globally, as for E2–E18.
  - The grid stage (E20) is model selection, not a hypothesis test, so it gets no p-values.

## 5. Code changes before any GPU run (CPU-tested)

1. `training/trainer.py`:
   - `training.selection_metric` (`val_loss` stays the default | `val_auroc`).
   - `training.early_stopping_patience: null` runs all epochs.
   - `training.save_epoch_adapters` saves the adapter after each epoch.
   - Defaults are unchanged, so every existing result and cache key stays valid.
2. `experiment.py`: when adapters are saved, score the test set at each rule's epoch and write every rule's metrics to
   `metrics.json`.
3. `scripts/run_grid.py`: add a filter to run a subset of points (S4 needs M at K=2 only) and to skip points that
   already exist.
4. Wiring:
   - Queue scripts for S1, S2 and S3.
   - `scripts/sel_analysis.py`.
   - Dashboard entries.
   - README, report and docx updates after each stage, as with E14–E18.

## 6. Decisions for the owner

1. **Scope:** minimal, **core (recommended)**, or core plus which optional stages?
2. **Seeds per grid point in S2:** **2 (recommended)**, or 1, which halves S2 to about 17 GPU-h but cannot separate
   configurations closer than about 1 AUROC.
3. **YelpChi:** take the AmazonVideo winner as it is, or add S2-Y?
