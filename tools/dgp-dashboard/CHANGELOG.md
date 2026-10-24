## 0.3.0 (2026-10-24)
Complete redesign, visual-first: graphite dark theme (light on a toggle), five sections — Overview (live study progress,
three-stage pipeline, runs / active jobs / remaining time, GPU summary, latest event), Experiments (one row per
configuration with a five-seed grid: queued / running / stalled / completed / failed; run manifests on demand),
Results (metric switch; paired change vs each study's baseline with seed dots, mean ± sd against baseline and the
published paper, per-seed consistency matrix, interpretation with uncorrected paired t-tests), Insights and diagnostics
(strongest result, largest regression, stability, gap to the paper, generated-text diagnostics with real completion
state, evidence links), Infrastructure and details (GPU history, runtimes, logs, token accounting, CSV, deviation report).
Studies kept separate and labelled: main reproduction and extension experiments (historical), FLAG-architecture study
(live). Server: studies / overview / insights / diagnostics / infra in /api/state (v0.2 keys kept), /api/results.csv,
/api/run, /api/log, /files/*; the neighbour analysis is no longer reported as running from a leftover log; FLAG-arch
caches labelled. Previous files: 2026-10-24-server-v2.py, 2026-10-24-index-v2.html.

## 0.2.0 (2026-10-12)
Redesign to the FLAG dashboard v4 layout: state-tinted sticky header with overall progress and ETA, Experiments strip,
five-stage pipeline cards (MDK trimming, LLM text, LoRA training, results, neighbour analysis) with a "limiting" marker,
results table with paired Δ against baseline A (win/loss/within-noise colouring, partial arms flagged), GPU rows with
utilisation sparkline, unfinished jobs grouped by arm with filter chips and sortable columns (idle groups collapsed),
neighbours-by-K table, cache coverage heatmap. Server adds ETA, experiment blocks, GPU history, per-seed results.
Previous files: 2026-10-12-server-v1.py, 2026-10-12-index-v1.html.

## 0.1.0 (2026-10-12)
First version: queue, results, generation progress, neighbour table.
