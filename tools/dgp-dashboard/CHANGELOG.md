## 0.2.0 (2026-10-12)
Redesign to the FLAG dashboard v4 layout: state-tinted sticky header with overall progress and ETA, Experiments strip,
five-stage pipeline cards (MDK trimming, LLM text, LoRA training, results, neighbour analysis) with a "limiting" marker,
results table with paired Δ against baseline A (win/loss/within-noise colouring, partial arms flagged), GPU rows with
utilisation sparkline, unfinished jobs grouped by arm with filter chips and sortable columns (idle groups collapsed),
neighbours-by-K table, cache coverage heatmap. Server adds ETA, experiment blocks, GPU history, per-seed results.
Previous files: 2026-10-12-server-v1.py, 2026-10-12-index-v1.html.

## 0.1.0 (2026-10-12)
First version: queue, results, generation progress, neighbour table.
