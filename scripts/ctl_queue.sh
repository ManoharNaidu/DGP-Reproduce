#!/bin/bash
# Controls queue (2026-11-01), one A100, two training lanes, resumable (train.py skips seeds with metrics.json).
#   bash scripts/ctl_queue.sh <name> <config> <experiment>      e.g. T0 configs/experiments/controls/T0_llm_target_only.yaml ctl_T0
# Builds the prompts once, then trains seeds 0,2,4 (lane A) and 1,3 (lane B). Events: results/controls/<name>/status.jsonl
set -u
cd "$(dirname "$0")/.."
NAME=$1 CFG=$2 EXP=$3
PY=../envs/dgp-fa/bin/python
export HF_HOME=${HF_HOME_DGP:-/workspace/hf_cache} HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${GPU:-0}
LOGS=logs/controls; OUT=results/controls/$NAME; mkdir -p "$LOGS" "$OUT"
stamp() { date -u +%FT%TZ; }
ev() { echo "{\"t\": \"$(stamp)\", $1}" >> "$OUT/status.jsonl"; echo "$(stamp) $1"; }
run() {   # job, seed
  local job=$1 seed=$2 t0=$SECONDS
  ev "\"job\": \"$job\", \"event\": \"start\""
  $PY -W ignore scripts/train.py --dataset amazonvideo --config "$CFG" --experiment "$EXP" --seeds "$seed" \
      --device cuda:0 > "$LOGS/$job.log" 2>&1
  local rc=$?
  ev "\"job\": \"$job\", \"event\": \"end\", \"rc\": $rc, \"secs\": $((SECONDS - t0))"
}
ev "\"job\": \"${NAME}_prompts\", \"event\": \"start\""
if ! $PY -W ignore scripts/generate_summaries.py --dataset amazonvideo --config "$CFG" --until prompts --device cuda:0 \
     > "$LOGS/${NAME}_prompts.log" 2>&1; then
  ev "\"job\": \"${NAME}_prompts\", \"event\": \"failed\""; exit 1
fi
ev "\"job\": \"${NAME}_prompts\", \"event\": \"end\""
( run ${NAME}_s0 0; run ${NAME}_s2 2; run ${NAME}_s4 4 ) &
sleep 120
( run ${NAME}_s1 1; run ${NAME}_s3 3 ) &
wait
ev "\"job\": \"queue\", \"event\": \"done\""
