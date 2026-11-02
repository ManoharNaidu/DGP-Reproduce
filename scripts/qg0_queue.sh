#!/bin/bash
# Exp 4, fourth cell (2026-11-02): gs_QG0 = Qwen-written text (dr_R0's cached summaries) + Gemma-2-9B-it classifier,
# 5 seeds. One A100, ONE training lane: two Gemma runs sharing a GPU were 2-3x slower per run than one alone (README §22).
# Resumable: train.py skips seeds whose results/raw/<run_id>/metrics.json exists.
#   setsid nohup bash scripts/qg0_queue.sh > /dev/null 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
PY=../envs/dgp-fa/bin/python
export HF_HOME=${HF_HOME_DGP:-/workspace/hf_cache} HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${GPU:-0}
CFG=configs/experiments/gemma_swap/QG0.yaml
LOGS=logs/gemma_swap; OUT=results/gemma_swap/qg0; mkdir -p "$LOGS" "$OUT"
stamp() { date -u +%FT%TZ; }
ev() { echo "{\"t\": \"$(stamp)\", $1}" >> "$OUT/status.jsonl"; echo "$(stamp) $1"; }

ev "\"job\": \"qg0_prompts\", \"event\": \"start\""
if ! $PY -W ignore scripts/generate_summaries.py --dataset amazonvideo --config $CFG --until prompts --device cuda:0 \
     > "$LOGS/qg0_prompts.log" 2>&1; then
  ev "\"job\": \"qg0_prompts\", \"event\": \"failed\""; exit 1
fi
ev "\"job\": \"qg0_prompts\", \"event\": \"end\""

for s in 0 1 2 3 4; do
  t0=$SECONDS
  ev "\"job\": \"qg0_s$s\", \"event\": \"start\""
  $PY -W ignore scripts/train.py --dataset amazonvideo --config $CFG --experiment gs_QG0 --seeds $s \
      --device cuda:0 > "$LOGS/qg0_s$s.log" 2>&1
  ev "\"job\": \"qg0_s$s\", \"event\": \"end\", \"rc\": $?, \"secs\": $((SECONDS - t0))"
done
ev "\"job\": \"queue\", \"event\": \"done\""
