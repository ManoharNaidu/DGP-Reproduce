#!/bin/bash
# Exp 4 (writer/classifier split, 2026-11-01): gs_X0 = Gemma-written text (gs_G0's cached summaries) + Qwen3-8B
# classifier, 5 seeds, plus one R0 seed-0 run on this machine's stack (par2_R0) to measure drift against dr_R0.
# One A100: prompts are built once, then two training lanes share the GPU (Qwen peak ~20 GB each).
# Resumable: train.py skips seeds whose results/raw/<run_id>/metrics.json exists.
#   nohup bash scripts/x0_queue.sh > logs/gemma_swap/x0_queue.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=../envs/dgp-fa/bin/python
# /workspace/.hf_home/hub is damaged on this instance (2026-11-01: stale file handles, broken blob links), so the
# weights were re-downloaded to a fresh cache: Qwen3-8B b968826d (the revision dr_R0 resolved), DeBERTa-v3-base 8ccc9b6f.
export HF_HOME=${HF_HOME_DGP:-/workspace/hf_cache} HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${GPU:-0}
X0=configs/experiments/gemma_swap/X0.yaml
R0=configs/experiments/disc_resid/R0_baseline.yaml
LOGS=logs/gemma_swap; OUT=results/gemma_swap/x0; mkdir -p "$LOGS" "$OUT"
stamp() { date -u +%FT%TZ; }
ev() { echo "{\"t\": \"$(stamp)\", $1}" >> "$OUT/status.jsonl"; echo "$(stamp) $1"; }

run() {   # name, config, experiment, seed
  local name=$1 cfg=$2 exp=$3 seed=$4 t0=$SECONDS
  ev "\"job\": \"$name\", \"event\": \"start\""
  $PY -W ignore scripts/train.py --dataset amazonvideo --config "$cfg" --experiment "$exp" --seeds "$seed" \
      --device cuda:0 > "$LOGS/$name.log" 2>&1
  local rc=$?
  ev "\"job\": \"$name\", \"event\": \"end\", \"rc\": $rc, \"secs\": $((SECONDS - t0))"
  return $rc
}

ev "\"job\": \"x0_prompts\", \"event\": \"start\""
if ! $PY -W ignore scripts/generate_summaries.py --dataset amazonvideo --config $X0 --until prompts --device cuda:0 \
     > "$LOGS/x0_prompts.log" 2>&1; then
  ev "\"job\": \"x0_prompts\", \"event\": \"failed\""; exit 1
fi
ev "\"job\": \"x0_prompts\", \"event\": \"end\""

( run x0_s0 $X0 gs_X0 0; run x0_s2 $X0 gs_X0 2; run x0_s4 $X0 gs_X0 4 ) &
sleep 120   # stagger model loading
( run x0_s1 $X0 gs_X0 1; run x0_s3 $X0 gs_X0 3; run par2_R0_s0 $R0 par2_R0 0 ) &
wait
ev "\"job\": \"queue\", \"event\": \"done\""
