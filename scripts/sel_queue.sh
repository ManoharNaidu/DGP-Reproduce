#!/bin/bash
# Hyperparameter search, owner-approved scope (2026-11-03, research/hparam_search_plan.md):
#   S1 = E19 epoch-selection rule, 5 seeds, YelpChi + AmazonVideo   (configs/experiments/sel/S1_*.yaml)
#   S2-Y = E20 YelpChi learning-rate sweep, lr 1e-5 and 3e-5, seeds 0-1 (configs/experiments/sel/S2Y_*.yaml)
# Every run trains all 10 epochs and scores the test set at the epoch each selection rule keeps.
# One A100, two training lanes (~20 GB each), YelpChi first. Resumable: train.py skips seeds whose metrics.json exists.
#   setsid nohup bash scripts/sel_queue.sh > /dev/null 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
PY=../envs/dgp-fa/bin/python
export HF_HOME=${HF_HOME_DGP:-/workspace/hf_cache} HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${GPU:-0}
C=configs/experiments/sel
LOGS=logs/sel; OUT=results/sel; mkdir -p "$LOGS" "$OUT"
stamp() { date -u +%FT%TZ; }
ev() { echo "{\"t\": \"$(stamp)\", $1}" >> "$OUT/status.jsonl"; echo "$(stamp) $1"; }
run() {   # job dataset config experiment seed
  local job=$1 t0=$SECONDS
  ev "\"job\": \"$job\", \"event\": \"start\""
  $PY -W ignore scripts/train.py --dataset "$2" --config "$3" --experiment "$4" --seeds "$5" --device cuda:0 \
      > "$LOGS/$job.log" 2>&1
  local rc=$?
  ev "\"job\": \"$job\", \"event\": \"end\", \"rc\": $rc, \"secs\": $((SECONDS - t0))"
}
for d in yelpchi:S1_yelp amazonvideo:S1_amazon; do   # summaries are cached: this only rebuilds and checks the prompts
  ds=${d%%:*} cfg=$C/${d##*:}.yaml
  ev "\"job\": \"prompts_$ds\", \"event\": \"start\""
  if ! $PY -W ignore scripts/generate_summaries.py --dataset "$ds" --config "$cfg" --until prompts --device cuda:0 \
       > "$LOGS/prompts_$ds.log" 2>&1; then
    ev "\"job\": \"prompts_$ds\", \"event\": \"failed\""; exit 1
  fi
  ev "\"job\": \"prompts_$ds\", \"event\": \"end\""
done
Y="yelpchi $C/S1_yelp.yaml sel_yelpchi"; A="amazonvideo $C/S1_amazon.yaml sel_amazonvideo"
L1="yelpchi $C/S2Y_lr1e5.yaml sel_yelp_lr1e5"; L3="yelpchi $C/S2Y_lr3e5.yaml sel_yelp_lr3e5"
( run sy_s0 $Y 0; run sy_s2 $Y 2; run sy_s4 $Y 4; run ly1_s0 $L1 0; run ly3_s0 $L3 0; run sa_s4 $A 4 ) &
sleep 180
( run sy_s1 $Y 1; run sy_s3 $Y 3; run ly1_s1 $L1 1; run ly3_s1 $L3 1; run sa_s0 $A 0; run sa_s1 $A 1; run sa_s2 $A 2; run sa_s3 $A 3 ) &
wait
ev "\"job\": \"queue\", \"event\": \"done\""
