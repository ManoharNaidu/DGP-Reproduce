#!/bin/bash
# Successor of sel_queue.sh (2026-11-03): the owner dropped E20 (YelpChi learning-rate sweep) to finish sooner; E19 continues
# on both datasets. Adopts the two runs sel_queue.sh had in progress (sel_yelpchi seeds 0 and 1, by PID), records their end,
# then runs the rest of E19 in two lanes. Resumable like sel_queue.sh (train.py skips seeds whose metrics.json exists).
#   setsid nohup bash scripts/sel_queue2.sh <pid_lane_A> <pid_lane_B> > /dev/null 2>&1 < /dev/null &
# After a machine restart (no PIDs left to adopt): bash scripts/sel_queue2.sh 0 0
set -u
cd "$(dirname "$0")/.."
PA=${1:-0} PB=${2:-0}
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
adopt() {   # job pid experiment seed: wait for a run started by sel_queue.sh and record its end.
            # Returns 0 if that seed is finished (nothing to run), 1 if it still has to be (re)run.
  local job=$1 pid=$2 rc=1 start
  if [ "$pid" -gt 0 ]; then
    while kill -0 "$pid" 2>/dev/null; do sleep 30; done
    ls results/raw/${3}_seed${4}_*/metrics.json > /dev/null 2>&1 && rc=0
    start=$(grep "\"job\": \"$job\", \"event\": \"start\"" "$OUT/status.jsonl" | tail -1 | sed 's/.*"t": "\([^"]*\)".*/\1/')
    ev "\"job\": \"$job\", \"event\": \"end\", \"rc\": $rc, \"secs\": $(( $(date +%s) - $(date -d "$start" +%s) ))"
  fi
  ls results/raw/${3}_seed${4}_*/metrics.json > /dev/null 2>&1
}
Y="yelpchi $C/S1_yelp.yaml sel_yelpchi"; A="amazonvideo $C/S1_amazon.yaml sel_amazonvideo"
ev "\"job\": \"queue\", \"event\": \"switched\", \"note\": \"E20 dropped by owner; sel_queue2.sh adopts sy_s0 (pid $PA) and sy_s1 (pid $PB)\""
( adopt sy_s0 "$PA" sel_yelpchi 0 || run sy_s0 $Y 0; run sy_s2 $Y 2; run sy_s4 $Y 4; run sa_s4 $A 4 ) &
sleep 5
( adopt sy_s1 "$PB" sel_yelpchi 1 || run sy_s1 $Y 1; run sy_s3 $Y 3; run sa_s0 $A 0; run sa_s1 $A 1; run sa_s2 $A 2; run sa_s3 $A 3 ) &
wait
ev "\"job\": \"queue\", \"event\": \"done\""
