#!/bin/bash
# Disc/resid study: train every arm x seed sequentially on one GPU. Resumable: a finished job leaves
# results/disc_resid/done/<job>; a stopped run restarts that seed from epoch 0 (train.py has no mid-run checkpoint).
# Seed-major order, so all arms of seed 0 finish first and paired comparisons are available early.
#   nohup scripts/dr_queue.sh > logs/disc_resid/queue.log 2>&1 &
cd "$(dirname "$0")/.."
export HF_HOME=/root/hf_home   # /workspace/.hf_home (system default) has stale file handles
DONE=results/disc_resid/done
mkdir -p "$DONE" logs/disc_resid
ARMS="R0_baseline:dr_R0 R1_disc:dr_R1 R1b_disc_nv:dr_R1b R2_disc_resid_aux:dr_R2 R2b_disc_nv_resid_aux:dr_R2b"
for seed in 0 1 2 3 4; do
  for arm in $ARMS; do
    cfg=${arm%%:*}; exp=${arm##*:}; job=${exp}_s${seed}
    [ -f "$DONE/$job" ] && continue
    echo "[$(date -u +%FT%T)] start $job"
    t0=$(date +%s)
    if /venv/main/bin/python -W ignore scripts/train.py --dataset amazonvideo \
         --config configs/experiments/disc_resid/$cfg.yaml --experiment $exp --seeds $seed --device cuda:0 \
         >> logs/disc_resid/train_$job.log 2>&1; then
      echo $(( $(date +%s) - t0 )) > "$DONE/$job"
      echo "[$(date -u +%FT%T)] done  $job ($(( ($(date +%s) - t0) / 60 )) min)"
    else
      echo "[$(date -u +%FT%T)] FAILED $job (see logs/disc_resid/train_$job.log)"
    fi
  done
done
echo "[$(date -u +%FT%T)] QUEUE_DONE"
