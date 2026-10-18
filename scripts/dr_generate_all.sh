#!/bin/bash
# Disc/resid study: generate target texts and build prompts for every arm (resumable; caches skip finished nodes).
set -e
cd "$(dirname "$0")/.."
export HF_HOME=/root/hf_home   # /workspace/.hf_home (system default) has stale file handles
for arm in R2_disc_resid_aux:dr_R2 R2b_disc_nv_resid_aux:dr_R2b R1_disc:dr_R1 R1b_disc_nv:dr_R1b; do
  cfg=${arm%%:*}; exp=${arm##*:}
  echo "[$(date -u +%FT%T)] generate $exp"
  /venv/main/bin/python -W ignore scripts/generate_summaries.py --dataset amazonvideo \
    --config configs/experiments/disc_resid/$cfg.yaml --set experiment=$exp --device cuda:0 > logs/disc_resid/gen_$exp.log 2>&1
  grep -E "target (disc|residual)|mean_final_prompt_tokens|prompts written" logs/disc_resid/gen_$exp.log
done
echo "[$(date -u +%FT%T)] GEN_DONE"
