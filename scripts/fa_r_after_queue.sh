#!/bin/bash
# FLAG-architecture study, R text: waits until scripts/fa_queue.sh has exited (the GPU is free), then generates FLAG's
# residual line per trimmed target subgraph (configs/experiments/flagarch/R_text.yaml, --until fa_text: subgraph text
# only, no summaries, no training). Resumable through the fa_subgraph_text cache. Times are UTC.
#   nohup scripts/fa_r_after_queue.sh > logs/flag_arch/r_text.log 2>&1 &
cd "$(dirname "$0")/.."
PY=${PY:-/workspace/Major_Project_Backup_Files/envs/dgp-fa/bin/python}
export HF_HOME=${HF_HOME:-/workspace/.hf_home} HF_HUB_OFFLINE=1
echo "[$(date -u +%FT%T)] waiting for scripts/fa_queue.sh to finish"
while pgrep -f "^(/bin/)?bash (\S*/)?scripts/fa_queue.sh( |$)" > /dev/null; do sleep 60; done
echo "[$(date -u +%FT%T)] generate fa_R"
if "$PY" -W ignore scripts/generate_summaries.py --dataset amazonvideo --config configs/experiments/flagarch/R_text.yaml \
     --until fa_text --device cuda:0 > logs/flag_arch/gen_fa_R.log 2>&1; then
  grep -E "FA residual:" logs/flag_arch/gen_fa_R.log | tail -1
  echo "[$(date -u +%FT%T)] R_TEXT_DONE"
else
  echo "[$(date -u +%FT%T)] R_TEXT_FAILED (see logs/flag_arch/gen_fa_R.log)"
fi
