#!/bin/bash
# FLAG-architecture study: (1) FLAG-style D text per trimmed target subgraph, the metapath summaries of those lines
# and the prompts, for F1 then F0; (2) train F0, F1 x seeds 0-4 on one GPU, seed-major so paired comparisons appear
# early. F2 (R text + FLAG's +FLAG* loss / fine-tuning phase) is out of scope for this experiment (user, 2026-10-23);
# its config stays in configs/experiments/flagarch/ but is not run. Resumable: caches skip finished items, a finished run leaves
# results/flag_arch/done/<job>, a stopped run restarts that seed from epoch 0. Times are UTC.
#   nohup scripts/fa_queue.sh > logs/flag_arch/queue.log 2>&1 &
cd "$(dirname "$0")/.."
PY=${PY:-/workspace/Major_Project_Backup_Files/envs/dgp-fa/bin/python}
export HF_HOME=${HF_HOME:-/workspace/.hf_home} HF_HUB_OFFLINE=1
DONE=results/flag_arch/done
GEN=logs/flag_arch/gen_all.log
mkdir -p "$DONE" logs/flag_arch

if ! grep -q GEN_DONE "$GEN" 2>/dev/null; then
  for arm in F1_disc:fa_F1 F0_baseline:fa_F0; do
    cfg=${arm%%:*}; exp=${arm##*:}
    echo "[$(date -u +%FT%T)] generate $exp" >> "$GEN"
    if ! "$PY" -W ignore scripts/generate_summaries.py --dataset amazonvideo \
         --config configs/experiments/flagarch/$cfg.yaml --set experiment=$exp --device cuda:0 \
         > logs/flag_arch/gen_$exp.log 2>&1; then
      echo "[$(date -u +%FT%T)] GEN_FAILED $exp (see logs/flag_arch/gen_$exp.log)" >> "$GEN"
      exit 1
    fi
    grep -E "FA (discriminative|residual):|mean_final_prompt_tokens|prompts written" logs/flag_arch/gen_$exp.log >> "$GEN"
  done
  echo "[$(date -u +%FT%T)] GEN_DONE" >> "$GEN"
fi

ARMS="F0_baseline:fa_F0 F1_disc:fa_F1"
for seed in 0 1 2 3 4; do
  for arm in $ARMS; do
    cfg=${arm%%:*}; exp=${arm##*:}; job=${exp}_s${seed}
    [ -f "$DONE/$job" ] && continue
    echo "[$(date -u +%FT%T)] start $job"
    t0=$(date +%s)
    if "$PY" -W ignore scripts/train.py --dataset amazonvideo \
         --config configs/experiments/flagarch/$cfg.yaml --experiment $exp --seeds $seed --device cuda:0 \
         >> logs/flag_arch/train_$job.log 2>&1; then
      echo $(( $(date +%s) - t0 )) > "$DONE/$job"
      echo "[$(date -u +%FT%T)] done  $job ($(( ($(date +%s) - t0) / 60 )) min)"
    else
      echo "[$(date -u +%FT%T)] FAILED $job (see logs/flag_arch/train_$job.log)"
    fi
  done
done
echo "[$(date -u +%FT%T)] QUEUE_DONE"
