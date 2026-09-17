#!/bin/bash
set -uo pipefail
cd /workspace/DGP-Reproduce
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
EXP=dgp_amazonvideo_seed0   # same experiment name as seed 0, so all 5 seeds aggregate together

echo "=== batch 1: seed 1 (cuda:0) + seed 2 (cuda:1) starting $(date -Iseconds) ==="
python scripts/train.py --dataset amazonvideo --device cuda:0 --seeds 1 --experiment "$EXP" > results/train_seed1.log 2>&1 &
P1=$!
python scripts/train.py --dataset amazonvideo --device cuda:1 --seeds 2 --experiment "$EXP" > results/train_seed2.log 2>&1 &
P2=$!
wait $P1; R1=$?
wait $P2; R2=$?
echo "=== batch 1 done: seed1 exit=$R1 seed2 exit=$R2 $(date -Iseconds) ==="

echo "=== batch 2: seed 3 (cuda:0) + seed 4 (cuda:1) starting $(date -Iseconds) ==="
python scripts/train.py --dataset amazonvideo --device cuda:0 --seeds 3 --experiment "$EXP" > results/train_seed3.log 2>&1 &
P3=$!
python scripts/train.py --dataset amazonvideo --device cuda:1 --seeds 4 --experiment "$EXP" > results/train_seed4.log 2>&1 &
P4=$!
wait $P3; R3=$?
wait $P4; R4=$?
echo "=== batch 2 done: seed3 exit=$R3 seed4 exit=$R4 $(date -Iseconds) ==="

echo "=== evaluate $(date -Iseconds) ==="
python scripts/evaluate.py

echo "=== compare with paper $(date -Iseconds) ==="
python scripts/compare_with_paper.py

echo "=== 5-SEED CHAIN COMPLETE $(date -Iseconds) ==="
