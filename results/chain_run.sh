#!/bin/bash
set -uo pipefail
cd /workspace/DGP-Reproduce
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home

# wait for the running summary-generation job (pid passed as $1) to exit
GEN_PID="$1"
while kill -0 "$GEN_PID" 2>/dev/null; do
  sleep 30
done
echo "=== generate_summaries.py (pid $GEN_PID) finished at $(date -Iseconds) ==="

echo "=== training seed 0 (fast path: single seed, reconstructed defaults) $(date -Iseconds) ==="
python scripts/train.py --dataset amazonvideo --device cuda:0 --seeds 0 --experiment dgp_amazonvideo_seed0

echo "=== evaluate $(date -Iseconds) ==="
python scripts/evaluate.py

echo "=== compare with paper $(date -Iseconds) ==="
python scripts/compare_with_paper.py

echo "=== CHAIN COMPLETE $(date -Iseconds) ==="
