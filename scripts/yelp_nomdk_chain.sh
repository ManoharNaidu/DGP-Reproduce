#!/bin/bash
# GPU chain (2026-11-01), one A100 80 GB. Restarts YelpChi's metapath summaries as 3 shards on the shared cache (the
# 30.7k already written are skipped), then trains YelpChi (scripts/ctl_queue.sh, 2 lanes) while one process generates
# the w/o-MDK summaries for AmazonVideo, then trains w/o MDK. Every step is resumable. Events: logs/controls/chain.log
set -u
cd "$(dirname "$0")/.."
PY=../envs/dgp-fa/bin/python
export HF_HOME=${HF_HOME_DGP:-/workspace/hf_cache} HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${GPU:-0}
LOG=logs/controls/chain.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
YCFG=configs/experiments/dgp_yelp.yaml
NCFG=configs/experiments/controls/A1_without_mdk.yaml

say "yelp shards start"
for i in 0 1 2; do
  DGP_GEN_SHARD=$i/3 $PY -W ignore scripts/generate_summaries.py --dataset yelpchi --config $YCFG \
      --until metapath_summaries --device cuda:0 > logs/controls/YELP_shard$i.log 2>&1 &
  sleep 60
done
wait
say "yelp shards end"

DATASET=yelpchi bash scripts/ctl_queue.sh YELP $YCFG dgp_yelpchi > logs/controls/YELP_queue2.log 2>&1 &
YQ=$!
say "nomdk generation start (1 process, alongside YelpChi training)"
$PY -W ignore scripts/generate_summaries.py --dataset amazonvideo --config $NCFG --until metapath_summaries \
    --device cuda:0 > logs/controls/NOMDK_gen.log 2>&1
say "nomdk generation end rc=$?"
wait $YQ
say "yelp queue end rc=$?"
bash scripts/ctl_queue.sh NOMDK $NCFG ctl_noMDK > logs/controls/NOMDK_queue.log 2>&1
say "nomdk queue end rc=$?"
say "CHAIN_DONE"
