#!/bin/bash
# 2026-11-04: owner stopped E19 after sel_amazonvideo seed 1. Waits for that run (PID $1), records its end, closes the queue.
cd "$(dirname "$0")/.."
OUT=results/sel; pid=$1
while kill -0 "$pid" 2>/dev/null; do sleep 30; done
rc=1; ls results/raw/sel_amazonvideo_seed1_*/metrics.json > /dev/null 2>&1 && rc=0
start=$(grep '"job": "sa_s1", "event": "start"' $OUT/status.jsonl | tail -1 | sed 's/.*"t": "\([^"]*\)".*/\1/')
t=$(date -u +%FT%TZ)
echo "{\"t\": \"$t\", \"job\": \"sa_s1\", \"event\": \"end\", \"rc\": $rc, \"secs\": $(( $(date +%s) - $(date -d "$start" +%s) ))}" >> $OUT/status.jsonl
echo "{\"t\": \"$t\", \"job\": \"queue\", \"event\": \"done\", \"note\": \"stopped by owner after sa_s1\"}" >> $OUT/status.jsonl
