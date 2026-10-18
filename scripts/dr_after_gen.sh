#!/bin/bash
# Wait for scripts/dr_generate_all.sh to finish, then run the text diagnostics (CPU) and start the training queue.
cd "$(dirname "$0")/.."
until grep -q GEN_DONE logs/disc_resid/gen_all.log 2>/dev/null; do
  if ! pgrep -f dr_generate_all.sh >/dev/null; then echo "generation stopped without GEN_DONE; not starting the queue"; exit 1; fi
  sleep 60
done
/venv/main/bin/python -W ignore scripts/dr_text_diagnostics.py > logs/disc_resid/text_diagnostics.log 2>&1 &
exec scripts/dr_queue.sh >> logs/disc_resid/queue.log 2>&1
