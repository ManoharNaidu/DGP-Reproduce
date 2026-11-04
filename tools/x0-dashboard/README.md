# x0-dashboard: DGP follow-up progress (read-only)

A plain-standard-library dashboard for the DGP follow-up queues of 2026-11-01 → 11-04: Exp 4 (writer vs classifier,
E14 and E18), the control queues (target-only LLM E15, YelpChi E16, w/o MDK E17), the hyperparameter search (E19,
stopped by the owner on 2026-11-04) with a per-rule results panel, baselines, and the overall plan.

```bash
/venv/main/bin/python -u tools/x0-dashboard/server.py 18790     # binds 127.0.0.1 only
supervisorctl restart x0_dashboard                               # it runs as a supervisor service on this instance
```

It only reads files: `plan.json` (edited by hand as items move), the queues' `status.jsonl` files, `results/raw/*/`
(training logs and metrics), `results/gemma_swap/exp4_2x2.json`, the log tails of whitelisted job names, and
`nvidia-smi`. Reach it through an SSH tunnel to 127.0.0.1:18790 (no token needed) or the instance's authenticated edge.
Restart without `pkill -f` (the pattern matches the calling shell): use `supervisorctl restart x0_dashboard`.
