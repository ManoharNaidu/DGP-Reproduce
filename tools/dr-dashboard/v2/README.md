# dr-dashboard v2

Read-only progress dashboard for the disc/resid study (FLAG's discriminative and residual text inside DGP,
AmazonVideo). Plain Python standard library on the server, plain HTML/CSS/JS with inline SVG in the browser; no
frameworks, no CDNs. Refreshes every 10 s; a refresh renders in about 40–70 ms.

## Run

```bash
cd DGP-Reproduce
python tools/dr-dashboard/v2/server.py 18782            # live data, http://127.0.0.1:18782 (loopback only)
```

Installed as the live dashboard (after the swap, see below) it runs as the supervisor service `dr_dashboard` on
127.0.0.1:18781, published by Caddy on external port 10100 (token-protected):

```bash
supervisorctl status dr_dashboard
supervisorctl restart dr_dashboard
```

Open it from the Vast Instance Portal ("DGP Disc-Resid Dashboard"), or with `?token=<OPEN_BUTTON_TOKEN>` in the URL;
the token is forwarded to every data request.

## What it reads (never writes)

| Section | Source |
|---|---|
| Generation | `datasets/cache/node_summaries/*/{metadata.json,summaries.jsonl}` for `dr_discriminative`, `dr_residual`, `dr_discriminative_nv`; `logs/disc_resid/gen_all.log`; whether `scripts/dr_generate_all.sh` is running |
| Training queue | `logs/disc_resid/queue.log`, `results/disc_resid/done/*`, `results/raw/dr_*/{training_log.jsonl,metrics.json}`; whether `scripts/dr_queue.sh` / `dr_after_gen.sh` are running |
| Results | `results/raw/dr_*/metrics.json`, the original run `results/raw/dgp_amazonvideo_seed0_seed*/metrics.json` |
| Diagnostics | `results/disc_resid/text_diagnostics.json`; `cache/text_insights.json` (below) |
| GPU | `nvidia-smi` |

It writes only its own files next to `server.py`: `history.json` (one hour of sampled generation counts and GPU readings,
so a restart keeps the sparklines and charts) and `cache/text_insights.json`, computed once by `text_insights.py`
(length histograms and top n-grams; train + val labels only) when all three texts are complete.

Endpoints: `/` page · `/api/state` JSON · `/api/results.csv` per-seed results · `/api/log?job=dr_R2_s0` tail of a
training log (only `dr_R<k>[b]_s<seed>` names are accepted) · `/files/deviation.md`, `/files/text_diagnostics.md`
once they exist.

## Known limits

- Training is logged once per epoch, so running cells show epoch, train/val loss and val AUROC, not per-step progress.
  Early stopping (patience 2) usually ends a run before epoch 10.
- Throughput sparklines come from the dashboard's own 10-second samples, so they start empty after a fresh install.
- ETAs use measured averages once available, otherwise 1.1 h per plain run and twice that with the residual branch
  (marked "est.").

## Test

```bash
python tools/dr-dashboard/v2/mock/make_mock.py tools/dr-dashboard/v2/mock/data   # full + empty mock trees
python tools/dr-dashboard/v2/server.py 18783 --root tools/dr-dashboard/v2/mock/data/full --no-insights &
cd tools/dr-dashboard/v2/test && npm install jsdom@24.1.3 && node render_test.js http://127.0.0.1:18783/
```

`full` covers every state (completed / failed generation, done / running / failed / queued runs, leakage alert,
GPU over 80 °C and over 90% VRAM, half-written cache lines); `empty` has no data at all and must render "—" everywhere
without script errors.

## Traceability and export (added 2026-10-23)

Everything below is read from files written at launch or run time; nothing is reconstructed at display time, and a
value that was never recorded is shown as "unknown". `history.json` is unchanged (no new keys).

| Feature | Source |
|---|---|
| Header: git hash / branch / dirty, run dates, hypothesis | latest `results/<study>/batch_*.json` (branch, hypothesis, command, dirty files) and each run's `results/raw/<run>/run_manifest.json` (commit + `-dirty`, written) |
| `/api/manifest?study=dr\|fa[&download=1]`, `server.py --manifest fa` | run manifests (config = hyperparameters, prompt_version, templates), `metrics.json` (val + test), batch records, generation cache `metadata.json` |
| View Config Diff | the two arms' flattened run-manifest configs; only differing fields |
| Failed runs | last error block of `logs/<study>/train_<job>.log` (kind: OOM / parse error / killed / error) |
| `/api/samples?study=&arm=&n=1..5`, View Samples | random byte offsets into `datasets/cache/prompts/<exp>_<hash>.jsonl` (newest file of the arm), text capped at 1500 chars; split from `split.json`, label from `labels.npy` for train/val only (test hidden), rating from `numeric.npy`, neighbour counts from `results/token_usage/token_usage_<exp>.csv` |
| Gold border | winner = highest mean **validation** AUROC over the arm's finished seeds (`pick_winner` in server.py) |
| Box plots | the finished seeds of each mean ± std (box for n ≥ 4, range + median for n 2–3, dots always) |

Record a batch at launch so branch, command and hypothesis are known:

```bash
python scripts/record_batch.py --study flag_arch --hypothesis "…" \
    --command "nohup scripts/fa_queue.sh > logs/flag_arch/queue.log 2>&1 &" --config configs/experiments/flagarch/F1_disc.yaml
```

Tests (mock trees only; the endpoint test also checks that the data tree is byte-for-byte unchanged afterwards):

```bash
/venv/main/bin/python -I tools/dr-dashboard/v2/test/test_trace_endpoints.py
python tools/dr-dashboard/v2/mock/make_mock.py /tmp/m && python tools/dr-dashboard/v2/mock/make_fa_mock.py /tmp/m/fa
python tools/dr-dashboard/v2/mock/make_trace_mock.py /tmp/m/full /tmp/m/fa
python tools/dr-dashboard/v2/server.py 18797 --root /tmp/m/full --no-insights &
node tools/dr-dashboard/v2/test/trace_ui_test.js http://127.0.0.1:18797/      # exit 1 on any failed check
```

Currently installed as supervisor program `dgp-dashboard` (127.0.0.1:18781, external port 10200, token); the
service name and port in "Run" above are from the first install.
