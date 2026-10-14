"""Writes HANDOFF_4GPU.md (what is done, what is left, how to resume) and packs the state into a tarball.
    python scripts/make_handoff.py [--no-tar]"""
import glob, json, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ext_queue
done = {p.name for p in (ROOT / "results/ext/done").iterdir()}
jobs = ext_queue.build_jobs()
left = [j["name"] for j in jobs if j["name"] not in done]
fin = [j["name"] for j in jobs if j["name"] in done]
caches = []
for m in sorted(glob.glob(str(ROOT / "datasets/cache/*_summaries/*/metadata.json"))):
    meta = json.load(open(m)); f = Path(m).parent / "summaries.jsonl"
    caches.append(f"- {Path(m).parent.parent.name}/{Path(m).parent.name}: {sum(1 for _ in open(f)) if f.exists() else 0} lines  ({meta.get('stage')}, template={meta.get('template')})")
res = []
for mp in sorted(glob.glob(str(ROOT / "results/raw/*/metrics.json"))):
    d = Path(mp).parent
    try:
        man = json.load(open(d / "run_manifest.json")); t = json.load(open(mp))["test"]
        res.append(f"- {man.get('experiment')} seed {json.load(open(mp)).get('seed')}: F1 {100*t['macro_f1']:.2f}  AUROC {100*t['auroc']:.2f}  AUPRC {100*t['auprc']:.2f}")
    except Exception:
        pass
group = lambda pre: [n for n in left if n.startswith(pre)]
md = f"""# Hand-off to the 4-GPU instance  ({time.strftime('%Y-%m-%d %H:%M')})

The 2-GPU queue was stopped on request; gen_B (about 52k of 119,829 metapath summaries) and train_armC_s3 were killed mid-run and restart from their caches (gen_B) or epoch 0 (armC_s3). Everything below is on disk
in this folder; copy it as-is and restart the queue.

## Finished jobs ({len(fin)} of {len(jobs)})
{chr(10).join('- ' + n for n in fin)}

## Still to run ({len(left)})
{chr(10).join('- ' + n for n in left)}

Totals left: {len([n for n in left if n.startswith('train_')])} training runs, {len([n for n in left if n.startswith('gen_')])} generation jobs
({', '.join(n for n in left if n.startswith('gen_')) or 'none'}).

## Generation caches (summaries are cached per node; a restarted job skips existing lines)
{chr(10).join(caches)}

## Result files already produced (results/raw/*/metrics.json)
{chr(10).join(res)}

## Resume on the new instance
1. Copy this folder (the tarball made by this script, or rsync) to the same path, plus the Qwen3-8B model into HF_HOME
   (or let it download: `huggingface-cli download Qwen/Qwen3-8B`). Install the same Python packages (torch, transformers,
   peft, accelerate, scipy, scikit-learn, pandas, sentencepiece).
2. Check the GPUs: `nvidia-smi` should list four. Optional: run `python -m pytest tests/unit -q` (84 tests).
3. Start the scheduler on all four GPUs (finished jobs are skipped through results/ext/done/):
   ```
   cd DGP-Reproduce
   mkdir -p logs/ext
   GPUS=0,1,2,3 SLOTS_PER_GPU=2 nohup python scripts/ext_queue.py > logs/ext/queue.log 2>&1 &
   ```
   Jobs are pinned with CUDA_VISIBLE_DEVICES; the K=3 jobs take a whole GPU. On this box two jobs per 80 GB GPU filled
   79 GB (allocator caching); use SLOTS_PER_GPU=1 if memory is tighter.
4. Dashboard: `python tools/dgp-dashboard/server.py 18780` (read-only, loopback); on a Vast box add it as a supervisor
   service and portal entry as done here.
5. Neighbour analysis (CPU, LLM-free): `python analysis/neighbor_analysis.py` if analysis/out/neighbor_analysis.json is missing.

## Notes
- Training runs have no mid-run checkpoint: a run that is stopped restarts from epoch 0.
- `gen_K3` (~390k metapath summaries) is a single job, so extra GPUs do not speed it up unless it is sharded.
- Baseline (5 seeds) and the parity run are unchanged; arm results are compared against them by seed.
"""
(ROOT / "HANDOFF_4GPU.md").write_text(md)
print("wrote HANDOFF_4GPU.md;", len(fin), "done,", len(left), "left")
if "--no-tar" not in sys.argv:
    out = ROOT.parent / f"DGP-Reproduce-state-{time.strftime('%Y%m%d-%H%M')}.tar.gz"
    subprocess.run(["tar", "-czf", str(out), "--exclude=__pycache__", "--exclude=.git", "-C", str(ROOT.parent), ROOT.name], check=True)
    print("packed", out, round(out.stat().st_size / 1e6), "MB")
