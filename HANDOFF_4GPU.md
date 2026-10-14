# Hand-off to the 4-GPU instance  (2026-10-12 10:52)

The 2-GPU queue was stopped on request; gen_B (about 52k of 119,829 metapath summaries) and train_armC_s3 were killed mid-run and restart from their caches (gen_B) or epoch 0 (armC_s3). Everything below is on disk
in this folder; copy it as-is and restart the queue.

## Finished jobs (6 of 46)
- train_parity_s0
- gen_T
- gen_K1
- train_armC_s0
- train_armC_s1
- train_armC_s2

## Still to run (40)
- gen_B
- train_armC_s3
- train_armC_s4
- train_armD_s0
- train_armD_s1
- train_armD_s2
- train_armD_s3
- train_armD_s4
- train_armE1_s0
- train_armE1_s1
- train_armE1_s2
- train_armE1_s3
- train_armE1_s4
- train_armE2_s0
- train_armE2_s1
- train_armE2_s2
- train_armE2_s3
- train_armE2_s4
- train_K1_s0
- train_K1_s1
- train_K1_s2
- train_K1_s3
- train_K1_s4
- train_armB_s0
- train_armB_s1
- train_armB_s2
- train_armB_s3
- train_armB_s4
- gen_BC
- train_armBC_s0
- train_armBC_s1
- train_armBC_s2
- train_armBC_s3
- train_armBC_s4
- gen_K3
- train_K3_s0
- train_K3_s1
- train_K3_s2
- train_K3_s3
- train_K3_s4

Totals left: 37 training runs, 3 generation jobs
(gen_B, gen_BC, gen_K3).

## Generation caches (summaries are cached per node; a restarted job skips existing lines)
- metapath_summaries/13769e2ed26ec021: 52448 lines  (metapath_summaries, template=metapath_summary)
- metapath_summaries/3e9ca191ca807f0b: 29978 lines  (metapath_summaries, template=metapath_summary)
- metapath_summaries/b19e6c18ffa0f14f: 119829 lines  (metapath_summaries, template=metapath_summary)
- node_summaries/242d1150cee3f44f: 10023 lines  (node_summaries, template=node_residual)
- node_summaries/6d167c62839cd689: 35747 lines  (node_summaries, template=node_summary)
- node_summaries/b4e5fae66e519bf7: 10023 lines  (node_summaries, template=node_discriminative)
- node_summaries/ed182ac19c83ee3e: 35747 lines  (node_summaries, template=node_discriminative)

## Result files already produced (results/raw/*/metrics.json)
- dgp_amazonvideo_seed0 seed 0: F1 6570.31  AUROC 7512.67  AUPRC 3278.43
- dgp_amazonvideo_seed0 seed 1: F1 6490.37  AUROC 7554.25  AUPRC 3246.83
- dgp_amazonvideo_seed0 seed 2: F1 6426.44  AUROC 7471.67  AUPRC 3170.44
- dgp_amazonvideo_seed0 seed 3: F1 6545.77  AUROC 7432.59  AUPRC 3148.98
- dgp_amazonvideo_seed0 seed 4: F1 6549.02  AUROC 7564.94  AUPRC 3320.50
- ext_armC seed 0: F1 6551.65  AUROC 7466.84  AUPRC 3217.49
- ext_armC seed 1: F1 6549.10  AUROC 7485.66  AUPRC 3233.53
- ext_armC seed 2: F1 6516.68  AUROC 7462.68  AUPRC 3150.61
- parity_dgp seed 0: F1 6542.58  AUROC 7514.70  AUPRC 3253.76

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
