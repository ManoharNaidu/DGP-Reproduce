"""Build mock data trees for testing the dashboard without touching the real study.

    python tools/dr-dashboard/v2/mock/make_mock.py <out-dir>

Creates <out-dir>/full (every state at once: completed + failed generation lines, done / running / failed / waiting
training cells, diagnostics, insights, an over-temperature + high-VRAM GPU) and <out-dir>/empty (no files at all, to
check that every field degrades to "—"). Values are synthetic and are never read by anything but the dashboard.
"""
import json
import os
import random
import sys
import time

N = 10023
rnd = random.Random(0)


def w(path, obj=None, text=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text if text is not None else json.dumps(obj, indent=1))


def iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t))


def full(root):
    now = time.time()
    t0 = now - 14 * 3600
    stores = [("aaaa", "dr_discriminative", N, t0, t0 + 1400), ("bbbb", "dr_residual", N, t0 + 1400, t0 + 3000),
              ("cccc", "dr_discriminative_nv", 6200, t0 + 3000, t0 + 4200)]
    for key, tmpl, n, start, end in stores:
        d = os.path.join(root, "datasets/cache/node_summaries", key)
        w(os.path.join(d, "metadata.json"), {"template": tmpl, "truncate_chars": 1200, "role": "target", "created": iso(start)})
        lines = []
        for i in range(n):
            text = "" if i in (5, 77) else (f"Mock {tmpl} line {i}: the review mentions plot, pacing and a personal opinion.\nsecond line"
                                             if i % 997 == 0 else f"Mock {tmpl} line {i}: the review mentions plot, pacing and a personal opinion.")
            lines.append(json.dumps({"key": str(i), "text": text, "generated_tokens": 176 if i % 1500 == 0 else rnd.randint(14, 48)}))
        w(os.path.join(d, "summaries.jsonl"), text="\n".join(lines) + "\n{\"key\": \"half-writ")
        os.utime(os.path.join(d, "summaries.jsonl"), (end, end))
    w(os.path.join(root, "logs/disc_resid/gen_all.log"), text=f"[{iso(t0)}] generate dr_R2\ntarget disc: 10023 targets, 0 cached\n"
      f"[{iso(t0 + 3000)}] generate dr_R2b\nTraceback (most recent call last):\nRuntimeError: CUDA out of memory (mock)\n")
    jobs = [("dr_R0", 0, "done"), ("dr_R1", 0, "done"), ("dr_R1b", 0, "done"), ("dr_R2", 0, "done"), ("dr_R2b", 0, "failed"),
            ("dr_R0", 1, "done"), ("dr_R1", 1, "running")]
    q, t = [], t0 + 3100
    for exp, s, st in jobs:
        job = f"{exp}_s{s}"
        q.append(f"[{iso(t)}] start {job}")
        run = os.path.join(root, f"results/raw/{exp}_seed{s}_mockhash")
        if st == "done":
            secs = 7600 if exp.startswith("dr_R2") else 3900
            t += secs
            q.append(f"[{iso(t)}] done  {job} ({secs // 60} min)")
            w(os.path.join(root, "results/disc_resid/done", job), text=str(secs))
            base = {"dr_R0": 65.2, "dr_R1": 65.6, "dr_R1b": 65.4, "dr_R2": 64.8}[exp]
            w(os.path.join(run, "metrics.json"), {"seed": s, "best_epoch": 2, "test": {
                "macro_f1": base + rnd.uniform(-.6, .6), "auroc": 75 + rnd.uniform(-.8, .8) + (base - 65.2),
                "auprc": 32.4 + rnd.uniform(-.7, .7) + (base - 65.2), "threshold": 0.25}})
        elif st == "failed":
            t += 600
            q.append(f"[{iso(t)}] FAILED {job} (see logs/disc_resid/train_{job}.log)")
            w(os.path.join(root, f"logs/disc_resid/train_{job}.log"), text="Traceback (most recent call last):\n  ...\ntorch.OutOfMemoryError: CUDA out of memory (mock)\n")
        else:
            w(os.path.join(run, "training_log.jsonl"), text="\n".join(json.dumps(
                {"epoch": e, "train_loss": 0.41 - .05 * e, "val_loss": 0.38 - .02 * e, "val_auroc": 74.1 + e, "val_auprc": 31.0 + e}) for e in (1, 2)) + "\n")
    w(os.path.join(root, "logs/disc_resid/queue.log"), text="\n".join(q) + "\n")
    for s in range(5):
        w(os.path.join(root, f"results/raw/dgp_amazonvideo_seed0_seed{s}_f12e6c750d/metrics.json"),
          {"seed": s, "test": {"macro_f1": 65.16 + rnd.uniform(-.6, .6), "auroc": 75.07 + rnd.uniform(-.5, .5), "auprc": 32.33 + rnd.uniform(-.6, .6)}})
    w(os.path.join(root, "results/disc_resid/text_diagnostics.json"), {
        "n_targets": N, "n_probe": 2598, "probe_prevalence": 11.8,
        "texts": {k: {"coverage": N, "empty": 0, "multi_line": 2, "hit_cap": 3, "tokens_mean": 28.1, "tokens_p95": 41,
                      "mentions_class_word": m, "jaccard_with_raw": 0.21,
                      **({"verdict_on_train_val": {"says_unhelpful_given_fraud": 51.0, "says_unhelpful_given_benign": 19.5,
                                                   "says_helpful_given_fraud": 47.0, "says_helpful_given_benign": 79.0}} if k != "residual" else {})}
                  for k, m in (("disc_flag", 9400), ("disc_noverdict", 410), ("residual", 0))},
        "probe": {"raw": {"auroc": 66.0, "auroc_std": 1.1, "auprc": 21.0, "auprc_std": 1.3},
                  "disc_flag": {"auroc": 64.2, "auroc_std": 1.0, "auprc": 20.1, "auprc_std": 1.2},
                  "disc_noverdict": {"auroc": 62.3, "auroc_std": 1.4, "auprc": 19.0, "auprc_std": 1.1},
                  "residual": {"auroc": 55.1, "auroc_std": 1.2, "auprc": 14.2, "auprc_std": 0.9}}})
    bins = [0, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 176]
    w(os.path.join(root, ".dashboard_state/cache/text_insights.json"), {
        "bins": bins, "n_fit": 2598, "fraud_pct": 11.8,
        "texts": {k: {"n": N, "length_hist": [rnd.randint(0, 3000) for _ in bins],
                      "tokens_fraud": [[f"fraud_tok_{i}", round(1.5 - i * .1, 3)] for i in range(15)],
                      "tokens_benign": [[f"benign_tok_{i}", round(-1.5 + i * .1, 3)] for i in range(15)]}
                  for k in ("disc_flag", "disc_noverdict", "residual")}})
    w(os.path.join(root, "mock_liveness.json"), {"dr_queue.sh": True, "dr_generate_all.sh": False, "dr_after_gen.sh": False})
    w(os.path.join(root, "mock_gpu.json"), [{"name": "NVIDIA A100-SXM4-80GB", "util": 97.0, "mem_used": 76500.0, "mem_total": 81920.0, "temp": 83.0}])
    hist = [[round(now - 3600 + i * 10), 60 + 35 * ((i // 20) % 2), 20000 + i * 150, 81920, 55 + i * 0.08] for i in range(360)]
    gen = {"disc_noverdict": [[round(now - 1800 + i * 30), 6200 - (60 - i) * 90] for i in range(61)]}
    w(os.path.join(root, ".dashboard_state/history.json"), {"gen": gen, "gpu": hist})


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "data")
    full(os.path.join(out, "full"))
    os.makedirs(os.path.join(out, "empty", "logs"), exist_ok=True)
    w(os.path.join(out, "empty", "mock_liveness.json"), {})
    print("mock trees written to", out)
