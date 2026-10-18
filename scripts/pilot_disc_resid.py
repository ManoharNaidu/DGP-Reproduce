"""Pilot for the disc/resid study: generate FLAG-style discriminative + residual lines for a small sample of
train reviews, into a throw-away cache, so the prompts can be checked before the full run.

    python scripts/pilot_disc_resid.py --n 200 --device cuda:0

Sample: n/2 fraud + n/2 benign train targets (seeded). Labels are used only to report the sample, never given
to the LLM. Output: results/disc_resid/pilot/pilot.jsonl and a format summary on stdout.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dgp_repro.config import load_experiment  # noqa: E402
from dgp_repro.data import load_graph, load_split  # noqa: E402
from dgp_repro.summarization.base import load_template, make_summarizer  # noqa: E402
from dgp_repro.summarization.pipeline import node_summaries  # noqa: E402
from dgp_repro.utils.cache import TextStore  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--config", default="configs/experiments/disc_resid/R2_disc_resid_aux.yaml")
    ap.add_argument("--out", default="pilot", help="sub-folder of results/disc_resid/ (one per prompt version)")
    args = ap.parse_args()

    cfg = load_experiment(ROOT / args.config)
    graph = load_graph(cfg["dataset"])
    split = load_split(cfg["dataset"], graph)
    train = np.asarray(split.train)
    y = graph.labels[train]
    rng = np.random.default_rng(0)
    pick = np.concatenate([rng.choice(train[y == 1], args.n // 2, replace=False),
                           rng.choice(train[y == 0], args.n // 2, replace=False)]).tolist()

    out_dir = ROOT / "results" / "disc_resid" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    summarizer = make_summarizer(cfg["summarizer"], args.device)
    texts, timing = {}, {}
    for kind, spec in cfg["target_text"].items():
        cut = spec.get("truncate_chars")
        source = [t[:cut] for t in graph.texts] if cut else graph.texts
        store = TextStore(out_dir / f"{kind}.jsonl")
        t0 = time.time()
        texts[kind] = node_summaries(source, pick, summarizer, load_template(spec["template"]), spec["budget"], store)
        timing[kind] = round(time.time() - t0, 1)
        texts[kind + "_tokens"] = {int(k): v.get("generated_tokens") for k, v in store.extra.items()}

    rows = []
    with open(out_dir / "pilot.jsonl", "w", encoding="utf-8") as f:
        for u in pick:
            row = {"node": u, "label": int(graph.labels[u]), "raw": graph.texts[u][:1200],
                   "disc": texts["disc"][u], "residual": texts["residual"][u],
                   "disc_tokens": texts["disc_tokens"].get(u), "residual_tokens": texts["residual_tokens"].get(u)}
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    for kind in ("disc", "residual"):
        outs = [r[kind] for r in rows]
        multi = sum(len([ln for ln in o.splitlines() if ln.strip()]) > 1 for o in outs)
        empty = sum(not o.strip() for o in outs)
        toks = [r[f"{kind}_tokens"] for r in rows if r[f"{kind}_tokens"] is not None]
        print(f"{kind}: n={len(outs)} empty={empty} multi-line={multi} tokens mean={np.mean(toks):.1f} "
              f"max={max(toks)} capped(176)={sum(t >= 176 for t in toks)} time={timing[kind]}s")


if __name__ == "__main__":
    main()
