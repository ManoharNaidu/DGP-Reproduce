"""Dashboard-side text insights for the disc/resid study (CPU, a few seconds). Launched by server.py once all three
generated texts are complete; reads the generation caches and labels of train + val targets only (never test).

    python tools/dr-dashboard/v2/text_insights.py --root <project root> --out <file.json>

Per text type (disc_flag, disc_noverdict, residual): a histogram of generated lengths (tokens) over all targets, and the
15 n-grams most predictive of each class from TF-IDF (1-2 grams) + logistic regression fit on train + val.
"""
import argparse
import json
import os
import sys

import numpy as np

TEXTS = {"disc_flag": "dr_discriminative", "disc_noverdict": "dr_discriminative_nv", "residual": "dr_residual"}
BINS = [0, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 176, 10_000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(args.root, "src"))
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    from dgp_repro.config import load_experiment
    from dgp_repro.data import load_graph, load_split

    cfg = load_experiment(os.path.join(args.root, "configs/experiments/disc_resid/R0_baseline.yaml"))
    cfg["dataset"]["processed_dir"] = os.path.join(args.root, cfg["dataset"]["processed_dir"])
    graph = load_graph(cfg["dataset"])
    split = load_split(cfg["dataset"], graph)
    labelled = [int(v) for v in list(split.train) + list(split.val)]
    y = graph.labels[labelled].astype(int)

    out = {"bins": BINS[:-1], "texts": {}, "n_fit": len(labelled), "fraud_pct": round(float(y.mean()) * 100, 2)}
    cache = os.path.join(args.root, "datasets/cache/node_summaries")
    for key, tmpl in TEXTS.items():
        store = None
        for d in os.listdir(cache):
            try:
                m = json.load(open(os.path.join(cache, d, "metadata.json")))
            except (OSError, ValueError):
                continue
            if m.get("template") == tmpl and m.get("truncate_chars") == 1200:
                store = os.path.join(cache, d, "summaries.jsonl")
        if not store:
            continue
        text, lens = {}, []
        for line in open(store, encoding="utf-8"):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            text[int(row["key"])] = row["text"]
            if isinstance(row.get("generated_tokens"), (int, float)):
                lens.append(row["generated_tokens"])
        hist = np.histogram(lens, bins=BINS)[0].tolist() if lens else []
        docs = [text.get(v, "") for v in labelled]
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
        x = vec.fit_transform(docs)
        clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(x, y)
        names = np.array(vec.get_feature_names_out())
        coef = clf.coef_[0]
        order = np.argsort(coef)
        out["texts"][key] = {
            "length_hist": hist, "n": len(lens),
            "tokens_fraud": [[str(names[i]), round(float(coef[i]), 3)] for i in order[::-1][:15]],
            "tokens_benign": [[str(names[i]), round(float(coef[i]), 3)] for i in order[:15]]}
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, args.out)


if __name__ == "__main__":
    main()
