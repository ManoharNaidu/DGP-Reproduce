"""Disc/resid study: checks on the generated texts, before and independent of DGP training (CPU only).

    python scripts/dr_text_diagnostics.py

For every generated text type (FLAG discriminative, no-verdict discriminative, residual) over the 10,023 targets:
format (empty, multi-line, generated length, outputs that hit the generation cap), whether it states a class, how much
it overlaps with the raw review and with the other text, and a probe: TF-IDF + logistic regression, 5-fold
cross-validation on train + val only, AUROC / AUPRC. Test labels are never read. A residual text that does its job
should probe near chance (AUROC ~ 50); a discriminative one should carry signal.
Writes results/disc_resid/text_diagnostics.{json,md}.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dgp_repro.config import load_experiment  # noqa: E402
from dgp_repro.data import load_graph, load_split  # noqa: E402

CACHE = ROOT / "datasets" / "cache" / "node_summaries"
CAP = 4 * 40 + 16          # max_new_tokens for budget 40 (summarizer: 4 * B + 16)
TEXTS = {"disc_flag": "dr_discriminative", "disc_noverdict": "dr_discriminative_nv", "residual": "dr_residual"}
CLASS_WORDS = re.compile(r"\b(unhelpful|not helpful|helpful|useful|useless|not useful)\b", re.I)
SAYS_UNHELPFUL = re.compile(r"\b(unhelpful|not helpful|not useful|useless|less helpful)\b", re.I)


def load_store(template: str) -> tuple[dict, dict]:
    for meta in CACHE.glob("*/metadata.json"):
        m = json.loads(meta.read_text())
        if m.get("template") == template and m.get("truncate_chars") == 1200 and m.get("role") == "target":
            text, extra = {}, {}
            for line in open(meta.parent / "summaries.jsonl", encoding="utf-8"):
                if line.strip():
                    row = json.loads(line)
                    text[int(row["key"])] = row["text"]
                    extra[int(row["key"])] = row.get("generated_tokens")
            return text, extra
    raise SystemExit(f"no cache for template {template}")


def words(s: str) -> set:
    return set(re.findall(r"[a-z']+", s.lower()))


def jaccard(a: str, b: str) -> float:
    x, y = words(a), words(b)
    return len(x & y) / len(x | y) if x | y else 0.0


def probe(docs: list[str], y: np.ndarray) -> dict:
    aucs, aps = [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(docs, y):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        xtr = vec.fit_transform([docs[i] for i in tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(xtr, y[tr])
        p = clf.predict_proba(vec.transform([docs[i] for i in te]))[:, 1]
        aucs.append(roc_auc_score(y[te], p) * 100)
        aps.append(average_precision_score(y[te], p) * 100)
    return {"auroc": round(float(np.mean(aucs)), 2), "auroc_std": round(float(np.std(aucs)), 2),
            "auprc": round(float(np.mean(aps)), 2), "auprc_std": round(float(np.std(aps)), 2)}


def main():
    cfg = load_experiment(ROOT / "configs/experiments/disc_resid/R0_baseline.yaml")
    graph = load_graph(cfg["dataset"])
    split = load_split(cfg["dataset"], graph)
    targets = [int(v) for v in list(split.train) + list(split.val) + list(split.test)]
    labelled = [int(v) for v in list(split.train) + list(split.val)]          # never test
    y = graph.labels[labelled].astype(int)
    raw = {v: graph.texts[v][:1200] for v in targets}

    gen = {k: load_store(t) for k, t in TEXTS.items()}
    out = {"n_targets": len(targets), "n_probe": len(labelled), "probe_prevalence": round(float(y.mean()) * 100, 2),
           "texts": {}}
    for k, (text, toks) in gen.items():
        outs = [text.get(v, "") for v in targets]
        lens = [toks.get(v) for v in targets if toks.get(v) is not None]
        stats = {
            "coverage": sum(v in text for v in targets),
            "empty": sum(not o.strip() for o in outs),
            "multi_line": sum(len([ln for ln in o.splitlines() if ln.strip()]) > 1 for o in outs),
            "hit_cap": sum(t >= CAP for t in lens),
            "tokens_mean": round(float(np.mean(lens)), 1), "tokens_p95": float(np.percentile(lens, 95)),
            "mentions_class_word": sum(bool(CLASS_WORDS.search(o)) for o in outs),
            "jaccard_with_raw": round(float(np.mean([jaccard(text.get(v, ""), raw[v]) for v in targets])), 3),
        }
        if k.startswith("disc"):
            says = np.array([bool(SAYS_UNHELPFUL.search(text.get(v, ""))) for v in labelled])
            other = np.array([bool(CLASS_WORDS.search(text.get(v, ""))) for v in labelled]) & ~says
            stats["verdict_on_train_val"] = {
                "says_unhelpful_given_fraud": round(float(says[y == 1].mean()) * 100, 1),
                "says_unhelpful_given_benign": round(float(says[y == 0].mean()) * 100, 1),
                "says_helpful_given_fraud": round(float(other[y == 1].mean()) * 100, 1),
                "says_helpful_given_benign": round(float(other[y == 0].mean()) * 100, 1)}
            stats["jaccard_with_residual"] = round(float(np.mean(
                [jaccard(text.get(v, ""), gen["residual"][0].get(v, "")) for v in targets])), 3)
        out["texts"][k] = stats

    docs = {"raw": [raw[v] for v in labelled]}
    for k, (text, _) in gen.items():
        docs[k] = [text.get(v, "") for v in labelled]
    docs["raw+disc_flag"] = [a + " " + b for a, b in zip(docs["raw"], docs["disc_flag"])]
    docs["raw+disc_noverdict"] = [a + " " + b for a, b in zip(docs["raw"], docs["disc_noverdict"])]
    out["probe"] = {k: probe(d, y) for k, d in docs.items()}

    dst = ROOT / "results" / "disc_resid"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "text_diagnostics.json").write_text(json.dumps(out, indent=2))
    lines = ["# Generated-text diagnostics (disc/resid study)", "",
             f"{out['n_targets']} targets; probe on train + val only ({out['n_probe']} reviews, "
             f"{out['probe_prevalence']}% fraud), TF-IDF 1-2 grams + logistic regression, 5-fold CV.", "",
             "| text | coverage | empty | multi-line | hit cap | tokens mean / p95 | mentions class word | Jaccard w/ raw |",
             "|---|---|---|---|---|---|---|---|"]
    for k, s in out["texts"].items():
        lines.append(f"| {k} | {s['coverage']} | {s['empty']} | {s['multi_line']} | {s['hit_cap']} | "
                     f"{s['tokens_mean']} / {s['tokens_p95']:.0f} | {s['mentions_class_word']} | {s['jaccard_with_raw']} |")
    lines += ["", "| probe input | AUROC | AUPRC |", "|---|---|---|"]
    for k, p in out["probe"].items():
        lines.append(f"| {k} | {p['auroc']} ± {p['auroc_std']} | {p['auprc']} ± {p['auprc_std']} |")
    for k in ("disc_flag", "disc_noverdict"):
        v = out["texts"][k]["verdict_on_train_val"]
        lines.append(f"\n{k}: says 'unhelpful' for {v['says_unhelpful_given_fraud']}% of fraud vs "
                     f"{v['says_unhelpful_given_benign']}% of benign train+val reviews; 'helpful' for "
                     f"{v['says_helpful_given_fraud']}% vs {v['says_helpful_given_benign']}%.")
    (dst / "text_diagnostics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
