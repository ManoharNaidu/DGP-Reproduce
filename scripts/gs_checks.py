"""Checks for the Gemma-swap / structured zero-shot study (CPU; read caches, write results/gemma_swap/checks/).

    python scripts/gs_checks.py mdk               # Gemma run's MDK neighbours == the dr study's (Qwen) neighbours
    python scripts/gs_checks.py text              # Gemma text caches: coverage, empties, length, class words; vs Qwen
    python scripts/gs_checks.py layout --exp 2    # zs_S* prompts carry exactly the fields of dr_R* prompts
    python scripts/gs_checks.py layout --exp 3    # zs_GS* prompts carry exactly the fields of gs_G* prompts

Exit code 1 when a check fails (the queue then stops dependent jobs); the JSON/MD files say why.
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "datasets" / "cache"
OUT = ROOT / "results" / "gemma_swap" / "checks"
QWEN, GEMMA = "Qwen/Qwen3-8B", "google/gemma-2-9b-it"
QWEN_MDK = "30447a444d0aa6e2"                     # the dr study's MDK cache (K=2, M=4, Qwen token counter)
N_TARGETS, N_NEIGHBOURS, N_PATHS = 10023, 35747, 119829
CLASS_WORDS = re.compile(r"\b(unhelpful|not helpful|helpful|useful|useless|not useful)\b", re.I)  # dr_text_diagnostics.py
ARMS = ["0", "1", "1b", "2", "2b"]


def meta(d):
    return json.loads((Path(d) / "metadata.json").read_text())


def write(name, payload, md_lines):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str))
    (OUT / f"{name}.md").write_text("\n".join(md_lines) + "\n")
    print("\n".join(md_lines))


# ---------------------------------------------------------------- MDK identity
def check_mdk():
    gem = [Path(m).parent for m in glob.glob(str(CACHE / "mdk" / "*" / "metadata.json"))
           if "gemma" in str(meta(Path(m).parent).get("counter", "")).lower() and meta(Path(m).parent).get("K") == 2]
    if len(gem) != 1:
        write("mdk_identity", {"ok": False, "error": f"expected 1 Gemma K=2 MDK cache, found {len(gem)}"}, ["MDK check: FAILED (cache not found)"])
        return 1
    g, q = gem[0], CACHE / "mdk" / QWEN_MDK
    mg, mq = meta(g), meta(q)
    same_settings = {k: mg.get(k) == mq.get(k) for k in mq if k not in ("counter", "cache_key", "git_commit", "created", "stage")}
    paths, rows = sorted(p.name for p in q.glob("*.npz")), []
    for name in paths:
        a, b = np.load(g / name), np.load(q / name)
        same = {k: bool(np.array_equal(a[k], b[k])) for k in ("targets", "offsets", "neighbors", "sizes")}
        rows.append({"metapath": name[:-4], **same, "raw_tokens_equal": bool(np.array_equal(a["raw_tokens"], b["raw_tokens"]))})
    ok = all(same_settings.values()) and len(rows) == len(list(g.glob("*.npz"))) == 12 and all(
        r[k] for r in rows for k in ("targets", "offsets", "neighbors", "sizes"))
    payload = {"ok": ok, "gemma_cache": g.name, "qwen_cache": q.name, "settings_equal": same_settings, "metapaths": rows,
               "note": "raw_tokens differ by design: they count review tokens with each model's tokenizer"}
    md = ["# MDK identity: Gemma run vs dr study (Qwen)", "",
          f"Result: **{'IDENTICAL' if ok else 'DIFFERENT'}** neighbour selection ({len(rows)} metapaths, caches "
          f"`{g.name}` vs `{q.name}`).", "",
          "| metapath | targets | neighbours | neighbourhood sizes | raw-token counts |", "|---|---|---|---|---|"]
    md += [f"| {r['metapath']} | {r['targets'] and r['offsets']} | {r['neighbors']} | {r['sizes']} | "
           f"{'same' if r['raw_tokens_equal'] else 'differ (tokenizer)'} |" for r in rows]
    write("mdk_identity", payload, md)
    return 0 if ok else 1


# ---------------------------------------------------------------- text caches
def stores(stage, model):
    out = {}
    for m in glob.glob(str(CACHE / stage / "*" / "metadata.json")):
        d = Path(m).parent
        x = meta(d)
        if x.get("summarizer") == model and (d / "summaries.jsonl").exists():
            out[d] = x
    return out


def scan(path, cap):
    rows = {}
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        rows[r["key"]] = r
    texts = [str(r["text"]) for r in rows.values()]
    lens = [r.get("generated_tokens") for r in rows.values() if isinstance(r.get("generated_tokens"), (int, float))]
    n = len(texts) or 1
    return rows, {"count": len(texts), "empty_pct": round(100 * sum(not t.strip() for t in texts) / n, 2),
                  "multi_line_pct": round(100 * sum(len([x for x in t.splitlines() if x.strip()]) > 1 for t in texts) / n, 2),
                  "tokens_mean": round(float(np.mean(lens)), 1) if lens else None,
                  "tokens_p95": round(float(np.percentile(lens, 95)), 1) if lens else None,
                  "hit_cap_pct": round(100 * sum(x >= cap for x in lens) / n, 2) if lens else None,
                  "class_word_pct": round(100 * sum(bool(CLASS_WORDS.search(t)) for t in texts) / n, 2)}


def check_text():
    kinds = [  # label, stage, selector, expected count, budget
        ("neighbour summaries (node_summary, B=10)", "node_summaries", lambda x: x.get("template") == "node_summary" and x.get("B_node") == 10, N_NEIGHBOURS, 10),
        ("metapath summaries (B=10)", "metapath_summaries", lambda x: x.get("template") == "metapath_summary" and x.get("node_template") == "node_summary" and x.get("B_meta") == 10 and not x.get("node_text"), N_PATHS, 10),
        ("D line · FLAG prompt (dr_discriminative)", "node_summaries", lambda x: x.get("template") == "dr_discriminative" and x.get("role") == "target", N_TARGETS, 40),
        ("D line · no-verdict (dr_discriminative_nv)", "node_summaries", lambda x: x.get("template") == "dr_discriminative_nv" and x.get("role") == "target", N_TARGETS, 40),
        ("R line (dr_residual)", "node_summaries", lambda x: x.get("template") == "dr_residual" and x.get("role") == "target", N_TARGETS, 40),
    ]
    payload, ok = {"kinds": []}, True
    md = ["# Text caches: Gemma-2-9B vs Qwen3-8B (same prompts, same inputs)", "",
          "class words = helpful / unhelpful / useful / useless / not helpful / not useful (as in dr_text_diagnostics.py). "
          "hit cap = generation stopped at max_new_tokens (4 x budget + 16).", "",
          "| text | model | lines | empty % | multi-line % | tokens mean / p95 | hit cap % | class words % |",
          "|---|---|---|---|---|---|---|---|"]
    for label, stage, sel, total, budget in kinds:
        entry = {"kind": label, "expected": total}
        samples = {}
        for model, tag in ((GEMMA, "gemma"), (QWEN, "qwen")):
            hits = [d for d, x in stores(stage, model).items() if sel(x)]
            if stage == "metapath_summaries" and model == QWEN:
                hits = [d for d in hits if meta(d).get("mdk_cache") == QWEN_MDK]
            if len(hits) != 1:
                entry[tag] = {"error": f"{len(hits)} matching caches"}
                ok &= tag != "gemma"
                continue
            rows, st = scan(hits[0] / "summaries.jsonl", 4 * budget + 16)
            entry[tag] = {**st, "cache": hits[0].name}
            samples[tag] = rows
            if tag == "gemma" and st["count"] < total:
                ok = False
            md.append(f"| {label} | {tag} | {st['count']:,} / {total:,} | {st['empty_pct']} | {st['multi_line_pct']} | "
                      f"{st['tokens_mean']} / {st['tokens_p95']} | {st['hit_cap_pct']} | {st['class_word_pct']} |")
        common = sorted(set(samples.get("gemma", {})) & set(samples.get("qwen", {})))
        pick = [common[i] for i in np.random.default_rng(0).choice(len(common), min(3, len(common)), replace=False)] if common else []
        entry["samples"] = [{"key": k, "gemma": samples["gemma"][k]["text"], "qwen": samples["qwen"][k]["text"]} for k in pick]
        payload["kinds"].append(entry)
    payload["ok"] = ok
    md += ["", "## Samples (same input, both models)", ""]
    for e in payload["kinds"]:
        for s in e["samples"]:
            md += [f"- **{e['kind']}**, key `{s['key']}`", f"  - Gemma: {s['gemma'][:300]!r}", f"  - Qwen: {s['qwen'][:300]!r}"]
    write("text_checks", payload, md)
    return 0 if ok else 1


# ---------------------------------------------------------------- prompt layout (zero-shot vs original)
def prompt_file(experiment):
    files = [p for p in (CACHE / "prompts").glob(f"{experiment}_*.jsonl") if re.fullmatch(re.escape(experiment) + r"_[0-9a-f]+\.jsonl", p.name)]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def fields_original(p):
    head, rest = p.split("\nMetapaths:\n", 1)
    blocks = rest.rsplit("\nQuestion: ", 1)[0]
    target, _, signals = head.partition("\nKey signals: ")
    return target.removeprefix("Target: "), (signals or None), blocks


def check_layout(exp):
    pairs = {"2": [(f"zs_S{a}", f"dr_R{a}") for a in ARMS], "3": [(f"zs_GS{a}", f"gs_G{a}") for a in ARMS]}[exp]
    payload, ok = {"exp": exp, "arms": []}, True
    md = [f"# Prompt layout check, Exp {exp}: structured zero-shot prompts vs the original-layout prompts", "",
          "Every field of the original prompt (target review, Key signals line, metapath lines) must appear verbatim in "
          "the matching tag of the zero-shot prompt, for every target.", "",
          "| arm | reference | targets | identical fields | mismatches | mean chars original / zero-shot |", "|---|---|---|---|---|---|"]
    for new, ref in pairs:
        fn, fr = prompt_file(new), prompt_file(ref)
        if not fn or not fr:
            payload["arms"].append({"arm": new, "error": "prompt file missing", "new": str(fn), "ref": str(fr)})
            ok = False
            md.append(f"| {new} | {ref} | — | — | prompt file missing | — |")
            continue
        P = {json.loads(l)["node"]: json.loads(l)["prompt"] for l in open(fn, encoding="utf-8")}
        Q = {json.loads(l)["node"]: json.loads(l)["prompt"] for l in open(fr, encoding="utf-8")}
        bad, good = [], 0
        for v, q in Q.items():
            p = P.get(v)
            target, signals, blocks = fields_original(q)
            fine = (p is not None and f"<target_review>\n{target}\n</target_review>" in p and
                    (signals is None or f"<key_signals>\n{signals}\n</key_signals>" in p) and
                    f"{blocks}\n</neighbourhood>" in p and
                    not any(ph in p for ph in ("{target_text}", "{target_disc}", "{metapath_blocks}", "{question}")))
            if fine:
                good += 1
            else:
                bad.append(v)
        arm_ok = not bad and len(P) == len(Q) == N_TARGETS
        ok &= arm_ok
        payload["arms"].append({"arm": new, "reference": ref, "new_file": fn.name, "ref_file": fr.name, "targets": len(Q),
                                "identical": good, "mismatches": len(bad), "first_mismatches": bad[:10], "ok": arm_ok})
        md.append(f"| {new} | {ref} | {len(Q):,} | {good:,} | {len(bad)} | "
                  f"{np.mean([len(x) for x in Q.values()]):.0f} / {np.mean([len(x) for x in P.values()]):.0f} |")
    payload["ok"] = ok
    md += ["", f"Result: **{'PASS' if ok else 'FAIL'}**"]
    write(f"layout_exp{exp}", payload, md)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=["mdk", "text", "layout"])
    ap.add_argument("--exp", choices=["2", "3"])
    a = ap.parse_args()
    if a.check == "layout" and not a.exp:
        ap.error("layout needs --exp")
    sys.exit({"mdk": check_mdk, "text": check_text}.get(a.check, lambda: check_layout(a.exp))())


if __name__ == "__main__":
    main()
