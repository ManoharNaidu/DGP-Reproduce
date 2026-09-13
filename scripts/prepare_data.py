"""Build a dataset graph, validate it against paper Table 1, and create the split.

    python scripts/prepare_data.py --dataset amazonvideo
    python scripts/prepare_data.py --dataset amazonvideo --probe      # compare relation definitions
    python scripts/prepare_data.py --dataset yelpchi                  # needs the email-gated release
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dgp_repro.config import REPO_ROOT, load_config  # noqa: E402
from dgp_repro.data import build_graph, load_split, processed_dir, validate_against_paper  # noqa: E402


def probe(cfg: dict) -> None:
    expected = cfg.get("expected", {}).get("edges")
    if cfg["name"] == "amazonvideo":
        from dgp_repro.data.amazon import download, probe_relation_definitions, read_reviews
        raw = download(cfg["source"]["url"], REPO_ROOT / cfg["source"]["raw_path"], cfg["source"].get("sha256"))
        counts = probe_relation_definitions(read_reviews(raw))
        combos = {"RSR = rating+week (default)": counts["RUR"] + counts["RPR"] + counts["RSR_rating_week"],
                  "RSR = product+rating+week (prose)": counts["RUR"] + counts["RPR"] + counts["RSR_product_rating_week"]}
    elif cfg["name"] == "yelpchi":
        from dgp_repro.data.yelpchi import probe_relation_definitions, read_canonical_reviews
        counts = probe_relation_definitions(read_canonical_reviews(REPO_ROOT / cfg["source"]["canonical_path"]))
        combos = {f"RSR {s} / RTR {t}": counts["RUR"] + counts[f"RSR_{s}"] + counts[f"RTR_{t}"]
                  for s in ("product", "global") for t in ("product", "global")}
    else:
        raise SystemExit(f"--probe is not defined for {cfg['name']}")
    print(json.dumps(counts, indent=2))
    for label, total in combos.items():
        flag = "MATCH" if total == expected else "no match"
        print(f"{label:40s} total={total:>12,}  paper={expected:,}  {flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", help="dataset config (default: configs/datasets/<dataset>.yaml)")
    parser.add_argument("--probe", action="store_true", help="only compare candidate relation definitions")
    parser.add_argument("--allow-mismatch", action="store_true",
                        help="save the graph even if it does not match paper Table 1")
    args = parser.parse_args()

    cfg = load_config(args.config or f"configs/datasets/{args.dataset}.yaml")
    if args.probe:
        probe(cfg)
        return

    graph = build_graph(cfg)
    print(json.dumps(graph.summary(), indent=2))
    mismatches = validate_against_paper(graph, cfg.get("expected"))
    if mismatches:
        print("MISMATCH against paper Table 1:\n  " + "\n  ".join(mismatches))
        if not args.allow_mismatch:
            raise SystemExit("refusing to save; rerun with --allow-mismatch to keep this graph")
    elif cfg.get("expected"):
        print("All Table 1 statistics match the paper exactly.")

    directory = processed_dir(cfg)
    graph.save(directory)
    split_path = directory / "split.json"
    if split_path.exists():
        split_path.unlink()  # a rebuilt graph gets a fresh split
    split = load_split(cfg, graph)
    print(f"saved to {directory}; split sizes {split.sizes()}")


if __name__ == "__main__":
    main()
