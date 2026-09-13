"""Stage 2: node + metapath summaries with the frozen Qwen3-8B, then prompts and token accounting.

    python scripts/generate_summaries.py --dataset amazonvideo --device cuda:0
    python scripts/generate_summaries.py --dataset amazonvideo --until node_summaries --device cuda:0

Resumable: summaries are appended to datasets/cache/*_summaries/<key>/summaries.jsonl as they finish,
so an interrupted run continues where it stopped.
Writes results/token_usage/token_usage_<experiment>.{csv,json} and the prompts to datasets/cache/prompts/.
"""

import json

from _common import ROOT, base_parser, device_for, experiment_config

from dgp_repro.evaluation.token_usage import summarize_token_usage, write_token_usage
from dgp_repro.experiment import build_prompts, config_hash


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--until", default="prompts", choices=["mdk", "node_summaries", "prompts"])
    args = parser.parse_args()
    cfg = experiment_config(args)
    device = device_for(args, cfg)
    prompt_set, graph, split = build_prompts(cfg, device, until=args.until)
    if args.until != "prompts":
        print(f"stopped after {args.until}; caches are filled")
        return
    experiment = cfg.get("experiment", cfg["dataset"]["name"])
    summary = summarize_token_usage(prompt_set.token_rows)
    write_token_usage(prompt_set.token_rows, summary, ROOT / "results" / "token_usage", tag=experiment)
    out = ROOT / "datasets" / "cache" / "prompts" / f"{experiment}_{config_hash(cfg)}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for node, prompt in prompt_set.prompts.items():
            f.write(json.dumps({"node": node, "prompt": prompt}, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"prompts written to {out}")


if __name__ == "__main__":
    main()
