"""Stage 1: DeBERTaV3 embeddings + MDK Top-M trimming for every metapath, cached.

    python scripts/build_mdk.py --dataset amazonvideo --device cuda:0
    python scripts/build_mdk.py --dataset amazonvideo --device cpu --set dgp.K=3 dgp.M=8

CPU is fine for the graph work; DeBERTa embedding of all nodes is much faster on a GPU.
"""

import json

from _common import base_parser, experiment_config

from dgp_repro.experiment import build_prompts
from dgp_repro.utils.device import resolve_device


def main():
    args = base_parser(__doc__).parse_args()
    cfg = experiment_config(args)
    device = resolve_device(args.device)  # graph work is CPU-friendly; no GPU warning here
    prompt_set, graph, split = build_prompts(cfg, device, until="mdk")
    print(json.dumps(prompt_set.trim_stats, indent=2))


if __name__ == "__main__":
    main()
