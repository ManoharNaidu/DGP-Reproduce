"""One command for a dataset: prepare -> DGP main (5 seeds) -> optional studies -> tables -> comparison.

    python scripts/reproduce.py --dataset amazonvideo --device cuda:0 --seeds 0 1 2 3 4
    python scripts/reproduce.py --dataset amazonvideo --device cuda:0 --with-llm-baseline --with-ablation --with-budget --with-task-aware
    python scripts/reproduce.py --dataset synthetic --device cpu --mode smoke           # CPU_DEBUG, a few seconds
    python scripts/reproduce.py --dataset amazonvideo --device cpu --mode smoke --seeds 0  # CPU_DEBUG on the real graph

Every stage is resumable: cached artefacts and finished seeds are reused.
"""

import subprocess
import sys

from _common import ROOT, base_parser

PY = sys.executable


def run(*args):
    cmd = [PY, *map(str, args)]
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--with-llm-baseline", action="store_true", help="Qwen3-8B target-only baseline")
    parser.add_argument("--with-ablation", action="store_true")
    parser.add_argument("--with-budget", action="store_true")
    parser.add_argument("--with-task-aware", action="store_true")
    args = parser.parse_args()

    if args.mode == "smoke" and args.dataset == "synthetic":
        run("scripts/smoke_test.py")
    common = ["--dataset", args.dataset, "--mode", args.mode, "--device", args.device]
    if args.seeds:
        common += ["--seeds", *args.seeds]
    if args.set:
        common += ["--set", *args.set]

    run("scripts/prepare_data.py", "--dataset", args.dataset)
    run("scripts/train.py", *common)
    if args.with_llm_baseline:
        run("scripts/train.py", *common, "--overlay", "configs/experiments/baselines/llm_target_only.yaml",
            "--experiment", f"llm_{args.dataset}" + ("_debug" if args.mode == "smoke" else ""))
    if args.with_ablation:
        run("scripts/run_ablation.py", *common)
    if args.with_budget:
        run("scripts/run_sensitivity.py", "--study", "budget", *common)
    if args.with_task_aware:
        run("scripts/run_sensitivity.py", "--study", "task_aware", *common)
    run("scripts/evaluate.py")
    if args.mode == "gpu":
        run("scripts/compare_with_paper.py")


if __name__ == "__main__":
    main()
