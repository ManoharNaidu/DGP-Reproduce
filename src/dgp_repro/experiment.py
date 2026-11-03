"""Shared experiment runner used by every script: prompts -> train -> evaluate -> record.

A run = one (config, seed). Its outputs go to results/raw/<run_id>/:
    run_manifest.json     git commit, versions, hardware, full config, runtime, peak memory
    metrics.json          val and test Macro-F1 / AUROC / AUPRC (percent)
    predictions.csv       node, split, label, p_fraud for val and test nodes
    training_log.jsonl    per-epoch train/val loss and val metrics

Modes:
    CPU_DEBUG         mock summarizer / hashing embedder / mock LLM (configs/experiments/smoke.yaml)
    GPU_REPRODUCTION  Qwen3-8B summarizer + DeBERTaV3 + Qwen3-8B LoRA classifier
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from dgp_repro.config import REPO_ROOT
from dgp_repro.data import load_graph, load_split
from dgp_repro.embeddings import make_embedder
from dgp_repro.evaluation.token_usage import summarize_token_usage, write_token_usage
from dgp_repro.metrics import aggregate_seeds, evaluate_split
from dgp_repro.pipeline import PromptSet, run_dgp_prompt_pipeline
from dgp_repro.summarization import make_summarizer
from dgp_repro.utils.provenance import peak_memory_gb, write_manifest
from dgp_repro.utils.seed import set_seed

RESULTS = REPO_ROOT / "results"


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


def make_token_counter(cfg: dict):
    """Returns (count_fn, name, tokenizer or None).

    Real runs count tokens with the Qwen3 tokenizer; debug runs count words and say so in the name.
    """
    if cfg.get("classifier", {}).get("backend") == "mock":
        return (lambda text: len(text.split())), "words-debug", None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["classifier"]["model_id"], revision=cfg["classifier"].get("revision"))
    return (lambda text: len(tok.encode(text, add_special_tokens=False))), f"{cfg['classifier']['model_id']}-tokenizer", tok


def build_prompts(cfg: dict, device: str = "cpu", log=print, until: str = "prompts") -> tuple[PromptSet, object, object]:
    graph = load_graph(cfg["dataset"])
    split = load_split(cfg["dataset"], graph)
    token_counter, counter_name, tokenizer = make_token_counter(cfg)
    truncate = None
    if tokenizer is not None and cfg["dgp"].get("max_target_tokens"):
        from dgp_repro.models.qwen_classifier import truncate_to_tokens
        limit = cfg["dgp"]["max_target_tokens"]
        truncate = lambda text: truncate_to_tokens(tokenizer, text, limit)  # noqa: E731

    comp = {"node_summary": True, "text_summary": True, "mdk": True, "path_summary": True,
            "numeric_summary": True, **cfg.get("components", {})}
    uses_neighbors = comp["text_summary"] or comp["numeric_summary"]
    # Build each model only if the requested stage uses it: stopping after MDK must never load Qwen3-8B.
    needs_summarizer = until != "mdk" and ((comp["text_summary"] and (comp["node_summary"] or comp["path_summary"]))
                                           or bool(cfg.get("target_text")))
    needs_embedder = uses_neighbors and comp["mdk"] and cfg.get("mdk", {}).get("features") != "raw"
    summarizer = make_summarizer(cfg["summarizer"], device) if needs_summarizer else None
    embedder = make_embedder(cfg["embedder"], device) if needs_embedder else None
    prompt_set = run_dgp_prompt_pipeline(graph, split, cfg, summarizer=summarizer, embedder=embedder,
                                         token_counter=token_counter, token_counter_name=counter_name,
                                         truncate_target=truncate, log=log, until=until)
    # Summaries are cached on disk, so release the frozen summarizer (~15 GiB for Qwen3-8B in bf16) and the
    # embedder before the classifier is loaded; otherwise two 8B models would sit in GPU memory together.
    del summarizer, embedder
    release_accelerator_memory()
    return prompt_set, graph, split


def release_accelerator_memory() -> None:
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def make_classifier(cfg: dict, device: str, seed: int):
    if cfg["classifier"].get("backend") == "mock":
        from dgp_repro.models.label_tokens import resolve_label_token_ids
        from dgp_repro.models.mock_llm import MockLLM
        model = MockLLM(seed=seed)
        return model, resolve_label_token_ids(model.tokenizer)
    from dgp_repro.models.qwen_classifier import QwenFraudClassifier
    model = QwenFraudClassifier(cfg["classifier"], device)
    return model, model.labels


def run_one_seed(cfg: dict, prompt_set: PromptSet, graph, split, seed: int, device: str, experiment: str) -> dict:
    from dgp_repro.training import predict, train

    run_id = f"{experiment}_seed{seed}_{config_hash(cfg)}"
    run_dir = RESULTS / "raw" / run_id
    if (run_dir / "metrics.json").exists():
        return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))  # resumable
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    set_seed(seed)
    model, label_tokens = make_classifier(cfg, device, seed)

    prompts = prompt_set.prompts
    ids = {name: getattr(split, name) for name in ("train", "val", "test")}
    text = {name: [prompts[int(v)] for v in ids[name]] for name in ids}
    labels = {name: graph.labels[ids[name]].astype(int) for name in ids}

    aux_prompts = None
    if cfg["training"].get("residual_aux", {}).get("enabled"):
        name = cfg["training"]["residual_aux"].get("template", "residual_only_prompt")
        template = (REPO_ROOT / "prompts" / "dgp" / f"{name}.txt").read_text(encoding="utf-8")
        question = cfg["dataset"].get("task", {}).get("question", "Is this fraud?")
        aux_prompts = [template.replace("{target_residual}", " ".join(prompt_set.extras[int(v)]["target_residual"].split()))
                       .replace("{question}", question) for v in ids["train"]]
    result = train(model, text["train"], labels["train"], text["val"], labels["val"], label_tokens,
                   cfg["training"], seed, log_path=run_dir / "training_log.jsonl", aux_prompts=aux_prompts)
    fa_finetune = None
    ft_cfg = (cfg.get("flag_arch") or {}).get("finetune") or {}
    if ft_cfg.get("enabled"):                     # FLAG-architecture study: FLAG's +FLAG* phase after normal training
        from dgp_repro.training.flag_finetune import flag_finetune
        usable_set = set(prompt_set.aux.get("usable", []))
        usable = [int(v) for v in ids["train"] if int(v) in usable_set]
        fa_finetune = flag_finetune(model, [prompt_set.aux["disc_only"][v] for v in usable],
                                    [prompt_set.aux["resid_only"][v] for v in usable], graph.labels[usable].astype(int),
                                    text["val"], labels["val"], label_tokens,
                                    {k: v for k, v in ft_cfg.items() if k != "enabled"}, seed,
                                    log_path=run_dir / "training_log.jsonl")
    batch = cfg["training"].get("eval_batch_size", 8)

    def write_predictions(path, probs):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["node", "split", "label", "p_fraud"])
            for name in ("val", "test"):
                for node, y, p in zip(ids[name], labels[name], probs[name]):
                    writer.writerow([int(node), name, int(y), f"{p:.6f}"])

    selection = None
    if result.rule_epochs:
        if fa_finetune:
            raise ValueError("training.selection_rules cannot be combined with flag_arch.finetune")
        # Selection-rule study (E19): one test pass per distinct epoch some rule kept; validation probabilities come
        # from the trainer's own pass at that epoch. Top-level val/test = the primary rule (result.best_epoch).
        per_epoch = {}
        for epoch in sorted(set(result.rule_epochs.values())):
            model.load_checkpoint_state(result.epoch_states[epoch])
            eprobs = {"val": result.epoch_val_probs[epoch], "test": predict(model, text["test"], label_tokens, batch)}
            per_epoch[epoch] = (eprobs, evaluate_split(labels["val"], eprobs["val"], labels["test"], eprobs["test"]))
            write_predictions(run_dir / f"predictions_epoch{epoch}.csv", eprobs)
        selection = {rule: {"epoch": e, **per_epoch[e][1]} for rule, e in result.rule_epochs.items()}
        probs = per_epoch[result.best_epoch][0]
        model.load_checkpoint_state(result.epoch_states[result.best_epoch])
        result.epoch_states.clear()
    else:
        probs = {name: predict(model, text[name], label_tokens, batch) for name in ("val", "test")}
    metrics = {"seed": seed, "best_epoch": result.best_epoch,
               **evaluate_split(labels["val"], probs["val"], labels["test"], probs["test"])}
    if selection is not None:
        metrics["selection"] = selection
        metrics["primary_rule"] = cfg["training"].get("primary_rule", cfg["training"]["selection_rules"][0])

    write_predictions(run_dir / "predictions.csv", probs)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_manifest(run_dir, cfg, {
        "run_id": run_id, "experiment": experiment, "seed": seed, "mode": cfg.get("mode", "GPU_REPRODUCTION"),
        "device": device, "runtime_seconds": round(time.time() - started, 1), "train_seconds": round(result.train_seconds, 1),
        "best_epoch": result.best_epoch, **({"rule_epochs": result.rule_epochs} if result.rule_epochs else {}),
        "label_tokens": label_tokens.report, "peak_memory": peak_memory_gb(),
        "prompt_metadata": prompt_set.metadata, "dataset_summary": graph.summary(),
        "split_sizes": split.sizes(), "split_seed": split.seed,
        **({"flag_finetune": {k: v for k, v in fa_finetune.items() if k != "history"}} if fa_finetune else {}),
    })
    if hasattr(model, "save_adapter") and cfg.get("save_checkpoints", True):
        model.save_adapter(REPO_ROOT / "checkpoints" / run_id)
    return metrics


def seed_metrics_on_disk(experiment: str, cfg_hash: str, raw_dir: Path | None = None) -> dict[int, dict]:
    """{seed: metrics.json} for every finished run results/raw/<experiment>_seed<k>_<cfg_hash>/."""
    out = {}
    for path in sorted((raw_dir or RESULTS / "raw").glob(f"{experiment}_seed*_{cfg_hash}/metrics.json")):
        seed_part = path.parent.name[len(experiment) + len("_seed"):-(len(cfg_hash) + 1)]
        if seed_part.isdigit():                                    # skips e.g. gs_G1b_* when experiment is gs_G1
            metrics = json.loads(path.read_text(encoding="utf-8"))
            out[int(seed_part)] = metrics
    return out


def run_experiment(cfg: dict, experiment: str, seeds: list[int], device: str, log=print) -> dict:
    """Build prompts once, train/evaluate every seed, aggregate test metrics as mean/std."""
    prompt_set, graph, split = build_prompts(cfg, device, log)
    token_summary = summarize_token_usage(prompt_set.token_rows)
    write_token_usage(prompt_set.token_rows, token_summary, RESULTS / "token_usage", tag=experiment)
    for seed in seeds:
        log(f"[{experiment}] seed {seed}")
        run_one_seed(cfg, prompt_set, graph, split, seed, device, experiment)
    # Aggregate every finished seed of this experiment and config on disk, not only the seeds of this call: the queues
    # run one seed per call, which used to leave a one-seed aggregate file (fixed 2026-11-01).
    finished = seed_metrics_on_disk(experiment, config_hash(cfg))
    aggregated = {"experiment": experiment, "dataset": cfg["dataset"]["name"], "mode": cfg.get("mode", "GPU_REPRODUCTION"),
                  "seeds": sorted(finished), "test": aggregate_seeds([finished[s]["test"] for s in sorted(finished)]),
                  "val": aggregate_seeds([finished[s]["val"] for s in sorted(finished)]), "token_usage": token_summary,
                  "config_hash": config_hash(cfg)}
    out = RESULTS / "aggregated" / f"{experiment}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")
    return aggregated


__all__ = ["build_prompts", "run_one_seed", "run_experiment", "seed_metrics_on_disk", "config_hash", "np"]
