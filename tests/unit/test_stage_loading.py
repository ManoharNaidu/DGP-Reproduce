"""Stages must not construct models they do not use (a MDK-only run once started downloading Qwen3-8B)."""

import dgp_repro.experiment as experiment
from dgp_repro.config import load_experiment
from dgp_repro.data import build_graph, processed_dir


def test_mdk_stage_never_builds_the_summarizer(monkeypatch):
    cfg = load_experiment("configs/experiments/smoke.yaml")
    cfg["summarizer"] = {"backend": "qwen"}  # would download 16 GB if constructed
    if not (processed_dir(cfg["dataset"]) / "graph.json").exists():
        graph = build_graph(cfg["dataset"])
        graph.save(processed_dir(cfg["dataset"]))

    built = []
    monkeypatch.setattr(experiment, "make_summarizer", lambda *a, **k: built.append("summarizer"))
    prompt_set, _, _ = experiment.build_prompts(cfg, "cpu", log=lambda *_: None, until="mdk")
    assert built == [] and prompt_set.metadata["stopped_after"] == "mdk"
