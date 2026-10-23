"""FLAG-architecture study: the prompt, parser and subgraph order must match FLAG; the hook must not change the
baseline; the whole pipeline (generation -> metapath summaries -> prompts -> +FLAG* phase) runs on CPU mocks."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from dgp_repro.config import REPO_ROOT, load_experiment
from dgp_repro.data import build_graph, load_graph, load_split, processed_dir
from dgp_repro.embeddings import make_embedder
from dgp_repro.pipeline import run_dgp_prompt_pipeline
from dgp_repro.summarization import make_summarizer
from dgp_repro.summarization.flag_subgraph import (FlagPrompts, SubgraphText, build_prompt, parse_response,
                                                   strip_numbering, subgraph_nodes)
from dgp_repro.training.flag_finetune import orthogonal_loss

FLAG_SRC = REPO_ROOT.parent / "FLAG-Reproduce" / "src"


def _flag_enhance():
    """FLAG's own module, if the companion repository is next to this one (comparison only, never imported at run time)."""
    if not FLAG_SRC.exists():
        pytest.skip("FLAG-Reproduce not present")
    sys.path.insert(0, str(FLAG_SRC))
    try:
        from flagbench.llm import enhance
    except Exception as exc:          # pragma: no cover - environment dependent
        pytest.skip(f"cannot import FLAG enhance: {exc}")
    finally:
        sys.path.remove(str(FLAG_SRC))
    return enhance


def test_prompt_is_byte_identical_to_flag_build_prompt():
    enhance = _flag_enhance()
    ours = FlagPrompts.load("fa")
    theirs = enhance.PromptSet(dataset="x", system_instruction=ours.system_instruction, global_prompt=ours.global_prompt,
                               discriminative=ours.discriminative, residual=ours.residual)
    texts = ["short review", "x" * 1500, "third [with] brackets"]
    for kind in ("discriminative", "residual"):
        assert build_prompt(ours, kind, texts) == enhance.build_prompt(theirs, kind, texts, enhance.LLMConfig(), "reviews")


def test_parser_matches_flag_parser():
    enhance = _flag_enhance()
    cases = [("1. a\n2. b\n3. c", 3), ("Answer: 1. a\n\n2. b", 2), ("1. a\n2. b", 3), ("  12) a  \n b", 2)]
    for answer, n in cases:
        assert parse_response(answer, n) == enhance.parse_response(answer, n)
        lines = parse_response(answer, n)
        if lines:
            assert strip_numbering(lines) == enhance.strip_numbering(lines)


def test_prompt_layout_and_text_cut():
    p = build_prompt(FlagPrompts.load("fa"), "discriminative", ["a" * 1300, "b"])
    assert p.startswith("Don't have extra blank lines and symbols after Answer! \nPrompt: You are provided with")
    assert "Question: The reviews of these users are as follows:\n1. [" + "a" * 1200 + "]\n2. [b]\n\nAnswer:" in p
    assert p.endswith("Answer:") and "unhelpful or helpful" in p and "fraudulent" not in p


def test_subgraph_order_target_first_metapath_then_rank_no_duplicates():
    trims = {"RUR": {"neighbors": {7: np.array([3, 5])}}, "RPR": {"neighbors": {7: np.array([5, 9, 7])}}}
    assert subgraph_nodes(7, trims) == [7, 3, 5, 9]


def test_failed_subgraph_falls_back_to_raw_text():
    raw = ["r0", "r1", "r2"]
    ok, bad = SubgraphText([2, 0], ["d2", "d0"]), SubgraphText([2, 0], None)
    assert ok.text_for(0, raw) == "d0" and bad.text_for(0, raw) == "r0" and not bad.ok


def test_orthogonal_loss_modes():
    a, b = torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 0.0]])
    assert orthogonal_loss(a, b, "squared_dot").item() == pytest.approx(9.0)
    assert orthogonal_loss(a, b, "squared_cosine").item() == pytest.approx((3 / (5 ** 0.5 * 3)) ** 2)


def _smoke_cfg(**flag_arch):
    cfg = load_experiment("configs/experiments/smoke.yaml")
    if not (processed_dir(cfg["dataset"]) / "graph.json").exists():
        build_graph(cfg["dataset"]).save(processed_dir(cfg["dataset"]))
    if flag_arch:
        cfg["flag_arch"] = {"enabled": True, "prompt_set": "fa", **flag_arch}
        cfg["templates"] = {"final": "final_prompt_disc"}
    return cfg


def _pipeline(cfg, tmp_path):
    graph = load_graph(cfg["dataset"])
    split = load_split(cfg["dataset"], graph)
    return run_dgp_prompt_pipeline(graph, split, cfg, summarizer=make_summarizer(cfg["summarizer"]),
                                   embedder=make_embedder(cfg["embedder"]), token_counter=lambda t: len(t.split()),
                                   token_counter_name="words", cache_root=tmp_path, log=lambda *_: None), graph


def test_hook_leaves_the_baseline_prompts_unchanged(tmp_path):
    base, _ = _pipeline(_smoke_cfg(), tmp_path / "a")
    again, _ = _pipeline(_smoke_cfg(), tmp_path / "b")
    assert base.prompts == again.prompts and base.aux == {} and "fa_coverage" not in base.metadata


def test_flag_arch_pipeline_end_to_end(tmp_path):
    ps, graph = _pipeline(_smoke_cfg(kinds=["discriminative", "residual"]), tmp_path)
    cov = ps.metadata["fa_coverage"]
    assert cov["discriminative"]["subgraph_success_rate"] == 1.0 and cov["residual"]["node_coverage"] == 1.0
    v = next(iter(ps.prompts))
    assert "Key signals: " in ps.prompts[v] and "{" not in ps.prompts[v]
    assert set(ps.aux["usable"]) == set(ps.prompts) and "Key signals" not in ps.aux["disc_only"][v]
    assert "neighbor mean" not in ps.aux["resid_only"][v]          # branch prompts carry that kind of text only
    answers = list((tmp_path / "fa_subgraph_text").glob("*/answers.jsonl"))
    assert len(answers) == 2                                       # one cache per kind, raw answers kept


def test_plus_flag_star_phase_runs_and_never_reports_worse_than_start(tmp_path):
    from dgp_repro.models.label_tokens import resolve_label_token_ids
    from dgp_repro.models.mock_llm import MockLLM
    from dgp_repro.training.flag_finetune import flag_finetune

    ps, graph = _pipeline(_smoke_cfg(kinds=["discriminative", "residual"]), tmp_path)
    model = MockLLM(seed=0)
    lt = resolve_label_token_ids(model.tokenizer)
    use = ps.aux["usable"][:12]
    stats = flag_finetune(model, [ps.aux["disc_only"][v] for v in use], [ps.aux["resid_only"][v] for v in use],
                          graph.labels[use].astype(int), [ps.prompts[v] for v in use], graph.labels[use].astype(int),
                          lt, {"outer_epochs": 1, "inner_epochs": 2, "orthogonality": "squared_cosine"}, seed=0)
    assert stats["usable_train_targets"] == 12 and len(stats["history"]) == 2
    assert stats.get("best_val_macro_f1", stats["start_val_macro_f1"]) >= stats["start_val_macro_f1"]
