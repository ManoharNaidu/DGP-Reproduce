"""Sharded text generation (DGP_GEN_SHARD), the target_text / metapath_summaries stops and the zero-shot templates."""

import json

import pytest

from dgp_repro.config import load_experiment
from dgp_repro.data import build_graph, load_graph, load_split, processed_dir
from dgp_repro.embeddings import make_embedder
from dgp_repro.pipeline import run_dgp_prompt_pipeline
from dgp_repro.summarization import make_summarizer
from dgp_repro.summarization.pipeline import in_shard
from dgp_repro.utils.cache import TextStore

TARGET_TEXT = {"disc": {"template": "dr_discriminative_nv", "budget": 8, "truncate_chars": 1200, "fallback_raw": True},
               "residual": {"template": "dr_residual", "budget": 8, "truncate_chars": 1200, "fallback_raw": True}}


def _cfg(**extra):
    cfg = load_experiment("configs/experiments/smoke.yaml")
    if not (processed_dir(cfg["dataset"]) / "graph.json").exists():
        build_graph(cfg["dataset"]).save(processed_dir(cfg["dataset"]))
    cfg.update(extra)
    return cfg


def _run(cfg, root, until="prompts"):
    graph = load_graph(cfg["dataset"])
    split = load_split(cfg["dataset"], graph)
    return run_dgp_prompt_pipeline(graph, split, cfg, summarizer=make_summarizer(cfg["summarizer"]),
                                   embedder=make_embedder(cfg["embedder"]), token_counter=lambda t: len(t.split()),
                                   token_counter_name="words", cache_root=root, log=lambda *_: None, until=until)


def test_shards_partition_the_keys():
    keys = [str(i) for i in range(500)] + [f"{i}|RUR|abc" for i in range(500)]
    owners = [[k for k in keys if in_shard(k, (i, 4))] for i in range(4)]
    assert sorted(sum(owners, [])) == sorted(keys)          # every key in exactly one shard
    assert all(len(o) > 150 for o in owners)                 # and the split is roughly even
    assert all(in_shard(k, None) for k in keys)


def test_store_append_is_line_atomic_and_reloads(tmp_path):
    store = TextStore(tmp_path / "s.jsonl")
    store.add_many([("a", "x\ny", {"n": 1}), ("b", "z", {"n": 2})])
    TextStore(tmp_path / "s.jsonl").add_many([("c", "w", {})])
    lines = (tmp_path / "s.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(ln)["key"] for ln in lines] == ["a", "b", "c"]
    again = TextStore(tmp_path / "s.jsonl")
    assert again["a"] == "x\ny" and len(again) == 3


def test_sharded_generation_gives_the_same_prompts_as_one_process(tmp_path, monkeypatch):
    cfg = _cfg(target_text=TARGET_TEXT, templates={"final": "dr_final_disc"})
    single = _run(cfg, tmp_path / "single")
    root = tmp_path / "sharded"
    _run(cfg, root, until="mdk")                             # MDK is built once, unsharded
    for stage in ("target_text", "node_summaries", "metapath_summaries"):
        for i in range(3):
            monkeypatch.setenv("DGP_GEN_SHARD", f"{i}/3")
            _run(cfg, root, until=stage)
    monkeypatch.delenv("DGP_GEN_SHARD")
    calls = []
    summarizer = make_summarizer(cfg["summarizer"])
    original = summarizer.summarize
    summarizer.summarize = lambda prompts, budget: calls.append(len(prompts)) or original(prompts, budget)
    graph = load_graph(cfg["dataset"])
    merged = run_dgp_prompt_pipeline(graph, load_split(cfg["dataset"], graph), cfg, summarizer=summarizer,
                                     embedder=make_embedder(cfg["embedder"]), token_counter=lambda t: len(t.split()),
                                     token_counter_name="words", cache_root=root, log=lambda *_: None)
    assert calls == []                                       # the shards filled every cache
    assert merged.prompts == single.prompts


def test_shard_refuses_to_build_mdk_or_prompts(tmp_path, monkeypatch):
    cfg = _cfg()
    monkeypatch.setenv("DGP_GEN_SHARD", "0/2")
    for until in ("mdk", "prompts"):
        with pytest.raises(ValueError, match="DGP_GEN_SHARD"):
            _run(cfg, tmp_path, until=until)
    with pytest.raises(RuntimeError, match="MDK cache"):     # caches must exist before shards start
        _run(cfg, tmp_path / "empty", until="node_summaries")


def test_target_text_stage_generates_only_target_lines(tmp_path):
    ps = _run(_cfg(target_text=TARGET_TEXT), tmp_path, until="target_text")
    assert ps.metadata["stopped_after"] == "target_text" and ps.prompts == {}
    stages = {p.parent.parent.name for p in tmp_path.glob("*/*/summaries.jsonl")}
    assert stages == {"node_summaries"}                      # D/R stores only; no metapath summaries yet
    roles = {json.loads((p.parent / "metadata.json").read_text()).get("role") for p in tmp_path.glob("*/*/summaries.jsonl")}
    assert roles == {"target"}


def test_zero_shot_templates_carry_the_same_fields(tmp_path):
    base = _cfg(target_text=TARGET_TEXT, templates={"final": "dr_final_disc"})
    zs = _cfg(target_text=TARGET_TEXT, templates={"final": "zs_final_disc"})
    a, b = _run(base, tmp_path), _run(zs, tmp_path)
    v = next(iter(a.prompts))
    pa, pb = a.prompts[v], b.prompts[v]
    assert "{" not in pb and pb.startswith("<task>") and pb.endswith("</output_format>")
    target = pa.split("\n", 1)[0].removeprefix("Target: ")
    signals = pa.split("\n")[1].removeprefix("Key signals: ")
    blocks = pa.split("Metapaths:\n", 1)[1].rsplit("\nQuestion:", 1)[0]
    assert f"<target_review>\n{target}\n</target_review>" in pb
    assert f"<key_signals>\n{signals}\n</key_signals>" in pb
    assert blocks in pb and "Respond with exactly one word, either Yes or No." in pb


def test_summarizer_batches_keep_order_and_respect_the_cap():
    from dgp_repro.summarization.qwen import QwenSummarizer

    class Tok:                                   # one token per word
        def __call__(self, texts, add_special_tokens=False):
            return {"input_ids": [t.split() for t in texts]}

    s = QwenSummarizer.__new__(QwenSummarizer)
    s.tokenizer, s.batch_size = Tok(), 4
    texts = ["a " * n for n in (2, 2, 10, 2, 2, 2, 2, 2, 2)]
    s.cfg = {}
    assert s._batches(texts) == [texts[0:4], texts[4:8], texts[8:9]]          # unset: old fixed-size batches
    s.cfg = {"max_batch_tokens_sq": 200}
    out = s._batches(texts)
    assert sum(out, []) == texts                                               # order and content unchanged
    assert all(len(b) * max(len(t.split()) for t in b) ** 2 <= 200 or len(b) == 1 for b in out)
    assert all(len(b) <= 4 for b in out)
