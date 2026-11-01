"""summarizer.backend=cached (gs_X0): reuses another writer's cached text under that writer's cache key, never
generates, and fails loudly on a cache miss."""

import pytest

from dgp_repro.config import load_experiment
from dgp_repro.summarization import make_summarizer
from dgp_repro.summarization.base import CachedOnlySummarizer
from dgp_repro.summarization.pipeline import node_summaries
from dgp_repro.utils.cache import TextStore


def test_cached_backend_keys_by_writer_and_never_loads_a_model():
    s = make_summarizer({"backend": "cached", "model_id": "google/gemma-2-9b-it"}, "cuda")
    assert isinstance(s, CachedOnlySummarizer) and s.name == "google/gemma-2-9b-it"
    with pytest.raises(ValueError):
        make_summarizer({"backend": "cached"})


def test_cached_backend_reads_hits_and_raises_on_miss(tmp_path):
    texts = ["first review", "second review"]
    store = TextStore(tmp_path / "s.jsonl")
    store.add_many([("0", "cached summary", {})])
    s = make_summarizer({"backend": "cached", "model_id": "w"})
    out = node_summaries(texts, [0], s, "{text}", 10, store)
    assert out == {0: "cached summary"}
    with pytest.raises(RuntimeError, match="not in the cache"):
        node_summaries(texts, [0, 1], s, "{text}", 10, store)


def test_x0_config_is_r0_writer_swapped_and_g0_classifier_swapped():
    x0 = load_experiment("configs/experiments/gemma_swap/X0.yaml")
    r0 = load_experiment("configs/experiments/disc_resid/R0_baseline.yaml")
    g0 = load_experiment("configs/experiments/gemma_swap/G0.yaml")
    assert x0["summarizer"]["model_id"] == g0["summarizer"]["model_id"] == "google/gemma-2-9b-it"
    assert x0["classifier"] == r0["classifier"]
    assert {k: v for k, v in x0.items() if k not in ("summarizer", "experiment", "variant")} == \
           {k: v for k, v in r0.items() if k not in ("summarizer", "experiment", "variant")}
