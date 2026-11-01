"""run_experiment aggregates every finished seed on disk (2026-11-01 fix for one-seed aggregate files)."""
import json

from dgp_repro.experiment import seed_metrics_on_disk


def _run(raw, name, seed, h, f1):
    d = raw / f"{name}_seed{seed}_{h}"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({"seed": seed, "test": {"macro_f1": f1}, "val": {"macro_f1": f1}}))


def test_collects_all_seeds_of_one_config(tmp_path):
    for s in range(3):
        _run(tmp_path, "gs_G1", s, "aaaaaaaaaa", 60.0 + s)
    _run(tmp_path, "gs_G1", 3, "bbbbbbbbbb", 99.0)          # other config: excluded
    _run(tmp_path, "gs_G1b", 0, "aaaaaaaaaa", 99.0)         # longer experiment name with the same prefix: excluded
    (tmp_path / "gs_G1_seed4_aaaaaaaaaa").mkdir()           # unfinished run (no metrics.json): excluded
    got = seed_metrics_on_disk("gs_G1", "aaaaaaaaaa", tmp_path)
    assert sorted(got) == [0, 1, 2]
    assert [got[s]["test"]["macro_f1"] for s in range(3)] == [60.0, 61.0, 62.0]


def test_experiment_names_containing_seed(tmp_path):
    _run(tmp_path, "dgp_amazonvideo_seed0", 2, "cccccccccc", 65.0)
    assert sorted(seed_metrics_on_disk("dgp_amazonvideo_seed0", "cccccccccc", tmp_path)) == [2]
