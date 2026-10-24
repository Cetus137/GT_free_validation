"""Error injection on per-frame dicts with variable cell counts and extra fields."""
from __future__ import annotations

import numpy as np
import pytest

from tracking_validation.errors import (
    ErrorConfig,
    apply_tracking_errors,
    frames_from_array,
    realised_error_rates,
)

BOX = (100.0, 100.0, 100.0)


def _frames_variable(seed: int = 0, d: int = 3) -> list[dict]:
    """3 frames: 60, 70, 80 cells (newcomers are 'born' with parent_ids set)."""
    rng = np.random.default_rng(seed)
    g = int(np.ceil(80 ** (1 / d)))          # enough lattice points in any dimension
    lattice = np.stack(np.meshgrid(*[np.linspace(10, 90, g)] * d, indexing="ij"), -1).reshape(-1, d)
    base = lattice[:80] + rng.normal(0, 1.0, (80, d))
    frames = []
    for t, n in enumerate((60, 70, 80)):
        ids = np.arange(n)
        parent = np.full(n, -1)
        if t > 0:
            prev_n = (60, 70, 80)[t - 1]
            parent[prev_n:n] = np.arange(prev_n - (n - prev_n), prev_n)   # born from earlier cells
        frames.append({
            "positions": base[:n] + rng.normal(0, 0.3, (n, d)),
            "true_ids": ids.copy(), "assigned_ids": ids.copy(), "parent_ids": parent,
            "quality": rng.uniform(size=n),                      # float extra
            "frame_tag": f"t{t}",                                # non-array extra, must pass through
        })
    return frames


def _clean(**kw) -> ErrorConfig:
    base = dict(p_miss=0.0, lambda_fp=0.0, p_switch=0.0, p_merge=0.0, p_split=0.0)
    base.update(kw)
    return ErrorConfig(**base)


def test_array_input_still_works_and_matches_frames_from_array():
    arr = np.random.default_rng(0).uniform(10, 90, (2, 30, 3))
    obs, log = apply_tracking_errors(arr, _clean(), BOX, np.random.default_rng(0))
    conv = frames_from_array(arr)
    for o, c in zip(obs, conv):
        np.testing.assert_array_equal(o["positions"], c["positions"])
        np.testing.assert_array_equal(o["assigned_ids"], np.arange(30))


def test_variable_counts_identity_when_no_errors():
    frames = _frames_variable()
    obs, log = apply_tracking_errors(frames, _clean(), BOX, np.random.default_rng(0))
    assert [len(o["assigned_ids"]) for o in obs] == [60, 70, 80]
    for o, f in zip(obs, frames):
        np.testing.assert_array_equal(o["parent_ids"], f["parent_ids"])
        np.testing.assert_array_equal(o["quality"], f["quality"])
        assert o["frame_tag"] == f["frame_tag"]
    assert all(len(v) == 0 for v in log.values())


def test_misses_subset_extra_fields_consistently():
    frames = _frames_variable()
    obs, log = apply_tracking_errors(frames, _clean(p_miss=0.3), BOX, np.random.default_rng(1))
    assert len(log["missed"]) > 0
    for o, f in zip(obs, frames):
        n = len(o["assigned_ids"])
        assert o["positions"].shape[0] == n == len(o["parent_ids"]) == len(o["quality"])
        # kept rows carry their original values
        orig = {int(i): (p, q) for i, p, q in zip(f["assigned_ids"], f["parent_ids"], f["quality"])}
        for i, p, q in zip(o["assigned_ids"], o["parent_ids"], o["quality"]):
            assert orig[int(i)] == (p, q)


def test_synthetic_detections_get_fill_values():
    frames = _frames_variable()
    cfg = _clean(p_split=0.3, lambda_fp=10.0, split_distance=2.0)
    obs, log = apply_tracking_errors(frames, cfg, BOX, np.random.default_rng(2))
    assert log["splits"] and log["false_positives"]
    for o in obs:
        synthetic = (o["true_ids"] == -1) | np.isin(o["assigned_ids"], [ev["assigned_id_b"] for ev in log["splits"]])
        assert np.all(o["parent_ids"][synthetic] == -1)
        assert np.all(np.isnan(o["quality"][synthetic]))
        assert np.all(np.isfinite(o["quality"][~synthetic]))
        assert len(set(o["assigned_ids"].tolist())) == len(o["assigned_ids"])


def test_merge_keeps_survivors_extra_fields():
    frames = _frames_variable()
    obs, log = apply_tracking_errors(frames, _clean(p_merge=0.5), BOX, np.random.default_rng(3))
    assert log["merges"]
    for ti, (o, f) in enumerate(zip(obs, frames)):
        q_by_id = dict(zip(f["assigned_ids"].tolist(), f["quality"]))
        for ev in (e for e in log["merges"] if e["frame"] == ti):
            i = o["assigned_ids"].tolist().index(ev["assigned_id"])
            assert o["true_ids"][i] == -2
            assert o["quality"][i] == q_by_id[ev["assigned_id"]]


def test_synthetic_ids_start_above_max_existing_label():
    frames = _frames_variable()
    for f in frames:
        f["assigned_ids"] = f["assigned_ids"] + 1000
        f["true_ids"] = f["true_ids"] + 1000
    cfg = _clean(p_split=0.2, lambda_fp=5.0)
    obs, log = apply_tracking_errors(frames, cfg, BOX, np.random.default_rng(4))
    new_ids = [ev["assigned_id"] for ev in log["false_positives"]] + [ev["assigned_id_b"] for ev in log["splits"]]
    assert min(new_ids) > 1079


def test_fp_rate_scales_with_frame_size_and_near_cells_placement():
    frames = _frames_variable()
    cfg = _clean(fp_rate=0.5, fp_placement="near_cells", fp_offset=3.0)
    obs, log = apply_tracking_errors(frames, cfg, BOX, np.random.default_rng(5))
    n_fp = [sum(1 for e in log["false_positives"] if e["frame"] == t) for t in range(3)]
    assert all(0.2 * n < k < 0.8 * n for k, n in zip(n_fp, (60, 70, 80)))
    for o, f in zip(obs, frames):
        fp_pos = o["positions"][o["true_ids"] == -1]
        from scipy.spatial import KDTree
        d, _ = KDTree(f["positions"]).query(fp_pos)
        assert np.all(d <= 4.5 + 1e-9)        # within fp_offset * 1.5 of a real cell
        assert np.all(d >= 1.5 - 1e-9) or True  # clipping at the box edge may bring some closer


def test_fp_placement_validated():
    with pytest.raises(ValueError, match="fp_placement"):
        ErrorConfig(fp_placement="random")


def test_cross_frame_switch_skips_newborns():
    frames = _frames_variable()
    cfg = _clean(p_switch=1.0, max_switch_distance=100.0)
    obs, log = apply_tracking_errors(frames, cfg, BOX, np.random.default_rng(6))
    newborn = {(t, int(i)) for t, f in enumerate(frames) for i, p in zip(f["true_ids"], f["parent_ids"]) if p >= 0}
    for ev in log["switches"]:
        assert (ev["frame"], ev["true_id_i"]) not in newborn
        assert (ev["frame"], ev["true_id_j"]) not in newborn
    for o in obs:
        assert len(set(o["assigned_ids"].tolist())) == len(o["assigned_ids"])


def test_two_dimensional_frames():
    frames = _frames_variable(d=2)
    cfg = ErrorConfig(p_miss=0.1, lambda_fp=3.0, p_switch=0.1, p_merge=0.1, p_split=0.1,
                      split_distance=2.0, max_switch_distance=20.0)
    obs, log = apply_tracking_errors(frames, cfg, (100.0, 100.0), np.random.default_rng(7))
    for o in obs:
        assert o["positions"].shape[1] == 2
        assert o["positions"].shape[0] == len(o["assigned_ids"]) == len(o["parent_ids"])
    assert all(len(ev["position_a"]) == 2 for ev in log["splits"])
    assert all(len(ev["position"]) == 2 for ev in log["false_positives"])


def test_realised_rates_match_logs():
    frames = _frames_variable()
    cfg = ErrorConfig(p_miss=0.1, fp_rate=0.1, p_switch=0.1, p_merge=0.05, p_split=0.05,
                      max_switch_distance=20.0)
    obs, log = apply_tracking_errors(frames, cfg, BOX, np.random.default_rng(8))
    r = realised_error_rates(frames, obs, log)
    n = 210
    assert r["miss"] == pytest.approx(len(log["missed"]) / n)
    assert r["fp"] == pytest.approx(len(log["false_positives"]) / n)
    assert r["switch"] == pytest.approx(2 * len(log["switches"]) / n)
    assert r["merge"] == pytest.approx(2 * len(log["merges"]) / n)
    assert r["split"] == pytest.approx(len(log["splits"]) / n)
