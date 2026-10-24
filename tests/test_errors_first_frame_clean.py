from __future__ import annotations

import numpy as np

from tracking_validation.errors import ErrorConfig, apply_tracking_errors, realised_error_rates


def _frames(n=120, t=3, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(10, 90, (n, 3))
    return [{"positions": base + rng.normal(0, 0.3, base.shape), "true_ids": np.arange(n),
             "assigned_ids": np.arange(n)} for _ in range(t)]


def test_first_frame_untouched_by_detection_errors():
    frames = _frames()
    cfg = ErrorConfig(p_miss=0.3, fp_rate=0.3, p_merge=0.2, p_split=0.2, p_switch=0.2,
                      max_switch_distance=30.0, first_frame_clean=True)
    obs, log = apply_tracking_errors(frames, cfg, (100, 100, 100), np.random.default_rng(1))
    np.testing.assert_array_equal(obs[0]["positions"], frames[0]["positions"])
    np.testing.assert_array_equal(obs[0]["assigned_ids"], frames[0]["assigned_ids"])
    for key in ("missed", "merges", "splits", "false_positives", "switches"):
        assert all(ev["frame"] >= 1 for ev in log[key]), key
    # later frames were corrupted
    assert len(log["missed"]) and len(log["false_positives"]) and len(log["splits"]) and len(log["merges"])
    assert len(obs[1]["assigned_ids"]) != 120


def test_first_frame_clean_default_false_corrupts_frame_zero():
    frames = _frames()
    cfg = ErrorConfig(p_miss=0.5, lambda_fp=0.0, p_switch=0.0)
    _, log = apply_tracking_errors(frames, cfg, (100, 100, 100), np.random.default_rng(2))
    assert any(ev["frame"] == 0 for ev in log["missed"])


def test_realised_rates_use_later_frames_when_first_is_clean():
    frames = _frames(n=100, t=2)
    cfg = ErrorConfig(p_miss=0.2, lambda_fp=0.0, p_switch=0.0, first_frame_clean=True)
    obs, log = apply_tracking_errors(frames, cfg, (100, 100, 100), np.random.default_rng(3))
    r = realised_error_rates(frames, obs, log, first_frame_clean=True)
    assert r["miss"] == len(log["missed"]) / 100          # denominator: frame 1 only
    r_all = realised_error_rates(frames, obs, log)
    assert r_all["miss"] == len(log["missed"]) / 200
