from __future__ import annotations

import numpy as np

from tracking_validation.errors import ErrorConfig, apply_tracking_errors, realised_error_rates


def _frames(n=100, t=4, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(10, 90, (n, 3))
    return [{"positions": base + rng.normal(0, 0.3, base.shape), "true_ids": np.arange(n),
             "assigned_ids": np.arange(n)} for _ in range(t)]


def _clean(**kw):
    base = dict(p_miss=0.0, lambda_fp=0.0, p_switch=0.0, p_merge=0.0, p_split=0.0)
    base.update(kw)
    return ErrorConfig(**base)


def test_fragmentation_relabels_persistently_and_never_in_frame_zero():
    frames = _frames()
    obs, log = apply_tracking_errors(frames, _clean(p_frag=0.2), (100, 100, 100), np.random.default_rng(1))
    evs = log["fragmentations"]
    assert evs and all(ev["frame"] >= 1 for ev in evs)
    np.testing.assert_array_equal(obs[0]["assigned_ids"], frames[0]["assigned_ids"])
    # positions and true ids untouched; only labels change
    for o, f in zip(obs, frames):
        np.testing.assert_array_equal(o["positions"], f["positions"])
        np.testing.assert_array_equal(o["true_ids"], f["true_ids"])
        assert len(set(o["assigned_ids"].tolist())) == len(o["assigned_ids"])
    # persistence: a cell fragmented at frame f shows its new label at every later frame
    # unless fragmented again (then a newer label)
    for ev in evs:
        t = ev["true_id"]
        later = [e for e in evs if e["true_id"] == t and e["frame"] > ev["frame"]]
        last_frame = min((e["frame"] for e in later), default=len(obs))
        for f in range(ev["frame"], last_frame):
            i = obs[f]["true_ids"].tolist().index(t)
            assert obs[f]["assigned_ids"][i] == ev["new_assigned_id"]
        # old label gone from that frame onward
        assert ev["old_assigned_id"] not in obs[ev["frame"]]["assigned_ids"].tolist()
    assert min(ev["new_assigned_id"] for ev in evs) >= 100


def test_fragmentation_skips_synthetic_and_newborn_detections():
    frames = _frames(t=3)
    # add a false-positive-like synthetic detection and a newborn at frame 1
    frames[1]["positions"] = np.vstack([frames[1]["positions"], [[50, 50, 50]]])
    frames[1]["true_ids"] = np.concatenate([frames[1]["true_ids"], [500]])      # newborn (absent at t0)
    frames[1]["assigned_ids"] = np.concatenate([frames[1]["assigned_ids"], [500]])
    obs, log = apply_tracking_errors(frames, _clean(p_frag=1.0, lambda_fp=5.0),
                                     (100, 100, 100), np.random.default_rng(2))
    frag_true = {ev["true_id"] for ev in log["fragmentations"] if ev["frame"] == 1}
    assert 500 not in frag_true                 # newborn cannot fragment in its first frame
    assert all(t >= 0 for t in frag_true)      # no synthetic detections fragmented
    # every continuing real cell fragmented at p=1
    assert frag_true == set(range(100))


def test_fragmentation_realised_rate():
    frames = _frames(n=100, t=2)
    obs, log = apply_tracking_errors(frames, _clean(p_frag=0.1), (100, 100, 100), np.random.default_rng(3))
    r = realised_error_rates(frames, obs, log)
    assert r["frag"] == len(log["fragmentations"]) / 100


def test_fragmentation_looks_like_vanish_plus_newcomer_but_not_positional_loss():
    """Fingerprint on a tight lattice: WIO drops, vanished and newcomer rise equally,
    positional overlap unchanged (the cell is still there under a new name)."""
    from tracking_validation.metrics import compute_neighbour_continuity_observed
    g = np.linspace(5, 95, 10)
    pos = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    rng = np.random.default_rng(4)
    frames = [{"positions": pos + rng.normal(0, 0.2, pos.shape), "true_ids": np.arange(1000),
               "assigned_ids": np.arange(1000)} for _ in range(2)]
    obs, log = apply_tracking_errors(frames, _clean(p_frag=0.1), (100, 100, 100), np.random.default_rng(5))
    kw = dict(k=6, max_neighbour_distance=15.0, extra_metrics=True, displacement_sigma=3.0)
    base = compute_neighbour_continuity_observed(frames, **kw)
    res = compute_neighbour_continuity_observed(obs, **kw)
    d = lambda key: float(np.nanmean(res[key][0]) - np.nanmean(base[key][0]))
    assert d("weighted_identity_overlap") < -0.05
    assert d("wio_vanished") > 0.05 and d("newcomer_fraction") > 0.05
    assert abs(d("wio_displaced")) < 0.01
    assert abs(d("positional_weighted_identity_overlap")) < 0.005
