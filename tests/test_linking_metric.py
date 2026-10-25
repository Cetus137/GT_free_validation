from __future__ import annotations

import numpy as np

from tracking_validation.linking import label_links, link_frames, link_lap, link_metric


def _lattice(n_side=5, spacing=10.0):
    g = np.arange(n_side) * spacing + 5.0
    return np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)


def test_metric_linker_perfect_on_slow_lattice():
    rng = np.random.default_rng(0); base = _lattice(); n = len(base)
    frames = [{"positions": base + rng.normal(0, 0.3, base.shape), "true_ids": np.arange(n), "assigned_ids": np.zeros(n, int)}
              for _ in range(3)]
    res = link_metric(frames, max_distance=4.0)
    rows = label_links(res)
    assert all(r["switch"] == 0 and r["frag"] == 0 and r["correct"] == n for r in rows)


def test_metric_linker_shear_case_is_not_worse_with_displacement_regulariser():
    """Half a lattice translates by 0.8 spacing past a static half.  At the shear
    boundary the TRUE labelling is less continuous than the smooth wrong one, so
    continuity alone (lam = 0) can make things worse; with the displacement
    regulariser the refinement must not fall below plain assignment."""
    from tracking_validation import linking
    rng = np.random.default_rng(1); base = _lattice(n_side=6, spacing=10.0); n = len(base)
    moved = base.copy(); moved[base[:, 0] < 35] += [8.0, 0, 0]; moved += rng.normal(0, 0.2, moved.shape)
    frames = [{"positions": base, "true_ids": np.arange(n), "assigned_ids": np.zeros(n, int)},
              {"positions": moved, "true_ids": np.arange(n), "assigned_ids": np.zeros(n, int)}]
    lap = label_links(link_lap(frames, max_distance=10.0))[0]
    old = linking.METRIC_DISPLACEMENT_WEIGHT
    try:
        linking.METRIC_DISPLACEMENT_WEIGHT = 1.0
        met = label_links(link_metric(frames, max_distance=10.0))[0]
    finally:
        linking.METRIC_DISPLACEMENT_WEIGHT = old
    assert met["switch"] + met["frag"] <= lap["switch"] + lap["frag"] + 1


def test_metric_linker_registered_and_returns_valid_matching():
    rng = np.random.default_rng(2)
    p0 = rng.uniform(0, 60, (80, 3)); p1 = p0 + rng.normal(0, 1.0, p0.shape)
    frames = [{"positions": p0, "true_ids": np.arange(80), "assigned_ids": np.zeros(80, int)},
              {"positions": p1, "true_ids": np.arange(80), "assigned_ids": np.zeros(80, int)}]
    res = link_frames(frames, 6.0, "metric")
    m = res.links[0]
    assert len(set(m[:, 0])) == len(m) and len(set(m[:, 1])) == len(m)   # one-to-one
    assert len(set(res.frames[1]["assigned_ids"].tolist())) == 80
