"""Dimension-agnostic behaviour: kNN and Delaunay neighbourhoods on 2D positions."""
from __future__ import annotations

import numpy as np

from tracking_validation.metrics import (
    _delaunay_neighbour_ranks_and_distances,
    _frame_neighbour_map_delaunay,
    compute_neighbour_continuity,
    compute_neighbour_continuity_observed,
)


def _pts2d(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0, 100, (n, 2))


def test_delaunay_ranks_2d_shape_sentinels_and_symmetry():
    pos = _pts2d(40)
    ranks, dists = _delaunay_neighbour_ranks_and_distances(pos)
    assert ranks.shape[0] == 40 and ranks.shape[1] >= 3
    assert np.array_equal(ranks == -1, np.isinf(dists))
    has_entry = {i for i in range(40) if np.any(ranks[i] != -1)}
    assert has_entry                                        # interior cells exist
    adj = {(i, int(j)) for i in range(40) for j in ranks[i] if j != -1}
    for i, j in adj:
        if j in has_entry:
            assert (j, i) in adj
    for i in range(40):
        for slot, j in enumerate(ranks[i]):
            if j != -1:
                assert abs(dists[i, slot] - np.linalg.norm(pos[i] - pos[j])) < 1e-9


def test_delaunay_2d_degenerate_below_three_points():
    for n in (1, 2):
        r, d = _delaunay_neighbour_ranks_and_distances(_pts2d(n))
        assert r.shape == (n, 0) and d.shape == (n, 0)
    assert _frame_neighbour_map_delaunay(_pts2d(2), np.arange(2)) == ({}, {})


def test_delaunay_2d_three_points_form_one_triangle_all_on_hull():
    # three points: a single triangle, every vertex on the hull -> no interior entries
    r, _ = _delaunay_neighbour_ranks_and_distances(_pts2d(3))
    assert r.shape[1] == 0


def test_continuity_2d_identical_frames_perfect_in_both_modes():
    base = _pts2d(60, seed=1)
    frames = np.stack([base, base])
    for mode in ("knn", "delaunay"):
        res = compute_neighbour_continuity(frames, k=6, neighbourhood_mode=mode)
        v = res["identity_overlap"][~np.isnan(res["identity_overlap"])]
        assert len(v) > 0 and np.allclose(v, 1.0)


def test_observed_2d_frames_both_modes_and_border_margin():
    rng = np.random.default_rng(2)
    base = rng.uniform(0, 100, (80, 2))
    f0 = {"positions": base, "assigned_ids": np.arange(80)}
    f1 = {"positions": base + rng.normal(0, 0.5, base.shape), "assigned_ids": np.arange(80)}
    for mode in ("knn", "delaunay"):
        res = compute_neighbour_continuity_observed([f0, f1], k=6, neighbourhood_mode=mode,
                                                    box_size=(100.0, 100.0), border_margin=5.0)
        v = res["weighted_identity_overlap"][0]
        v = v[~np.isnan(v)]
        assert len(v) > 0 and np.all(v >= 0) and np.all(v <= 1 + 1e-9)
