from __future__ import annotations

import numpy as np
import pytest

from tracking_validation.linking import label_links, link_frames, link_greedy, link_lap


def _lattice(n_side=4, spacing=10.0):
    g = np.arange(n_side) * spacing + 5.0
    return np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)


def _frames(t=4, jitter=0.3, seed=0):
    rng = np.random.default_rng(seed)
    base = _lattice()
    n = len(base)
    return [{"positions": base + rng.normal(0, jitter, base.shape), "true_ids": np.arange(n),
             "assigned_ids": np.zeros(n, dtype=int)} for _ in range(t)]


@pytest.mark.parametrize("method", ["greedy", "lap"])
def test_perfect_linking_on_slow_lattice(method):
    frames = _frames()
    res = link_frames(frames, max_distance=4.0, method=method)
    for fr in res.frames:
        np.testing.assert_array_equal(np.sort(fr["assigned_ids"]), np.arange(64))
    # same label follows the same true cell throughout
    for fr in res.frames[1:]:
        np.testing.assert_array_equal(fr["assigned_ids"], res.frames[0]["assigned_ids"])
    rows = label_links(res)
    assert all(r["switch"] == 0 and r["frag"] == 0 and r["correct"] == 64 for r in rows)
    assert res.n_tracks == 64


@pytest.mark.parametrize("method", ["greedy", "lap"])
def test_tight_cap_fragments_tracks(method):
    frames = _frames(jitter=1.0)
    res = link_frames(frames, max_distance=0.5, method=method)
    rows = label_links(res)
    assert sum(r["frag"] for r in rows) > 0
    assert res.n_tracks > 64
    assert all(r["switch"] == 0 for r in rows)      # cells 10 apart never confused


def test_dropped_detection_causes_fragmentation_not_switch():
    frames = _frames()
    keep = np.arange(64) != 10
    frames[1] = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in frames[1].items()}
    res = link_lap(frames, max_distance=4.0)
    rows = label_links(res)
    assert rows[0]["switch"] == 0 and rows[1]["switch"] == 0
    assert rows[1]["frag"] == 0                       # cell 10 absent at t=1, so not "present at both"
    # at t=2 cell 10 reappears and starts a new track
    lab2 = res.frames[2]["assigned_ids"][res.frames[2]["true_ids"] == 10][0]
    assert lab2 >= 64


def test_lap_prefers_global_optimum_over_greedy():
    # two cells shifting right by 6: greedy (closest first) links each to its neighbour's
    # old spot or not at all depending on order; LAP finds the consistent shift.
    p0 = np.array([[0.0, 0, 0], [10.0, 0, 0]]); p1 = p0 + [6.0, 0, 0]
    frames = [{"positions": p0, "true_ids": np.array([0, 1]), "assigned_ids": np.zeros(2, int)},
              {"positions": p1, "true_ids": np.array([0, 1]), "assigned_ids": np.zeros(2, int)}]
    lap = label_links(link_lap(frames, max_distance=7.0))[0]
    assert lap["correct"] == 2 and lap["switch"] == 0
    greedy = label_links(link_greedy(frames, max_distance=7.0))[0]
    assert greedy["switch"] >= 1 or greedy["frag"] >= 1   # greedy takes the 4-unit cross link first


def test_switch_labelled_when_cells_cross():
    p0 = np.array([[0.0, 0, 0], [3.0, 0, 0]]); p1 = np.array([[3.0, 0, 0], [0.0, 0, 0]])  # swap places
    frames = [{"positions": p0, "true_ids": np.array([0, 1]), "assigned_ids": np.zeros(2, int)},
              {"positions": p1, "true_ids": np.array([0, 1]), "assigned_ids": np.zeros(2, int)}]
    res = link_lap(frames, max_distance=1.0)        # cap smaller than the jump -> links to the nearer, wrong, cell
    r = label_links(res)[0]
    assert r["switch"] == 2 and r["switched_true_ids"] == {0, 1}


def test_evaluate_filter_restricts_scoring():
    frames = _frames()
    res = link_lap(frames, max_distance=4.0)
    rows = label_links(res, evaluate=[{0, 1, 2}] * (len(frames) - 1))
    assert all(r["n1"] == 3 and r["correct"] == 3 for r in rows)


def test_unknown_method():
    with pytest.raises(ValueError):
        link_frames(_frames(), 1.0, method="magic")
