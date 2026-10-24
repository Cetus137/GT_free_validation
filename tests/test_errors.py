"""Tests for tracking-error injection (errors.py)."""
from __future__ import annotations

import numpy as np
import pytest

from tracking_validation.errors import ErrorConfig, apply_tracking_errors

BOX = (100.0, 100.0, 100.0)


def _frames(t: int = 3, n: int = 200, seed: int = 0, step: float = 0.5) -> np.ndarray:
    """Well-separated cells on a jittered lattice, moving slightly each frame."""
    rng = np.random.default_rng(seed)
    g = int(np.ceil(n ** (1 / 3)))
    lattice = np.stack(np.meshgrid(*[np.linspace(10, 90, g)] * 3, indexing="ij"), -1).reshape(-1, 3)
    base = lattice[:n] + rng.normal(0, 1.0, (n, 3))
    frames = [base]
    for _ in range(1, t):
        frames.append(frames[-1] + rng.normal(0, step, (n, 3)))
    return np.asarray(frames)


def _true_to_assigned(frame: dict) -> dict[int, int]:
    return {int(t): int(a) for t, a in zip(frame["true_ids"], frame["assigned_ids"]) if t >= 0}


def _clean(**kw) -> ErrorConfig:
    base = dict(p_miss=0.0, lambda_fp=0.0, p_switch=0.0, p_merge=0.0, p_split=0.0)
    base.update(kw)
    return ErrorConfig(**base)


# --------------------------------------------------------------------------- no errors

def test_no_errors_is_identity() -> None:
    frames = _frames()
    obs, log = apply_tracking_errors(frames, _clean(), BOX, np.random.default_rng(0))
    assert len(obs) == frames.shape[0]
    for f, o in zip(frames, obs):
        np.testing.assert_array_equal(o["positions"], f)
        np.testing.assert_array_equal(o["true_ids"], np.arange(f.shape[0]))
        np.testing.assert_array_equal(o["assigned_ids"], np.arange(f.shape[0]))
    assert all(len(v) == 0 for v in log.values())


# --------------------------------------------------------------------------- misses

def test_misses_drop_cells_and_are_logged() -> None:
    frames = _frames()
    obs, log = apply_tracking_errors(frames, _clean(p_miss=0.2), BOX, np.random.default_rng(1))
    n = frames.shape[1]
    total_missed = 0
    for ti, o in enumerate(obs):
        missed_here = {ev["true_id"] for ev in log["missed"] if ev["frame"] == ti}
        assert len(o["true_ids"]) + len(missed_here) == n
        assert set(o["true_ids"].tolist()).isdisjoint(missed_here)
        total_missed += len(missed_here)
    assert 0 < total_missed < n * len(obs)
    assert len(log["missed"]) == total_missed


def test_all_missed_gives_empty_frames() -> None:
    frames = _frames(t=2, n=30)
    obs, _ = apply_tracking_errors(frames, _clean(p_miss=1.0), BOX, np.random.default_rng(0))
    assert all(o["positions"].shape == (0, 3) for o in obs)


# --------------------------------------------------------------------------- false positives

def test_false_positives_have_sentinel_true_id_and_unique_fresh_ids() -> None:
    frames = _frames()
    n = frames.shape[1]
    obs, log = apply_tracking_errors(frames, _clean(lambda_fp=5.0), BOX, np.random.default_rng(2))
    n_fp = 0
    for o in obs:
        fp = o["true_ids"] == -1
        n_fp += int(fp.sum())
        assert np.all(o["assigned_ids"][fp] >= n)
        assert np.all((o["positions"][fp] >= 0) & (o["positions"][fp] <= np.asarray(BOX)))
        assert len(set(o["assigned_ids"].tolist())) == len(o["assigned_ids"])
    assert n_fp == len(log["false_positives"]) > 0
    fp_ids = [ev["assigned_id"] for ev in log["false_positives"]]
    assert len(set(fp_ids)) == len(fp_ids)          # globally unique across frames


# --------------------------------------------------------------------------- splits

def test_splits_duplicate_cells_at_the_requested_separation() -> None:
    frames = _frames(t=2, n=100)
    n = frames.shape[1]
    cfg = _clean(p_split=1.0, split_distance=3.0)
    obs, log = apply_tracking_errors(frames, cfg, BOX, np.random.default_rng(3))
    assert len(log["splits"]) == 2 * n
    for o in obs:
        assert o["positions"].shape == (2 * n, 3)
        # every true id appears exactly twice
        _, counts = np.unique(o["true_ids"], return_counts=True)
        assert np.all(counts == 2)
        assert len(set(o["assigned_ids"].tolist())) == 2 * n
    for ev in log["splits"]:
        sep = np.linalg.norm(np.asarray(ev["position_a"]) - np.asarray(ev["position_b"]))
        assert abs(sep - 3.0) < 1e-6   # direction normalised with a +1e-9 guard
        assert ev["assigned_id_a"] == ev["original_assigned_id"]   # primary keeps its ID
        assert ev["assigned_id_b"] >= n                             # fragment gets fresh ID


# --------------------------------------------------------------------------- merges

def test_merges_collapse_pairs_to_midpoint_and_keep_one_parent_id() -> None:
    frames = _frames(t=2, n=100)
    n = frames.shape[1]
    obs, log = apply_tracking_errors(frames, _clean(p_merge=0.5), BOX, np.random.default_rng(4))
    assert len(log["merges"]) > 0
    for ti, o in enumerate(obs):
        merged = o["true_ids"] == -2
        evs = [ev for ev in log["merges"] if ev["frame"] == ti]
        assert int(merged.sum()) == len(evs)
        assert o["positions"].shape[0] == n - len(evs)          # each merge removes one detection
        assert len(set(o["assigned_ids"].tolist())) == len(o["assigned_ids"])
        for ev in evs:
            assert ev["assigned_id"] in (ev["true_id_i"], ev["true_id_j"])
            assert ev["lost_assigned_id"] in (ev["true_id_i"], ev["true_id_j"])
            assert ev["assigned_id"] != ev["lost_assigned_id"]
            expected_mid = 0.5 * (frames[ti, ev["true_id_i"]] + frames[ti, ev["true_id_j"]])
            np.testing.assert_allclose(ev["position"], expected_mid)


# --------------------------------------------------------------------------- ID switches

def test_same_frame_switches_keep_ids_a_permutation_and_revert_next_frame() -> None:
    frames = _frames(t=3, n=200, step=0.1)
    obs, log = apply_tracking_errors(frames, _clean(p_switch=0.1), BOX, np.random.default_rng(5))
    assert len(log["switches"]) > 0
    for o in obs:
        assert sorted(o["assigned_ids"].tolist()) == sorted(o["true_ids"].tolist())
    # same-frame rule: a swap at frame f does not carry to frame f+1 unless re-drawn there
    m1 = _true_to_assigned(obs[1])
    switched_at_1 = {ev["true_id_i"] for ev in log["switches"] if ev["frame"] == 1} | \
                    {ev["true_id_j"] for ev in log["switches"] if ev["frame"] == 1}
    for t, a in m1.items():
        assert (t != a) == (t in switched_at_1)


def test_cross_frame_switches_persist_and_respect_distance_cap() -> None:
    frames = _frames(t=3, n=200, step=0.1)
    cap = 20.0
    cfg = _clean(p_switch=0.1, max_switch_distance=cap)
    obs, log = apply_tracking_errors(frames, cfg, BOX, np.random.default_rng(6))
    events = log["switches"]
    assert len(events) > 0
    assert all(ev["frame"] >= 1 for ev in events)          # never at frame 0
    for o in obs:
        assert len(set(o["assigned_ids"].tolist())) == len(o["assigned_ids"])

    # both mis-links within the cap
    for ev in events:
        f, i, j = ev["frame"], ev["true_id_i"], ev["true_id_j"]
        assert np.linalg.norm(frames[f - 1, i] - frames[f, j]) <= cap
        assert np.linalg.norm(frames[f - 1, j] - frames[f, i]) <= cap

    # persistence: a pair swapped at frame 1 and not touched again at frame 2 stays swapped
    touched_at_2 = {ev["true_id_i"] for ev in events if ev["frame"] == 2} | \
                   {ev["true_id_j"] for ev in events if ev["frame"] == 2}
    m2 = _true_to_assigned(obs[2])
    checked = 0
    for ev in events:
        if ev["frame"] != 1:
            continue
        i, j = ev["true_id_i"], ev["true_id_j"]
        if i in touched_at_2 or j in touched_at_2:
            continue
        assert m2[i] == j and m2[j] == i
        checked += 1
    assert checked > 0


def test_cross_frame_switch_with_tiny_cap_never_fires() -> None:
    frames = _frames(t=3, n=200, step=0.1)
    cfg = _clean(p_switch=1.0, max_switch_distance=1e-6)
    _, log = apply_tracking_errors(frames, cfg, BOX, np.random.default_rng(0))
    assert log["switches"] == []


# --------------------------------------------------------------------------- combined

def test_combined_errors_keep_assigned_ids_unique_per_frame() -> None:
    frames = _frames(t=4, n=200)
    cfg = ErrorConfig(p_miss=0.05, lambda_fp=3.0, p_switch=0.05, p_merge=0.05,
                      p_split=0.05, max_switch_distance=15.0)
    obs, log = apply_tracking_errors(frames, cfg, BOX, np.random.default_rng(7))
    for o in obs:
        ids = o["assigned_ids"].tolist()
        assert len(set(ids)) == len(ids)
        assert o["positions"].shape == (len(ids), 3)
        assert o["true_ids"].shape == (len(ids),)
    assert set(log) == {"missed", "merges", "splits", "false_positives", "switches"}


def test_error_config_rejects_nothing_but_runs_with_defaults() -> None:
    frames = _frames(t=2, n=50)
    obs, _ = apply_tracking_errors(frames, ErrorConfig(), BOX, np.random.default_rng(0))
    assert len(obs) == 2
