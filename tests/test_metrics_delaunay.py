"""Tests for Delaunay-based neighbourhood functions in metrics.py."""
from __future__ import annotations

import numpy as np
import pytest

from tracking_validation.metrics import (
    _delaunay_neighbour_ranks_and_distances,
    _frame_neighbour_map_delaunay,
    compute_neighbour_continuity,
    compute_neighbour_continuity_observed,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _random_3d(n: int, seed: int = 0, scale: float = 100.0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0, scale, (n, 3))


def _adjacency_set(ranks: np.ndarray) -> set[tuple[int, int]]:
    """All non-sentinel (i, j) directed edges from a ranks array."""
    adj = set()
    for i in range(ranks.shape[0]):
        for j in ranks[i]:
            if j != -1:
                adj.add((i, int(j)))
    return adj


# ---------------------------------------------------------------------------
# _delaunay_neighbour_ranks_and_distances
# ---------------------------------------------------------------------------

class TestDelaunayRanksAndDistances:

    def test_basic_shape_and_sentinel_pairing(self):
        pos = _random_3d(20)
        ranks, dists = _delaunay_neighbour_ranks_and_distances(pos)
        n, k_max = ranks.shape
        assert n == 20
        assert k_max >= 3
        assert dists.shape == (20, k_max)
        # sentinel -1 and inf must always appear together
        assert np.array_equal(ranks == -1, np.isinf(dists))

    def test_non_sentinel_values_are_valid(self):
        pos = _random_3d(20)
        ranks, dists = _delaunay_neighbour_ranks_and_distances(pos)
        valid_mask = ranks != -1
        assert np.all(dists[valid_mask] > 0)
        assert np.all(np.isfinite(dists[valid_mask]))
        assert np.all(ranks[valid_mask] >= 0)
        assert np.all(ranks[valid_mask] < 20)
        # no self-neighbours
        for i in range(20):
            assert i not in ranks[i][ranks[i] != -1]

    def test_symmetry(self):
        # Symmetry holds among non-hull cells only: hull cells appear as
        # neighbours of interior cells but have no entries of their own
        # (their Voronoi regions are open — they are excluded by design).
        pos = _random_3d(30, seed=1)
        ranks, _ = _delaunay_neighbour_ranks_and_distances(pos)
        has_entry = {i for i in range(30) if np.any(ranks[i] != -1)}
        adj = _adjacency_set(ranks)
        for i, j in adj:
            if j in has_entry:
                assert (j, i) in adj, f"Asymmetry among interior cells: ({i},{j}) present but ({j},{i}) absent"

    def test_max_distance_cutoff(self):
        pos = _random_3d(20, scale=10.0)
        cutoff = 2.0
        ranks, dists = _delaunay_neighbour_ranks_and_distances(pos, max_distance=cutoff)
        valid_mask = ranks != -1
        assert np.all(dists[valid_mask] <= cutoff)
        # some sentinels expected at this tight cutoff in a 10-unit cube
        assert np.any(ranks == -1)

    def test_degenerate_n_less_than_4(self):
        for n_pts in (1, 2, 3):
            pos = _random_3d(n_pts)
            ranks, dists = _delaunay_neighbour_ranks_and_distances(pos)
            assert ranks.shape == (n_pts, 0)
            assert dists.shape == (n_pts, 0)

    def test_distances_match_positions(self):
        pos = _random_3d(15, seed=2)
        ranks, dists = _delaunay_neighbour_ranks_and_distances(pos)
        for i in range(15):
            for slot, j in enumerate(ranks[i]):
                if j == -1:
                    continue
                expected = float(np.linalg.norm(pos[i] - pos[j]))
                assert abs(dists[i, slot] - expected) < 1e-9

    def test_max_degree_cap(self):
        # dense cube — most cells will have Delaunay degree > 4
        pos = _random_3d(50, scale=10.0)
        cap = 4
        ranks, dists = _delaunay_neighbour_ranks_and_distances(pos, max_degree=cap)
        valid_counts = np.sum(ranks != -1, axis=1)
        assert np.all(valid_counts <= cap)

    def test_distances_sorted_ascending(self):
        pos = _random_3d(20)
        ranks, dists = _delaunay_neighbour_ranks_and_distances(pos)
        for i in range(20):
            valid_d = dists[i][ranks[i] != -1]
            assert np.all(np.diff(valid_d) >= 0), f"Row {i} distances not sorted"

    def test_max_degree_keeps_closest(self):
        """With max_degree=3, retained neighbours must be the 3 nearest Delaunay neighbours."""
        pos = _random_3d(20, seed=3)
        ranks_all, dists_all = _delaunay_neighbour_ranks_and_distances(pos)
        ranks_cap, dists_cap = _delaunay_neighbour_ranks_and_distances(pos, max_degree=3)
        for i in range(20):
            valid_all = dists_all[i][ranks_all[i] != -1]
            valid_cap = dists_cap[i][ranks_cap[i] != -1]
            expected = valid_all[:3]
            np.testing.assert_array_almost_equal(valid_cap, expected)


# ---------------------------------------------------------------------------
# _frame_neighbour_map_delaunay
# ---------------------------------------------------------------------------

class TestFrameNeighbourMapDelaunay:

    def test_basic_output_format(self):
        n = 15
        pos = _random_3d(n)
        ids = np.arange(n)
        nb_ids, nb_dists = _frame_neighbour_map_delaunay(pos, ids)
        assert isinstance(nb_ids, dict)
        assert isinstance(nb_dists, dict)
        for cid in nb_ids:
            assert cid in ids
            assert len(nb_ids[cid]) == len(nb_dists[cid])
            assert cid not in nb_ids[cid]  # no self-neighbour

    def test_distances_sorted_ascending(self):
        pos = _random_3d(15, seed=4)
        ids = np.arange(15)
        _, nb_dists = _frame_neighbour_map_delaunay(pos, ids)
        for cid, dists in nb_dists.items():
            assert dists == sorted(dists), f"Cell {cid} distances not sorted"

    def test_symmetry_with_integer_ids(self):
        # Symmetry holds only among non-hull cells: hull cells may appear as
        # neighbours of interior cells but have no entries of their own.
        n = 20
        pos = _random_3d(n, seed=5)
        ids = np.arange(n)
        nb_ids, _ = _frame_neighbour_map_delaunay(pos, ids)
        for a, neighbours in nb_ids.items():
            for b in neighbours:
                if b in nb_ids:  # only assert symmetry if b is also a non-hull cell
                    assert a in nb_ids[b], f"Asymmetry among interior cells: {b}→{a} missing"

    def test_max_distance_cutoff(self):
        pos = _random_3d(20, scale=10.0)
        ids = np.arange(20)
        cutoff = 2.5
        nb_ids, nb_dists = _frame_neighbour_map_delaunay(pos, ids, max_distance=cutoff)
        for cid, dists in nb_dists.items():
            assert all(d <= cutoff for d in dists)

    def test_degenerate_n_less_than_4(self):
        for n_pts in (1, 2, 3):
            pos = _random_3d(n_pts)
            ids = np.arange(n_pts)
            nb_ids, nb_dists = _frame_neighbour_map_delaunay(pos, ids)
            assert nb_ids == {}
            assert nb_dists == {}

    def test_non_contiguous_assigned_ids(self):
        pos = _random_3d(10, seed=6)
        ids = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        nb_ids, _ = _frame_neighbour_map_delaunay(pos, ids)
        for cid, neighbours in nb_ids.items():
            assert cid in ids
            for nb in neighbours:
                assert nb in ids

    def test_max_degree_cap(self):
        pos = _random_3d(30, scale=10.0)
        ids = np.arange(30)
        cap = 3
        nb_ids, _ = _frame_neighbour_map_delaunay(pos, ids, max_degree=cap)
        for neighbours in nb_ids.values():
            assert len(neighbours) <= cap


# ---------------------------------------------------------------------------
# compute_neighbour_continuity — delaunay mode
# ---------------------------------------------------------------------------

class TestComputeNeighbourContinuityDelaunay:

    def test_output_shape(self):
        frames = _random_3d(50, seed=7).reshape(1, 50, 3)
        frames = np.stack([frames[0], frames[0] + np.random.default_rng(8).normal(0, 1, (50, 3)),
                           frames[0] + np.random.default_rng(9).normal(0, 1, (50, 3))], axis=0)
        result = compute_neighbour_continuity(frames, neighbourhood_mode="delaunay")
        t, n = frames.shape[:2]
        assert result["identity_overlap"].shape == (t - 1, n)
        assert result["weighted_identity_overlap"].shape == (t - 1, n)
        assert result["rank_correlation"].shape == (t - 1, n)
        assert result["ranks"].shape[0] == t
        assert result["ranks"].shape[1] == n
        assert result["distances"].shape == result["ranks"].shape
        assert result["interior_mask"].shape == (t, n)

    def test_metric_values_in_range(self):
        rng = np.random.default_rng(10)
        base = rng.uniform(0, 100, (50, 3))
        frames = np.stack([base, base + rng.normal(0, 2, (50, 3))], axis=0)
        result = compute_neighbour_continuity(frames, neighbourhood_mode="delaunay")
        for key in ("identity_overlap", "weighted_identity_overlap"):
            vals = result[key]
            valid = vals[~np.isnan(vals)]
            # weighted_identity_overlap can exceed 1.0 by a small float epsilon
            assert np.all(valid >= 0.0) and np.all(valid <= 1.0 + 1e-9)

    def test_identical_frames_give_perfect_score(self):
        rng = np.random.default_rng(11)
        base = rng.uniform(0, 100, (30, 3))
        frames = np.stack([base, base], axis=0)
        result = compute_neighbour_continuity(frames, neighbourhood_mode="delaunay")
        valid_io = result["identity_overlap"][~np.isnan(result["identity_overlap"])]
        assert np.allclose(valid_io, 1.0)

    def test_with_max_neighbour_distance(self):
        rng = np.random.default_rng(12)
        base = rng.uniform(0, 100, (40, 3))
        frames = np.stack([base, base + rng.normal(0, 1, (40, 3))], axis=0)
        result = compute_neighbour_continuity(
            frames, neighbourhood_mode="delaunay", max_neighbour_distance=20.0
        )
        # ranks array should have some -1 sentinels due to distance cutoff
        assert np.any(result["ranks"] == -1)

    def test_unknown_mode_raises(self):
        frames = np.random.default_rng(0).uniform(0, 10, (2, 10, 3))
        with pytest.raises(ValueError, match="Unknown neighbourhood_mode"):
            compute_neighbour_continuity(frames, neighbourhood_mode="voronoi")

    def test_knn_and_delaunay_both_produce_valid_output(self):
        rng = np.random.default_rng(13)
        base = rng.uniform(0, 100, (50, 3))
        frames = np.stack([base, base + rng.normal(0, 2, (50, 3))], axis=0)
        res_knn = compute_neighbour_continuity(frames, k=7, neighbourhood_mode="knn")
        res_del = compute_neighbour_continuity(frames, neighbourhood_mode="delaunay")
        for key in ("identity_overlap", "weighted_identity_overlap"):
            for res in (res_knn, res_del):
                valid = res[key][~np.isnan(res[key])]
                assert np.all(valid >= 0.0) and np.all(valid <= 1.0 + 1e-9)

    def test_degenerate_small_n(self):
        # n < 4 — Delaunay cannot triangulate; should return empty arrays gracefully
        frames = np.random.default_rng(0).uniform(0, 10, (2, 3, 3))
        result = compute_neighbour_continuity(frames, neighbourhood_mode="delaunay")
        assert result["ranks"].shape[2] == 0


# ---------------------------------------------------------------------------
# compute_neighbour_continuity_observed — delaunay mode
# ---------------------------------------------------------------------------

class TestComputeNeighbourContinuityObservedDelaunay:

    def _make_observed_frames(self, t: int = 3, n: int = 20, seed: int = 0) -> list[dict]:
        rng = np.random.default_rng(seed)
        base = rng.uniform(0, 100, (n, 3))
        frames = []
        for ti in range(t):
            pos = base + rng.normal(0, 1, (n, 3))
            frames.append({
                "positions":    pos,
                "assigned_ids": np.arange(n),
            })
        return frames

    def test_output_keys_and_shapes(self):
        frames = self._make_observed_frames()
        result = compute_neighbour_continuity_observed(
            frames, neighbourhood_mode="delaunay"
        )
        assert "weighted_identity_overlap" in result
        assert "rank_correlation" in result
        assert "cell_ids" in result
        assert "track_survival" in result
        t_minus_1 = len(frames) - 1
        n_ids = len(result["cell_ids"])
        assert result["weighted_identity_overlap"].shape == (t_minus_1, n_ids)
        assert result["rank_correlation"].shape == (t_minus_1, n_ids)
        assert result["track_survival"].shape == (t_minus_1,)

    def test_metric_values_in_range(self):
        frames = self._make_observed_frames(n=30, seed=1)
        result = compute_neighbour_continuity_observed(
            frames, neighbourhood_mode="delaunay"
        )
        vals = result["weighted_identity_overlap"]
        valid = vals[~np.isnan(vals)]
        assert np.all(valid >= 0.0) and np.all(valid <= 1.0 + 1e-9)

    def test_cell_ids_match_assigned_ids(self):
        frames = self._make_observed_frames(n=15, seed=2)
        result = compute_neighbour_continuity_observed(
            frames, neighbourhood_mode="delaunay"
        )
        all_ids = sorted({int(i) for f in frames for i in f["assigned_ids"]})
        np.testing.assert_array_equal(result["cell_ids"], all_ids)

    def test_unknown_mode_raises(self):
        frames = self._make_observed_frames()
        with pytest.raises(ValueError, match="Unknown neighbourhood_mode"):
            compute_neighbour_continuity_observed(frames, neighbourhood_mode="bad")

    def test_with_max_neighbour_distance(self):
        frames = self._make_observed_frames(n=25, seed=3)
        result = compute_neighbour_continuity_observed(
            frames, neighbourhood_mode="delaunay", max_neighbour_distance=30.0
        )
        vals = result["weighted_identity_overlap"]
        valid = vals[~np.isnan(vals)]
        assert np.all(valid >= 0.0) and np.all(valid <= 1.0 + 1e-9)

    def test_identical_frames_give_perfect_score(self):
        rng = np.random.default_rng(4)
        pos = rng.uniform(0, 100, (20, 3))
        ids = np.arange(20)
        frame = {"positions": pos, "assigned_ids": ids}
        result = compute_neighbour_continuity_observed(
            [frame, frame], neighbourhood_mode="delaunay"
        )
        valid = result["weighted_identity_overlap"][~np.isnan(
            result["weighted_identity_overlap"]
        )]
        assert np.allclose(valid, 1.0)


class TestTrackSurvival:
    """Tests for the track_survival metric in compute_neighbour_continuity_observed."""

    def _frame(self, ids, seed=0):
        pos = np.random.default_rng(seed).uniform(0, 100, (len(ids), 3))
        return {"positions": pos, "assigned_ids": np.array(ids)}

    def test_survival_shape(self):
        frames = [self._frame(range(20), s) for s in range(4)]
        result = compute_neighbour_continuity_observed(frames)
        assert result["track_survival"].shape == (3,)

    def test_full_survival(self):
        # same IDs in every frame → survival = 1.0
        frame = self._frame(range(20))
        result = compute_neighbour_continuity_observed([frame, frame, frame])
        np.testing.assert_allclose(result["track_survival"], 1.0)

    def test_zero_survival(self):
        # completely disjoint IDs between frames → survival = 0.0
        f0 = self._frame(range(0,  20), seed=0)
        f1 = self._frame(range(20, 40), seed=1)
        result = compute_neighbour_continuity_observed([f0, f1])
        assert result["track_survival"][0] == 0.0

    def test_partial_survival(self):
        # 10 of 20 cells carry forward → survival = 0.5
        rng = np.random.default_rng(0)
        pos = rng.uniform(0, 100, (20, 3))
        f0 = {"positions": pos,                              "assigned_ids": np.arange(20)}
        f1 = {"positions": rng.uniform(0, 100, (10, 3)),    "assigned_ids": np.arange(10)}
        result = compute_neighbour_continuity_observed([f0, f1])
        assert abs(result["track_survival"][0] - 0.5) < 1e-9

    def test_survival_in_range(self):
        frames = [self._frame(range(20), s) for s in range(5)]
        result = compute_neighbour_continuity_observed(frames)
        assert np.all(result["track_survival"] >= 0.0)
        assert np.all(result["track_survival"] <= 1.0)

    def test_survival_returned_by_both_modes(self):
        frames = [self._frame(range(20), s) for s in range(3)]
        for mode in ("knn", "delaunay"):
            result = compute_neighbour_continuity_observed(frames, neighbourhood_mode=mode)
            assert "track_survival" in result
            assert result["track_survival"].shape == (2,)

    def test_survival_consistent_with_penalise_absence(self):
        # WIO_penalised should be <= WIO_standard when some cells are lost
        rng = np.random.default_rng(1)
        pos = rng.uniform(0, 100, (20, 3))
        f0 = {"positions": pos,                           "assigned_ids": np.arange(20)}
        f1 = {"positions": rng.uniform(0, 100, (15, 3)), "assigned_ids": np.arange(15)}
        res_std = compute_neighbour_continuity_observed([f0, f1], penalise_absence=False)
        res_pen = compute_neighbour_continuity_observed([f0, f1], penalise_absence=True)
        survival = res_std["track_survival"][0]
        assert survival < 1.0  # some cells were lost
        wio_std = np.nanmean(res_std["weighted_identity_overlap"])
        wio_pen = np.nanmean(res_pen["weighted_identity_overlap"])
        assert wio_pen <= wio_std
