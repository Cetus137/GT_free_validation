from __future__ import annotations

import numpy as np
import pytest

from tracking_validation.ctc import CTCTrack, CTCTracks
from tracking_validation.lineage import (
    division_affected_cells,
    divisions_per_frame,
    inherit_parent_labels,
    mask_cells,
)
from tracking_validation.metrics import compute_neighbour_continuity_observed


def _data() -> CTCTracks:
    """Tracks: 1 (t0..t1) -> single child 2 (t2) ; 2 divides -> 3, 4 at t3.
    Bystanders 10..15 throughout, well separated.  Mother 2 at t2 sits at x=0;
    daughter 3 is born nearer to it than daughter 4."""
    tracks = {
        1: CTCTrack(1, 0, 1, 0), 2: CTCTrack(2, 2, 2, 1),
        3: CTCTrack(3, 3, 3, 2), 4: CTCTrack(4, 3, 3, 2),
    }
    by = {lab: CTCTrack(lab, 0, 3, 0) for lab in range(10, 16)}
    tracks.update(by)
    by_pos = np.array([[20, 0, 0], [40, 0, 0], [0, 20, 0], [0, 40, 0], [0, 0, 20], [0, 0, 40]], float)
    frames = []
    for t in range(4):
        pos = [by_pos]; ids = list(range(10, 16)); par = [-1] * 6
        if t <= 1:
            pos.append([[0, 0, 0]]); ids.append(1); par.append(-1)
        elif t == 2:
            pos.append([[0, 0, 0]]); ids.append(2); par.append(1)
        else:
            pos.append([[1, 0, 0], [5, 0, 0]]); ids += [3, 4]; par += [2, 2]
        frames.append({"positions": np.vstack(pos), "true_ids": np.array(ids),
                       "assigned_ids": np.array(ids), "parent_ids": np.array(par)})
    return CTCTracks(frames=frames, frame_indices=[0, 1, 2, 3], tracks=tracks,
                     voxel_size=(1, 1, 1), image_shape=(50, 50, 50), box_size=(50, 50, 50), ndim=3)


def test_divisions_per_frame_counts_only_multi_child_parents():
    assert divisions_per_frame(_data().tracks) == {3: 1}


def test_inherit_single_child_and_nearest_daughter():
    out = inherit_parent_labels(_data(), prefer="nearest")
    m = out.label_map
    assert m[2] == 1          # single child continues under 1
    assert m[3] == 1          # nearest daughter inherits (chained to root 1)
    assert m[4] == 4          # other daughter keeps its label
    f3 = out.frames[3]
    ids = f3["assigned_ids"].tolist()
    assert 1 in ids and 4 in ids and 3 not in ids and 2 not in ids
    # heir's parent cleared; sister's parent now points at inherited label
    assert f3["parent_ids"][ids.index(1)] == -1
    assert f3["parent_ids"][ids.index(4)] == 1
    # frame 2: label 2 -> 1, and its single-child parent link cleared
    f2 = out.frames[2]
    assert 1 in f2["assigned_ids"].tolist() and np.all(f2["parent_ids"] == -1)
    # original untouched
    assert 2 in _data().frames[2]["assigned_ids"].tolist()


def test_inherit_prefer_first_picks_lowest_label():
    d = _data()
    # move daughter 4 nearer than 3 so "nearest" would pick 4, "first" still picks 3
    f3 = d.frames[3]; i3 = f3["assigned_ids"].tolist().index(3); i4 = f3["assigned_ids"].tolist().index(4)
    f3["positions"][i3] = [9, 0, 0]; f3["positions"][i4] = [0.5, 0, 0]
    assert inherit_parent_labels(d, prefer="nearest").label_map[4] == 1
    assert inherit_parent_labels(d, prefer="first").label_map[3] == 1
    with pytest.raises(ValueError):
        inherit_parent_labels(d, prefer="bogus")


def test_division_affected_cells_raw_and_inherited():
    d = _data()
    raw = division_affected_cells(d.frames, k=2)
    # pair (2,3): mother 2, newborns 3,4 and any bystander with them as neighbours
    assert {2, 3, 4} <= raw[2]
    # pair (1,2): single-child relabel 1->2 counts as newborn with parent at t2
    assert {1, 2} <= raw[1]
    assert raw[0] == set()
    inh = division_affected_cells(inherit_parent_labels(d).frames, k=2)
    assert inh[1] == set()                    # relabel removed by inheritance
    assert {1, 4} <= inh[2] and 3 not in inh[2]


def test_mask_cells_sets_nan_only_for_excluded():
    d = inherit_parent_labels(_data())
    res = compute_neighbour_continuity_observed(d.frames, k=3)
    exc = division_affected_cells(d.frames, k=3)
    masked = mask_cells(res, exc)
    col = {int(c): i for i, c in enumerate(res["cell_ids"])}
    for ti, bad in enumerate(exc):
        for c in bad:
            assert np.isnan(masked["weighted_identity_overlap"][ti, col[c]])
    # non-excluded entries unchanged
    keep = ~np.isnan(res["weighted_identity_overlap"])
    for ti, bad in enumerate(exc):
        for c in bad:
            keep[ti, col[c]] = False
    np.testing.assert_array_equal(masked["weighted_identity_overlap"][keep],
                                  res["weighted_identity_overlap"][keep])
    assert masked["track_survival"] is res["track_survival"]
