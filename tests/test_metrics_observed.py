"""Tests for compute_neighbour_continuity_observed options (border margin, absence penalty)."""
from __future__ import annotations

import numpy as np

import tracking_validation
from tracking_validation.metrics import compute_neighbour_continuity_observed

BOX = (100.0, 100.0, 100.0)
MARGIN = 10.0


def test_tests_import_local_source() -> None:
    """Guard against an editable install of another copy shadowing this repo."""
    here = __file__.rsplit("/tests/", 1)[0]
    assert tracking_validation.__file__.startswith(here), tracking_validation.__file__


def _interior_frames(n: int = 40, seed: int = 0) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    pos0 = rng.uniform(MARGIN + 10, 100 - MARGIN - 10, (n, 3))   # all interior
    pos1 = pos0 + rng.normal(0, 0.5, (n, 3))
    ids = np.arange(n)
    return (
        {"positions": pos0, "assigned_ids": ids},
        {"positions": pos1, "assigned_ids": ids.copy()},
    )


def test_absence_penalty_not_applied_to_cell_that_drifts_into_border() -> None:
    f0, f1 = _interior_frames()
    f1["positions"][0] = [MARGIN / 2, 50.0, 50.0]   # still PRESENT at t1, but in border band
    res = compute_neighbour_continuity_observed(
        [f0, f1], k=5, box_size=BOX, border_margin=MARGIN, penalise_absence=True
    )
    col = int(np.where(res["cell_ids"] == 0)[0][0])
    # tracked but not interior in both frames -> excluded (NaN), NOT penalised (0.0)
    assert np.isnan(res["weighted_identity_overlap"][0, col])
    assert res["track_survival"][0] == 1.0


def test_absence_penalty_applied_to_interior_cell_missing_at_t1() -> None:
    f0, f1 = _interior_frames()
    keep = np.arange(1, len(f1["assigned_ids"]))             # drop cell 0 at t1
    f1 = {"positions": f1["positions"][keep], "assigned_ids": f1["assigned_ids"][keep]}
    res = compute_neighbour_continuity_observed(
        [f0, f1], k=5, box_size=BOX, border_margin=MARGIN, penalise_absence=True
    )
    col = int(np.where(res["cell_ids"] == 0)[0][0])
    assert res["weighted_identity_overlap"][0, col] == 0.0
    assert np.isnan(res["rank_correlation"][0, col])
    n = len(f0["assigned_ids"])
    assert abs(res["track_survival"][0] - (n - 1) / n) < 1e-12


def test_absence_penalty_without_border_margin_matches_set_difference() -> None:
    f0, f1 = _interior_frames()
    keep = np.arange(2, len(f1["assigned_ids"]))             # drop cells 0 and 1 at t1
    f1 = {"positions": f1["positions"][keep], "assigned_ids": f1["assigned_ids"][keep]}
    res = compute_neighbour_continuity_observed([f0, f1], k=5, penalise_absence=True)
    wio = res["weighted_identity_overlap"][0]
    assert wio[0] == 0.0 and wio[1] == 0.0
    assert np.all(np.isfinite(wio[2:]))
