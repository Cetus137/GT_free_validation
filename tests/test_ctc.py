"""Tests for the CTC reader, using a synthetic CTC folder written with tifffile."""
from __future__ import annotations

import csv

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from tracking_validation.ctc import (
    KNOWN_VOXEL_SIZES,
    CTCTracks,
    read_ctc_tracks,
    read_track_table,
    write_tracks_csv,
    summarize,
)


def _ball(img: np.ndarray, centre_zyx: tuple[int, ...], r: int, label: int) -> None:
    grids = np.ogrid[tuple(slice(0, s) for s in img.shape)]
    d2 = sum((g - c) ** 2 for g, c in zip(grids, centre_zyx))
    img[d2 <= r * r] = label


def _write_ctc_3d(folder, table: str, markers: dict[int, dict[int, tuple[int, int, int]]],
                  shape=(8, 40, 40), name="man_track"):
    """markers: frame -> {label: (z, y, x) centre}"""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.txt").write_text(table)
    for t, cells in markers.items():
        img = np.zeros(shape, dtype=np.uint16)
        for lab, c in cells.items():
            _ball(img, c, 2, lab)
        tifffile.imwrite(folder / f"{name}{t:03d}.tif", img)


@pytest.fixture
def ctc_folder(tmp_path):
    # track 1 lives t0..t1, divides into 2 and 3 at t2; track 4 lives throughout;
    # track 5 has a gap at t1 (listed t0..t2 but no marker at t1)
    table = "1 0 1 0\n2 2 2 1\n3 2 2 1\n4 0 2 0\n5 0 2 0\n"
    markers = {
        0: {1: (4, 10, 10), 4: (4, 30, 30), 5: (4, 10, 30)},
        1: {1: (4, 11, 11), 4: (4, 30, 29), 5: None},
        2: {2: (4, 8, 8), 3: (4, 14, 14), 4: (4, 30, 28), 5: (4, 10, 30)},
    }
    markers[1] = {k: v for k, v in markers[1].items() if v is not None}
    folder = tmp_path / "01_GT" / "TRA"
    _write_ctc_3d(folder, table, markers)
    return folder


def test_read_track_table_parses_and_validates(tmp_path):
    p = tmp_path / "man_track.txt"
    p.write_text("1 0 5 0\n2 6 9 1\n3 6 9 1\n")
    tr = read_track_table(p)
    assert set(tr) == {1, 2, 3}
    assert tr[2].parent == 1 and tr[2].has_parent and not tr[1].has_parent
    p.write_text("1 0 5 0\n2 6 9 7\n")
    with pytest.raises(ValueError, match="parent 7"):
        read_track_table(p)
    p.write_text("1 5 0 0\n")
    with pytest.raises(ValueError, match="ends"):
        read_track_table(p)


def test_reader_centroids_scaled_and_in_xyz_order(ctc_folder):
    vs = (0.5, 0.5, 2.0)
    data = read_ctc_tracks(ctc_folder, voxel_size=vs)
    assert isinstance(data, CTCTracks)
    assert data.ndim == 3 and data.num_frames == 3 and data.frame_indices == [0, 1, 2]
    assert data.image_shape == (8, 40, 40)
    assert data.box_size == pytest.approx((40 * 0.5, 40 * 0.5, 8 * 2.0))
    f0 = data.frames[0]
    i = int(np.where(f0["assigned_ids"] == 1)[0][0])
    # marker centre (z=4, y=10, x=10) -> (x, y, z) * voxel = (5.0, 5.0, 8.0)
    np.testing.assert_allclose(f0["positions"][i], (5.0, 5.0, 8.0), atol=1e-9)
    np.testing.assert_array_equal(f0["true_ids"], f0["assigned_ids"])


def test_reader_known_dataset_voxel_size(ctc_folder):
    data = read_ctc_tracks(ctc_folder, dataset="Fluo-N3DH-CE")
    assert data.voxel_size == KNOWN_VOXEL_SIZES["Fluo-N3DH-CE"]


def test_reader_default_voxel_size_is_unity(ctc_folder):
    data = read_ctc_tracks(ctc_folder)
    assert data.voxel_size == (1.0, 1.0, 1.0)


def test_parent_ids_set_only_in_birth_frame(ctc_folder):
    data = read_ctc_tracks(ctc_folder)
    f2 = data.frames[2]
    for lab in (2, 3):
        i = int(np.where(f2["assigned_ids"] == lab)[0][0])
        assert f2["parent_ids"][i] == 1
    i4 = int(np.where(f2["assigned_ids"] == 4)[0][0])
    assert f2["parent_ids"][i4] == -1
    assert np.all(data.frames[0]["parent_ids"] == -1)
    assert data.divisions() == [(1, 2, 3, 2)]
    assert data.children_of() == {1: [2, 3]}


def test_gaps_are_reported_not_fatal(ctc_folder):
    data = read_ctc_tracks(ctc_folder)
    assert data.gaps == [(5, 1)]
    assert data.unlisted == []
    assert 5 not in data.frames[1]["assigned_ids"].tolist()


def test_unlisted_marker_is_kept_and_reported(ctc_folder):
    img = tifffile.imread(ctc_folder / "man_track000.tif")
    img[0:2, 0:3, 0:3] = 99
    tifffile.imwrite(ctc_folder / "man_track000.tif", img)
    data = read_ctc_tracks(ctc_folder)
    assert data.unlisted == [(99, 0)]
    assert 99 in data.frames[0]["assigned_ids"].tolist()


def test_frame_subset_and_missing_frame_error(ctc_folder):
    data = read_ctc_tracks(ctc_folder, frames=[0, 2])
    assert data.frame_indices == [0, 2]
    with pytest.raises(FileNotFoundError):
        read_ctc_tracks(ctc_folder, frames=[0, 7])


def test_result_folder_naming_is_accepted(tmp_path):
    folder = tmp_path / "01_RES"
    _write_ctc_3d(folder, "1 0 0 0\n", {0: {1: (4, 20, 20)}}, name="mask")
    # result folders use res_track.txt
    (folder / "mask.txt").rename(folder / "res_track.txt")
    data = read_ctc_tracks(folder)
    assert data.frames[0]["assigned_ids"].tolist() == [1]


def test_2d_images_and_pad_to_3d(tmp_path):
    folder = tmp_path / "TRA"
    folder.mkdir()
    (folder / "man_track.txt").write_text("1 0 0 0\n2 0 0 0\n")
    img = np.zeros((30, 50), dtype=np.uint16)
    img[10:13, 20:23] = 1
    img[20:23, 40:43] = 2
    tifffile.imwrite(folder / "man_track000.tif", img)
    data = read_ctc_tracks(folder, voxel_size=(0.645, 0.645))
    assert data.ndim == 2 and data.frames[0]["positions"].shape == (2, 2)
    i = int(np.where(data.frames[0]["assigned_ids"] == 1)[0][0])
    np.testing.assert_allclose(data.frames[0]["positions"][i], (21 * 0.645, 11 * 0.645))
    padded = read_ctc_tracks(folder, voxel_size=(0.645, 0.645), pad_to_3d=True)
    assert padded.ndim == 3 and padded.frames[0]["positions"].shape == (2, 3)
    assert np.all(padded.frames[0]["positions"][:, 2] == 0.0)
    assert len(padded.box_size) == 3
    with pytest.raises(ValueError, match="voxel_size"):
        read_ctc_tracks(folder, voxel_size=(1.0, 1.0, 1.0))


def test_write_tracks_csv_roundtrip(ctc_folder, tmp_path):
    data = read_ctc_tracks(ctc_folder, voxel_size=(1.0, 1.0, 1.0))
    out = write_tracks_csv(data, tmp_path / "out" / "tracks.csv")
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0]) == {"track_id", "t", "x", "y", "z", "parent_id"}
    assert len(rows) == int(data.counts().sum())
    birth = [r for r in rows if r["track_id"] == "2"]
    assert len(birth) == 1 and birth[0]["t"] == "2" and birth[0]["parent_id"] == "1"


def test_summarize_mentions_divisions(ctc_folder):
    s = summarize(read_ctc_tracks(ctc_folder))
    assert "two-daughter divisions: 1" in s
    assert "gaps" in s


def test_reader_feeds_observed_metrics(ctc_folder):
    """Frames from the reader plug straight into the observed-frames metric."""
    from tracking_validation.metrics import compute_neighbour_continuity_observed
    data = read_ctc_tracks(ctc_folder)
    res = compute_neighbour_continuity_observed(data.frames, k=2)
    assert res["weighted_identity_overlap"].shape == (2, len(res["cell_ids"]))
