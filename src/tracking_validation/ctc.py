"""Reader for Cell Tracking Challenge (CTC) tracking annotations and results.

A CTC tracking folder (``01_GT/TRA``, ``01_ST/TRA`` or a result folder ``01_RES``)
contains

* a track table, ``man_track.txt`` (ground truth) or ``res_track.txt`` (results),
  one row per track: ``L B E P`` = label, first frame, last frame, parent label
  (0 = no parent);
* one label image per frame, ``man_trackTTT.tif`` or ``maskTTT.tif``, in which
  every voxel of a cell marker carries that track's label.  Images are 2D
  ``(y, x)`` or 3D ``(z, y, x)``.

This module converts such a folder into the per-frame dict format used by
``metrics.compute_neighbour_continuity_observed`` and ``errors.py``:

    {"positions": (N, d) float  — centroids in physical units, axis order (x, y[, z])
     "true_ids":  (N,)   int    — track label
     "assigned_ids": (N,) int   — track label (identical for GT; a tracker's labels
                                  for results)
     "parent_ids": (N,)  int    — parent track label for a detection in the FIRST
                                  frame of a track that has a parent, else -1.
                                  Used by the division policy in metrics.}

Positions are scaled by the voxel size so that every distance parameter in the
package can be given in microns.  Voxel sizes for the common CTC datasets are
tabulated in ``KNOWN_VOXEL_SIZES``; pass ``voxel_size`` explicitly otherwise.

Requires ``tifffile`` (``pip install -e ".[ctc]"``).  Command line:
``python -m tracking_validation ctc <folder> --dataset Fluo-N3DH-CE --out tracks.csv``.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage

# Voxel sizes in microns, axis order (x, y, z) or (x, y) for 2D, from
# https://celltrackingchallenge.net/3d-datasets/ and /2d-datasets/.
KNOWN_VOXEL_SIZES: dict[str, tuple[float, ...]] = {
    "Fluo-N3DH-CE":   (0.09, 0.09, 1.0),
    "Fluo-N3DL-TRIF": (0.38, 0.38, 0.38),
    "Fluo-N3DL-DRO":  (0.406, 0.406, 2.03),
    "Fluo-N2DL-HeLa": (0.645, 0.645),
    "PhC-C2DL-PSC":   (1.6, 1.6),
}

# Time step in minutes, same sources.
KNOWN_TIME_STEPS_MIN: dict[str, float] = {
    "Fluo-N3DH-CE":   1.5,
    "Fluo-N3DL-TRIF": 1.5,
    "Fluo-N3DL-DRO":  0.5,
    "Fluo-N2DL-HeLa": 30.0,
    "PhC-C2DL-PSC":   10.0,
}

_FRAME_FILE_RE = re.compile(r"^(?:man_track|mask)(\d+)\.tiff?$", re.IGNORECASE)


@dataclass(slots=True)
class CTCTrack:
    label: int
    begin: int
    end: int
    parent: int           # 0 = no parent (CTC convention)

    @property
    def has_parent(self) -> bool:
        return self.parent > 0


@dataclass(slots=True)
class CTCTracks:
    """A CTC tracking folder converted to physical-unit centroids."""
    frames: list[dict]                       # per-frame dicts, see module docstring
    frame_indices: list[int]                 # CTC frame index of each entry in ``frames``
    tracks: dict[int, CTCTrack]              # label -> track table row
    voxel_size: tuple[float, ...]            # (x, y[, z]) microns per voxel
    image_shape: tuple[int, ...]             # array shape of one label image (z, y, x)
    box_size: tuple[float, ...]              # image extent in microns, (x, y[, z])
    ndim: int
    gaps: list[tuple[int, int]] = field(default_factory=list)   # (label, frame) in
                                                                # [begin, end] with no marker
    unlisted: list[tuple[int, int]] = field(default_factory=list)  # (label, frame) marker
                                                                   # present but not in table

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def counts(self) -> np.ndarray:
        return np.array([len(f["assigned_ids"]) for f in self.frames])

    def divisions(self) -> list[tuple[int, int, int, int]]:
        """(mother, daughter_a, daughter_b, daughters' first frame), one per division.

        Only pairs of daughters are returned; a parent with a single child (a track
        that was merely re-labelled) or more than two children is reported as-is
        in ``children_of`` but not here.
        """
        kids = self.children_of()
        out = []
        for mother, ds in kids.items():
            if len(ds) == 2:
                a, b = sorted(ds)
                out.append((mother, a, b, self.tracks[a].begin))
        return sorted(out, key=lambda r: (r[3], r[0]))

    def children_of(self) -> dict[int, list[int]]:
        kids: dict[int, list[int]] = {}
        for tr in self.tracks.values():
            if tr.has_parent:
                kids.setdefault(tr.parent, []).append(tr.label)
        return kids


# --------------------------------------------------------------------------- track table

def read_track_table(path: str | Path) -> dict[int, CTCTrack]:
    """Parse ``man_track.txt`` / ``res_track.txt`` (``L B E P`` per line)."""
    tracks: dict[int, CTCTrack] = {}
    with Path(path).open() as f:
        for line_no, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 4:
                raise ValueError(f"{path}:{line_no}: expected 4 columns (L B E P), got {parts!r}")
            label, begin, end, parent = (int(p) for p in parts)
            if label <= 0:
                raise ValueError(f"{path}:{line_no}: track label must be positive, got {label}")
            if end < begin:
                raise ValueError(f"{path}:{line_no}: track {label} ends ({end}) before it begins ({begin})")
            if label in tracks:
                raise ValueError(f"{path}:{line_no}: duplicate track label {label}")
            tracks[label] = CTCTrack(label, begin, end, parent)
    for tr in tracks.values():
        if tr.has_parent and tr.parent not in tracks:
            raise ValueError(f"track {tr.label} has parent {tr.parent} which is not in the table")
    return tracks


def find_frame_files(folder: str | Path) -> dict[int, Path]:
    """Map CTC frame index -> label image path for every ``man_trackTTT.tif``/``maskTTT.tif``."""
    out: dict[int, Path] = {}
    for p in Path(folder).iterdir():
        m = _FRAME_FILE_RE.match(p.name)
        if m:
            out[int(m.group(1))] = p
    return dict(sorted(out.items()))


def find_track_table(folder: str | Path) -> Path:
    folder = Path(folder)
    for name in ("man_track.txt", "res_track.txt"):
        if (folder / name).exists():
            return folder / name
    raise FileNotFoundError(f"no man_track.txt or res_track.txt in {folder}")


# --------------------------------------------------------------------------- label images

def _read_label_image(path: Path) -> np.ndarray:
    try:
        import tifffile
    except ImportError as e:  # pragma: no cover
        raise ImportError("reading CTC label images requires tifffile: pip install tifffile") from e
    img = tifffile.imread(str(path))
    img = np.squeeze(img)
    if img.ndim not in (2, 3):
        raise ValueError(f"{path}: expected a 2D or 3D label image, got shape {img.shape}")
    return img


def label_centroids(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, centroids) for every non-zero label in ``img``.

    Centroids are in array index order (z, y, x) or (y, x), in voxels.
    """
    labels = np.unique(img)
    labels = labels[labels > 0]
    if labels.size == 0:
        return labels.astype(int), np.zeros((0, img.ndim))
    com = ndimage.center_of_mass(np.ones_like(img, dtype=np.uint8), labels=img, index=labels)
    return labels.astype(int), np.asarray(com, dtype=float)


def _to_xyz(centroids_zyx: np.ndarray, voxel_size_xy_z: tuple[float, ...]) -> np.ndarray:
    """(N, d) array-order centroids -> (N, d) physical (x, y[, z])."""
    xyz = centroids_zyx[:, ::-1]                       # (z,y,x) -> (x,y,z) ; (y,x) -> (x,y)
    return xyz * np.asarray(voxel_size_xy_z, dtype=float)


# --------------------------------------------------------------------------- main entry point

def read_ctc_tracks(
    folder: str | Path,
    voxel_size: tuple[float, ...] | None = None,
    dataset: str | None = None,
    frames: Iterable[int] | None = None,
    pad_to_3d: bool = False,
) -> CTCTracks:
    """Read a CTC tracking folder into physical-unit per-frame centroids.

    Parameters
    ----------
    folder:
        ``.../01_GT/TRA``, ``.../01_ST/TRA`` or ``.../01_RES``.
    voxel_size:
        Microns per voxel, axis order (x, y[, z]).  If None, looked up from
        ``dataset`` in ``KNOWN_VOXEL_SIZES``; if both are None, 1.0 per axis
        (positions stay in voxels).
    dataset:
        CTC dataset name, e.g. ``"Fluo-N3DH-CE"``, used to look up voxel size.
    frames:
        Subset of CTC frame indices to read (default: all present).
    pad_to_3d:
        For 2D data, append a z column of zeros so the result can be fed to
        code that assumes three coordinates.

    Notes
    -----
    * ``true_ids`` and ``assigned_ids`` are both the CTC track label.  For a
      ground-truth folder they are the truth; for a result folder they are the
      tracker's labels and you must match them to the GT separately (e.g. with
      ``traccuracy``) to obtain ``true_ids``.
    * ``parent_ids`` is set only in the first frame of a track that has a parent.
      This is the hook used by the division policy in ``metrics``.
    * Frames in ``[begin, end]`` with no marker are recorded in ``gaps``; markers
      whose label is missing from the table are recorded in ``unlisted`` and kept
      (with ``parent_ids = -1``).
    """
    folder = Path(folder)
    tracks = read_track_table(find_track_table(folder))
    files = find_frame_files(folder)
    if not files:
        raise FileNotFoundError(f"no man_trackTTT.tif / maskTTT.tif label images in {folder}")
    if frames is not None:
        wanted = set(int(t) for t in frames)
        missing = wanted - set(files)
        if missing:
            raise FileNotFoundError(f"requested frames not present in {folder}: {sorted(missing)}")
        files = {t: p for t, p in files.items() if t in wanted}

    first_path = next(iter(files.values()))
    first = _read_label_image(first_path)
    ndim = first.ndim
    if voxel_size is None:
        voxel_size = KNOWN_VOXEL_SIZES.get(dataset, None) if dataset else None
    if voxel_size is None:
        voxel_size = (1.0,) * ndim
    voxel_size = tuple(float(v) for v in voxel_size)
    if len(voxel_size) != ndim:
        raise ValueError(f"voxel_size has {len(voxel_size)} entries but images are {ndim}D")

    out_frames: list[dict] = []
    frame_indices: list[int] = []
    seen: dict[int, set[int]] = {}       # label -> frames where a marker was found
    unlisted: list[tuple[int, int]] = []

    for t, path in files.items():
        img = first if path == first_path else _read_label_image(path)
        if img.shape != first.shape:
            raise ValueError(f"{path}: shape {img.shape} differs from first frame {first.shape}")
        labels, com = label_centroids(img)
        pos = _to_xyz(com, voxel_size)
        if pad_to_3d and ndim == 2:
            pos = np.hstack([pos, np.zeros((pos.shape[0], 1))])

        parent_ids = np.full(labels.shape, -1, dtype=int)
        for i, lab in enumerate(labels.tolist()):
            seen.setdefault(lab, set()).add(t)
            tr = tracks.get(lab)
            if tr is None:
                unlisted.append((lab, t))
                continue
            if tr.has_parent and t == tr.begin:
                parent_ids[i] = tr.parent

        out_frames.append({
            "positions":    pos,
            "true_ids":     labels.copy(),
            "assigned_ids": labels.copy(),
            "parent_ids":   parent_ids,
        })
        frame_indices.append(t)

    gaps: list[tuple[int, int]] = []
    read_set = set(frame_indices)
    for tr in tracks.values():
        present = seen.get(tr.label, set())
        for t in range(tr.begin, tr.end + 1):
            if t in read_set and t not in present:
                gaps.append((tr.label, t))

    extent = tuple(float(s * v) for s, v in zip(first.shape[::-1], voxel_size))
    if pad_to_3d and ndim == 2:
        extent = extent + (0.0,)

    return CTCTracks(
        frames=out_frames,
        frame_indices=frame_indices,
        tracks=tracks,
        voxel_size=voxel_size,
        image_shape=tuple(first.shape),
        box_size=extent,
        ndim=3 if (pad_to_3d and ndim == 2) else ndim,
        gaps=sorted(gaps),
        unlisted=sorted(unlisted),
    )


# --------------------------------------------------------------------------- export

def write_tracks_csv(data: CTCTracks, output_path: str | Path) -> Path:
    """Write ``track_id, t, x, y, z, parent_id`` rows (z = 0 for 2D data).

    The column set is a superset of the ``track_id, t, x, y, z`` layout used by
    the real-data scripts in the original ``tracking_validation`` project.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "t", "x", "y", "z", "parent_id"])
        for t, fr in zip(data.frame_indices, data.frames):
            pos = fr["positions"]
            z = pos[:, 2] if pos.shape[1] >= 3 else np.zeros(pos.shape[0])
            for i in range(pos.shape[0]):
                w.writerow([int(fr["assigned_ids"][i]), int(t),
                            float(pos[i, 0]), float(pos[i, 1]), float(z[i]),
                            int(fr["parent_ids"][i])])
    return out


def summarize(data: CTCTracks) -> str:
    c = data.counts()
    divs = data.divisions()
    lines = [
        f"frames read       : {data.num_frames} (CTC indices {data.frame_indices[0]}..{data.frame_indices[-1]})",
        f"image shape       : {data.image_shape}  voxel size (x,y[,z]) µm: {data.voxel_size}",
        f"extent (x,y[,z]) µm: {tuple(round(v, 2) for v in data.box_size)}",
        f"tracks in table   : {len(data.tracks)}  with parent: {sum(t.has_parent for t in data.tracks.values())}",
        f"two-daughter divisions: {len(divs)}",
        f"cells per frame   : min {c.min()}  median {int(np.median(c))}  max {c.max()}",
        f"gaps (label in [B,E] but no marker): {len(data.gaps)}",
        f"unlisted markers (not in table)    : {len(data.unlisted)}",
    ]
    return "\n".join(lines)
