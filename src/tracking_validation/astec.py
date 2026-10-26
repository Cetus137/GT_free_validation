"""Reader for ASTEC / MorphoNet lineage property files (Guignard et al. 2020,
*Phallusia mammillata* embryos).

The XML holds per-cell properties keyed by ``cell-id = time * 10000 + label``:
``cell_barycenter`` (voxel coordinates, [x, y, z]), ``cell_lineage`` (list of
successor cell-ids in the next time point: one = continuation, two = division),
``cell_volume``, ``cell_name`` and others.

:func:`read_astec_tracks` follows the lineage to build tracks: a chain of
single-successor links keeps one track label; a division ends the mother's
track and starts two daughters with ``parent_ids`` set in their birth frame.
Output is the package's per-frame dict list (``positions`` in physical units,
``true_ids`` = ``assigned_ids`` = track label, ``parent_ids``), wrapped in the
same :class:`~tracking_validation.ctc.CTCTracks` container so every downstream
tool (lineage helpers, metrics, sweeps) works unchanged.
"""
from __future__ import annotations

import ast
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import numpy as np

from .ctc import CTCTrack, CTCTracks


def read_properties(path: str | Path, keys: Iterable[str] = ("cell_barycenter", "cell_lineage", "cell_volume")) -> dict[str, dict[int, object]]:
    """Parse the requested property blocks into {property: {cell_id: value}}."""
    root = ET.parse(str(path)).getroot()
    out: dict[str, dict[int, object]] = {}
    for key in keys:
        node = root.find(key)
        if node is None:
            continue
        d: dict[int, object] = {}
        for c in node:
            cid = int(c.get("cell-id"))
            txt = (c.text or "").strip()
            try:
                d[cid] = ast.literal_eval(txt) if txt else None
            except (ValueError, SyntaxError):
                d[cid] = txt
        out[key] = d
    return out


def read_astec_tracks(
    path: str | Path,
    voxel_size: tuple[float, float, float] | float = 1.0,
    frames: Iterable[int] | None = None,
) -> CTCTracks:
    """Build per-frame tracks from an ASTEC properties XML.

    Parameters
    ----------
    path:
        ``<embryo>_properties.xml``.
    voxel_size:
        Microns per voxel for (x, y, z), or one number for isotropic data.
        Barycentres are multiplied by it.
    frames:
        Subset of ASTEC time points to keep (default all).
    """
    props = read_properties(path)
    bary = props["cell_barycenter"]
    lineage = props.get("cell_lineage", {})
    vs = np.array([voxel_size] * 3 if np.isscalar(voxel_size) else voxel_size, dtype=float)

    times = sorted({cid // 10000 for cid in bary})
    if frames is not None:
        wanted = set(int(t) for t in frames)
        times = [t for t in times if t in wanted]
    time_set = set(times)

    # --- follow the lineage to assign track labels -------------------------------------
    label_of: dict[int, int] = {}        # cell_id -> track label
    parent_of_track: dict[int, int] = {} # track label -> parent track label (0 = none)
    begin: dict[int, int] = {}; end: dict[int, int] = {}
    next_label = 1
    for t in times:
        cells_t = [cid for cid in bary if cid // 10000 == t]
        for cid in cells_t:
            if cid not in label_of:                      # a track starts here (root or unexplained start)
                label_of[cid] = next_label; parent_of_track[next_label] = 0; begin[next_label] = t
                next_label += 1
            lab = label_of[cid]; end[lab] = t
            succ = [s for s in (lineage.get(cid) or []) if s // 10000 in time_set]
            if len(succ) == 1:
                label_of[succ[0]] = lab                  # continuation
            elif len(succ) >= 2:
                for s in succ:                           # division: new tracks with parent
                    label_of[s] = next_label; parent_of_track[next_label] = lab; begin[next_label] = s // 10000
                    next_label += 1

    tracks = {lab: CTCTrack(lab, begin[lab], end.get(lab, begin[lab]), parent_of_track[lab]) for lab in begin}

    out_frames: list[dict] = []
    for t in times:
        cells_t = sorted(cid for cid in bary if cid // 10000 == t)
        pos = np.array([np.asarray(bary[c], dtype=float) * vs for c in cells_t]).reshape(-1, 3)
        labels = np.array([label_of[c] for c in cells_t], dtype=int)
        parent = np.array([parent_of_track[label_of[c]] if tracks[label_of[c]].begin == t and parent_of_track[label_of[c]] > 0 else -1
                           for c in cells_t], dtype=int)
        out_frames.append({"positions": pos, "true_ids": labels.copy(), "assigned_ids": labels.copy(),
                           "parent_ids": parent, "astec_ids": np.array(cells_t, dtype=int)})

    all_pos = np.vstack([f["positions"] for f in out_frames])
    extent = tuple(float(v) for v in all_pos.max(axis=0))
    return CTCTracks(frames=out_frames, frame_indices=times, tracks=tracks, voxel_size=tuple(vs),
                     image_shape=(0, 0, 0), box_size=extent, ndim=3)
