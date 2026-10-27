"""Lineage helpers for real ground truth with divisions.

Two cheap ways to look at division-containing data *before* a division-aware
metric exists:

* :func:`inherit_parent_labels` — relabel tracks so that a child track continues
  under its parent's label.  A single child (CTC tables often re-label a cell
  that merely continues) always inherits.  For a division, ONE daughter inherits
  (the one nearest the mother's last position by default) and the other keeps
  its own label with ``parent_ids`` pointing at the inherited label.  This is the
  TrackMate-style convention and removes most of the "mother vanished" artefact.

* :func:`division_affected_cells` — for each consecutive frame pair, the set of
  focal labels whose neighbourhood touches a division (the mother, the newborn,
  or any cell that has one of them among its neighbours at either time).
  :func:`mask_cells` sets those entries to NaN in a metrics result so what
  remains is a division-free sample at every frame.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import numpy as np

from .ctc import CTCTracks, CTCTrack
from .metrics import _frame_neighbour_map, _frame_neighbour_map_delaunay


# --------------------------------------------------------------------------- tables

def children_by_parent(tracks: dict[int, CTCTrack]) -> dict[int, list[CTCTrack]]:
    kids: dict[int, list[CTCTrack]] = defaultdict(list)
    for tr in tracks.values():
        if tr.has_parent:
            kids[tr.parent].append(tr)
    return {p: sorted(v, key=lambda t: (t.begin, t.label)) for p, v in kids.items()}


def divisions_per_frame(tracks: dict[int, CTCTrack]) -> dict[int, int]:
    """Birth frame -> number of parents with two or more children born there."""
    out: dict[int, int] = defaultdict(int)
    for p, ch in children_by_parent(tracks).items():
        if len(ch) >= 2:
            out[ch[0].begin] += 1
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------- relabelling

def inherit_parent_labels(data: CTCTracks, prefer: str = "nearest") -> CTCTracks:
    """Return a copy of ``data`` in which child tracks continue their parent's label.

    Parameters
    ----------
    prefer:
        Which daughter inherits at a division: ``"nearest"`` (closest to the
        mother's last detected position; falls back to lowest label if the
        mother has no detection in her last frame) or ``"first"`` (lowest label).

    Notes
    -----
    * ``true_ids`` and ``assigned_ids`` are both relabelled (they are identical
      for ground truth).
    * ``parent_ids`` of the inheriting child is cleared to -1 (it is now a
      continuation); the non-inheriting daughter's ``parent_ids`` is rewritten
      to the *inherited* label of the mother so a later division-aware metric
      still finds the right mother.
    * ``label_map`` on the result maps original label -> final label.
    """
    if prefer not in ("nearest", "first"):
        raise ValueError("prefer must be 'nearest' or 'first'")

    kids = children_by_parent(data.tracks)
    label_map: dict[int, int] = {lab: lab for lab in data.tracks}
    inherits: set[int] = set()      # child labels that continue the parent

    # position lookup for "nearest": (label, frame) -> position
    pos_at: dict[tuple[int, int], np.ndarray] = {}
    if prefer == "nearest":
        for t, fr in zip(data.frame_indices, data.frames):
            for i, lab in enumerate(fr["assigned_ids"].tolist()):
                pos_at[(lab, t)] = fr["positions"][i]

    # parents in time order so chains resolve root-first
    for parent in sorted(kids, key=lambda p: data.tracks[p].end):
        children = kids[parent]
        root = label_map[parent]
        if len(children) == 1:
            heir = children[0]
        else:
            heir = None
            if prefer == "nearest":
                p_last = pos_at.get((parent, data.tracks[parent].end))
                if p_last is not None:
                    cands = [(c, pos_at.get((c.label, c.begin))) for c in children]
                    cands = [(c, p) for c, p in cands if p is not None]
                    if cands:
                        heir = min(cands, key=lambda cp: float(np.linalg.norm(cp[1] - p_last)))[0]
            if heir is None:
                heir = children[0]           # lowest label (children sorted by begin, label)
        label_map[heir.label] = root
        inherits.add(heir.label)

    # resolve chains: a child of an heir maps to the heir's root
    def _resolve(lab: int) -> int:
        seen = set()
        while label_map.get(lab, lab) != lab and lab not in seen:
            seen.add(lab)
            lab = label_map[lab]
        return lab
    label_map = {lab: _resolve(lab) for lab in label_map}

    new_frames: list[dict] = []
    for fr in data.frames:
        ids = fr["assigned_ids"]
        new_ids = np.array([label_map.get(int(l), int(l)) for l in ids], dtype=int)
        new_parent = fr["parent_ids"].copy()
        for i, lab in enumerate(ids.tolist()):
            if new_parent[i] >= 0:
                if lab in inherits:
                    new_parent[i] = -1
                else:
                    new_parent[i] = label_map.get(int(new_parent[i]), int(new_parent[i]))
        new_frames.append({**fr, "true_ids": new_ids.copy(), "assigned_ids": new_ids,
                           "parent_ids": new_parent})

    return replace(data, frames=new_frames, label_map=label_map)


# --------------------------------------------------------------------------- exclusion

def division_affected_cells(
    frames: list[dict],
    k: int = 8,
    max_neighbour_distance: float | None = None,
    neighbourhood_mode: str = "knn",
) -> list[set[int]]:
    """Per consecutive frame pair, the focal labels whose neighbourhood touches a division.

    A pair (t, t+1) is affected for focal label c if any of:
      * c is a mother (its label is the ``parent_ids`` of a detection at t+1),
      * c is a newborn at t+1 (``parent_ids >= 0``) or the continuing heir
        (its label equals a newborn's parent label at t+1),
      * one of c's neighbours at t is a mother,
      * one of c's neighbours at t+1 is a newborn or heir.

    Works on raw frames (both daughters newborn) and on frames after
    :func:`inherit_parent_labels` (one newborn, one heir).
    """
    if neighbourhood_mode not in ("knn", "delaunay"):
        raise ValueError("neighbourhood_mode must be 'knn' or 'delaunay'")
    excluded: list[set[int]] = []
    for ti in range(len(frames) - 1):
        f0, f1 = frames[ti], frames[ti + 1]
        ids0, ids1 = f0["assigned_ids"], f1["assigned_ids"]
        p1 = f1["parent_ids"]
        newborn = {int(l) for l, p in zip(ids1, p1) if p >= 0}
        mothers = {int(p) for p in p1 if p >= 0}
        affected0 = mothers
        affected1 = newborn | mothers          # "mothers" at t+1 are heirs under inheritance
        bad: set[int] = set(affected0) | set(affected1)
        if not bad:
            excluded.append(set())
            continue
        if neighbourhood_mode == "knn":
            k0 = min(k, len(ids0) - 1); k1 = min(k, len(ids1) - 1)
            nb0 = _frame_neighbour_map(f0["positions"], ids0, k0, max_neighbour_distance)[0] if k0 >= 1 else {}
            nb1 = _frame_neighbour_map(f1["positions"], ids1, k1, max_neighbour_distance)[0] if k1 >= 1 else {}
        else:
            nb0 = _frame_neighbour_map_delaunay(f0["positions"], ids0, max_neighbour_distance)[0]
            nb1 = _frame_neighbour_map_delaunay(f1["positions"], ids1, max_neighbour_distance)[0]
        for c, nbs in nb0.items():
            if affected0.intersection(nbs):
                bad.add(c)
        for c, nbs in nb1.items():
            if affected1.intersection(nbs):
                bad.add(c)
        excluded.append(bad)
    return excluded


def subsample_frames(frames: list[dict], stride: int) -> list[dict]:
    """Every ``stride``-th frame, with division records repaired.

    ``parent_ids`` is set only in a daughter's FIRST frame, so plain slicing loses every
    division whose daughter is born in a skipped frame; :func:`division_affected_cells`
    then misses it and the daughter looks like a newcomer.  Here a label that appears in
    a kept frame without being in the previous kept frame gets its recorded parent back."""
    if stride <= 1:
        return list(frames)
    parent_of: dict[int, int] = {}
    for f in frames:
        for lab, p in zip(f["assigned_ids"].tolist(), f["parent_ids"].tolist()):
            if p >= 0:
                parent_of.setdefault(int(lab), int(p))
    kept = list(frames[::stride])
    out = [kept[0]]
    for prev, f in zip(kept[:-1], kept[1:]):
        before = set(int(l) for l in prev["assigned_ids"])
        p = f["parent_ids"].copy()
        for i, lab in enumerate(f["assigned_ids"].tolist()):
            if p[i] < 0 and lab not in before and lab in parent_of:
                p[i] = parent_of[lab]
        out.append({**f, "parent_ids": p})
    return out


def mask_cells(result: dict, excluded: list[set[int]]) -> dict:
    """Return a copy of a ``compute_neighbour_continuity_observed`` result with the
    excluded (pair, label) entries set to NaN in every per-cell array."""
    cell_ids = result["cell_ids"]
    col = {int(c): i for i, c in enumerate(cell_ids)}
    out = {}
    for key, val in result.items():
        if isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[1] == len(cell_ids):
            v = val.copy()
            for ti, bad in enumerate(excluded):
                cols = [col[c] for c in bad if c in col]
                if cols:
                    v[ti, cols] = np.nan
            out[key] = v
        else:
            out[key] = val
    return out
