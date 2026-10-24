"""Injection of segmentation and tracking errors into clean tracks.

Input may be either a dense ground-truth array of shape ``(T, N, d)`` (every cell
present in every frame, as produced by the simulator) or a list of per-frame
dicts with a variable number of detections (as produced by ``ctc.read_ctc_tracks``
or the lineage helpers)::

    {"positions":    (N_t, d) float,
     "true_ids":     (N_t,)   int,
     "assigned_ids": (N_t,)   int,
     ...any other per-detection arrays of length N_t, e.g. "parent_ids"}

All steps operate on the dict form.  Extra per-detection arrays are carried
through: kept detections keep their values, a merged detection takes the
surviving parent's values, and synthetic detections (split fragments, false
positives) receive a fill value (-1 for integer arrays, NaN for floats, False
for booleans).

Sentinels in ``true_ids``: ``-1`` = false positive, ``-2`` = merged detection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree

_CORE_KEYS = ("positions", "true_ids", "assigned_ids")


@dataclass(slots=True)
class ErrorConfig:
    p_miss: float = 0.05        # per-cell per-frame probability of missed segmentation
    lambda_fp: float = 1.0      # expected false positives per frame (Poisson rate) ...
    fp_rate: float | None = None  # ... or, if set, FPs per frame = Poisson(fp_rate * N_t):
                                  # a fraction of the detections present in that frame
    fp_placement: str = "uniform"   # "uniform": anywhere in box_size;
                                    # "near_cells": at a random detection plus a random
                                    # offset of fp_offset * U(0.5, 1.5)
    fp_offset: float = 5.0          # distance scale for "near_cells" (position units)
    p_switch: float = 0.02      # per-cell per-frame ID switch probability
    switch_among_k: int = 3     # switches only with this many nearest spatial neighbours
    p_merge: float = 0.0        # per-cell probability of merging with nearest neighbour
    p_split: float = 0.0        # per-cell probability of splitting into two detections
    split_distance: float = 5.0 # full separation between the two split detections
    max_switch_distance: float | None = None  # if set: cross-frame switch rule, both
                                              # mis-links (A(t)->B(t+1), B(t)->A(t+1))
                                              # must be within this distance; None =
                                              # original same-frame k-nearest swaps

    def __post_init__(self) -> None:
        if self.fp_placement not in ("uniform", "near_cells"):
            raise ValueError("fp_placement must be 'uniform' or 'near_cells'")


# --------------------------------------------------------------------------- frame helpers

def frames_from_array(frames: np.ndarray) -> list[dict]:
    """Dense ``(T, N, d)`` ground truth -> list of per-frame dicts (ids = 0..N-1)."""
    frames = np.asarray(frames, dtype=float)
    if frames.ndim != 3:
        raise ValueError(f"expected (T, N, d) array, got shape {frames.shape}")
    t, n, _ = frames.shape
    ids = np.arange(n, dtype=int)
    return [{"positions": frames[i].copy(), "true_ids": ids.copy(), "assigned_ids": ids.copy()}
            for i in range(t)]


def _as_frames(frames) -> list[dict]:
    if isinstance(frames, np.ndarray):
        return frames_from_array(frames)
    out = []
    for i, f in enumerate(frames):
        missing = [k for k in _CORE_KEYS if k not in f]
        if missing:
            raise ValueError(f"frame {i} lacks keys {missing}")
        out.append(f)
    return out


def _extra_keys(frame: dict) -> list[str]:
    n = frame["positions"].shape[0]
    return [k for k, v in frame.items()
            if k not in _CORE_KEYS and isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n]


def _fill_value(arr: np.ndarray):
    if arr.dtype.kind in "iu":
        return -1
    if arr.dtype.kind == "b":
        return False
    return np.nan


def _subset(frame: dict, idx: np.ndarray) -> dict:
    """Rows ``idx`` of every per-detection array (other keys copied as-is)."""
    out = {}
    n = frame["positions"].shape[0]
    for k, v in frame.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n:
            out[k] = v[idx].copy()
        else:
            out[k] = v
    return out


def _rebuild(frame: dict, positions, true_ids, assigned_ids, src_rows: list[int | None]) -> dict:
    """New frame from lists; ``src_rows[i]`` = source row for extra arrays (None = fill)."""
    d = frame["positions"].shape[1]
    out = dict(frame)
    out["positions"]    = np.asarray(positions, dtype=float).reshape(-1, d)
    out["true_ids"]     = np.asarray(true_ids, dtype=int)
    out["assigned_ids"] = np.asarray(assigned_ids, dtype=int)
    for k in _extra_keys(frame):
        src = frame[k]
        fill = _fill_value(src)
        new = np.empty((len(src_rows),) + src.shape[1:], dtype=src.dtype)
        for i, r in enumerate(src_rows):
            new[i] = src[r] if r is not None else fill
        out[k] = new
    return out


def _append(frame: dict, positions: np.ndarray, true_ids: np.ndarray, assigned_ids: np.ndarray) -> dict:
    """Append synthetic detections (extras filled)."""
    out = dict(frame)
    out["positions"]    = np.vstack([frame["positions"], positions])
    out["true_ids"]     = np.concatenate([frame["true_ids"], true_ids])
    out["assigned_ids"] = np.concatenate([frame["assigned_ids"], assigned_ids])
    m = len(true_ids)
    for k in _extra_keys(frame):
        src = frame[k]
        pad = np.full((m,) + src.shape[1:], _fill_value(src), dtype=src.dtype)
        out[k] = np.concatenate([src, pad])
    return out


# --------------------------------------------------------------------------- 1. misses

def _apply_missed_segmentations(
    frames,
    p_miss: float,
    rng: np.random.Generator,
) -> tuple[list[dict], list[list[int]]]:
    """Drop each detection independently with probability p_miss.

    Accepts a dense ``(T, N, d)`` array or a list of frame dicts.  The log holds
    the ``true_ids`` of dropped detections per frame.
    """
    observed_in = _as_frames(frames)
    observed: list[dict] = []
    missed_log: list[list[int]] = []
    for frame in observed_in:
        n = frame["positions"].shape[0]
        keep = rng.random(n) >= p_miss if p_miss > 0.0 else np.ones(n, dtype=bool)
        missed_log.append(frame["true_ids"][~keep].astype(int).tolist())
        observed.append(_subset(frame, np.where(keep)[0]))
    return observed, missed_log


# --------------------------------------------------------------------------- 2. merges

def _apply_merges(
    observed: list[dict],
    p_merge: float,
    merge_id_start: int,
    rng: np.random.Generator,
) -> tuple[list[dict], list[list[dict]], int]:
    """Merge nearby detection pairs with probability p_merge.

    For each detection (random order), with probability p_merge it is merged
    with its nearest unmerged neighbour.  Both are replaced by one detection at
    their midpoint which keeps one parent's ``assigned_id`` (chosen at random)
    and that parent's extra fields; the other parent's track simply ends.
    ``true_id`` is set to -2.  ``merge_id_start`` is returned unchanged (no fresh
    IDs are allocated); it is kept so callers can chain it into ``_apply_splits``.
    """
    merge_log: list[list[dict]] = []
    result: list[dict] = []

    for frame_idx, frame in enumerate(observed):
        positions, true_ids, assigned = frame["positions"], frame["true_ids"], frame["assigned_ids"]
        n_obs = positions.shape[0]
        frame_merges: list[dict] = []
        if n_obs < 2 or p_merge <= 0.0:
            result.append(frame); merge_log.append(frame_merges); continue

        _, nn_idx = KDTree(positions).query(positions, k=2, workers=-1)
        nearest = nn_idx[:, 1].astype(int)
        merged: set[int] = set()
        pos_l, true_l, asg_l, src_l = [], [], [], []

        for i in rng.permutation(n_obs):
            i = int(i)
            if i in merged:
                continue
            j = int(nearest[i])
            if j in merged or rng.random() >= p_merge:
                continue
            survivor, lost = (i, j) if rng.random() < 0.5 else (j, i)
            midpoint = (positions[i] + positions[j]) / 2.0
            pos_l.append(midpoint); true_l.append(-2); asg_l.append(int(assigned[survivor])); src_l.append(survivor)
            frame_merges.append({
                "frame": frame_idx, "true_id_i": int(true_ids[i]), "true_id_j": int(true_ids[j]),
                "assigned_id": int(assigned[survivor]), "lost_assigned_id": int(assigned[lost]),
                "position": midpoint.tolist(),
            })
            merged.update((i, j))

        for i in range(n_obs):
            if i not in merged:
                pos_l.append(positions[i]); true_l.append(int(true_ids[i])); asg_l.append(int(assigned[i])); src_l.append(i)

        merge_log.append(frame_merges)
        result.append(_rebuild(frame, pos_l, true_l, asg_l, src_l))

    return result, merge_log, merge_id_start


# --------------------------------------------------------------------------- 3. splits

def _apply_splits(
    observed: list[dict],
    p_split: float,
    split_distance: float,
    split_id_start: int,
    rng: np.random.Generator,
) -> tuple[list[dict], list[list[dict]], int]:
    """Replace a detection by two at ±split_distance/2 along a random direction.

    Both fragments carry the original ``true_id``.  The primary keeps the
    original ``assigned_id`` and extra fields; the spurious fragment gets a
    fresh ID and filled extras (over-segmentation, not a missed detection).
    """
    split_log: list[list[dict]] = []
    result: list[dict] = []
    split_id = split_id_start

    for frame_idx, frame in enumerate(observed):
        positions, true_ids, assigned = frame["positions"], frame["true_ids"], frame["assigned_ids"]
        n_obs, d = positions.shape
        frame_splits: list[dict] = []
        if n_obs == 0 or p_split <= 0.0:
            result.append(frame); split_log.append(frame_splits); continue

        pos_l, true_l, asg_l, src_l = [], [], [], []
        for i in range(n_obs):
            if rng.random() < p_split:
                direction = rng.standard_normal(d)
                direction /= np.linalg.norm(direction) + 1e-12
                offset = direction * (split_distance / 2.0)
                pos_a, pos_b = positions[i] + offset, positions[i] - offset
                pos_l.extend([pos_a, pos_b]); true_l.extend([int(true_ids[i])] * 2)
                asg_l.extend([int(assigned[i]), split_id]); src_l.extend([i, None])
                frame_splits.append({
                    "frame": frame_idx, "true_id": int(true_ids[i]),
                    "original_assigned_id": int(assigned[i]),
                    "assigned_id_a": int(assigned[i]), "assigned_id_b": split_id,
                    "position_a": pos_a.tolist(), "position_b": pos_b.tolist(),
                })
                split_id += 1
            else:
                pos_l.append(positions[i]); true_l.append(int(true_ids[i])); asg_l.append(int(assigned[i])); src_l.append(i)

        split_log.append(frame_splits)
        result.append(_rebuild(frame, pos_l, true_l, asg_l, src_l))

    return result, split_log, split_id


# --------------------------------------------------------------------------- 4. false positives

def _apply_false_positives(
    observed: list[dict],
    lambda_fp: float,
    box_size: tuple[float, ...],
    fp_id_start: int,
    rng: np.random.Generator,
    fp_rate: float | None = None,
    fp_placement: str = "uniform",
    fp_offset: float = 5.0,
) -> tuple[list[dict], list[list[dict]], int]:
    """Add spurious detections per frame.

    Count per frame is Poisson(lambda_fp), or Poisson(fp_rate * N_t) if ``fp_rate``
    is given.  Placement is uniform in the box, or (``"near_cells"``) at a random
    existing detection plus a random direction scaled by fp_offset * U(0.5, 1.5),
    which keeps spurious detections inside a tissue that does not fill its box.
    """
    box = np.asarray(box_size, dtype=float)
    fp_log: list[list[dict]] = []
    fp_id = fp_id_start
    result: list[dict] = []

    for frame_idx, frame in enumerate(observed):
        n_obs, d = frame["positions"].shape
        rate = fp_rate * n_obs if fp_rate is not None else lambda_fp
        n_fp = int(rng.poisson(rate)) if rate > 0 else 0
        frame_fps: list[dict] = []
        if n_fp == 0 or (fp_placement == "near_cells" and n_obs == 0):
            result.append(frame); fp_log.append(frame_fps); continue

        if fp_placement == "near_cells":
            anchors = frame["positions"][rng.integers(0, n_obs, size=n_fp)]
            dirs = rng.standard_normal((n_fp, d))
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
            fp_pos = anchors + dirs * (fp_offset * rng.uniform(0.5, 1.5, size=(n_fp, 1)))
            fp_pos = np.clip(fp_pos, 0.0, box[:d])
        else:
            fp_pos = rng.uniform(np.zeros(d), box[:d], size=(n_fp, d))

        fp_true = np.full(n_fp, -1, dtype=int)
        fp_assigned = np.arange(fp_id, fp_id + n_fp, dtype=int)
        for fi in range(n_fp):
            frame_fps.append({"frame": frame_idx, "assigned_id": int(fp_assigned[fi]),
                              "position": fp_pos[fi].tolist()})
        fp_id += n_fp
        result.append(_append(frame, fp_pos, fp_true, fp_assigned))
        fp_log.append(frame_fps)

    return result, fp_log, fp_id


# --------------------------------------------------------------------------- 5a. same-frame switches

def _apply_id_switches(
    observed: list[dict],
    p_switch: float,
    switch_among_k: int,
    rng: np.random.Generator,
) -> tuple[list[dict], list[list[dict]]]:
    """Swap assigned_ids between spatially nearest neighbours, per frame (not persistent)."""
    switch_log: list[list[dict]] = []
    result: list[dict] = []

    for frame_idx, frame in enumerate(observed):
        positions = frame["positions"]
        assigned_ids = frame["assigned_ids"].copy()
        n_obs = positions.shape[0]
        frame_switches: list[dict] = []

        if n_obs >= 2 and p_switch > 0.0:
            k = min(switch_among_k, n_obs - 1)
            _, nn_idx = KDTree(positions).query(positions, k=k + 1, workers=-1)
            nearest = nn_idx[:, 1:].astype(int)
            swapped: set[int] = set()
            for i in range(n_obs):
                if i in swapped or rng.random() >= p_switch:
                    continue
                candidates = [j for j in nearest[i].tolist() if j not in swapped]
                if not candidates:
                    continue
                j = int(rng.choice(candidates))
                assigned_ids[i], assigned_ids[j] = int(assigned_ids[j]), int(assigned_ids[i])
                swapped.update((i, j))
                frame_switches.append({
                    "frame": frame_idx, "cell_idx_i": i, "cell_idx_j": j,
                    "true_id_i": int(frame["true_ids"][i]), "true_id_j": int(frame["true_ids"][j]),
                })

        switch_log.append(frame_switches)
        result.append({**frame, "assigned_ids": assigned_ids})

    return result, switch_log


# --------------------------------------------------------------------------- 5b. cross-frame switches

def _apply_id_switches_cross_frame(
    observed: list[dict],
    p_switch: float,
    max_switch_distance: float,
    rng: np.random.Generator,
) -> tuple[list[dict], list[list[dict]]]:
    """ID switches as a tracker would make them: decided AFTER the cells have moved.

    At each frame f >= 1, detection A may swap identity with B only if both
    mis-links are within ``max_switch_distance``:

        |A(f-1) - B(f)| <= cap   and   |B(f-1) - A(f)| <= cap

    where X(f-1) is that cell's detected position in the previous frame (matched
    by ``true_ids``; detections absent from f-1, including newborn daughters,
    cannot switch).  Each eligible detection attempts a switch with probability
    ``p_switch``; the partner is drawn at random among eligible cells.  A switch
    persists: the two labels stay exchanged in every later frame.
    """
    relabel: dict[int, int] = {}
    switch_log: list[list[dict]] = []
    result: list[dict] = []
    prev_pos: dict[int, np.ndarray] = {}

    for frame_idx, frame in enumerate(observed):
        positions = frame["positions"]
        true_ids = frame["true_ids"]
        assigned = np.array([relabel.get(int(a), int(a)) for a in frame["assigned_ids"]], dtype=int)
        frame_switches: list[dict] = []

        if frame_idx > 0 and p_switch > 0.0 and len(positions) >= 2 and prev_pos:
            has_prev = np.array([int(t) in prev_pos for t in true_ids])
            d = positions.shape[1]
            prev = np.array([prev_pos[int(t)] if h else np.full(d, np.nan)
                             for t, h in zip(true_ids, has_prev)], dtype=float)
            tree = KDTree(positions)
            swapped: set[int] = set()
            for i in rng.permutation(len(positions)):
                i = int(i)
                if i in swapped or not has_prev[i] or rng.random() >= p_switch:
                    continue
                cand = [j for j in tree.query_ball_point(prev[i], max_switch_distance)
                        if j != i and j not in swapped and has_prev[j]]
                cand = [j for j in cand
                        if np.linalg.norm(prev[j] - positions[i]) <= max_switch_distance]
                if not cand:
                    continue
                j = int(rng.choice(cand))
                a, b = int(assigned[i]), int(assigned[j])
                assigned[i], assigned[j] = b, a
                src_a = [k for k, v in relabel.items() if v == a] + ([a] if a not in relabel else [])
                src_b = [k for k, v in relabel.items() if v == b] + ([b] if b not in relabel else [])
                for k in src_a:
                    relabel[k] = b
                for k in src_b:
                    relabel[k] = a
                swapped.update((i, j))
                frame_switches.append({
                    "frame": frame_idx, "cell_idx_i": i, "cell_idx_j": j,
                    "true_id_i": int(true_ids[i]), "true_id_j": int(true_ids[j]),
                })

        switch_log.append(frame_switches)
        result.append({**frame, "assigned_ids": assigned})
        prev_pos = {int(t): positions[k] for k, t in enumerate(true_ids) if int(t) >= 0}

    return result, switch_log


# --------------------------------------------------------------------------- driver

def apply_tracking_errors(
    frames,
    config: ErrorConfig,
    box_size: tuple[float, ...],
    rng: np.random.Generator,
) -> tuple[list[dict], dict]:
    """Apply a full suite of segmentation and tracking errors to clean tracks.

    Errors are applied in order:
        1. Missed segmentations  — detections randomly dropped
        2. Merges                — nearby pairs collapsed to one midpoint detection
        3. Splits                — detections duplicated into two nearby ones
        4. False positives       — spurious detections added
        5. ID switches           — assigned IDs swapped (same-frame or cross-frame rule)

    Split and FP IDs are allocated from one counter starting above the largest
    existing ``assigned_id``, so synthetic IDs never collide with real ones.

    Two ID-switch rules are available (``config.max_switch_distance``):

    * ``None`` (same-frame rule): swaps are NOT persistent — labels revert at the
      next frame — and the rule fires in every frame, so over a single frame pair
      the fraction of cells with a label change is roughly ``4 * p_switch``.
    * a distance (cross-frame rule): switches are decided after the cells have
      moved, both mis-links must be within the cap, and the swap persists, as a
      real tracker's wrong link would.  Across one jump the label-change fraction
      is roughly ``2 * p_switch``.

    Parameters
    ----------
    frames:
        Dense ``(T, N, d)`` array, or list of per-frame dicts with ``positions``,
        ``true_ids``, ``assigned_ids`` and optional extra per-detection arrays
        (e.g. ``parent_ids``), which are carried through.
    config, box_size, rng:
        Error configuration, box used to place/clip false positives, random generator.

    Returns
    -------
    observed_frames:
        List of T dicts with ``positions``, ``true_ids`` (-1 = false positive,
        -2 = merged), ``assigned_ids`` and the carried extra arrays.
    error_log:
        ``missed``, ``merges``, ``splits``, ``false_positives``, ``switches``:
        flat lists of event dicts, each with a ``frame`` key.
    """
    clean = _as_frames(frames)
    next_id = int(max((int(f["assigned_ids"].max()) for f in clean if len(f["assigned_ids"])), default=-1)) + 1

    observed, missed_log = _apply_missed_segmentations(clean, config.p_miss, rng)
    observed, merge_log, next_id = _apply_merges(observed, config.p_merge, merge_id_start=next_id, rng=rng)
    observed, split_log, next_id = _apply_splits(
        observed, config.p_split, config.split_distance, split_id_start=next_id, rng=rng)
    observed, fp_log, _ = _apply_false_positives(
        observed, config.lambda_fp, box_size, fp_id_start=next_id, rng=rng,
        fp_rate=config.fp_rate, fp_placement=config.fp_placement, fp_offset=config.fp_offset)
    if config.max_switch_distance is None:
        observed, switch_log = _apply_id_switches(observed, config.p_switch, config.switch_among_k, rng)
    else:
        observed, switch_log = _apply_id_switches_cross_frame(
            observed, config.p_switch, config.max_switch_distance, rng)

    error_log = {
        "missed": [{"frame": fi, "true_id": ci} for fi, missed in enumerate(missed_log) for ci in missed],
        "merges": [ev for evs in merge_log for ev in evs],
        "splits": [ev for evs in split_log for ev in evs],
        "false_positives": [fp for fps in fp_log for fp in fps],
        "switches": [sw for sws in switch_log for sw in sws],
    }
    return observed, error_log


def realised_error_rates(clean, observed: list[dict], error_log: dict) -> dict[str, float]:
    """Events per clean detection, pooled over frames: what actually happened.

    ``miss``: dropped detections / clean detections.  ``fp``: added / clean.
    ``switch``: cells involved (2 per event) / clean.  ``merge``: cells involved
    (2 per event) / clean.  ``split``: split events / clean.
    """
    clean = _as_frames(clean)
    n_clean = sum(len(f["assigned_ids"]) for f in clean)
    if n_clean == 0:
        return {k: float("nan") for k in ("miss", "fp", "switch", "merge", "split")}
    return {
        "miss":   len(error_log["missed"]) / n_clean,
        "fp":     len(error_log["false_positives"]) / n_clean,
        "switch": 2 * len(error_log["switches"]) / n_clean,
        "merge":  2 * len(error_log["merges"]) / n_clean,
        "split":  len(error_log["splits"]) / n_clean,
    }
