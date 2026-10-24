from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree


@dataclass(slots=True)
class ErrorConfig:
    p_miss: float = 0.05        # per-cell per-frame probability of missed segmentation
    lambda_fp: float = 1.0      # expected false positives per frame (Poisson rate)
    p_switch: float = 0.02      # per-cell per-frame ID switch probability
    switch_among_k: int = 3     # switches only with this many nearest spatial neighbours
    p_merge: float = 0.0        # per-cell probability of merging with nearest neighbour
    p_split: float = 0.0        # per-cell probability of splitting into two detections
    split_distance: float = 5.0 # full separation between the two split detections
    max_switch_distance: float | None = None  # if set: cross-frame switch rule, both
                                              # mis-links (A(t)->B(t+1), B(t)->A(t+1))
                                              # must be within this distance; None =
                                              # original same-frame k-nearest swaps


def _pairwise_distances(positions: np.ndarray) -> np.ndarray:
    diffs = positions[:, None, :] - positions[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(dists, np.inf)
    return dists


def _apply_missed_segmentations(
    frames: np.ndarray,
    p_miss: float,
    rng: np.random.Generator,
) -> tuple[list[dict], list[list[int]]]:
    """Drop each cell independently with probability p_miss."""
    t, n, _ = frames.shape
    observed: list[dict] = []
    missed_log: list[list[int]] = []

    for frame_idx in range(t):
        keep_mask = rng.random(n) >= p_miss
        kept = np.where(keep_mask)[0]
        missed = np.where(~keep_mask)[0].tolist()
        missed_log.append(missed)

        observed.append({
            "positions":    frames[frame_idx][kept].copy(),
            "true_ids":     kept.copy().astype(int),
            "assigned_ids": kept.copy().astype(int),
        })

    return observed, missed_log


def _apply_merges(
    observed: list[dict],
    p_merge: float,
    merge_id_start: int,
    rng: np.random.Generator,
) -> tuple[list[dict], list[list[dict]], int]:
    """Merge nearby cell pairs with probability p_merge.

    For each cell (processed in random order), with probability p_merge it is
    merged with its nearest unprocessed neighbour.  Both true centroids are
    removed and replaced by a single detection at their midpoint.

    The merged detection retains one parent's assigned_id (chosen at random),
    mirroring real tracker behaviour where the merged blob is linked to one
    existing track.  The other parent's track simply ends (equivalent to a
    miss).  ``true_id`` is set to -2 to distinguish merged detections from
    false positives (``true_id = -1``).
    """
    merge_log: list[list[dict]] = []
    result: list[dict] = []
    merge_id = merge_id_start  # kept for API compatibility; no longer consumed

    for frame_idx, frame in enumerate(observed):
        positions    = frame["positions"]
        true_ids     = frame["true_ids"]
        assigned_ids = frame["assigned_ids"]
        n_obs = positions.shape[0]
        frame_merges: list[dict] = []

        if n_obs < 2 or p_merge <= 0.0:
            result.append(frame)
            merge_log.append(frame_merges)
            continue

        # KDTree: O(N log N) vs O(N²) pairwise matrix
        tree = KDTree(positions)
        _, nn_idx = tree.query(positions, k=2, workers=-1)
        nearest = nn_idx[:, 1].astype(int)  # index of nearest neighbour for each cell

        merged: set[int] = set()
        new_positions:    list[np.ndarray] = []
        new_true_ids:     list[int]        = []
        new_assigned_ids: list[int]        = []

        for i in rng.permutation(n_obs):
            i = int(i)
            if i in merged:
                continue
            j = int(nearest[i])
            if j in merged:
                continue
            if rng.random() < p_merge:
                midpoint = (positions[i] + positions[j]) / 2.0
                # one parent's ID survives; the other ends (acts as a miss)
                survivor, lost = (i, j) if rng.random() < 0.5 else (j, i)
                retained_id = int(assigned_ids[survivor])
                new_positions.append(midpoint)
                new_true_ids.append(-2)          # -2 = merged detection
                new_assigned_ids.append(retained_id)
                frame_merges.append({
                    "frame":              frame_idx,
                    "true_id_i":          int(true_ids[i]),
                    "true_id_j":          int(true_ids[j]),
                    "assigned_id":        retained_id,
                    "lost_assigned_id":   int(assigned_ids[lost]),
                    "position":           midpoint.tolist(),
                })
                merged.add(i)
                merged.add(j)

        # Retain all cells that were not merged
        for i in range(n_obs):
            if i not in merged:
                new_positions.append(positions[i])
                new_true_ids.append(int(true_ids[i]))
                new_assigned_ids.append(int(assigned_ids[i]))

        merge_log.append(frame_merges)
        result.append({
            "positions":    np.array(new_positions, dtype=float).reshape(-1, 3),
            "true_ids":     np.array(new_true_ids,     dtype=int),
            "assigned_ids": np.array(new_assigned_ids, dtype=int),
        })

    return result, merge_log, merge_id


def _apply_splits(
    observed: list[dict],
    p_split: float,
    split_distance: float,
    split_id_start: int,
    rng: np.random.Generator,
) -> tuple[list[dict], list[list[dict]], int]:
    """Replace each cell with two nearby detections with probability p_split.

    The two detections are placed at ±``split_distance``/2 from the true
    centroid along a random unit direction.  Both carry the original ``true_id``
    (they spatially correspond to the same cell).  The primary fragment retains
    the original ``assigned_id``; only the spurious second fragment receives a
    fresh ID, modelling over-segmentation rather than a missed detection.
    """
    split_log: list[list[dict]] = []
    result: list[dict] = []
    split_id = split_id_start

    for frame_idx, frame in enumerate(observed):
        positions    = frame["positions"]
        true_ids     = frame["true_ids"]
        assigned_ids = frame["assigned_ids"]
        n_obs = positions.shape[0]
        frame_splits: list[dict] = []

        new_positions:    list[np.ndarray] = []
        new_true_ids:     list[int]        = []
        new_assigned_ids: list[int]        = []

        for i in range(n_obs):
            if p_split > 0.0 and rng.random() < p_split:
                direction = rng.standard_normal(3)
                direction /= np.linalg.norm(direction) + 1e-9
                offset = direction * (split_distance / 2.0)

                pos_a = positions[i] + offset
                pos_b = positions[i] - offset

                new_positions.extend([pos_a, pos_b])
                new_true_ids.extend([int(true_ids[i]), int(true_ids[i])])
                # primary fragment retains the original assigned_id so the
                # tracker can still follow it; only the spurious second
                # fragment gets a fresh ID (over-segmentation, not a miss)
                new_assigned_ids.extend([int(assigned_ids[i]), split_id])

                frame_splits.append({
                    "frame":                frame_idx,
                    "true_id":              int(true_ids[i]),
                    "original_assigned_id": int(assigned_ids[i]),
                    "assigned_id_a":        int(assigned_ids[i]),
                    "assigned_id_b":        split_id,
                    "position_a":           pos_a.tolist(),
                    "position_b":           pos_b.tolist(),
                })
                split_id += 1
            else:
                new_positions.append(positions[i])
                new_true_ids.append(int(true_ids[i]))
                new_assigned_ids.append(int(assigned_ids[i]))

        split_log.append(frame_splits)
        result.append({
            "positions":    np.array(new_positions, dtype=float).reshape(-1, 3),
            "true_ids":     np.array(new_true_ids,     dtype=int),
            "assigned_ids": np.array(new_assigned_ids, dtype=int),
        })

    return result, split_log, split_id


def _apply_false_positives(
    observed: list[dict],
    lambda_fp: float,
    box_size: tuple[float, float, float],
    fp_id_start: int,
    rng: np.random.Generator,
) -> tuple[list[dict], list[list[dict]], int]:
    """Add Poisson(lambda_fp) random detections per frame."""
    box = np.asarray(box_size, dtype=float)
    fp_log: list[list[dict]] = []
    fp_id = fp_id_start

    result: list[dict] = []
    for frame_idx, frame in enumerate(observed):
        n_fp = int(rng.poisson(lambda_fp))
        frame_fps: list[dict] = []

        if n_fp > 0:
            fp_pos      = rng.uniform(np.zeros(3), box, size=(n_fp, 3))
            fp_true     = np.full(n_fp, -1, dtype=int)
            fp_assigned = np.arange(fp_id, fp_id + n_fp, dtype=int)

            for fi in range(n_fp):
                frame_fps.append({
                    "frame":       frame_idx,
                    "assigned_id": int(fp_assigned[fi]),
                    "position":    fp_pos[fi].tolist(),
                })
            fp_id += n_fp

            result.append({
                "positions":    np.vstack([frame["positions"], fp_pos]),
                "true_ids":     np.concatenate([frame["true_ids"], fp_true]),
                "assigned_ids": np.concatenate([frame["assigned_ids"], fp_assigned]),
            })
        else:
            result.append(frame)

        fp_log.append(frame_fps)

    return result, fp_log, fp_id


def _apply_id_switches(
    observed: list[dict],
    p_switch: float,
    switch_among_k: int,
    rng: np.random.Generator,
) -> tuple[list[dict], list[list[dict]]]:
    """Swap assigned_ids between spatially nearest neighbours only."""
    switch_log: list[list[dict]] = []
    result: list[dict] = []

    for frame_idx, frame in enumerate(observed):
        positions    = frame["positions"]
        assigned_ids = frame["assigned_ids"].copy()
        n_obs = positions.shape[0]
        frame_switches: list[dict] = []

        if n_obs >= 2 and p_switch > 0.0:
            k = min(switch_among_k, n_obs - 1)
            # KDTree: O(N log N) vs O(N²) pairwise matrix
            tree = KDTree(positions)
            _, nn_idx = tree.query(positions, k=k + 1, workers=-1)
            nearest = nn_idx[:, 1:].astype(int)  # drop self (col 0)

            swapped: set[int] = set()
            for i in range(n_obs):
                if i in swapped:
                    continue
                if rng.random() < p_switch:
                    candidates = [j for j in nearest[i].tolist() if j not in swapped]
                    if not candidates:
                        continue
                    j = int(rng.choice(candidates))
                    assigned_ids[i], assigned_ids[j] = int(assigned_ids[j]), int(assigned_ids[i])
                    swapped.add(i)
                    swapped.add(j)
                    frame_switches.append({
                        "frame":       frame_idx,
                        "cell_idx_i":  i,
                        "cell_idx_j":  j,
                        "true_id_i":   int(frame["true_ids"][i]),
                        "true_id_j":   int(frame["true_ids"][j]),
                    })

        switch_log.append(frame_switches)
        result.append({**frame, "assigned_ids": assigned_ids})

    return result, switch_log


def _apply_id_switches_cross_frame(
    observed: list[dict],
    p_switch: float,
    max_switch_distance: float,
    rng: np.random.Generator,
) -> tuple[list[dict], list[list[dict]]]:
    """ID switches as a tracker would make them: decided AFTER the cells have moved.

    At each frame f >= 1, cell A may swap identity with cell B only if both
    mis-links are within ``max_switch_distance`` (the tracker's maximum link
    distance):

        |A(f-1) - B(f)| <= cap   and   |B(f-1) - A(f)| <= cap

    where X(f-1) is that cell's detected position in the previous frame (matched by
    ``true_ids``; detections absent from f-1 cannot switch).  Each cell attempts a
    switch with probability ``p_switch``; the partner is drawn at random among the
    eligible cells.  A switch persists: the two labels stay exchanged in every later
    frame, as when a tracker follows the wrong cell from then on.
    """
    relabel: dict[int, int] = {}          # label -> label, accumulated swaps
    switch_log: list[list[dict]] = []
    result: list[dict] = []
    prev_pos: dict[int, np.ndarray] = {}

    for frame_idx, frame in enumerate(observed):
        positions = frame["positions"]
        true_ids = frame["true_ids"]
        assigned = np.array([relabel.get(int(a), int(a)) for a in frame["assigned_ids"]],
                            dtype=int)
        frame_switches: list[dict] = []

        if frame_idx > 0 and p_switch > 0.0 and len(positions) >= 2 and prev_pos:
            has_prev = np.array([int(t) in prev_pos for t in true_ids])
            prev = np.array([prev_pos[int(t)] if h else (np.nan, np.nan, np.nan)
                             for t, h in zip(true_ids, has_prev)], dtype=float)
            tree = KDTree(positions)
            swapped: set[int] = set()
            for i in rng.permutation(len(positions)):
                i = int(i)
                if i in swapped or not has_prev[i] or rng.random() >= p_switch:
                    continue
                # B(f) within the cap of A(f-1) ...
                cand = [j for j in tree.query_ball_point(prev[i], max_switch_distance)
                        if j != i and j not in swapped and has_prev[j]]
                # ... and A(f) within the cap of B(f-1)
                cand = [j for j in cand
                        if np.linalg.norm(prev[j] - positions[i]) <= max_switch_distance]
                if not cand:
                    continue
                j = int(rng.choice(cand))
                a, b = int(assigned[i]), int(assigned[j])
                assigned[i], assigned[j] = b, a
                # persist: from now on, every original label currently shown as a is
                # shown as b and vice versa (simultaneous swap of the two classes)
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


def apply_tracking_errors(
    frames: np.ndarray,
    config: ErrorConfig,
    box_size: tuple[float, float, float],
    rng: np.random.Generator,
) -> tuple[list[dict], dict]:
    """Apply a full suite of segmentation and tracking errors to ground truth frames.

    Errors are applied in order:
        1. Missed segmentations  — cells randomly dropped
        2. Merges                — nearby cell pairs collapsed to one midpoint detection
        3. Splits                — cells duplicated into two nearby detections
        4. False positives       — random spurious detections added
        5. ID switches           — assigned IDs swapped between spatial neighbours

    Merge and split IDs are allocated before FP IDs so all synthetic detection
    IDs are globally unique.

    Parameters
    ----------
    frames:
        Ground truth frames, shape (T, N, 3).
    config:
        Error configuration.
    box_size:
        Box dimensions used to place false positive detections.
    rng:
        Random generator.

    Returns
    -------
    observed_frames:
        List of T dicts, each with:
        - ``positions``    (N_obs, 3) — detected positions
        - ``true_ids``     (N_obs,)   — ground truth cell index;
                                        -1 = false positive,
                                        -2 = merged detection
        - ``assigned_ids`` (N_obs,)   — tracker-assigned ID
    error_log:
        Dict recording each error event:
        - ``missed``          list of {frame, true_id}
        - ``merges``          list of {frame, true_id_i, true_id_j, assigned_id, position}
        - ``splits``          list of {frame, true_id, original_assigned_id,
                                        assigned_id_a, assigned_id_b,
                                        position_a, position_b}
        - ``false_positives`` list of {frame, assigned_id, position}
        - ``switches``        list of {frame, cell_idx_i, cell_idx_j,
                                        true_id_i, true_id_j}
    """
    _, n, _ = frames.shape

    # 1. Missed segmentations
    observed, missed_log = _apply_missed_segmentations(frames, config.p_miss, rng)

    # 2. Merges (IDs start after the N ground-truth cells)
    observed, merge_log, next_id = _apply_merges(
        observed, config.p_merge, merge_id_start=n, rng=rng
    )

    # 3. Splits (IDs continue after merge IDs)
    observed, split_log, next_id = _apply_splits(
        observed, config.p_split, config.split_distance,
        split_id_start=next_id, rng=rng
    )

    # 4. False positives (IDs continue after split IDs)
    observed, fp_log, _ = _apply_false_positives(
        observed, config.lambda_fp, box_size, fp_id_start=next_id, rng=rng
    )

    # 5. ID switches
    if config.max_switch_distance is None:
        observed, switch_log = _apply_id_switches(
            observed, config.p_switch, config.switch_among_k, rng
        )
    else:
        observed, switch_log = _apply_id_switches_cross_frame(
            observed, config.p_switch, config.max_switch_distance, rng
        )

    error_log = {
        "missed": [
            {"frame": fi, "true_id": ci}
            for fi, missed in enumerate(missed_log)
            for ci in missed
        ],
        "merges": [ev for frame_evs in merge_log for ev in frame_evs],
        "splits": [ev for frame_evs in split_log for ev in frame_evs],
        "false_positives": [fp for frame_fps in fp_log for fp in frame_fps],
        "switches": [sw for frame_sws in switch_log for sw in frame_sws],
    }

    return observed, error_log
