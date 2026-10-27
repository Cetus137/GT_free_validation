"""Ground-truth error counts for a real tracker's output, per frame pair, in the same
units as the injected-error sweeps (events per evaluated GT cell at t+1).

Unlike the injection experiments, frame t is NOT clean: detections can be missing or
spurious in both frames.  Counts are therefore defined by what the neighbourhood
metrics of pair (t, t+1) actually see, i.e. by the tracker's LABELS in the two frames.

For pair (t, t+1) and evaluated GT cells E (division-free cells present at t and t+1):
  miss      cell detected at t, not detected at t+1                (label vanishes)
  switch    cell detected at t+1 whose label belonged to a DIFFERENT detection at t
            (a real cell or a false detection); both cells of a swap count, as in the sweeps
  frag      cell detected at t and t+1, new label at t+1           (track broken)
  reappear  cell not detected at t, detected at t+1 with a new label (looks like a newcomer)
  absent    cell detected in neither frame                         (invisible to the pair)
  fp        false detection at t+1 with a label not present at t, whose nearest GT cell
            at t+1 is in E (so it can enter an evaluated neighbourhood)
  fp_carry  false detection at t+1 carrying a label present at t (persistent FP, or a
            track that jumped onto a false detection)
All rates use |E| as denominator.

Inputs are the per-frame dicts used throughout the package: ``positions``,
``assigned_ids`` (tracker labels) and ``true_ids`` (GT label after daughter inheritance,
-1 for a false detection).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

COUNTS = ("miss", "switch", "frag", "reappear", "absent", "fp", "fp_carry")


def pair_errors(f0: dict, f1: dict, evaluate: set[int], gt1_pos: np.ndarray, gt1_ids: np.ndarray) -> dict:
    """Error counts for one frame pair.  ``gt1_pos``/``gt1_ids``: all GT cells at t+1
    (used to attribute false detections to a neighbourhood)."""
    t0, l0 = f0["true_ids"], f0["assigned_ids"]
    t1, l1 = f1["true_ids"], f1["assigned_ids"]
    D0 = {int(c): int(l) for c, l in zip(t0, l0) if c >= 0}
    D1 = {int(c): int(l) for c, l in zip(t1, l1) if c >= 0}
    labels0 = set(int(l) for l in l0)
    out = dict.fromkeys(COUNTS, 0)
    out["n_eval"] = len(evaluate)
    for c in evaluate:
        in0, in1 = c in D0, c in D1
        if in0 and not in1:
            out["miss"] += 1
        elif in1:
            lab = D1[c]
            if in0 and lab == D0[c]:
                continue
            if lab in labels0:
                out["switch"] += 1
            elif in0:
                out["frag"] += 1
            else:
                out["reappear"] += 1
        else:
            out["absent"] += 1
    fp_idx = np.nonzero(np.asarray(t1) < 0)[0]
    if len(fp_idx) and len(gt1_pos):
        _, nn = cKDTree(gt1_pos).query(f1["positions"][fp_idx])
        for i, k in zip(fp_idx, nn):
            if int(gt1_ids[k]) not in evaluate:
                continue
            if int(l1[i]) in labels0:
                out["fp_carry"] += 1
            else:
                out["fp"] += 1
    return out


def sequence_errors(frames: list[dict], gt_frames: list[dict], keep: list[set[int]]) -> list[dict]:
    """``pair_errors`` for every consecutive pair.  ``keep[t]``: evaluated GT labels for
    pair (t, t+1); cells absent from GT at t+1 are dropped from it."""
    rows = []
    for t in range(len(frames) - 1):
        g1 = gt_frames[t + 1]
        ev = keep[t] & set(int(c) for c in g1["true_ids"])
        r = pair_errors(frames[t], frames[t + 1], ev, g1["positions"], g1["true_ids"])
        rows.append(dict(pair=t, **r))
    return rows
