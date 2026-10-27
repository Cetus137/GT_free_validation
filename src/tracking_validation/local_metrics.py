"""Reference-free local metrics: every test is referenced to the unaffected cells of the SAME
frame pair, so no external clean baseline is needed.

1. ``classify_events``  identity events by position.
   Each vanished label (at t, not at t+1) gets a predicted t+1 position from its neighbours'
   motion.  A NEW label (at t+1, not at t) close to that prediction -> fragmentation (same
   body, new label); no new body there -> vanished without a body (a miss, or a false
   detection that ended).  New labels left unmatched -> new without a predecessor (a false
   positive, a cell re-detected after a miss, or a missed division).
   "Close" is a structural radius, 0.75 median spacings: nearer than any other cell would be.

2. ``cell_scores`` + ``bulk_tail``  per-cell disruption scored against the pair's own bulk.
   Scores per persisting label: displaced share, 1 - rank correlation, motion residual.
   Biology shifts the whole distribution; errors add a tail.  The tail threshold is
   q50 + K * (q90 - q50) of the same pair's scores (fixed K, stated a priori).

3. Motion residual (inside ``cell_scores``): a label's step minus the median step of its
   retained Delaunay neighbours, in units of the pair's median spacing.  Embryo motion is
   locally coherent; a wrong link moves a label out of step with the local flow, which may
   expose hop chains that keep neighbourhoods intact.

Frames are the per-frame dicts used throughout (``positions``, ``assigned_ids``).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .metrics import _frame_neighbour_map_delaunay, compute_neighbour_continuity_observed

K_TAIL = 3.0       # tail threshold: q50 + K_TAIL * (q90 - q50) of the pair's own scores
R_BODY = 0.75      # body-present radius, in median spacings (structural; see classify_events)
MIN_SPREAD = 0.05  # floor on (q90 - q50), in the score's units, so a pair with a degenerate bulk is not all tail


def _robust_threshold(x: np.ndarray, k: float, floor: float) -> float:
    x = x[np.isfinite(x)]
    if x.size < 10:
        return np.inf
    q50, q90 = np.percentile(x, [50, 90])
    return float(q50 + k * max(q90 - q50, floor))


def predicted_positions(f0: dict, f1: dict, cap: float):
    """For every label at t: predicted t+1 position = own t position + median step of its
    Delaunay neighbours whose labels persist (falls back to the pair's median step).
    Returns (pred: {label: xyz}, step: {label: xyz} for persisting labels, spacing)."""
    p0, l0 = f0["positions"], f0["assigned_ids"]
    p1, l1 = f1["positions"], f1["assigned_ids"]
    at1 = {int(l): p for l, p in zip(l1, p1)}
    step = {int(l): at1[int(l)] - p for l, p in zip(l0, p0) if int(l) in at1}
    nb, _ = _frame_neighbour_map_delaunay(p0, l0, max_distance=cap)
    glob = np.median(np.array(list(step.values())), axis=0) if step else np.zeros(p0.shape[1])
    pred = {}
    for l, p in zip(l0, p0):
        s = [step[j] for j in nb.get(int(l), []) if j in step]
        pred[int(l)] = p + (np.median(np.array(s), axis=0) if s else glob)
    sp = float(np.median(cKDTree(p0).query(p0, k=2)[0][:, 1])) if len(p0) > 2 else 1.0
    return pred, step, sp


def classify_events(f0: dict, f1: dict, cap: float) -> dict:
    """Identity events of one pair.  Returns label lists:
    ``frag`` (vanished label, new label) pairs; ``vanished_nobody`` labels at t;
    ``new_unmatched`` labels at t+1; plus the body radius used."""
    l0 = [int(l) for l in f0["assigned_ids"]]; l1 = [int(l) for l in f1["assigned_ids"]]
    s0, s1 = set(l0), set(l1)
    pred, step, sp = predicted_positions(f0, f1, cap)
    at0 = {l: p for l, p in zip(l0, f0["positions"])}
    at1 = {l: p for l, p in zip(l1, f1["positions"])}
    # structural radius: closer than any OTHER cell would be (the closest 10% of neighbours sit
    # ~0.75 median spacings apart).  A radius fitted to retained cells is biased small: with a
    # short link cap, only short steps are retained, while fragmented cells are exactly those
    # that moved further (0.5-0.7 spacings in the C. elegans test).
    radius = R_BODY * sp
    van = [l for l in l0 if l not in s1]
    new = [l for l in l1 if l not in s0]
    cand = []
    for v in van:
        for n in new:
            d = float(np.linalg.norm(at1[n] - pred[v]))
            if d <= radius:
                cand.append((d, v, n))
    cand.sort()
    used_v, used_n, frag = set(), set(), []
    for d, v, n in cand:
        if v in used_v or n in used_n:
            continue
        used_v.add(v); used_n.add(n); frag.append((v, n))
    return dict(frag=frag, vanished_nobody=[v for v in van if v not in used_v],
                new_unmatched=[n for n in new if n not in used_n], radius=radius, spacing=sp,
                pos0=at0, pos1=at1)


def cell_scores(f0: dict, f1: dict, cap: float) -> dict:
    """Per persisting label at t: displaced share, 1 - rank correlation (Delaunay
    neighbourhoods) and motion residual / spacing.  Returns {label: (disp, 1-rc, resid)}."""
    res = compute_neighbour_continuity_observed([f0, f1], neighbourhood_mode="delaunay",
                                                max_neighbour_distance=cap, extra_metrics=True)
    disp = dict(zip(res["cell_ids"].tolist(), res["wio_displaced"][0]))
    rc = dict(zip(res["cell_ids"].tolist(), res["rank_correlation"][0]))
    pred, step, sp = predicted_positions(f0, f1, cap)
    p0 = {int(l): p for l, p in zip(f0["assigned_ids"], f0["positions"])}
    out = {}
    for l, s in step.items():
        r = float(np.linalg.norm(p0[l] + s - pred[l])) / sp
        out[l] = (float(disp.get(l, np.nan)), 1.0 - float(rc.get(l, np.nan)), r)
    return out


def bulk_tail(values: np.ndarray, k: float = K_TAIL, floor: float = MIN_SPREAD) -> np.ndarray:
    """Boolean mask of tail cells relative to the same pair's bulk."""
    thr = _robust_threshold(np.asarray(values, float), k, floor)
    return np.asarray(values, float) > thr
