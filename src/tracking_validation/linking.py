"""Two simple frame-to-frame linkers, and exact labelling of their link errors.

Both linkers take the per-frame dict list used throughout the package (with
``positions`` and ``true_ids``; any existing ``assigned_ids`` are ignored) and
return a copy in which ``assigned_ids`` are TRACK LABELS assigned by the
linker: a detection linked from the previous frame inherits that track's
label, an unlinked detection starts a new track.  Labels persist across the
whole sequence, as a real tracker's would.

* :func:`link_greedy` — nearest-neighbour: candidate pairs within the cap are
  taken in order of increasing distance, each detection used at most once.
* :func:`link_lap` — frame-to-frame linear assignment (TrackMate's LAP step
  without gap closing): squared-distance costs, a "no link" alternative at the
  cost of the cap, solved with the Hungarian algorithm.

Neither handles divisions (one-to-one only); daughters start new tracks.

:func:`label_links` classifies every link against ``true_ids`` so the error
composition of the linker output is known exactly, without external tools.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree


@dataclass(slots=True)
class LinkResult:
    frames: list[dict]                       # with linker-assigned ``assigned_ids``
    links: list[np.ndarray]                  # per pair: (M, 2) int array of (idx_t, idx_t1)
    n_tracks: int
    params: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- matching

def _match_greedy(p0: np.ndarray, p1: np.ndarray, cap: float) -> np.ndarray:
    if len(p0) == 0 or len(p1) == 0:
        return np.zeros((0, 2), dtype=int)
    tree = KDTree(p1)
    pairs = tree.query_ball_point(p0, cap)
    cand = [(float(np.linalg.norm(p0[i] - p1[j])), i, j) for i, js in enumerate(pairs) for j in js]
    cand.sort()
    used0: set[int] = set(); used1: set[int] = set(); out = []
    for d, i, j in cand:
        if i in used0 or j in used1:
            continue
        used0.add(i); used1.add(j); out.append((i, j))
    return np.array(out, dtype=int).reshape(-1, 2)


def _match_lap(p0: np.ndarray, p1: np.ndarray, cap: float) -> np.ndarray:
    """Linear assignment with a no-link alternative priced at cap² (squared costs)."""
    n0, n1 = len(p0), len(p1)
    if n0 == 0 or n1 == 0:
        return np.zeros((0, 2), dtype=int)
    big = 1e9
    d2 = ((p0[:, None, :] - p1[None, :, :]) ** 2).sum(-1)
    cost = np.full((n0 + n1, n0 + n1), big)
    cost[:n0, :n1] = np.where(d2 <= cap ** 2, d2, big)       # real links, squared distance
    cost[:n0, n1:] = np.where(np.eye(n0, dtype=bool), cap ** 2, big)   # t cell ends its track
    cost[n0:, :n1] = np.where(np.eye(n1, dtype=bool), cap ** 2, big)   # t+1 cell starts a track
    cost[n0:, n1:] = 0.0      # auxiliary block: free, so every real cost is counted exactly once
    r, c = linear_sum_assignment(cost)
    out = [(int(i), int(j)) for i, j in zip(r, c) if i < n0 and j < n1 and cost[i, j] < big]
    return np.array(out, dtype=int).reshape(-1, 2)


_MATCHERS = {"greedy": _match_greedy, "lap": _match_lap}


# --------------------------------------------------------------------------- linking

def link_frames(frames: list[dict], max_distance: float, method: str = "lap") -> LinkResult:
    """Assign track labels across ``frames`` with the chosen matcher."""
    if method not in _MATCHERS:
        raise ValueError(f"method must be one of {sorted(_MATCHERS)}")
    match = _MATCHERS[method]
    out: list[dict] = []
    links: list[np.ndarray] = []
    next_label = 0
    prev_labels: np.ndarray | None = None
    for t, fr in enumerate(frames):
        n = fr["positions"].shape[0]
        labels = np.full(n, -1, dtype=int)
        if t > 0 and prev_labels is not None:
            m = match(frames[t - 1]["positions"], fr["positions"], max_distance)
            links.append(m)
            for i, j in m:
                labels[j] = prev_labels[i]
        new = labels < 0
        labels[new] = np.arange(next_label, next_label + int(new.sum()))
        next_label += int(new.sum())
        out.append({**fr, "assigned_ids": labels})
        prev_labels = labels
    return LinkResult(frames=out, links=links, n_tracks=next_label,
                      params=dict(method=method, max_distance=max_distance))


def link_greedy(frames: list[dict], max_distance: float) -> LinkResult:
    return link_frames(frames, max_distance, "greedy")


def link_lap(frames: list[dict], max_distance: float) -> LinkResult:
    return link_frames(frames, max_distance, "lap")


# --------------------------------------------------------------------------- labelling

def label_links(result: LinkResult, evaluate: list[set[int]] | None = None) -> list[dict]:
    """Classify the linker's output per frame pair against ``true_ids``.

    For pair (t, t+1), restricted to cells whose true id is in ``evaluate[t]``
    (all cells if None):
      ``switch``   links whose two detections have different real true ids
      ``frag``     real cells detected at both t and t+1 but not linked to each other
      ``correct``  links joining the same real cell
      ``n1``       real detections at t+1 (denominator for rates, as in the sweeps)
    plus the sets of switched true ids at t+1 (for scoring a locator) and the
    affected labels.
    """
    rows = []
    for t, m in enumerate(result.links):
        f0, f1 = result.frames[t], result.frames[t + 1]
        true0, true1 = f0["true_ids"], f1["true_ids"]
        ok = (lambda tid: True) if evaluate is None else (lambda tid, s=evaluate[t]: tid in s)
        linked_from = {int(i): int(j) for i, j in m}
        real1 = {int(tid): k for k, tid in enumerate(true1) if tid >= 0 and ok(int(tid))}
        real0 = {int(tid): k for k, tid in enumerate(true0) if tid >= 0 and ok(int(tid))}
        switch_ids: set[int] = set(); correct = 0; switch = 0
        for i, j in m:
            a, b = int(true0[i]), int(true1[j])
            if a < 0 or b < 0 or not (ok(a) and ok(b)):
                continue                      # synthetic detections: not scored here
            if a == b:
                correct += 1
            else:
                switch += 1; switch_ids.update((a, b))
        both = set(real0) & set(real1)
        frag = sum(1 for tid in both if linked_from.get(real0[tid]) != real1[tid]
                   and int(true0[real0[tid]]) == tid)
        rows.append(dict(pair=t, n1=len(real1), correct=correct, switch=switch, frag=frag,
                         switch_rate=switch / max(len(real1), 1), frag_rate=frag / max(len(real1), 1),
                         switched_true_ids=switch_ids))
    return rows
