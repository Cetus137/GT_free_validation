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


def _delaunay_graph(pos: np.ndarray, cap: float) -> dict[int, list[tuple[int, float]]]:
    """index -> [(neighbour index, distance)] from the Delaunay triangulation (hull cells
    get no entry), edges longer than cap dropped."""
    from .metrics import _frame_neighbour_map_delaunay
    idx = np.arange(len(pos))
    nb, nd = _frame_neighbour_map_delaunay(pos, idx, max_distance=cap)
    return {int(i): list(zip(nb[i], nd[i])) for i in nb}


METRIC_DISPLACEMENT_WEIGHT = 1.0   # lambda: continuity gain must exceed lambda x displacement
                                   # cost increase (in units of cap^2); 0 = continuity alone


def _match_metric(p0: np.ndarray, p1: np.ndarray, cap: float, sweeps: int = 6,
                  rc_weight: float = 1.0, lam: float | None = None) -> np.ndarray:
    """Linear assignment refined by local exchanges that increase neighbourhood
    continuity: for each cell i at t the objective is its weighted identity
    overlap (fraction of its t-neighbours, 1/distance weighted, whose linked
    bodies are neighbours of i's linked body at t+1) plus ``rc_weight`` times the
    Spearman rank correlation of the shared neighbours' distance order.  Two
    cells whose t+1 targets are both within ``cap`` of each other's t position
    may exchange targets; the exchange is kept if the summed objective over the
    affected cells rises.  Repeats until no exchange helps or ``sweeps`` is hit."""
    from scipy.stats import rankdata
    lam = METRIC_DISPLACEMENT_WEIGHT if lam is None else lam
    m = _match_lap(p0, p1, cap)
    if len(m) < 4 or len(p0) < 5 or len(p1) < 5:
        return m
    sigma: dict[int, int] = {int(i): int(j) for i, j in m}
    g0 = _delaunay_graph(p0, cap)
    g1 = _delaunay_graph(p1, cap)
    n1_sets = {j: {k for k, _ in nbrs} for j, nbrs in g1.items()}
    n1_rank = {j: {k: r for r, (k, _) in enumerate(nbrs)} for j, nbrs in g1.items()}
    cap2 = cap * cap

    def term(i: int) -> float:
        j = sigma.get(i)
        if j is None or i not in g0 or j not in n1_sets:
            return 0.0
        nbrs = g0[i]
        w = np.array([1.0 / max(d, 1e-9) for _, d in nbrs]); w /= w.sum()
        tgt = [sigma.get(a) for a, _ in nbrs]
        inset = np.array([t is not None and t in n1_sets[j] for t in tgt])
        wio = float((w * inset).sum())
        rc = 0.0
        if rc_weight and inset.sum() >= 3:
            r0 = np.arange(len(nbrs))[inset]
            r1 = np.array([n1_rank[j][t] for t, ok in zip(tgt, inset) if ok])
            a = rankdata(r0) - r0.size / 2 - 0.5; b = rankdata(r1) - r1.size / 2 - 0.5
            den = float(np.sqrt((a * a).sum() * (b * b).sum()))
            rc = float((a * b).sum() / den) if den > 0 else 0.0
        return wio + rc_weight * rc

    def affected(i: int, k: int) -> set[int]:
        s = {i, k}
        for c in (i, k):
            s.update(a for a, _ in g0.get(c, []))
        # cells whose t-neighbours include i or k also change (their target sets change)
        return s

    for _ in range(sweeps):
        improved = 0
        for i in list(sigma):
            j = sigma.get(i)
            if j is None:
                continue
            for k, _ in g0.get(i, []):
                jk = sigma.get(k)
                if jk is None or jk == j:
                    continue
                # both exchanged links must respect the cap
                if ((p1[jk] - p0[i]) ** 2).sum() > cap2 or ((p1[j] - p0[k]) ** 2).sum() > cap2:
                    continue
                cells = affected(i, k)
                before = sum(term(c) for c in cells)
                d_before = ((p1[j] - p0[i]) ** 2).sum() + ((p1[jk] - p0[k]) ** 2).sum()
                d_after = ((p1[jk] - p0[i]) ** 2).sum() + ((p1[j] - p0[k]) ** 2).sum()
                sigma[i], sigma[k] = jk, j
                after = sum(term(c) for c in cells)
                if (after - before) - lam * (d_after - d_before) / cap2 > 1e-9:
                    improved += 1
                    j = sigma[i]
                else:
                    sigma[i], sigma[k] = j, jk
        if not improved:
            break
    return np.array(sorted(sigma.items()), dtype=int).reshape(-1, 2)


_MATCHERS = {"greedy": _match_greedy, "lap": _match_lap, "metric": _match_metric}


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


def link_metric(frames: list[dict], max_distance: float) -> LinkResult:
    """Linear assignment refined to maximise neighbourhood overlap + rank correlation."""
    return link_frames(frames, max_distance, "metric")


# --------------------------------------------------------------------------- gap closing

def close_gaps(result: LinkResult, max_gap: int, max_distance: float) -> LinkResult:
    """Bridge track ends to track starts up to ``max_gap`` frames later and within
    ``max_distance`` (TrackMate's gap-closing step, by linear assignment per
    (end frame, gap length)).  The later track takes the earlier track's label.
    ``max_gap`` = 0 returns the input unchanged."""
    if max_gap <= 0:
        return result
    frames = [dict(f) for f in result.frames]
    T = len(frames)
    first: dict[int, tuple[int, int]] = {}; last: dict[int, tuple[int, int]] = {}
    for t, f in enumerate(frames):
        for k, lab in enumerate(f["assigned_ids"].tolist()):
            first.setdefault(lab, (t, k)); last[lab] = (t, k)
    parent: dict[int, int] = {}

    def root(l):
        while l in parent:
            l = parent[l]
        return l

    closed = 0
    for gap in range(2, max_gap + 2):            # start frame = end frame + gap
        for t in range(T - gap):
            ends = [l for l, (tl, _) in last.items() if tl == t and root(l) == l]
            starts = [l for l, (ts, _) in first.items() if ts == t + gap and root(l) == l and l not in parent]
            if not ends or not starts:
                continue
            pe = np.array([frames[t]["positions"][last[l][1]] for l in ends])
            ps = np.array([frames[t + gap]["positions"][first[l][1]] for l in starts])
            m = _match_lap(pe, ps, max_distance)
            for i, j in m:
                e, s = ends[i], starts[j]
                parent[s] = e                   # the later track joins the earlier one
                last[e] = last[s]               # the merged track now ends where s ended
                closed += 1
    if not closed:
        return result
    mapping = {l: root(l) for l in list(first)}
    out = []
    for f in frames:
        out.append({**f, "assigned_ids": np.array([mapping.get(int(l), int(l)) for l in f["assigned_ids"]], dtype=int)})
    return LinkResult(frames=out, links=result.links, n_tracks=len({mapping[l] for l in mapping}),
                      params={**result.params, "max_gap": max_gap, "gap_distance": max_distance, "gaps_closed": closed})


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
