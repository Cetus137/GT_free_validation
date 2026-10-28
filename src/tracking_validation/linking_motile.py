"""Global linking with motile (min-cost selection of nodes and edges over the whole sequence),
no divisions.  Requires the ``motile`` package (motile_env).

Why node selection matters: with one parent / one child per detection and only distance,
appear and disappear costs, the global problem splits into independent frame pairs and is
exactly frame-to-frame LAP.  Here every detection also carries a selection reward that grows
with its detector response, so a weak detection is kept only if the track it belongs to is
worth its appear + disappear cost.  That couples the frames: a weak, isolated detection (a
likely false positive) is dropped, a weak detection inside a long track is kept.

Costs (all in units of median spacing squared, so LAP's are matched exactly):
  edge       (d / spacing)^2 for candidate links within ``cap`` (as LAP's squared cost)
  appear     cap^2 per track start;  disappear  cap^2 per track end   (LAP's no-link prices)
  node       -GAMMA * (appear + disappear) * q,  q = min(response / median response of the
             frame, 1): a strong isolated detection is kept (GAMMA > 1), a weak isolated one
             (q < 1 / GAMMA) is dropped
Constraints: MaxParents(1), MaxChildren(1).

Long sequences are solved in overlapping windows (SCIP is the available backend); each
window keeps only the edges whose source frame lies in its central part.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .linking import LinkResult

GAMMA = 1.5


def _window_solve(frames: list[dict], t0: int, t1: int, cap: float, spacing: float, q: list[np.ndarray]):
    import motile
    import networkx as nx
    from motile.constraints import MaxChildren, MaxParents
    from motile.costs import Appear, Disappear, EdgeSelection, NodeSelection

    G = nx.DiGraph()
    nid = {}
    unit = (cap / spacing) ** 2                     # appear = disappear = cap^2 (spacing units)
    for t in range(t0, t1):
        for i, p in enumerate(frames[t]["positions"]):
            n = len(nid); nid[(t, i)] = n
            G.add_node(n, t=t, node_cost=-GAMMA * 2 * unit * float(q[t][i]))
    for t in range(t0, t1 - 1):
        p0, p1 = frames[t]["positions"], frames[t + 1]["positions"]
        if len(p0) == 0 or len(p1) == 0:
            continue
        for i, js in enumerate(cKDTree(p1).query_ball_point(p0, cap)):
            for j in js:
                d = float(np.linalg.norm(p1[j] - p0[i])) / spacing
                G.add_edge(nid[(t, i)], nid[(t + 1, j)], edge_cost=d * d)
    tg = motile.TrackGraph(G, frame_attribute="t")
    solver = motile.Solver(tg)
    solver.add_cost(NodeSelection(weight=1.0, attribute="node_cost"))
    solver.add_cost(EdgeSelection(weight=1.0, attribute="edge_cost"))
    solver.add_cost(Appear(constant=unit))
    solver.add_cost(Disappear(constant=unit))
    solver.add_constraint(MaxParents(1))
    solver.add_constraint(MaxChildren(1))
    solver.solve(num_threads=1)
    sub = solver.get_selected_subgraph()
    inv = {v: k for k, v in nid.items()}
    nodes = {inv[n] for n in sub.nodes}
    edges = [(inv[u], inv[v]) for u, v in sub.edges]
    return nodes, edges


def link_motile(frames: list[dict], max_distance: float, response: list[np.ndarray],
                window: int = 24, overlap: int = 8) -> LinkResult:
    """Track labels from a global motile solution.  ``response``: detector response per
    detection per frame (the node-selection reward).  Unselected detections are removed from
    the returned frames (they are the tracker's rejected detections)."""
    T = len(frames)
    spacing = float(np.median([np.median(cKDTree(f["positions"]).query(f["positions"], k=2)[0][:, 1])
                               for f in frames if len(f["positions"]) > 2]))
    q = []
    for r in response:
        r = np.asarray(r, float)
        q.append(np.clip(r / np.median(r), 0, 1) if len(r) else r)
    keep_nodes: set[tuple[int, int]] = set()
    keep_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    step = window - overlap
    starts = list(range(0, max(T - overlap, 1), step))
    for k, s in enumerate(starts):
        e = min(s + window, T)
        nodes, edges = _window_solve(frames, s, e, max_distance, spacing, q)
        lo = s if k == 0 else s + overlap // 2                     # this window owns frames [lo, hi)
        hi = e if k == len(starts) - 1 else s + step + overlap // 2
        keep_nodes |= {n for n in nodes if lo <= n[0] < hi}
        keep_edges += [(a, b) for a, b in edges if lo <= a[0] < hi]
        if e >= T:
            break
    for a, b in keep_edges:                                         # an edge's ends are selected
        keep_nodes.add(a); keep_nodes.add(b)
    child = {a: b for a, b in keep_edges}
    has_parent = {b for _, b in keep_edges}
    label: dict[tuple[int, int], int] = {}
    nxt = 0
    for n in sorted(keep_nodes):
        if n in has_parent:
            continue
        cur = n
        while cur is not None and cur not in label:
            label[cur] = nxt
            cur = child.get(cur)
        nxt += 1
    out, links = [], []
    for t, f in enumerate(frames):
        idx = [i for i in range(len(f["positions"])) if (t, i) in keep_nodes]
        new = {k: v[idx] if isinstance(v, np.ndarray) and len(v) == len(f["positions"]) else v for k, v in f.items()}
        new["assigned_ids"] = np.array([label[(t, i)] for i in idx], dtype=int)
        out.append(new)
    for t in range(T - 1):
        pos_t = {i: k for k, i in enumerate([i for i in range(len(frames[t]["positions"])) if (t, i) in keep_nodes])}
        pos_t1 = {i: k for k, i in enumerate([i for i in range(len(frames[t + 1]["positions"])) if (t + 1, i) in keep_nodes])}
        m = [(pos_t[a[1]], pos_t1[b[1]]) for a, b in keep_edges if a[0] == t]
        links.append(np.array(m, dtype=int).reshape(-1, 2))
    return LinkResult(frames=out, links=links, n_tracks=nxt,
                      params=dict(method="motile", max_distance=max_distance, gamma=GAMMA, window=window, overlap=overlap))
