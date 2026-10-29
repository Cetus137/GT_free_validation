"""Schematics for Figure 1a: the neighbourhood-continuity metrics and the six
error types, drawn as packed cells (bounded Voronoi tessellation of centroids).

    python scripts/make_schematics.py

Writes figures/panels/schematic_metrics.{png,pdf} and
figures/panels/schematic_errors.{png,pdf}.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, Voronoi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402

OUT = ROOT / "figures" / "panels"
fs.apply()
fs.SURFACE = "#ffffff"          # schematics on a pure white background
plt.rcParams.update({"figure.facecolor": fs.SURFACE, "axes.facecolor": fs.SURFACE, "savefig.facecolor": fs.SURFACE})


def save_hi(fig, stem):
    """Vector PDF and SVG (print), plus a 600 dpi PNG (preview)."""
    fig.savefig(f"{stem}.pdf"); fig.savefig(f"{stem}.svg")
    fig.savefig(f"{stem}.png", dpi=600)
    plt.close(fig)
    print("wrote", f"{stem}.pdf / .svg / .png")

FATE = {"retained": fs.MUTED, "vanished": "#e34948", "displaced": fs.SERIES["switch"],
        "newcomer": fs.SERIES["split"]}
CELL_FILL = "#efeeea"        # unremarkable cell
CELL_EDGE = "#b9b7ad"


# --------------------------------------------------------------------------- geometry

def bounded_voronoi(pts: np.ndarray, box=(0, 1, 0, 1)) -> list[np.ndarray]:
    """Voronoi cells of ``pts`` clipped to ``box`` by mirroring the points across
    the four sides.  Returns one closed polygon (vertex array) per input point."""
    x0, x1, y0, y1 = box
    mirrors = [pts * [-1, 1] + [2 * x0, 0], pts * [-1, 1] + [2 * x1, 0],
               pts * [1, -1] + [0, 2 * y0], pts * [1, -1] + [0, 2 * y1]]
    allp = np.vstack([pts] + mirrors)
    vor = Voronoi(allp)
    polys = []
    for i in range(len(pts)):
        region = vor.regions[vor.point_region[i]]
        polys.append(vor.vertices[region])
    return polys


def tint(hexcolor: str, f=0.72) -> str:
    """Mix a colour with the surface colour (opaque light tint)."""
    import matplotlib.colors as mc
    a = np.array(mc.to_rgb(hexcolor)); b = np.array(mc.to_rgb(fs.SURFACE))
    return mc.to_hex(b + (1 - f) * (a - b) if False else a * (1 - f) + b * f)


def shrink(poly: np.ndarray, f=0.93) -> np.ndarray:
    """Pull the polygon toward its centroid (membrane gap) and round its corners."""
    c = poly.mean(axis=0)
    q = c + f * (poly - c)
    return _round(q, passes=1)


def _round(q: np.ndarray, passes=3) -> np.ndarray:
    """Chaikin corner cutting on a closed polygon."""
    for _ in range(passes):
        n = len(q); out = []
        for k in range(n):
            a, b = q[k], q[(k + 1) % n]
            out.append(0.75 * a + 0.25 * b); out.append(0.25 * a + 0.75 * b)
        q = np.array(out)
    return q


def delaunay_neighbours(pts: np.ndarray, i: int) -> set[int]:
    tri = Delaunay(pts)
    ptr, idx = tri.vertex_neighbor_vertices
    return set(idx[ptr[i]:ptr[i + 1]].tolist())


def clip_line(poly: np.ndarray, c: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Segment of the line through c with direction u that lies inside poly."""
    ts = []
    n = len(poly)
    for k in range(n):
        a, b = poly[k], poly[(k + 1) % n]
        e = b - a
        M = np.array([u, -e]).T
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        t, sgm = np.linalg.solve(M, a - c)
        if 0 <= sgm <= 1:
            ts.append(t)
    ts = sorted(ts)
    return np.array([c + ts[0] * u, c + ts[-1] * u])


def draw_cells(ax, polys, fills, edges, lws=None, ls=None, z=2, f=0.93):
    for k, p in enumerate(polys):
        ax.add_patch(Polygon(shrink(p, f), closed=True, facecolor=fills[k], edgecolor=edges[k],
                             lw=(lws[k] if lws else 0.9), ls=(ls[k] if ls else "-"), zorder=z,
                             joinstyle="round"))


def frame(ax, title, box=(0, 1, 0, 1)):
    x0, x1, y0, y1 = box
    ax.set_xlim(x0 - 0.02, x1 + 0.02); ax.set_ylim(y0 - 0.02, y1 + 0.02); ax.set_aspect("equal"); ax.axis("off")
    ax.text(x0, y1 + 0.01, title, ha="left", va="bottom", fontsize=8.5, color=fs.INK)


# --------------------------------------------------------------------------- metrics

def _metric_points():
    """Hand-placed centroids.  Index 0 = focal cell i.  Returned as (t, t+1, names, fates)."""
    rng = np.random.default_rng(7)
    base = np.array([
        [0.50, 0.50],              # 0 i
        [0.69, 0.56],              # 1 a
        [0.56, 0.72],              # 2 b
        [0.36, 0.66],              # 3 c
        [0.30, 0.46],              # 4 d
        [0.42, 0.30],              # 5 e
        [0.64, 0.34],              # 6 f
        [0.17, 0.60],              # 7 g  (becomes newcomer)
        [0.86, 0.72], [0.80, 0.22], [0.15, 0.20], [0.20, 0.84], [0.50, 0.90], [0.88, 0.46],
        [0.70, 0.86], [0.38, 0.10], [0.62, 0.10], [0.10, 0.40], [0.90, 0.90], [0.08, 0.90], [0.92, 0.08], [0.06, 0.06],
    ])
    t1 = base + rng.normal(0, 0.012, base.shape)
    t1[0] = base[0]
    t1[4] = [0.14, 0.50]           # d slides away from i ...
    t1[7] = [0.30, 0.52]           # ... and g slides in between: d displaced, g newcomer
    t1[17] = [0.10, 0.36]
    names = {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f", 7: "g"}
    return base, t1, names


def schematic_metrics(stem="schematic_metrics"):
    p0, p1, names = _metric_points()
    n0 = delaunay_neighbours(p0, 0)
    n1 = delaunay_neighbours(p1, 0)
    vanished = {2}                           # b: not detected at t+1
    p1v = np.delete(p1, list(vanished), axis=0)      # detections present at t+1
    keep = [k for k in range(len(p1)) if k not in vanished]
    n1v = {keep[j] for j in delaunay_neighbours(p1v, 0)}
    fate = {}
    for j in n0:
        fate[j] = "vanished" if j in vanished else ("retained" if j in n1v else "displaced")
    for j in n1v - n0:
        fate[j] = "newcomer"
    assert fate.get(4) == "displaced" and fate.get(7) == "newcomer" and fate.get(2) == "vanished", fate

    fig, axes = plt.subplots(1, 2, figsize=(fs.DOUBLE * 0.5, 2.05),
                             gridspec_kw=dict(left=0.01, right=0.99, top=0.93, bottom=0.11, wspace=0.06))

    # ---- t
    ax = axes[0]; frame(ax, "t")
    polys = bounded_voronoi(p0)
    fills = [CELL_FILL] * len(p0); edges = [CELL_EDGE] * len(p0); lws = [0.8] * len(p0)
    for j in n0:
        fills[j] = "#dcdbd5"; edges[j] = FATE["retained"]; lws[j] = 1.2
    fills[0] = fs.INK; edges[0] = fs.INK
    draw_cells(ax, polys, fills, edges, lws)
    for j in n0:
        ax.text(*p0[j], names[j], ha="center", va="center", fontsize=7, color=fs.INK, zorder=5)
    ax.text(*p0[0], "i", ha="center", va="center", fontsize=8, color=fs.SURFACE, zorder=5)
    # 1/d weights as link width
    w = {j: 1 / np.linalg.norm(p0[j] - p0[0]) for j in n0}; s = sum(w.values())
    for j in n0:
        q = p0[0] + 0.75 * (p0[j] - p0[0])
        ax.plot([p0[0][0], q[0]], [p0[0][1], q[1]], color=fs.INK, lw=0.4 + 11 * w[j] / s,
                zorder=4, solid_capstyle="round", alpha=0.75)

    # ---- t+1
    ax = axes[1]; frame(ax, "t + 1")
    polys = bounded_voronoi(p1v)
    fills = [CELL_FILL] * len(keep); edges = [CELL_EDGE] * len(keep); lws = [0.8] * len(keep)
    for jj, j in enumerate(keep):
        if j in fate:
            edges[jj] = FATE[fate[j]]; lws[jj] = 1.4
            fills[jj] = "#dcdbd5" if fate[j] == "retained" else FATE[fate[j]] + "33"
    fills[0] = fs.INK; edges[0] = fs.INK
    draw_cells(ax, polys, fills, edges, lws)
    # the vanished cell: its true outline at t+1, dashed, no detection
    polys_true = bounded_voronoi(p1)
    for j in vanished:
        ax.add_patch(Polygon(shrink(polys_true[j], 0.90), closed=True, facecolor=fs.SURFACE,
                             edgecolor=FATE["vanished"], lw=1.2, ls=(0, (2, 1.5)), zorder=3))
        ax.text(*p1[j], names[j], ha="center", va="center", fontsize=7, color=FATE["vanished"], zorder=5)
    for j, f in fate.items():
        if f != "vanished":
            ax.text(*p1[j], names[j], ha="center", va="center", fontsize=7,
                    color=fs.INK if f == "retained" else FATE[f], zorder=5, fontweight="bold" if f != "retained" else None)
    ax.text(*p1[0], "i", ha="center", va="center", fontsize=8, color=fs.SURFACE, zorder=5)
    for j in n1v:
        q = p1[0] + 0.75 * (p1[j] - p1[0])
        ax.plot([p1[0][0], q[0]], [p1[0][1], q[1]], color=fs.INK, lw=0.9, zorder=4, alpha=0.6,
                solid_capstyle="round")

    handles = [fs.dot_handle(t, FATE[k]) for k, t in
               [("retained", "retained"), ("vanished", "vanished"), ("displaced", "displaced"), ("newcomer", "newcomer")]]
    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4,
                     handlelength=1.0, columnspacing=1.4, handletextpad=0.4, fontsize=7)
    for t in leg.get_texts():
        t.set_color(fs.INK)
    save_hi(fig, OUT / stem)


# --------------------------------------------------------------------------- errors

# a small patch of 8 cells used in every error panel; index 0 and 1 are the two
# cells the error involves, chosen adjacent and near the centre
PATCH = np.array([[0.38, 0.50], [0.64, 0.52], [0.48, 0.84], [0.14, 0.76], [0.12, 0.24],
                  [0.46, 0.14], [0.86, 0.22], [0.88, 0.80]])
PATCH1 = PATCH.copy()        # t+1 identical to t: the only change in each panel is the error
FP_XY = [0.50, 0.66]         # the false positive sits on the membrane between cells 1, 2 and 3


def _tess(ax, pts, labels, colors, extra=None, lws=None, ls=None, f=0.93):
    polys = bounded_voronoi(pts)
    fills = [c + "55" if c != CELL_FILL else c for c in colors]
    edges = [c if c != CELL_FILL else CELL_EDGE for c in colors]
    draw_cells(ax, polys, fills, edges, lws, ls, f=f)
    for k, lab in enumerate(labels):
        if lab:
            ax.text(*pts[k], lab, ha="center", va="center", fontsize=6.5, zorder=5,
                    color=fs.INK if colors[k] == CELL_FILL else colors[k], fontweight=None if colors[k] == CELL_FILL else "bold")
    return polys


def _panel(axl, axr, title, color):
    for ax, t in ((axl, "t"), (axr, "t + 1")):
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal"); ax.axis("off")
        ax.text(0.5, -0.03, t, ha="center", va="top", fontsize=6.5, color=fs.INK)
    axl.text(0.0, 1.03, title, ha="left", va="bottom", fontsize=7.5, color=color, fontweight="bold")


def schematic_errors(stem="schematic_errors"):
    """Six panels, each two frames of the same patch.  Cell outlines are the true
    cells; the number is the ID the tracker assigned; colour marks the error."""
    fig = plt.figure(figsize=(fs.DOUBLE * 0.38, 3.1))
    gs = fig.add_gridspec(3, 4, left=0.01, right=0.99, top=0.94, bottom=0.03, wspace=0.08, hspace=0.32)
    C = fs.SERIES; N = CELL_FILL
    plain = [N] * 8

    def pair(r, c):
        return fig.add_subplot(gs[r, 2 * c]), fig.add_subplot(gs[r, 2 * c + 1])

    base_lab = ["1", "2", "3", "4", "5", "6", "7", "8"]

    # identity switch: cells 1 and 2 swap IDs at t+1
    al, ar = pair(0, 0); _panel(al, ar, fs.LABELS["switch"], C["switch"])
    _tess(al, PATCH, base_lab, plain)
    lab = base_lab.copy(); lab[0], lab[1] = "2", "1"
    _tess(ar, PATCH1, lab, [C["switch"], C["switch"]] + [N] * 6)

    # track fragmentation: cell 1 gets a new ID at t+1
    al, ar = pair(0, 1); _panel(al, ar, fs.LABELS["frag"], C["frag"])
    _tess(al, PATCH, base_lab, plain)
    lab = base_lab.copy(); lab[0] = "9"
    _tess(ar, PATCH1, lab, [C["frag"]] + [N] * 7)

    # missed detection: cell 1 has no detection at t+1 (dashed outline, no ID)
    al, ar = pair(1, 0); _panel(al, ar, fs.LABELS["miss"], C["miss"])
    _tess(al, PATCH, base_lab, plain)
    lab = base_lab.copy(); lab[0] = ""
    _tess(ar, PATCH1, lab, [C["miss"]] + [N] * 7, ls=[(0, (2, 1.5))] + ["-"] * 7)

    # split: cell 1 is detected as two at t+1
    al, ar = pair(1, 1); _panel(al, ar, fs.LABELS["split"], C["split"])
    _tess(al, PATCH, base_lab, plain)
    lab = base_lab.copy(); lab[0] = ""
    polys = _tess(ar, PATCH1, lab, [C["split"]] + [N] * 7)
    q = shrink(polys[0], 0.93); c = q.mean(axis=0)
    seg = clip_line(q, c, np.array([0.2, 1.0]))
    ar.plot(seg[:, 0], seg[:, 1], color=C["split"], lw=1.2, zorder=4)
    nrm = np.array([1.0, -0.2]); nrm /= np.linalg.norm(nrm)
    for off, txt in ((-0.075, "1"), (0.075, "9")):
        ar.text(*(c + off * nrm), txt, ha="center", va="center", fontsize=6, color=C["split"], fontweight="bold", zorder=5)

    # merge: cells 1 and 2 detected as one at t+1 (their shared face is gone)
    al, ar = pair(2, 0); _panel(al, ar, fs.LABELS["merge"], C["merge"])
    _tess(al, PATCH, base_lab, plain)
    lab = base_lab.copy(); lab[0] = ""; lab[1] = ""
    polys = _tess(ar, PATCH1, lab, plain)
    from matplotlib.patches import Polygon as _P
    from scipy.spatial import ConvexHull
    tc = tint(C["merge"])
    for k in (0, 1):   # repaint the two cells in the tint with no visible edge
        ar.add_patch(_P(shrink(polys[k], 0.93), closed=True, facecolor=tc, edgecolor=tc, lw=1.0, zorder=3))
    u = np.vstack([shrink(polys[0], 0.93), shrink(polys[1], 0.93)])
    hull = u[ConvexHull(u).vertices]
    ar.add_patch(_P(hull, closed=True, facecolor=tc, edgecolor=C["merge"], lw=1.2, zorder=4))
    ar.text(*(0.5 * (PATCH1[0] + PATCH1[1])), "1", ha="center", va="center", fontsize=6.5, color=C["merge"], fontweight="bold", zorder=6)

    # false positive: a spurious detection carves a small extra cell out of the junction
    al, ar = pair(2, 1); _panel(al, ar, fs.LABELS["fp"], C["fp"])
    _tess(al, PATCH, base_lab, plain)
    p = np.vstack([PATCH1, [FP_XY]])
    _tess(ar, p, base_lab + ["9"], plain + [C["fp"]])

    save_hi(fig, OUT / stem)


def schematic_errors_separate(prefix="schematic_error_"):
    """One file per error type: schematic_error_<key>.{pdf,svg,png}."""
    C = fs.SERIES; N = CELL_FILL; plain = [N] * 8
    base_lab = ["1", "2", "3", "4", "5", "6", "7", "8"]

    def switch(ar):
        lab = base_lab.copy(); lab[0], lab[1] = "2", "1"
        _tess(ar, PATCH1, lab, [C["switch"], C["switch"]] + [N] * 6)

    def frag(ar):
        lab = base_lab.copy(); lab[0] = "9"
        _tess(ar, PATCH1, lab, [C["frag"]] + [N] * 7)

    def miss(ar):
        lab = base_lab.copy(); lab[0] = ""
        _tess(ar, PATCH1, lab, [C["miss"]] + [N] * 7, ls=[(0, (2, 1.5))] + ["-"] * 7)

    def split(ar):
        lab = base_lab.copy(); lab[0] = ""
        polys = _tess(ar, PATCH1, lab, [C["split"]] + [N] * 7)
        q = shrink(polys[0], 0.93); c = q.mean(axis=0)
        seg = clip_line(q, c, np.array([0.2, 1.0]))
        ar.plot(seg[:, 0], seg[:, 1], color=C["split"], lw=1.2, zorder=4)
        nrm = np.array([1.0, -0.2]); nrm /= np.linalg.norm(nrm)
        for off, txt in ((-0.075, "1"), (0.075, "9")):
            ar.text(*(c + off * nrm), txt, ha="center", va="center", fontsize=6, color=C["split"], fontweight="bold", zorder=5)

    def merge(ar):
        from matplotlib.patches import Polygon as _P
        from scipy.spatial import ConvexHull
        lab = base_lab.copy(); lab[0] = ""; lab[1] = ""
        polys = _tess(ar, PATCH1, lab, plain)
        tc = tint(C["merge"])
        for k in (0, 1):
            ar.add_patch(_P(shrink(polys[k], 0.93), closed=True, facecolor=tc, edgecolor=tc, lw=1.0, zorder=3))
        u = np.vstack([shrink(polys[0], 0.93), shrink(polys[1], 0.93)])
        hull = u[ConvexHull(u).vertices]
        ar.add_patch(_P(hull, closed=True, facecolor=tc, edgecolor=C["merge"], lw=1.2, zorder=4))
        ar.text(*(0.5 * (PATCH1[0] + PATCH1[1])), "1", ha="center", va="center", fontsize=6.5, color=C["merge"], fontweight="bold", zorder=6)

    def fp(ar):
        p = np.vstack([PATCH1, [FP_XY]])
        _tess(ar, p, base_lab + ["9"], plain + [C["fp"]])

    for key, draw in [("switch", switch), ("frag", frag), ("miss", miss), ("split", split), ("merge", merge), ("fp", fp)]:
        fig, (al, ar) = plt.subplots(1, 2, figsize=(fs.SINGLE * 0.55, 1.15),
                                     gridspec_kw=dict(left=0.01, right=0.99, top=0.84, bottom=0.08, wspace=0.08))
        _panel(al, ar, fs.LABELS[key], C[key])
        _tess(al, PATCH, base_lab, plain)
        draw(ar)
        save_hi(fig, OUT / f"{prefix}{key}")


# --------------------------------------------------------------------------- metric panels

RC_COLOR = "#7b5cd6"


def schematic_metric_panels(prefix="schematic_metric_"):
    """One small two-frame panel per concept, in the style of the error panels:
    neighbourhood + weights, retained, vanished, displaced, newcomer, rank correlation."""
    N = CELL_FILL; plain = [N] * 8
    NAMES = {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f", 7: "g"}
    nb0 = sorted(delaunay_neighbours(PATCH, 0))            # [1..5]

    def labels_for(pts_n, present=None):
        lab = [""] * pts_n
        for k, nm in NAMES.items():
            if k < pts_n: lab[k] = nm
        return lab

    def base_frame(ax, pts, present_nb, colors=None, lws=None, ls=None, spokes=True, weighted=False):
        cols = list(colors) if colors else [N] * len(pts)
        for j in present_nb:
            if cols[j] == N: cols[j] = "#dcdbd5"
        cols[0] = fs.INK
        lab = labels_for(len(pts)); lab[0] = ""
        polys = bounded_voronoi(pts)
        fills = [c + "55" if c not in (N, fs.INK, "#dcdbd5") else c for c in cols]
        edges = [CELL_EDGE if c == N else (FATE["retained"] if c == "#dcdbd5" else c) for c in cols]
        lw = lws or [1.2 if j in present_nb or j == 0 else 0.9 for j in range(len(pts))]
        draw_cells(ax, polys, fills, edges, lw, ls)
        if spokes:
            w = {j: 1 / np.linalg.norm(pts[j] - pts[0]) for j in present_nb}; sw = sum(w.values())
            for j in present_nb:
                q = pts[0] + 0.72 * (pts[j] - pts[0])
                ax.plot([pts[0][0], q[0]], [pts[0][1], q[1]], color=fs.INK,
                        lw=(0.4 + 9 * w[j] / sw) if weighted else 0.9, zorder=4, alpha=0.7, solid_capstyle="round")
        for j, nm in enumerate(lab):
            if nm:
                c = cols[j]
                ax.text(*pts[j], nm, ha="center", va="center", fontsize=6.5, zorder=5,
                        color=fs.INK if c in (N, "#dcdbd5") else c, fontweight=None if c in (N, "#dcdbd5") else "bold",
                        path_effects=[pe.withStroke(linewidth=2.0, foreground=fs.SURFACE)])
        ax.text(*pts[0], "i", ha="center", va="center", fontsize=7.5, color=fs.SURFACE, zorder=5)
        return polys

    panels = []

    # 1 neighbourhood and weights: t plain tessellation with i; t+1 slot shows neighbours shaded + weighted spokes
    def neighbourhood(al, ar):
        base_frame(al, PATCH, [], spokes=False)
        base_frame(ar, PATCH, nb0, spokes=True, weighted=True)
    panels.append(("neighbourhood", "Neighbourhood", fs.INK, neighbourhood, ("t", "t")))

    # 2 retained: small drift, everyone stays
    def retained(al, ar):
        base_frame(al, PATCH, nb0, weighted=True)
        base_frame(ar, PATCH1, nb0, weighted=True)
    panels.append(("retained", "Retained", fs.INK, retained, ("t", "t + 1")))

    # 3 vanished: neighbour b (index 2) has no detection at t+1
    def vanished(al, ar):
        base_frame(al, PATCH, nb0, weighted=True)
        cols = [N] * 8; cols[2] = FATE["vanished"]
        base_frame(ar, PATCH1, [j for j in nb0 if j != 2], colors=cols,
                   ls=["-"] * 2 + [(0, (2, 1.5))] + ["-"] * 5)
    panels.append(("vanished", "Vanished", FATE["vanished"], vanished, ("t", "t + 1")))

    # 4 displaced: neighbour a (index 1) slides away, e slides between
    def displaced(al, ar):
        base_frame(al, PATCH, nb0, weighted=True)
        p = PATCH1.copy(); p[1] = [0.86, 0.50]; p[5] = [0.64, 0.40]
        nb = sorted(delaunay_neighbours(p, 0)); assert 1 not in nb
        cols = [N] * 8; cols[1] = FATE["displaced"]
        base_frame(ar, p, nb, colors=cols)
    panels.append(("displaced", "Displaced", FATE["displaced"], displaced, ("t", "t + 1")))

    # 5 newcomer: cell f (index 6) slides in and touches i
    def newcomer(al, ar):
        base_frame(al, PATCH, nb0, weighted=True)
        p = PATCH1.copy(); p[6] = [0.66, 0.30]
        nb = sorted(delaunay_neighbours(p, 0)); assert 6 in nb
        cols = [N] * 8; cols[6] = FATE["newcomer"]
        base_frame(ar, p, nb, colors=cols)
    panels.append(("newcomer", "Newcomer", FATE["newcomer"], newcomer, ("t", "t + 1")))

    # 6 rank correlation: neighbours numbered by distance rank; two swap order at t+1
    def rank(al, ar):
        for ax, pts in ((al, PATCH), (ar, PATCH1.copy())):
            if ax is ar:
                pts[2] = [0.50, 0.70]; pts[3] = [0.20, 0.70]      # b comes closer, c moves off
            cols = [N] * 8
            for j in nb0: cols[j] = RC_COLOR
            polys = bounded_voronoi(pts)
            fills = [tint(RC_COLOR, 0.8) if j in nb0 else N for j in range(8)]; fills[0] = fs.INK
            edges = [RC_COLOR if j in nb0 else CELL_EDGE for j in range(8)]; edges[0] = fs.INK
            draw_cells(ax, polys, fills, edges, [1.2 if j in nb0 or j == 0 else 0.9 for j in range(8)])
            d = {j: np.linalg.norm(pts[j] - pts[0]) for j in nb0}
            order = sorted(nb0, key=d.get)
            for r, j in enumerate(order, 1):
                ax.text(*pts[j], str(r), ha="center", va="center", fontsize=6.5, color=RC_COLOR, fontweight="bold", zorder=5)
            ax.text(*pts[0], "i", ha="center", va="center", fontsize=7.5, color=fs.SURFACE, zorder=5)
    panels.append(("rank", "Rank correlation", RC_COLOR, rank, ("t", "t + 1")))

    # individual files
    for key, title, color, draw, (tl, tr) in panels:
        fig, (al, ar) = plt.subplots(1, 2, figsize=(fs.SINGLE * 0.55, 1.15),
                                     gridspec_kw=dict(left=0.01, right=0.99, top=0.84, bottom=0.08, wspace=0.08))
        _panel(al, ar, title, color)
        if tr != "t + 1":
            for t in ar.texts:
                if t.get_text() == "t + 1": t.set_text(tr)
        draw(al, ar)
        save_hi(fig, OUT / f"{prefix}{key}")

    # combined 3 x 2
    fig = plt.figure(figsize=(fs.DOUBLE * 0.38, 3.1))
    gs = fig.add_gridspec(3, 4, left=0.01, right=0.99, top=0.94, bottom=0.03, wspace=0.08, hspace=0.32)
    for n, (key, title, color, draw, (tl, tr)) in enumerate(panels):
        r, c = divmod(n, 2)
        al, ar = fig.add_subplot(gs[r, 2 * c]), fig.add_subplot(gs[r, 2 * c + 1])
        _panel(al, ar, title, color)
        if tr != "t + 1":
            for t in ar.texts:
                if t.get_text() == "t + 1": t.set_text(tr)
        draw(al, ar)
    save_hi(fig, OUT / "schematic_metric_panels")


# --------------------------------------------------------------------------- WIO and RC

def schematic_wio_rc(prefix="schematic_"):
    """Two rows: WIO (weights at t, retained weights at t+1) and RC (distance ranks
    at t and t+1).  Values printed in the right frame are computed from the drawing."""
    from scipy.stats import spearmanr
    N = CELL_FILL
    NAMES = {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f", 7: "g"}
    nb0 = sorted(delaunay_neighbours(PATCH, 0))

    def tess_focal(ax, pts, highlight, color, fill_hi, labels, spokes=None, lws_hi=1.3):
        polys = bounded_voronoi(pts)
        fills = [fill_hi if j in highlight else N for j in range(len(pts))]; fills[0] = fs.INK
        edges = [color if j in highlight else CELL_EDGE for j in range(len(pts))]; edges[0] = fs.INK
        draw_cells(ax, polys, fills, edges, [lws_hi if j in highlight or j == 0 else 0.9 for j in range(len(pts))])
        if spokes:
            for j, lw in spokes.items():
                q = pts[0] + 0.70 * (pts[j] - pts[0])
                ax.plot([pts[0][0], q[0]], [pts[0][1], q[1]], color=color, lw=lw, zorder=4, solid_capstyle="round")
        for j, (txt, col, bold) in labels.items():
            ax.text(*pts[j], txt, ha="center", va="center", fontsize=6.5, color=col, zorder=5,
                    fontweight="bold" if bold else None,
                    path_effects=[pe.withStroke(linewidth=2.0, foreground=fs.SURFACE)])
        ax.text(*pts[0], "i", ha="center", va="center", fontsize=7.5, color=fs.SURFACE, zorder=5)

    # ---- WIO: weights at t; at t+1 a slides out (displaced) and b is undetected (vanished)
    p1 = PATCH1.copy(); p1[1] = [0.86, 0.50]; p1[5] = [0.64, 0.40]
    vanished = {2}
    keep = [k for k in range(8) if k not in vanished]
    p1v = p1[keep]
    nb1 = {keep[j] for j in delaunay_neighbours(p1v, 0)}
    retained = [j for j in nb0 if j in nb1]
    w = {j: 1 / np.linalg.norm(PATCH[j] - PATCH[0]) for j in nb0}; sw = sum(w.values())
    w = {j: v / sw for j, v in w.items()}
    wio = sum(w[j] for j in retained)
    WC = fs.INK2

    def wio_left(ax):
        tess_focal(ax, PATCH, nb0, WC, "#dcdbd5",
                   {j: (f"{w[j]:.2f}", fs.INK, False) for j in nb0},
                   spokes={j: 0.4 + 9 * w[j] for j in nb0})

    def wio_right(ax):
        labels = {j: (f"{w[j]:.2f}", fs.INK, False) for j in retained}
        labels[1] = ("0", FATE["displaced"], True)
        hl = set(retained)
        polys = bounded_voronoi(p1v)
        fills = [("#dcdbd5" if keep[j] in hl else N) for j in range(len(keep))]; fills[0] = fs.INK
        edges = [(WC if keep[j] in hl else CELL_EDGE) for j in range(len(keep))]; edges[0] = fs.INK
        edges[keep.index(1)] = FATE["displaced"]; fills[keep.index(1)] = tint(FATE["displaced"], 0.8)
        draw_cells(ax, polys, fills, edges, [1.3 if keep[j] in hl or j == 0 or keep[j] == 1 else 0.9 for j in range(len(keep))])
        for j in retained:
            q = p1[0] + 0.70 * (p1[j] - p1[0])
            ax.plot([p1[0][0], q[0]], [p1[0][1], q[1]], color=WC, lw=0.4 + 9 * w[j], zorder=4, solid_capstyle="round")
        # vanished b: true outline, dashed, weight lost
        pt = bounded_voronoi(p1)
        ax.add_patch(Polygon(shrink(pt[2], 0.90), closed=True, facecolor=tint(FATE["vanished"], 0.8),
                             edgecolor=FATE["vanished"], lw=1.2, ls=(0, (2, 1.5)), zorder=3))
        labels[2] = ("0", FATE["vanished"], True)
        for j, (txt, col, bold) in labels.items():
            ax.text(*p1[j], txt, ha="center", va="center", fontsize=6.5, color=col, zorder=5,
                    fontweight="bold" if bold else None, path_effects=[pe.withStroke(linewidth=2.0, foreground=fs.SURFACE)])
        ax.text(*p1[0], "i", ha="center", va="center", fontsize=7.5, color=fs.SURFACE, zorder=5)
        ax.text(0.5, -0.17, f"WIO = {wio:.2f}", ha="center", va="top", fontsize=7, color=fs.INK, zorder=6)

    # ---- RC: ranks of neighbour distance at t and t+1 (all retained)
    pr = PATCH1.copy(); pr[2] = [0.50, 0.70]; pr[3] = [0.20, 0.70]
    assert set(delaunay_neighbours(pr, 0)) >= set(nb0)
    d0 = {j: np.linalg.norm(PATCH[j] - PATCH[0]) for j in nb0}
    d1 = {j: np.linalg.norm(pr[j] - pr[0]) for j in nb0}
    r0 = {j: r for r, j in enumerate(sorted(nb0, key=d0.get), 1)}
    r1 = {j: r for r, j in enumerate(sorted(nb0, key=d1.get), 1)}
    rc = spearmanr([r0[j] for j in nb0], [r1[j] for j in nb0]).statistic
    RC = RC_COLOR

    def rc_left(ax):
        tess_focal(ax, PATCH, nb0, RC, tint(RC, 0.8), {j: (str(r0[j]), RC, True) for j in nb0})

    def rc_right(ax):
        tess_focal(ax, pr, nb0, RC, tint(RC, 0.8), {j: (str(r1[j]), RC, True) for j in nb0})
        ax.text(0.5, -0.17, f"RC = {rc:.2f}", ha="center", va="top", fontsize=7, color=fs.INK, zorder=6)

    rows = [("wio", "Weighted identity overlap", fs.INK, wio_left, wio_right),
            ("rc", "Rank correlation", RC, rc_left, rc_right)]

    for key, title, color, left, right in rows:
        fig, (al, ar) = plt.subplots(1, 2, figsize=(fs.SINGLE * 0.55, 1.3),
                                     gridspec_kw=dict(left=0.01, right=0.99, top=0.86, bottom=0.2, wspace=0.08))
        _panel(al, ar, title, color); left(al); right(ar)
        save_hi(fig, OUT / f"{prefix}{key}")

    fig = plt.figure(figsize=(fs.SINGLE * 0.55, 2.6))
    gs = fig.add_gridspec(2, 2, left=0.01, right=0.99, top=0.93, bottom=0.09, wspace=0.08, hspace=0.5)
    for r, (key, title, color, left, right) in enumerate(rows):
        al, ar = fig.add_subplot(gs[r, 0]), fig.add_subplot(gs[r, 1])
        _panel(al, ar, title, color); left(al); right(ar)
    save_hi(fig, OUT / f"{prefix}wio_rc")


# --------------------------------------------------------------------------- errors + metric shift

def _example_frames():
    """The drawn tissue as (positions, ids) at t and, per error type, at t+1.
    Positions are the Voronoi seeds used for the drawing; ids are the tracker's."""
    ids0 = np.arange(1, 9)
    frames = {}
    # identity switch: cells 1 and 2 swap IDs
    i1 = ids0.copy(); i1[0], i1[1] = 2, 1
    frames["switch"] = (PATCH1.copy(), i1)
    # fragmentation: cell 1 continues under a new ID
    i1 = ids0.copy(); i1[0] = 9
    frames["frag"] = (PATCH1.copy(), i1)
    # miss: cell 1 not detected
    frames["miss"] = (PATCH1[1:].copy(), ids0[1:].copy())
    # split: cell 1 detected twice, either side of the dividing line
    nrm = np.array([1.0, -0.2]); nrm /= np.linalg.norm(nrm)
    p = PATCH1.copy(); c = p[0].copy(); p[0] = c - 0.06 * nrm
    frames["split"] = (np.vstack([p, c + 0.06 * nrm]), np.append(ids0, 9))
    # merge: cells 1 and 2 detected as one, at their midpoint, under ID 1
    p = np.delete(PATCH1, 1, axis=0); p[0] = 0.5 * (PATCH1[0] + PATCH1[1])
    frames["merge"] = (p, np.delete(ids0, 1))
    # false positive: an extra detection at the junction
    frames["fp"] = (np.vstack([PATCH1, [FP_XY]]), np.append(ids0, 9))
    return (PATCH.copy(), ids0), frames


def _continuity_2d(p0, id0, p1, id1):
    """WIO, RC and track survival for one frame pair, 2-D Delaunay neighbourhoods,
    the definitions of tracking_validation.metrics (penalise_absence=False)."""
    from scipy.stats import spearmanr
    pos1 = {int(i): p1[k] for k, i in enumerate(id1)}
    nb0 = {int(id0[i]): {int(id0[j]) for j in delaunay_neighbours(p0, i)} for i in range(len(p0))}
    nb1 = {int(id1[i]): {int(id1[j]) for j in delaunay_neighbours(p1, i)} for i in range(len(p1))}
    wio, rc = [], []
    for i, pi in zip(id0, p0):
        i = int(i)
        if i not in pos1:
            continue
        d0 = {j: np.linalg.norm(p0[list(id0).index(j)] - pi) for j in nb0[i]}
        w = {j: 1 / d for j, d in d0.items()}; sw = sum(w.values())
        wio.append(sum(w[j] for j in nb0[i] if j in nb1[i]) / sw)
        shared = [j for j in nb0[i] if j in nb1[i]]
        if len(shared) >= 3:
            d1 = [np.linalg.norm(pos1[j] - pos1[i]) for j in shared]
            rc.append(spearmanr([d0[j] for j in shared], d1).statistic)
    surv = np.mean([int(i) in pos1 for i in id0])
    return dict(WIO=float(np.mean(wio)), RC=float(np.nanmean(rc)), Surv=float(surv))


def _example_metrics():
    (p0, id0), frames = _example_frames()
    return {key: _continuity_2d(p0, id0, p1, id1) for key, (p1, id1) in frames.items()}


SHIFT = _example_metrics()


def _shift_bars(ax, key, color):
    """Two horizontal bars, WIO and RC: bar length = size of the drop, scaled to the largest."""
    """Three bars: the metric values on the drawn tissue (0 to 1); the open box is the full scale."""
    ax.set_xlim(-0.30, 1.22); ax.set_ylim(-0.7, 2.7); ax.axis("off")
    for y, m in ((2, "WIO"), (1, "RC"), (0, "Surv")):
        v = SHIFT[key][m]
        ax.barh(y, 1.0, height=0.58, facecolor="none", edgecolor=CELL_EDGE, lw=0.6, zorder=1)
        ax.barh(y, max(v, 0.0), height=0.58, color=color, zorder=2)
        ax.text(-0.04, y, m, ha="right", va="center", fontsize=6.5, color=fs.INK, zorder=3)
        ax.text(1.03, y, f"{v:.2f}", ha="left", va="center", fontsize=6, color=fs.INK, zorder=3)
    ax.plot([0, 0], [-0.45, 2.45], color=fs.INK, lw=0.6, zorder=1)


def schematic_errors_with_shift(stem="schematic_errors_shift", separate_prefix="schematic_error_shift_"):
    """The six error panels, each with the WIO / RC shift drawn underneath."""
    C = fs.SERIES; N = CELL_FILL; plain = [N] * 8
    base_lab = ["1", "2", "3", "4", "5", "6", "7", "8"]

    def switch(ar):
        lab = base_lab.copy(); lab[0], lab[1] = "2", "1"
        _tess(ar, PATCH1, lab, [C["switch"], C["switch"]] + [N] * 6)

    def frag(ar):
        lab = base_lab.copy(); lab[0] = "9"
        _tess(ar, PATCH1, lab, [C["frag"]] + [N] * 7)

    def miss(ar):
        lab = base_lab.copy(); lab[0] = ""
        _tess(ar, PATCH1, lab, [C["miss"]] + [N] * 7, ls=[(0, (2, 1.5))] + ["-"] * 7)

    def split(ar):
        lab = base_lab.copy(); lab[0] = ""
        polys = _tess(ar, PATCH1, lab, [C["split"]] + [N] * 7)
        q = shrink(polys[0], 0.93); c = q.mean(axis=0)
        seg = clip_line(q, c, np.array([0.2, 1.0]))
        ar.plot(seg[:, 0], seg[:, 1], color=C["split"], lw=1.2, zorder=4)
        nrm = np.array([1.0, -0.2]); nrm /= np.linalg.norm(nrm)
        for off, txt in ((-0.075, "1"), (0.075, "9")):
            ar.text(*(c + off * nrm), txt, ha="center", va="center", fontsize=6, color=C["split"], fontweight="bold", zorder=5)

    def merge(ar):
        from matplotlib.patches import Polygon as _P
        from scipy.spatial import ConvexHull
        lab = base_lab.copy(); lab[0] = ""; lab[1] = ""
        polys = _tess(ar, PATCH1, lab, plain)
        tc = tint(C["merge"])
        for k in (0, 1):
            ar.add_patch(_P(shrink(polys[k], 0.93), closed=True, facecolor=tc, edgecolor=tc, lw=1.0, zorder=3))
        u = np.vstack([shrink(polys[0], 0.93), shrink(polys[1], 0.93)])
        hull = u[ConvexHull(u).vertices]
        ar.add_patch(_P(hull, closed=True, facecolor=tc, edgecolor=C["merge"], lw=1.2, zorder=4))
        ar.text(*(0.5 * (PATCH1[0] + PATCH1[1])), "1", ha="center", va="center", fontsize=6.5, color=C["merge"], fontweight="bold", zorder=6)

    def fp(ar):
        p = np.vstack([PATCH1, [FP_XY]])
        _tess(ar, p, base_lab + ["9"], plain + [C["fp"]])

    draws = [("switch", switch), ("frag", frag), ("miss", miss), ("split", split), ("merge", merge), ("fp", fp)]

    def one(fig, sub, key, draw):
        """sub = a SubplotSpec; inside it: two frames on top, bars underneath."""
        g = sub.subgridspec(2, 2, height_ratios=[1.0, 0.42], wspace=0.08, hspace=0.10)
        al, ar = fig.add_subplot(g[0, 0]), fig.add_subplot(g[0, 1])
        ab = fig.add_subplot(g[1, :])
        _panel(al, ar, fs.LABELS[key], C[key])
        _tess(al, PATCH, base_lab, plain); draw(ar)
        _shift_bars(ab, key, C[key])

    # separate files
    for key, draw in draws:
        fig = plt.figure(figsize=(fs.SINGLE * 0.55, 1.7))
        gs = fig.add_gridspec(1, 1, left=0.01, right=0.99, top=0.89, bottom=0.02)
        one(fig, gs[0, 0], key, draw)
        save_hi(fig, OUT / f"{separate_prefix}{key}")

    # combined 3 x 2
    fig = plt.figure(figsize=(fs.DOUBLE * 0.38, 4.3))
    gs = fig.add_gridspec(3, 2, left=0.01, right=0.99, top=0.96, bottom=0.01, wspace=0.12, hspace=0.32)
    for n, (key, draw) in enumerate(draws):
        r, c = divmod(n, 2)
        one(fig, gs[r, c], key, draw)
    save_hi(fig, OUT / stem)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    schematic_metrics()
    schematic_errors()
    schematic_errors_separate()
    schematic_metric_panels()
    schematic_wio_rc()
    schematic_errors_with_shift()
