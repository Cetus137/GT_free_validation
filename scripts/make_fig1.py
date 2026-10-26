"""Figure 1 (simulation-driven), one file per panel in figures/fig1/.

    python scripts/make_fig1.py

  a_metrics       what the metrics measure: one focal cell's neighbourhood at t and
                  t+1, the fate of each neighbour, and how the neighbour weight
                  partitions into overlap / vanished / displaced
  b_errors        the four tracking errors as two-frame cartoons
  c_fingerprint   which channels respond to which error (simulated tissue, 10% error)
  d_inversion     error proportions recovered from the metric vector on simulated
                  tissue: pooled fixed mixes (all four types at once) and per-type
                  recovery against injected rate
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import figstyle as fs  # noqa: E402
import make_panels as mp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

R = ROOT / "data" / "_results"
OUT = ROOT / "figures" / "fig1"
fs.apply()
# Figure 1 panel order (also used by the shared panel functions called from here)
fs.ORDER4[:] = ["miss", "switch", "frag", "fp"]
# Figure 1 style: black, bold ticks and labels
# Arial: Helvetica's bold sits inside a .ttc that matplotlib reads as regular only
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"],
                     "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold",
                     "xtick.color": fs.INK, "ytick.color": fs.INK, "axes.labelcolor": fs.INK,
                     "axes.edgecolor": fs.INK, "xtick.major.width": 0.8, "ytick.major.width": 0.8})
FATE = {"retained": fs.INK2, "vanished": fs.SERIES["miss"], "displaced": fs.SERIES["switch"], "newcomer": fs.SERIES["fp"]}


# --------------------------------------------------------------------------- a: the metrics

def a_metrics():
    rng = np.random.default_rng(3)
    focal = np.array([0.0, 0.0])
    ang = np.linspace(0, 2 * np.pi, 7)[:-1] + 0.2
    radii = np.array([1.0, 1.15, 0.95, 1.05, 1.1, 1.0])
    nb0 = np.c_[np.cos(ang), np.sin(ang)] * radii[:, None]
    labels = list("ABCDEF")
    fate = {"A": "retained", "B": "retained", "C": "retained", "D": "retained", "E": "vanished", "F": "displaced"}
    nb1 = nb0 + rng.normal(0, 0.07, nb0.shape)
    far_F = np.array([2.3, -0.9]); newcomer = np.array([0.73, 0.68])
    w = 1 / radii; w = w / w.sum()

    fig = plt.figure(figsize=(fs.SINGLE, 2.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1], left=0.02, right=0.98, top=0.9, bottom=0.17, wspace=0.08, hspace=0.35)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    for ax, title in zip(axes, ("t", "t + 1")):
        ax.set_aspect("equal"); ax.set_xlim(-1.6, 2.6); ax.set_ylim(-1.6, 1.6); ax.axis("off")
        ax.set_title(title, loc="left", pad=2)
        ax.scatter(*focal, s=120, color=fs.INK, zorder=4)
    for p, l in zip(nb0, labels):
        axes[0].plot([0, p[0]], [0, p[1]], color=fs.GRID, lw=1, zorder=1)
        axes[0].scatter(*p, s=70, color=fs.INK2, edgecolors=fs.SURFACE, linewidths=1, zorder=3)
        axes[0].text(p[0] * 1.3, p[1] * 1.3, l, ha="center", va="center", fontsize=7.5, color=fs.INK2)
    for p, l in zip(nb1, labels):
        f = fate[l]
        if f == "vanished":
            axes[1].scatter(*p, s=70, facecolors="none", edgecolors=FATE[f], linewidths=1.2, linestyle="--", zorder=3)
            axes[1].text(p[0] * 1.3, p[1] * 1.3, l, ha="center", va="center", fontsize=7.5, color=FATE[f])
        elif f == "displaced":
            axes[1].scatter(*far_F, s=70, color=FATE[f], edgecolors=fs.SURFACE, linewidths=1, zorder=3)
            axes[1].text(far_F[0] + 0.25, far_F[1], l, ha="left", va="center", fontsize=7.5, color=FATE[f])
            axes[1].annotate("", xy=far_F, xytext=p, arrowprops=dict(arrowstyle="->", color=FATE[f], lw=1))
        else:
            axes[1].plot([0, p[0]], [0, p[1]], color=fs.GRID, lw=1, zorder=1)
            axes[1].scatter(*p, s=70, color=fs.INK2, edgecolors=fs.SURFACE, linewidths=1, zorder=3)
            axes[1].text(p[0] * 1.3, p[1] * 1.3, l, ha="center", va="center", fontsize=7.5, color=fs.INK2)
    axes[1].plot([0, newcomer[0]], [0, newcomer[1]], color=fs.GRID, lw=1, zorder=1)
    axes[1].scatter(*newcomer, s=70, color=FATE["newcomer"], edgecolors=fs.SURFACE, linewidths=1, zorder=3)
    axes[1].text(newcomer[0] * 1.3, newcomer[1] * 1.3, "G", ha="center", va="center", fontsize=7.5, color=FATE["newcomer"])

    # weight partition of the t-neighbourhood: overlap (retained) | vanished | displaced
    axb = fig.add_subplot(gs[1, :]); axb.axis("off"); axb.set_xlim(0, 1); axb.set_ylim(0, 1)
    parts = [("retained", sum(w[i] for i, l in enumerate(labels) if fate[l] == "retained")),
             ("vanished", sum(w[i] for i, l in enumerate(labels) if fate[l] == "vanished")),
             ("displaced", sum(w[i] for i, l in enumerate(labels) if fate[l] == "displaced"))]
    x = 0.0
    for name, v in parts:
        axb.add_patch(FancyBboxPatch((x + 0.004, 0.55), v - 0.008, 0.4, boxstyle="square,pad=0", color=FATE[name], lw=0))
        axb.text(x + v / 2, 0.75, f"{v:.2f}", ha="center", va="center", fontsize=7, color="#ffffff")
        lab = "Overlap" if name == "retained" else name.capitalize()
        axb.text(x + v / 2, 0.38, lab, ha="center", va="center", fontsize=7, color=fs.INK2)
        x += v
    axb.text(0, 0.08, "Weight of t-neighbours (1/distance): overlap + vanished + displaced = 1", fontsize=7, color=fs.INK2, va="center")
    fig.legend(handles=[fs.dot_handle(k.capitalize(), c) for k, c in FATE.items()], loc="lower left",
               bbox_to_anchor=(0.02, 0.0), ncol=4, handletextpad=0.3, columnspacing=0.9)
    fs.save(fig, OUT / "a_metrics")


# --------------------------------------------------------------------------- b: the errors

def _cells(ax, pts, labels, colors, hollow=()):
    for (x, y), l, c in zip(pts, labels, colors):
        if l in hollow:
            ax.scatter(x, y, s=160, facecolors="none", edgecolors=c, linewidths=1.2, linestyle="--", zorder=3)
        else:
            ax.scatter(x, y, s=160, color=c, edgecolors=fs.SURFACE, linewidths=1, zorder=3)
        ax.text(x, y, l, ha="center", va="center", fontsize=7, color="#ffffff" if l not in hollow else c, zorder=4)


def b_errors():
    pal = {"1": "#2a78d6", "2": "#eb6834", "3": "#1baf7a", "4": "#eda100", "n": fs.SERIES["fp"]}
    base = np.array([[0.0, 0.0], [1.0, 0.3], [0.5, -0.9]])
    moved = base + np.array([0.08, 0.05])
    cases = [
        ("Missed detection", ["1", "2", "3"], ["1", "2"], [pal["1"], pal["2"], pal["3"]], [pal["1"], pal["2"]], "3", None),
        ("Identity switch", ["1", "2", "3"], ["1", "3", "2"], [pal["1"], pal["2"], pal["3"]], [pal["1"], pal["3"], pal["2"]], None, None),
        ("Track fragmentation", ["1", "2", "3"], ["1", "2", "4"], [pal["1"], pal["2"], pal["3"]], [pal["1"], pal["2"], pal["4"]], None, "4"),
        ("False positive", ["1", "2", "3"], ["1", "2", "3", "n"], [pal["1"], pal["2"], pal["3"]], [pal["1"], pal["2"], pal["3"], pal["n"]], None, "n"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(fs.DOUBLE, 2.7), gridspec_kw=dict(left=0.03, right=0.99, top=0.86, bottom=0.03, wspace=0.08, hspace=0.1))
    for col, (title, l0, l1, c0, c1, ghost, new) in enumerate(cases):
        ax0, ax1 = axes[0, col], axes[1, col]
        for ax in (ax0, ax1):
            ax.set_aspect("equal"); ax.set_xlim(-0.6, 1.9); ax.set_ylim(-1.5, 1.0); ax.axis("off")
        ax0.set_title(title, loc="left", pad=8)
        ax0.text(-0.55, -1.35, "t", fontsize=7.5, color=fs.MUTED); ax1.text(-0.55, -1.35, "t + 1", fontsize=7.5, color=fs.MUTED)
        _cells(ax0, base, l0, c0)
        if title == "Missed detection":
            _cells(ax1, moved[:2], l1, c1)
            ax1.scatter(*moved[2], s=160, facecolors="none", edgecolors=fs.MUTED, linewidths=1.0, linestyle="--", zorder=3)
        elif title == "False positive":
            _cells(ax1, list(moved) + [np.array([1.5, -0.4])], l1, c1)
        elif title == "Identity switch":
            _cells(ax1, moved, l1, c1)
            # curved double arrow to the right of the two swapped cells
            ax1.annotate("", xy=moved[1] + [0.32, -0.05], xytext=moved[2] + [0.32, 0.05],
                         arrowprops=dict(arrowstyle="<->", color=fs.INK2, lw=0.9, connectionstyle="arc3,rad=-0.4"))
        else:
            _cells(ax1, list(moved), l1, c1)
    fs.save(fig, OUT / "b_errors")


# --------------------------------------------------------------------------- c: fingerprint with survival

COMPS = [("d_survival", "Survival"), ("d_vanished", "Vanished"), ("d_displaced", "Displaced"),
         ("d_newcomer", "Newcomer"), ("d_rc", "Rank corr.")]


def _sim10():
    df = mp._load_sweep("slab")
    if df is None:
        return None
    sim = df[(df.src == "sim") & (df.rate == 0.10)].copy()
    sim["d_survival"] = sim.survival - 1.0
    return sim


def c_fingerprint():
    """Grouped bars: for each channel, the change at 10% error, one bar per error type."""
    sim = _sim10()
    if sim is None:
        return
    order = fs.ORDER4
    fig, ax = plt.subplots(figsize=(fs.DOUBLE * 0.6, 2.7), gridspec_kw=dict(left=0.15, right=0.99, top=0.95, bottom=0.36))
    w = 0.19
    for i, t in enumerate(order):
        g = sim[sim.error == t]
        means = [100 * g[c].mean() for c, _ in COMPS]
        sds = [100 * g[c].std() for c, _ in COMPS]
        x = np.arange(len(COMPS)) + (i - 1.5) * w
        ax.bar(x, means, width=w - 0.02, color=fs.SERIES[t], lw=0, zorder=3)
        ax.errorbar(x, means, yerr=sds, fmt="none", ecolor=fs.INK2, elinewidth=0.8, capsize=0, zorder=4)
    ax.axhline(0, color=fs.AXIS, lw=0.8)
    ax.set_xticks(range(len(COMPS))); ax.set_xticklabels([n for _, n in COMPS])
    ax.set_ylabel("Change at 10% error (×100)"); fs.grid(ax)
    ax.spines["bottom"].set_visible(False); ax.tick_params(axis="x", length=0)
    fig.legend(handles=[fs.dot_handle(fs.LABELS[t], fs.SERIES[t]) for t in order], loc="lower left",
               bbox_to_anchor=(0.15, 0.0), ncol=2, handletextpad=0.3, columnspacing=1.2)
    fs.save(fig, OUT / "c_fingerprint")


def c_fingerprint_table():
    sim = _sim10()
    if sim is None:
        return
    comps = COMPS
    order = fs.ORDER4
    M = np.array([[sim[sim.error == k][c].mean() * 100 for c, _ in comps] for k in order])
    vmax = np.nanmax(np.abs(M)); norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    h = 0.5 * len(order) + 0.9
    fig, ax = plt.subplots(figsize=(fs.SINGLE, h), gridspec_kw=dict(left=0.42, right=0.98, top=0.98, bottom=0.8 / h))
    ax.imshow(M, cmap=fs.DIVERGING, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(-0.5, M.shape[1]), minor=True); ax.set_yticks(np.arange(-0.5, M.shape[0]), minor=True)
    ax.grid(which="minor", color=fs.SURFACE, lw=2); ax.tick_params(which="both", length=0)
    ax.set_xticks(range(len(comps))); ax.set_xticklabels([n for _, n in comps], rotation=35, ha="right", rotation_mode="anchor", color=fs.INK2)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([fs.LABELS[k] for k in order], color=fs.INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]; rgba = fs.DIVERGING(norm(v)); lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(j, i, f"{v:+.0f}" if abs(v) >= 0.5 else "0", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if lum < 0.55 else fs.INK)
    fs.save(fig, OUT / "c_fingerprint_table")


# --------------------------------------------------------------------------- calibration curves

def c_calibration():
    """Simulation calibration: change in each channel vs realised error rate, one panel per
    channel, one line per error type (mean over simulated pairs, 95% CI band)."""
    df = mp._load_sweep("slab")
    if df is None:
        return
    sim = df[df.src == "sim"].copy()
    sim["d_survival"] = sim.survival - 1.0
    # channel colours: vanished / displaced / newcomer match panel a; survival and rank corr. added
    ch_col = {"d_survival": "#eda100", "d_vanished": FATE["vanished"], "d_displaced": FATE["displaced"],
              "d_newcomer": FATE["newcomer"], "d_rc": "#6250d6"}
    fig, axes = plt.subplots(1, 4, figsize=(fs.DOUBLE, 2.3), sharey=True,
                             gridspec_kw=dict(left=0.07, right=0.99, top=0.88, bottom=0.36, wspace=0.12))
    for ax, t in zip(axes, fs.ORDER4):
        g = sim[sim.error == t]
        for col, name in COMPS:
            pp = g.groupby(["source", "pair", "rate"]).agg(x=("rate_realised", "mean"), y=(col, "mean")).reset_index()
            a = pp.groupby("rate").agg(x=("x", "mean"), y=("y", "mean"), sd=("y", "std"), n=("y", "size"))
            ci = 1.96 * a.sd / np.sqrt(a.n)
            x, y = np.r_[0, 100 * a.x], np.r_[0, 100 * a.y]
            ax.fill_between(x, y - np.r_[0, 100 * ci], y + np.r_[0, 100 * ci], color=ch_col[col], alpha=0.15, lw=0)
            ax.plot(x, y, color=ch_col[col], **fs.LINE, **{**fs.MARK, "ms": 3.5})
        ax.axhline(0, color=fs.AXIS, lw=0.8)
        ax.set_title(fs.LABELS[t], loc="left", pad=3); fs.grid(ax)
        ax.set_xlim(0, 21); ax.set_xticks([0, 5, 10, 20]); ax.set_xlabel("Cells affected (%)")
    axes[0].set_ylabel("Change (×100)")
    fig.legend(handles=[fs.dot_handle(name, ch_col[col]) for col, name in COMPS], loc="lower left",
               bbox_to_anchor=(0.07, 0.0), ncol=5, handletextpad=0.3, columnspacing=1.2)
    fs.save(fig, OUT / "c_calibration_channels")


def _calibration_fine(metric: str, ylabel: str, stem: str, ylim=(-6, 0.5)):
    """Fine sweep (0.25-5%, held-out simulated tissues): change in one metric vs realised
    rate, one panel per error type.  No grid, no annotation; mean with 95% CI over pairs."""
    f = R / "ce_error_sweep_fine_delaunay.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    # evenly spaced points: bin each simulated pair by its REALISED rate into whole-percent bins
    # (1%: 0.5-1.5, 2%: 1.5-2.5, ...), so crosses sit near 1..5% for every error type
    sim = df[df.source.str.startswith("SIM") & (df.error != "none")].copy()
    sim["rate"] = np.round(sim.rate_realised * 100) / 100
    sim = sim[sim.rate.between(0.01, 0.05)]
    def draw(ax, t):
        g = sim[sim.error == t]
        pp = g.groupby(["source", "pair", "rate"]).agg(x=("rate_realised", "mean"), y=(metric, "mean")).reset_index()
        a = pp.groupby("rate").agg(x=("x", "mean"), y=("y", "mean"), sd=("y", "std"), n=("y", "size"))
        ci = 1.96 * a.sd / np.sqrt(a.n)
        x, y = np.r_[0, 100 * a.x], np.r_[0, 100 * a.y]
        ax.errorbar(x[1:], y[1:], yerr=100 * ci.values, fmt="none", ecolor=fs.SERIES[t], elinewidth=1.0,
                    capsize=2.5, capthick=1.0, zorder=3)
        ax.plot(x[1:], y[1:], color=fs.SERIES[t], ls="", marker="x", ms=5.5, mew=1.4, zorder=4)
        ax.plot(x, y, color=fs.SERIES[t], lw=1.2, solid_capstyle="round", zorder=2)
        ax.set_title(fs.LABELS[t], loc="left", pad=3)
        ax.set_xlim(0, 5.3); ax.set_xticks([0, 1, 2, 3, 4, 5]); ax.set_ylim(*ylim)
        ax.set_xlabel("Cells affected (%)")

    # combined row
    fig, axes = plt.subplots(1, 4, figsize=(fs.DOUBLE, 2.0), sharey=True,
                             gridspec_kw=dict(left=0.08, right=0.99, top=0.88, bottom=0.24, wspace=0.12))
    for ax, t in zip(axes, fs.ORDER4):
        draw(ax, t)
    axes[0].set_ylabel(ylabel)
    fs.save(fig, OUT / stem)

    # one file per error type, numbered in figure order
    sub = OUT / "calibration"; sub.mkdir(exist_ok=True)
    tag = "wio" if metric == "d_wio" else "rc"
    for i, t in enumerate(fs.ORDER4, 1):
        fig, ax = plt.subplots(figsize=(1.85, 2.0), gridspec_kw=dict(left=0.3, right=0.96, top=0.88, bottom=0.24))
        draw(ax, t); ax.set_ylabel(ylabel)
        fs.save(fig, sub / f"{tag}_{i}_{t}")


def c_calibration_wio():
    """Simulation calibration: overlap and rank correlation, fine sweep, shared y range."""
    _calibration_fine("d_wio", "Δ WIO (×100)", "c_calibration")
    _calibration_fine("d_rc", "Δ RC (×100)", "c_calibration_rc")


# --------------------------------------------------------------------------- d: inversion

def d_inversion():
    pooled = R / "error_mix_sim_pooled.csv"; on = R / "error_mix_sim_on_sweep.csv"
    if pooled.exists():
        mp.OUT = OUT
        mp.inversion_pooled("error_mix_sim_pooled.csv", "d_inversion_pooled")
    if on.exists():
        mp.OUT = OUT
        mp.inversion_rate("error_mix_sim_on_sweep.csv", "d_inversion_rate")
    fine = R / "error_mix_sim_fine_on_sweep.csv"
    if fine.exists():
        mp.OUT = OUT
        mp.inversion_rate("error_mix_sim_fine_on_sweep.csv", "d_inversion_rate_fine", lim=5.5, ticks=(0, 1, 2, 3, 4, 5))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    a_metrics(); b_errors(); c_calibration(); c_calibration_wio(); c_fingerprint(); c_fingerprint_table(); d_inversion()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
