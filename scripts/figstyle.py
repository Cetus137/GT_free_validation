"""Shared figure style: one place for fonts, colours and mark specs.

Import and call ``apply()`` before plotting.  Single-column width is 88 mm
(3.46 in); double column 180 mm (7.09 in).  All text 8 pt, titles 8.5 pt.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

SINGLE = 3.46
DOUBLE = 7.09

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

# fixed categorical slots (identity follows the entity, never cycled)
SERIES = {
    "switch": "#2a78d6", "frag": "#eb6834", "miss": "#1baf7a",
    "split": "#eda100", "merge": "#e87ba4", "fp": "#008300", "hop": "#5b3a9e",
}
LABELS = {
    "switch": "Identity switch", "frag": "Track fragmentation", "miss": "Missed detection",
    "split": "Split", "merge": "Merge", "fp": "False positive", "hop": "Single hop",
}
ORDER4 = ["switch", "frag", "miss", "fp"]
ORDER6 = ["switch", "frag", "miss", "split", "merge", "fp"]
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
SLAB, CUBE = "#2a78d6", "#eb6834"

LINE = dict(lw=2, solid_capstyle="round")
DASH = dict(lw=1.6, ls=(0, (3, 2)))
MARK = dict(marker="o", ms=4.5, mec=SURFACE, mew=1.2)
SCAT = dict(s=18, edgecolors=SURFACE, linewidths=0.8, zorder=3)


def apply() -> None:
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.labelcolor": INK2, "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": SURFACE, "pdf.fonttype": 42, "savefig.dpi": 300,
        "legend.frameon": False, "axes.grid": False,
    })


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def identity(ax, lim):
    ax.plot([0, lim], [0, lim], color=AXIS, lw=0.8, zorder=1)


def real_handle(label="Embryo", color=INK2):
    return Line2D([], [], color=color, **LINE, **MARK, label=label)


def sim_handle(label="Simulation", color=INK2):
    return Line2D([], [], color=color, **DASH, label=label)


def dot_handle(label, color, marker="o"):
    return Line2D([], [], ls="", marker=marker, ms=5, color=color, mec=SURFACE, label=label)


def save(fig, path_stem):
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}")
    plt.close(fig)
    print("wrote", f"{path_stem}.png")
