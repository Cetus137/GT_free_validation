"""Supplementary figure: identity-switch response depends on tissue geometry.

    python scripts/make_figure_switch_geometry.py

Real embryo vs cube-shaped and slab-shaped stage-matched simulations, one
panel per stage. Reads data/_results/ce_error_sweep_t1only_delaunay.csv (cube)
and data/_results/ce_slab_switch.csv (slab + real). Writes
figures/fig_switch_geometry.{png,pdf}.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "_results"
OUT = ROOT / "figures"

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"          # slots 1 and 2: slab, cube
STAGES = [("50-100", "Early: 50–100 cells"), ("100-200", "Middle: 100–200 cells"), ("200-360", "Late: 200–360 cells")]

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK2, "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": SURFACE, "pdf.fonttype": 42,
})


def real_curve(df: pd.DataFrame, stage: str):
    g = df[(df.error == "switch") & df.source.str.startswith("CE") & (df.stage == stage)]
    per_pair = g.groupby(["source", "pair", "rate"]).agg(x=("rate_realised", "mean"), y=("d_wio", "mean")).reset_index()
    agg = per_pair.groupby("rate").agg(x=("x", "mean"), y=("y", "mean"), sd=("y", "std"), n=("y", "size"))
    agg["ci"] = 1.96 * agg.sd / np.sqrt(agg.n)
    return agg


def sim_curve(df: pd.DataFrame, source: str):
    g = df[(df.error == "switch") & (df.source == source)]
    per_seed = g.groupby(["pair", "rate"]).agg(x=("rate_realised", "mean"), y=("d_wio", "mean")).reset_index()
    return per_seed.groupby("rate").agg(x=("x", "mean"), y=("y", "mean"), lo=("y", "min"), hi=("y", "max"))


def slope(c) -> float:
    return float(-np.sum(c.x * c.y) / np.sum(c.x ** 2))     # WIO points (x100) lost per 1% switched


def main() -> int:
    cube = pd.read_csv(RESULTS / "ce_error_sweep_t1only_delaunay.csv")
    slab = pd.read_csv(RESULTS / "ce_slab_switch.csv")
    OUT.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.5), sharey=True,
                             gridspec_kw=dict(left=0.09, right=0.99, top=0.82, bottom=0.3, wspace=0.18))
    for ax, (stage, title) in zip(axes, STAGES):
        real = real_curve(slab, stage)
        cu = sim_curve(cube, f"SIM-{stage}")
        sl = sim_curve(slab, f"SIM-slab-{stage}")
        thick = slab[(slab.source == f"SIM-slab-{stage}")].iloc[0]
        for c, col in ((cu, ORANGE), (sl, BLUE)):
            x, y = np.r_[0, c.x * 100], np.r_[0, c.y * 100]
            ax.fill_between(x, np.r_[0, c.lo * 100], np.r_[0, c.hi * 100], color=col, alpha=0.12, lw=0)
            ax.plot(x, y, ls=(0, (3, 2)), lw=1.6, color=col)
        xr, yr = np.r_[0, real.x * 100], np.r_[0, real.y * 100]
        ax.fill_between(xr, yr - np.r_[0, real.ci * 100], yr + np.r_[0, real.ci * 100], color=INK2, alpha=0.15, lw=0)
        ax.plot(xr, yr, lw=2, color=INK2, marker="o", ms=4.5, mec=SURFACE, mew=1.2)
        ax.set_title(title, loc="left", color=INK, pad=4)
        ax.text(0.03, 0.04, f"embryo {slope(real):.2f}\nslab {slope(sl):.2f}\ncube {slope(cu):.2f}  pt / %",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color=INK2, linespacing=1.3)
        ax.text(0.97, 0.96, f"slab {thick.sim_box_z / thick.sim_nn_med:.1f} spacings thick",
                transform=ax.transAxes, ha="right", va="top", fontsize=7, color=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_xlim(0, 21); ax.set_xticks([0, 5, 10, 20])
        ax.set_xlabel("Cells switched (%)")
    axes[0].set_ylim(-31, 1); axes[0].set_yticks([0, -10, -20, -30])
    axes[0].set_ylabel("Δ weighted identity\noverlap (×100)")
    handles = [Line2D([], [], color=INK2, lw=2, marker="o", ms=4.5, mec=SURFACE, mew=1.2, label="Real embryo (mean, 95% CI)"),
               Line2D([], [], color=BLUE, lw=1.6, ls=(0, (3, 2)), label="Slab simulation (embryo thickness)"),
               Line2D([], [], color=ORANGE, lw=1.6, ls=(0, (3, 2)), label="Cube simulation")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.09, 0.0), ncol=3, frameon=False,
               handlelength=2.6, columnspacing=1.4)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_switch_geometry.{ext}", dpi=300 if ext == "png" else None)
    print("wrote", OUT / "fig_switch_geometry.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
