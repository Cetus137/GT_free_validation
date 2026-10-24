"""Figure: error-type rates estimated from the metric vector vs the true rates.

    python scripts/make_figure_error_mix.py

Reads data/_results/error_mix_validation.csv (single held-out frame pairs,
random mixtures) and data/_results/error_mix_validation_pooled.csv (fixed
mixes pooled over 30 pairs). Writes figures/fig_error_mix.{png,pdf}.

Panel a: one small multiple per error type, estimated vs realised rate per
frame pair (global baseline), identity line, RMSE.
Panel b: pooled scenarios, estimated vs true, all types on one axis.
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
SERIES = {"switch": "#2a78d6", "frag": "#eb6834", "miss": "#1baf7a", "fp": "#008300"}   # same slots as fig 1
LABELS = {"switch": "Identity switch", "frag": "Track fragmentation", "miss": "Missed detection", "fp": "False positive"}
TYPES = ["switch", "frag", "miss", "fp"]

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK2, "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": SURFACE, "pdf.fonttype": 42,
})


def main() -> int:
    single = pd.read_csv(RESULTS / "error_mix_validation.csv")
    pooled = pd.read_csv(RESULTS / "error_mix_validation_pooled.csv")
    OUT.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(6.3, 3.7))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 0.35, 1.9], wspace=0.35, hspace=0.55,
                          left=0.09, right=0.98, top=0.9, bottom=0.27)

    # ---- a: per-pair small multiples -------------------------------------------------
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
    lim = 4 * np.ceil((100 * single[[f"true_{t}" for t in TYPES] + [f"glob_{t}" for t in TYPES]].max().max() + 0.5) / 4)
    ticks = np.arange(0, lim + 0.1, 4)
    for ax, t in zip(axes, TYPES):
        x, y = 100 * single[f"true_{t}"], 100 * single[f"glob_{t}"]
        ax.plot([0, lim], [0, lim], color=AXIS, lw=0.8)
        ax.scatter(x, y, s=14, color=SERIES[t], edgecolors=SURFACE, linewidths=0.8, zorder=3)
        rmse = np.sqrt(np.mean((y - x) ** 2))
        ax.set_title(LABELS[t], loc="left", pad=3, color=INK)
        ax.text(0.97, 0.05, f"±{rmse:.1f} pt", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK2)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_aspect("equal"); ax.grid(color=GRID, lw=0.6)
    for i, ax in enumerate(axes):
        if i % 2:
            ax.tick_params(labelleft=False)
        if i >= 2:
            ax.set_xlabel("True rate (% cells)")
    axes[0].set_ylabel("Estimated (%)"); axes[2].set_ylabel("Estimated (%)")
    fig.text(0.01, 0.96, "a", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.09, 0.96, f"Single frame pairs, random mixtures (n = {len(single)})", fontsize=8, color=INK2, va="center")

    # ---- b: pooled scenarios ----------------------------------------------------------
    axb = fig.add_subplot(gs[:, 3])
    limb = 100 * pooled[[f"true_{t}" for t in TYPES] + [f"est_{t}" for t in TYPES]].max().max() + 0.5
    axb.plot([0, limb], [0, limb], color=AXIS, lw=0.8)
    for t in TYPES:
        for seq, marker in (("01", "o"), ("02", "s")):
            g = pooled[pooled.seq.astype(str).str.zfill(2) == seq]
            axb.scatter(100 * g[f"true_{t}"], 100 * g[f"est_{t}"], s=26, marker=marker, color=SERIES[t],
                        edgecolors=SURFACE, linewidths=0.8, zorder=3)
    e = np.concatenate([100 * (pooled[f"est_{t}"] - pooled[f"true_{t}"]).values for t in TYPES])
    axb.text(0.97, 0.05, f"RMSE {np.sqrt(np.mean(e ** 2)):.2f} pt\nmax {np.abs(e).max():.2f} pt",
             transform=axb.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK2, linespacing=1.3)
    axb.set_xlim(0, limb); axb.set_ylim(0, limb); axb.set_aspect("equal"); axb.grid(color=GRID, lw=0.6)
    axb.set_xlabel("True rate (% cells)"); axb.set_ylabel("Estimated (%)")
    axb.set_title(f"Fixed mix pooled over {int(pooled.n_pairs.iloc[0])} pairs,\n{pooled.scenario.nunique()} scenarios × 2 recordings",
                  loc="left", pad=4, color=INK)
    handles = [Line2D([], [], ls="", marker="o", ms=5, color=SERIES[t], mec=SURFACE, label=LABELS[t]) for t in TYPES]
    handles += [Line2D([], [], ls="", marker="o", ms=5, color=INK2, mec=SURFACE, label="Recording 01"),
                Line2D([], [], ls="", marker="s", ms=5, color=INK2, mec=SURFACE, label="Recording 02")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.09, 0.0), ncol=3, frameon=False,
               handletextpad=0.3, columnspacing=1.2)
    fig.text(0.60, 0.96, "b", fontsize=11, fontweight="bold", color=INK)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_error_mix.{ext}", dpi=300 if ext == "png" else None)
    print("wrote", OUT / "fig_error_mix.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
