"""Figures: error-type rates estimated from the metric vector vs the true rates.

    python scripts/make_figure_error_mix.py

Main figure  figures/fig_error_mix.{png,pdf}
  a: accuracy vs true rate — for each error type injected alone at 1–20% into
     real embryo frame pairs (the calibration sweep rows, global baseline):
     estimated rate of that type (solid, mean ± 95% CI over pairs) and the total
     falsely attributed to the other three types (dashed), against the realised rate.
  b: fixed error mixes pooled over 30 held-out pairs per recording, all types on
     one axis.
Supplementary  figures/fig_error_mix_pairs.{png,pdf}
  single held-out frame pairs with random mixtures, one small multiple per type.

Inputs: data/_results/error_mix_validation{,_pooled,_on_sweep}.csv.
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


def _ci(g: pd.DataFrame, col: str) -> pd.DataFrame:
    """mean and 95% CI over frame pairs (seeds averaged within a pair first)."""
    per_pair = g.groupby(["source", "pair", "rate"]).agg(x=("rate_realised", "mean"), y=(col, "mean")).reset_index()
    agg = per_pair.groupby("rate").agg(x=("x", "mean"), y=("y", "mean"), sd=("y", "std"), n=("y", "size"))
    agg["ci"] = 1.96 * agg.sd / np.sqrt(agg.n)
    return agg


def main_figure(on_sweep: pd.DataFrame, pooled: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.1, 2.75))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 0.35, 1.6], wspace=0.25,
                          left=0.07, right=0.985, top=0.83, bottom=0.38)

    # ---- a: accuracy vs true rate, one small multiple per type ---------------------
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    lim = 22
    for ax, t in zip(axes, TYPES):
        g = on_sweep[on_sweep.error == t]
        same = _ci(g, "est_same"); other = _ci(g, "est_other")
        x = np.r_[0, 100 * same.x]
        ax.plot([0, lim], [0, lim], color=AXIS, lw=0.8)
        ax.fill_between(x, np.r_[0, 100 * (same.y - same.ci)], np.r_[0, 100 * (same.y + same.ci)],
                        color=SERIES[t], alpha=0.18, lw=0)
        ax.plot(x, np.r_[0, 100 * same.y], lw=2, color=SERIES[t], marker="o", ms=4.5, mec=SURFACE, mew=1.2)
        ax.plot(np.r_[0, 100 * other.x], np.r_[0, 100 * other.y], lw=1.4, ls=(0, (3, 2)), color=INK2)
        ax.set_title(LABELS[t], loc="left", pad=3, color=INK)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
        ax.set_xticks([0, 5, 10, 20]); ax.set_yticks([0, 5, 10, 20])
        ax.grid(color=GRID, lw=0.6)
        ax.set_xlabel("True rate (% cells)")
        err10 = 100 * (same.y - same.x)
        ax.text(0.05, 0.95, f"spill ≤ {100 * other.y.max():.1f} pt", transform=ax.transAxes, ha="left",
                va="top", fontsize=7, color=INK2)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    axes[0].set_ylabel("Estimated rate (%)")
    fig.text(0.01, 0.95, "a", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.07, 0.95, "One error type injected alone into real embryo frame pairs, global baseline",
             fontsize=8, color=INK2, va="center")
    handles_a = [Line2D([], [], color=INK2, lw=2, marker="o", ms=4.5, mec=SURFACE, mew=1.2,
                        label="Estimated rate of the injected type (mean, 95% CI over pairs)"),
                 Line2D([], [], color=INK2, lw=1.4, ls=(0, (3, 2)), label="Total falsely attributed to the other three types")]

    # ---- b: pooled scenarios ---------------------------------------------------------
    axb = fig.add_subplot(gs[0, 5])
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
    axb.set_xlabel("True rate (% cells)"); axb.set_ylabel("Estimated rate (%)")
    axb.set_title(f"Mixed errors,\npooled over {int(pooled.n_pairs.iloc[0])} pairs", loc="left", pad=3, color=INK)
    fig.text(0.70, 0.95, "b", fontsize=11, fontweight="bold", color=INK)
    handles_b = [Line2D([], [], ls="", marker="o", ms=5, color=SERIES[t], mec=SURFACE, label=LABELS[t]) for t in TYPES]
    handles_b += [Line2D([], [], ls="", marker="o", ms=5, color=INK2, mec=SURFACE, label="Recording 01"),
                  Line2D([], [], ls="", marker="s", ms=5, color=INK2, mec=SURFACE, label="Recording 02")]

    fig.legend(handles=handles_a, loc="lower left", bbox_to_anchor=(0.07, 0.10), ncol=1, frameon=False,
               handlelength=2.6, labelspacing=0.2)
    fig.legend(handles=handles_b, loc="lower left", bbox_to_anchor=(0.07, 0.0), ncol=6, frameon=False,
               handletextpad=0.3, columnspacing=1.0)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_error_mix.{ext}", dpi=300 if ext == "png" else None)
    print("wrote", OUT / "fig_error_mix.png")


def pairs_figure(single: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(3.6, 3.9), sharex=True, sharey=True,
                             gridspec_kw=dict(left=0.15, right=0.97, top=0.9, bottom=0.12, wspace=0.15, hspace=0.4))
    lim = 4 * np.ceil((100 * single[[f"true_{t}" for t in TYPES] + [f"glob_{t}" for t in TYPES]].max().max() + 0.5) / 4)
    for ax, t in zip(axes.ravel(), TYPES):
        x, y = 100 * single[f"true_{t}"], 100 * single[f"glob_{t}"]
        ax.plot([0, lim], [0, lim], color=AXIS, lw=0.8)
        ax.scatter(x, y, s=14, color=SERIES[t], edgecolors=SURFACE, linewidths=0.8, zorder=3)
        ax.set_title(LABELS[t], loc="left", pad=3, color=INK)
        ax.text(0.97, 0.05, f"±{np.sqrt(np.mean((y - x) ** 2)):.1f} pt", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.5, color=INK2)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal"); ax.grid(color=GRID, lw=0.6)
        ax.set_xticks(np.arange(0, lim + 0.1, 4)); ax.set_yticks(np.arange(0, lim + 0.1, 4))
    for ax in axes[1]:
        ax.set_xlabel("True rate (% cells)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Estimated (%)")
    fig.suptitle(f"Single frame pairs, random mixtures (n = {len(single)})", x=0.15, ha="left", fontsize=8, color=INK2)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_error_mix_pairs.{ext}", dpi=300 if ext == "png" else None)
    print("wrote", OUT / "fig_error_mix_pairs.png")


def main() -> int:
    OUT.mkdir(exist_ok=True)
    main_figure(pd.read_csv(RESULTS / "error_mix_validation_on_sweep.csv"),
                pd.read_csv(RESULTS / "error_mix_validation_pooled.csv"))
    pairs_figure(pd.read_csv(RESULTS / "error_mix_validation.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
