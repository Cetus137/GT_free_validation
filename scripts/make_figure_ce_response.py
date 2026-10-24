"""Figure: neighbourhood-continuity response to injected tracking errors on
Fluo-N3DH-CE ground truth versus stage-matched simulations.

    python scripts/make_figure_ce_response.py

Reads data/_results/ce_error_sweep_t1only_delaunay.csv (+ the fragmentation file)
and writes figures/fig_ce_response.{png,pdf}.

Panel a: small multiples, one per error type — change in weighted identity
overlap (WIO, x100) against the realised fraction of cells affected. Solid line
with markers = real embryo frames (mean over frame pairs, 95% CI band); dashed
line = simulations (mean over the three stage-matched tissues, min–max wash).
Panel b: fingerprint heat map at a nominal 10% error rate on the real embryo —
change in each decomposition component.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "_results"
OUT = ROOT / "figures"

# --- palette (validated default; see dataviz skill references/palette.md) ----------
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {  # fixed categorical order -> error type (identity, never cycled)
    "switch": "#2a78d6", "frag": "#eb6834", "miss": "#1baf7a",
    "split": "#eda100", "merge": "#e87ba4", "fp": "#008300",
}
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
LABELS = {
    "switch": "Identity switch", "frag": "Track fragmentation", "miss": "Missed detection",
    "split": "Split", "merge": "Merge", "fp": "False positive",
}
ORDER_ALL = ["switch", "frag", "miss", "split", "merge", "fp"]
ORDER_NUCLEAR = ["switch", "frag", "miss", "fp"]        # errors that matter for a nuclear label
COMPONENTS_ALL = [("d_vanished", "Vanished"), ("d_displaced", "Displaced"), ("d_newcomer", "Newcomer"),
                  ("d_pwio", "Positional"), ("d_rc", "Rank corr.")]
COMPONENTS_MAIN = [c for c in COMPONENTS_ALL if c[0] != "d_pwio"]

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK2, "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": SURFACE, "pdf.fonttype": 42,
})


def load(sim_shape: str) -> pd.DataFrame:
    """Slab: one file holds real + slab simulations for all six types.
    Cube: the original cube run plus the fragmentation-only cube run."""
    if sim_shape == "slab":
        files = [RESULTS / "ce_error_sweep_slab_delaunay.csv"]
    else:
        files = [RESULTS / "ce_error_sweep_t1only_delaunay.csv", RESULTS / "ce_error_sweep_frag_delaunay.csv"]
    files = [f for f in files if f.exists()]
    if not files:
        sys.exit(f"no sweep results for {sim_shape} in {RESULTS}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df.error != "none"].copy()
    df["src"] = np.where(df.source.str.startswith("CE"), "real", "sim")
    return df


def curve(df: pd.DataFrame, kind: str, src: str):
    g = df[(df.error == kind) & (df.src == src)]
    if src == "real":
        # per (pair, rate) mean over seeds, then mean and 95% CI over pairs
        per_pair = g.groupby(["source", "pair", "rate"]).agg(x=("rate_realised", "mean"), y=("d_wio", "mean")).reset_index()
        agg = per_pair.groupby("rate").agg(x=("x", "mean"), y=("y", "mean"), sd=("y", "std"), n=("y", "size"))
        agg["ci"] = 1.96 * agg.sd / np.sqrt(agg.n)
        return agg
    per_stage = g.groupby(["source", "rate"]).agg(x=("rate_realised", "mean"), y=("d_wio", "mean")).reset_index()
    agg = per_stage.groupby("rate").agg(x=("x", "mean"), y=("y", "mean"), lo=("y", "min"), hi=("y", "max"))
    return agg


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--all-types", action="store_true", help="six error types incl. split and merge")
    p.add_argument("--positional", action="store_true", help="include the positional-overlap column")
    p.add_argument("--sim-shape", choices=("slab", "cube"), default="slab",
                   help="which simulations to overlay (default slab = embryo thickness)")
    a = p.parse_args(argv)
    ORDER = ORDER_ALL if a.all_types else ORDER_NUCLEAR
    COMPONENTS = COMPONENTS_ALL if a.positional else COMPONENTS_MAIN
    ncol = 3 if a.all_types else 2
    stem = ("fig_ce_response_all" if a.all_types else "fig_ce_response") + ("" if a.sim_shape == "slab" else "_cube")

    df = load(a.sim_shape)
    OUT.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(7.1 if a.all_types else 6.3, 4.6))
    widths = [1] * ncol + [0.25, 1.55 if a.all_types else 1.35]
    gs = fig.add_gridspec(2, ncol + 2, width_ratios=widths, wspace=0.45, hspace=0.6,
                          left=0.08 if a.all_types else 0.09, right=0.985, top=0.93, bottom=0.19)

    # ---- panel a: small multiples -------------------------------------------------
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(ncol)]
    ymin = 0.0
    for ax, kind in zip(axes, ORDER):
        col = SERIES[kind]
        real = curve(df, kind, "real")
        sim = curve(df, kind, "sim")
        xr, yr = np.r_[0, real.x * 100], np.r_[0, real.y * 100]
        xs, ys = np.r_[0, sim.x * 100], np.r_[0, sim.y * 100]
        ax.fill_between(xs, np.r_[0, sim.lo * 100], np.r_[0, sim.hi * 100], color=col, alpha=0.12, lw=0)
        ax.plot(xs, ys, ls=(0, (3, 2)), lw=1.6, color=col, solid_capstyle="round")
        ax.fill_between(xr, yr - np.r_[0, real.ci * 100], yr + np.r_[0, real.ci * 100], color=col, alpha=0.18, lw=0)
        ax.plot(xr, yr, lw=2, color=col, marker="o", ms=4.5, mec=SURFACE, mew=1.2, solid_capstyle="round")
        slope = -np.sum(real.x * real.y) / np.sum(real.x ** 2)      # WIO per unit rate, through origin
        ax.set_title(LABELS[kind], loc="left", color=INK, pad=4)
        ax.text(0.98, 0.95, f"{slope:.2f} pt / %", transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color=INK2)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_xlim(0, 21); ax.set_xticks([0, 5, 10, 20])
        ymin = min(ymin, float(np.min(np.r_[yr, ys, np.r_[0, sim.lo * 100]])))
    for i, ax in enumerate(axes):
        ax.set_ylim(np.floor(ymin / 5) * 5 - 1, 1)
        ax.set_yticks(np.arange(0, np.floor(ymin / 5) * 5 - 1, -5))
        if i % ncol:
            ax.tick_params(labelleft=False)
        if i >= ncol:
            ax.set_xlabel("Cells affected (%)")
    axes[0].set_ylabel("Δ weighted identity overlap (×100)")
    axes[ncol].set_ylabel("Δ weighted identity overlap (×100)")
    # legend for the two line styles (identity of the panels is in the titles)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=INK2, lw=2, marker="o", ms=4.5, mec=SURFACE, mew=1.2, label="Real embryo (mean, 95% CI)"),
               Line2D([], [], color=INK2, lw=1.6, ls=(0, (3, 2)),
                      label=f"Stage-matched {'slab' if a.sim_shape == 'slab' else 'cube'} simulations (mean, range)")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.08 if a.all_types else 0.09, 0.0),
               ncol=2, frameon=False, handlelength=2.6, columnspacing=1.5)
    fig.text(0.01, 0.975, "a", fontsize=11, fontweight="bold", color=INK)

    # ---- panel b: fingerprint heat map ---------------------------------------------
    axb = fig.add_subplot(gs[:, ncol + 1])
    real10 = df[(df.src == "real") & (df.rate == 0.10)]
    M = np.array([[real10[real10.error == k][c].mean() * 100 for c, _ in COMPONENTS] for k in ORDER])
    vmax = np.nanmax(np.abs(M))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    axb.imshow(M, cmap=DIVERGING, norm=norm, aspect="auto")
    # 2px surface gap between cells
    axb.set_xticks(np.arange(-0.5, M.shape[1], 1), minor=True); axb.set_yticks(np.arange(-0.5, M.shape[0], 1), minor=True)
    axb.grid(which="minor", color=SURFACE, lw=2); axb.tick_params(which="minor", length=0)
    axb.set_xticks(range(len(COMPONENTS)))
    axb.set_xticklabels([n for _, n in COMPONENTS], color=INK2, rotation=35, ha="right", rotation_mode="anchor")
    axb.set_yticks(range(len(ORDER))); axb.set_yticklabels([LABELS[k] for k in ORDER], color=INK2)
    axb.tick_params(length=0)
    for s in axb.spines.values():
        s.set_visible(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            rgba = DIVERGING(norm(v)); lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            axb.text(j, i, f"{v:+.0f}" if abs(v) >= 0.5 else "0", ha="center", va="center", fontsize=7.5,
                     color=("#ffffff" if lum < 0.55 else INK))
    axb.set_title("Decomposition at 10% error\n(real embryo, ×100)", loc="left", color=INK, pad=6)
    bx = axb.get_position().x0 - 0.075
    fig.text(bx, 0.975, "b", fontsize=11, fontweight="bold", color=INK)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300 if ext == "png" else None)
    print("wrote", OUT / f"{stem}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
