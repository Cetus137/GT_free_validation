"""Every figure panel as its own file, shared style, minimal text.

    python scripts/make_panels.py            # writes figures/panels/*.png and *.pdf

Main
  response_curves      Δ overlap vs error rate, 4 types, embryo over slab simulations
  fingerprint          decomposition change at 10% error, 4 types x 4 channels
  inversion_rate       estimated vs true rate per type (single-type sweeps, global baseline)
  inversion_pooled     fixed mixes pooled over 30 pairs, all types
  blindspot            real linkers: true wrong-link rate vs what the metrics report, by frame interval
Supplementary
  division_handling    clean baseline under raw / inherit / inherit+exclude, division-free vs division pairs
  switch_geometry      switch slope: embryo vs slab vs cube, per stage
  response_curves_all  6 types (slab simulations)
  fingerprint_all      6 types x 5 channels
  lowrate_outliers     0.25–2% with outlier channels
  lowrate_knn          0.25–2% with kNN neighbourhoods
  cross_embryo         estimator error when calibrated on one recording and tested on the other
Panels whose result file is missing are skipped.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

R = ROOT / "data" / "_results"
OUT = ROOT / "figures" / "panels"
fs.apply()


def _load_sweep(shape="slab"):
    f = R / f"ce_error_sweep_{shape}_delaunay.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f); df = df[df.error != "none"].copy()
    df["src"] = np.where(df.source.str.startswith("CE"), "real", "sim")
    return df


def _curve(df, kind, src):
    g = df[(df.error == kind) & (df.src == src)]
    if src == "real":
        pp = g.groupby(["source", "pair", "rate"]).agg(x=("rate_realised", "mean"), y=("d_wio", "mean")).reset_index()
        a = pp.groupby("rate").agg(x=("x", "mean"), y=("y", "mean"), sd=("y", "std"), n=("y", "size"))
        a["lo"] = a.y - 1.96 * a.sd / np.sqrt(a.n); a["hi"] = a.y + 1.96 * a.sd / np.sqrt(a.n)
        return a
    ps = g.groupby(["source", "rate"]).agg(x=("rate_realised", "mean"), y=("d_wio", "mean")).reset_index()
    return ps.groupby("rate").agg(x=("x", "mean"), y=("y", "mean"), lo=("y", "min"), hi=("y", "max"))


# --------------------------------------------------------------------------- response curves

def response_curves(order, stem):
    df = _load_sweep("slab")
    if df is None:
        return
    n = len(order); ncol = 4 if n == 4 else 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(fs.DOUBLE, 1.9 * nrow + 0.5), sharey=True,
                             gridspec_kw=dict(left=0.07, right=0.99, top=0.9 if nrow == 1 else 0.93,
                                              bottom=0.3 if nrow == 1 else 0.18, wspace=0.15, hspace=0.55))
    axes = np.atleast_1d(axes).ravel()
    ymin = 0
    for ax, kind in zip(axes, order):
        col = fs.SERIES[kind]; real = _curve(df, kind, "real"); sim = _curve(df, kind, "sim")
        xs, ys = np.r_[0, sim.x * 100], np.r_[0, sim.y * 100]
        ax.fill_between(xs, np.r_[0, sim.lo * 100], np.r_[0, sim.hi * 100], color=col, alpha=0.12, lw=0)
        ax.plot(xs, ys, color=col, **fs.DASH)
        xr, yr = np.r_[0, real.x * 100], np.r_[0, real.y * 100]
        ax.fill_between(xr, np.r_[0, real.lo * 100], np.r_[0, real.hi * 100], color=col, alpha=0.18, lw=0)
        ax.plot(xr, yr, color=col, **fs.LINE, **fs.MARK)
        slope = -np.sum(real.x * real.y) / np.sum(real.x ** 2)
        ax.set_title(fs.LABELS[kind], loc="left", pad=3)
        ax.text(0.97, 0.95, f"{slope:.2f} pt / %", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=fs.INK2)
        fs.grid(ax); ax.set_xlim(0, 21); ax.set_xticks([0, 5, 10, 20]); ax.set_xlabel("Cells affected (%)")
        ymin = min(ymin, float(np.min(np.r_[yr, ys, np.r_[0, sim.lo * 100]])))
    for ax in axes[n:]:
        ax.axis("off")
    lo = np.floor(ymin / 5) * 5 - 1
    for ax in axes[:n]:
        ax.set_ylim(lo, 1); ax.set_yticks(np.arange(0, lo, -5))
    for i in range(0, n, ncol):
        axes[i].set_ylabel("Δ overlap (×100)")
    fig.legend(handles=[fs.real_handle("Embryo"), fs.sim_handle("Matched simulation")], loc="lower left",
               bbox_to_anchor=(0.07, 0.0), ncol=2, handlelength=2.6)
    fs.save(fig, OUT / stem)


# --------------------------------------------------------------------------- fingerprint

def fingerprint(order, comps, stem):
    df = _load_sweep("slab")
    if df is None:
        return
    real = df[(df.src == "real") & (df.rate == 0.10)]
    M = np.array([[real[real.error == k][c].mean() * 100 for c, _ in comps] for k in order])
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
    fs.save(fig, OUT / stem)


# --------------------------------------------------------------------------- inversion

def _ci(g, col):
    pp = g.groupby(["source", "pair", "rate"]).agg(x=("rate_realised", "mean"), y=(col, "mean")).reset_index()
    a = pp.groupby("rate").agg(x=("x", "mean"), y=("y", "mean"), sd=("y", "std"), n=("y", "size"))
    a["ci"] = 1.96 * a.sd / np.sqrt(a.n)
    return a


def inversion_rate(file, stem, lim=22, ticks=(0, 5, 10, 20)):
    f = R / file
    if not f.exists():
        return
    on = pd.read_csv(f)
    fig, axes = plt.subplots(1, 4, figsize=(fs.DOUBLE, 2.35), sharey=True,
                             gridspec_kw=dict(left=0.07, right=0.99, top=0.9, bottom=0.34, wspace=0.15))
    for ax, t in zip(axes, fs.ORDER4):
        g = on[on.error == t]; same = _ci(g, "est_same"); other = _ci(g, "est_other")
        x = np.r_[0, 100 * same.x]
        fs.identity(ax, lim)
        ax.fill_between(x, np.r_[0, 100 * (same.y - same.ci)], np.r_[0, 100 * (same.y + same.ci)], color=fs.SERIES[t], alpha=0.18, lw=0)
        ax.plot(x, np.r_[0, 100 * same.y], color=fs.SERIES[t], **fs.LINE, **fs.MARK)
        ax.plot(np.r_[0, 100 * other.x], np.r_[0, 100 * other.y], color=fs.INK2, lw=1.4, ls=(0, (3, 2)))
        ax.set_title(fs.LABELS[t], loc="left", pad=3)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
        ax.set_xticks(list(ticks)); ax.set_yticks(list(ticks)); fs.grid(ax, "both")
        ax.set_xlabel("True rate (%)")
    axes[0].set_ylabel("Estimated rate (%)")
    fig.legend(handles=[fs.real_handle("Injected type"), plt.Line2D([], [], color=fs.INK2, lw=1.4, ls=(0, (3, 2)), label="Attributed to other types")],
               loc="lower left", bbox_to_anchor=(0.07, 0.0), ncol=2, handlelength=2.6)
    fs.save(fig, OUT / stem)


def inversion_pooled(file="error_mix_validation_pooled.csv", stem="inversion_pooled"):
    f = R / file
    if not f.exists():
        return
    p = pd.read_csv(f)
    lim = 100 * p[[f"true_{t}" for t in fs.ORDER4] + [f"est_{t}" for t in fs.ORDER4]].max().max() + 0.5
    fig, ax = plt.subplots(figsize=(fs.SINGLE, fs.SINGLE + 0.4), gridspec_kw=dict(left=0.17, right=0.97, top=0.97, bottom=0.3))
    fs.identity(ax, lim)
    for t in fs.ORDER4:
        for seq, marker in (("01", "o"), ("02", "s")):
            g = p[p.seq.astype(str).str.zfill(2) == seq]
            ax.scatter(100 * g[f"true_{t}"], 100 * g[f"est_{t}"], marker=marker, color=fs.SERIES[t], **{**fs.SCAT, "s": 26})
    e = np.concatenate([100 * (p[f"est_{t}"] - p[f"true_{t}"]).values for t in fs.ORDER4])
    ax.text(0.97, 0.04, f"RMSE {np.sqrt(np.mean(e ** 2)):.2f} pt", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=fs.INK2)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal"); fs.grid(ax, "both")
    ax.set_xlabel("True rate (%)"); ax.set_ylabel("Estimated rate (%)")
    h = [fs.dot_handle(fs.LABELS[t], fs.SERIES[t]) for t in fs.ORDER4] + [fs.dot_handle("Recording 01", fs.INK2), fs.dot_handle("Recording 02", fs.INK2, "s")]
    fig.legend(handles=h, loc="lower left", bbox_to_anchor=(0.05, 0.0), ncol=2, handletextpad=0.3, columnspacing=1.0)
    fs.save(fig, OUT / stem)


# --------------------------------------------------------------------------- blind spot

def blindspot(stem="blindspot"):
    files = [R / "linker_test_clean.csv", R / "linker_test_stride.csv"]
    if not all(f.exists() for f in files):
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "stride" not in df:
        return
    df["stride"] = df["stride"].fillna(1).astype(int)
    df = df[(df.cap_spacings == 1.0) & (df.dropout == 0)]
    df["true_total"] = df.true_switch + df.true_frag
    df["est_total"] = df.est_switch + df.est_frag + df.est_miss + df.est_fp
    df["interval_min"] = 1.5 * df.stride
    fig, ax = plt.subplots(figsize=(fs.SINGLE, 2.6), gridspec_kw=dict(left=0.16, right=0.97, top=0.96, bottom=0.36))
    for method, col in (("greedy", fs.CUBE), ("lap", fs.SLAB)):
        g = df[df.method == method].groupby("interval_min")[["true_total", "est_total"]].mean()
        ax.plot(g.index, g.true_total, color=col, **fs.LINE, **fs.MARK)
        ax.plot(g.index, g.est_total, color=col, **fs.DASH)
    fs.grid(ax); ax.set_xticks(sorted(df.interval_min.unique()))
    ax.set_xlabel("Frame interval (min)"); ax.set_ylabel("Cells with a tracking error (%)")
    ax.set_ylim(0, None)
    h = [fs.real_handle("True, from link labels"), fs.sim_handle("Reported by the metrics"),
         fs.dot_handle("Nearest-neighbour linker", fs.CUBE), fs.dot_handle("Linear-assignment linker", fs.SLAB)]
    fig.legend(handles=h, loc="lower left", bbox_to_anchor=(0.05, 0.0), ncol=2, handlelength=2.4, columnspacing=1.0)
    fs.save(fig, OUT / stem)


# --------------------------------------------------------------------------- supplementary

def division_handling(stem="division_handling"):
    f = R / "ce_division_handling.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    variants = ["raw", "inherit", "inherit + exclude"]
    names = {"raw": "Divisions ignored", "inherit": "Daughter inherits", "inherit + exclude": "Inherit + exclude"}
    fig, axes = plt.subplots(1, 2, figsize=(fs.DOUBLE * 0.75, 2.5), gridspec_kw=dict(left=0.13, right=0.99, top=0.9, bottom=0.4, wspace=0.35))
    for ax, (col, label) in zip(axes, (("wio", "Weighted identity overlap"), ("newcomer", "Newcomer share"))):
        for i, v in enumerate(variants):
            for j, (dp, mk, c) in enumerate(((False, "o", fs.INK2), (True, "s", fs.SERIES["frag"]))):
                g = df[(df.variant == v) & (df.division_pair == dp)][col]
                x = i + (j - 0.5) * 0.25
                ax.errorbar(x, g.mean(), yerr=g.std(), fmt=mk, color=c, ms=5, mec=fs.SURFACE, mew=1, capsize=0, elinewidth=1.2)
        ax.set_xticks(range(3)); ax.set_xticklabels([names[v] for v in variants], rotation=20, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.6, 2.6)
        ax.set_title(label, loc="left", pad=3); fs.grid(ax)
    fig.legend(handles=[fs.dot_handle("Frame pairs without a division", fs.INK2), fs.dot_handle("Frame pairs with a division", fs.SERIES["frag"], "s")],
               loc="lower left", bbox_to_anchor=(0.09, 0.0), ncol=2)
    fs.save(fig, OUT / stem)


def switch_geometry(stem="switch_geometry"):
    cube_f, slab_f = R / "ce_error_sweep_t1only_delaunay.csv", R / "ce_slab_switch.csv"
    if not (cube_f.exists() and slab_f.exists()):
        return
    cube = pd.read_csv(cube_f); slab = pd.read_csv(slab_f)
    stages = [("50-100", "50–100 cells"), ("100-200", "100–200 cells"), ("200-360", "200–360 cells")]

    def realc(stage):
        g = slab[(slab.error == "switch") & slab.source.str.startswith("CE") & (slab.stage == stage)]
        return _ci(g, "d_wio")

    def simc(df, src):
        g = df[(df.error == "switch") & (df.source == src)]
        ps = g.groupby(["pair", "rate"]).agg(x=("rate_realised", "mean"), y=("d_wio", "mean")).reset_index()
        return ps.groupby("rate").agg(x=("x", "mean"), y=("y", "mean"), lo=("y", "min"), hi=("y", "max"))

    fig, axes = plt.subplots(1, 3, figsize=(fs.DOUBLE * 0.85, 2.3), sharey=True,
                             gridspec_kw=dict(left=0.09, right=0.99, top=0.88, bottom=0.34, wspace=0.15))
    for ax, (stage, title) in zip(axes, stages):
        real = realc(stage); cu = simc(cube, f"SIM-{stage}"); sl = simc(slab, f"SIM-slab-{stage}")
        for c, col in ((cu, fs.CUBE), (sl, fs.SLAB)):
            x = np.r_[0, c.x * 100]
            ax.fill_between(x, np.r_[0, c.lo * 100], np.r_[0, c.hi * 100], color=col, alpha=0.12, lw=0)
            ax.plot(x, np.r_[0, c.y * 100], color=col, **fs.DASH)
        xr, yr = np.r_[0, real.x * 100], np.r_[0, real.y * 100]
        ax.fill_between(xr, yr - np.r_[0, real.ci * 100], yr + np.r_[0, real.ci * 100], color=fs.INK2, alpha=0.15, lw=0)
        ax.plot(xr, yr, color=fs.INK2, **fs.LINE, **fs.MARK)
        ax.set_title(title, loc="left", pad=3); fs.grid(ax)
        ax.set_xlim(0, 21); ax.set_xticks([0, 5, 10, 20]); ax.set_xlabel("Cells switched (%)")
    axes[0].set_ylim(-31, 1); axes[0].set_yticks([0, -10, -20, -30]); axes[0].set_ylabel("Δ overlap (×100)")
    fig.legend(handles=[fs.real_handle("Embryo"), fs.sim_handle("Slab simulation", fs.SLAB), fs.sim_handle("Cube simulation", fs.CUBE)],
               loc="lower left", bbox_to_anchor=(0.09, 0.0), ncol=3, handlelength=2.6)
    fs.save(fig, OUT / stem)


def cross_embryo(stem="cross_embryo"):
    f = R / "error_mix_cross.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    settings = ["01 → 02", "02 → 01", "pooled"]
    fig, ax = plt.subplots(figsize=(fs.SINGLE, 2.3), gridspec_kw=dict(left=0.16, right=0.98, top=0.95, bottom=0.36))
    w = 0.25
    for i, t in enumerate(fs.ORDER4):
        for j, s in enumerate(settings):
            v = df[(df.error == t) & (df.setting == s)].pooled_rmse.values
            if len(v):
                ax.bar(i + (j - 1) * w, v[0], width=w - 0.03, color=fs.SERIES[t], alpha=(0.55, 0.8, 1.0)[j], lw=0)
    ax.set_xticks(range(4)); ax.set_xticklabels([fs.LABELS[t] for t in fs.ORDER4], rotation=20, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Error, pooled over movie (pt)"); fs.grid(ax)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=fs.INK2, alpha=a, label=l) for a, l in ((0.55, "Calibrate 01, test 02"), (0.8, "Calibrate 02, test 01"), (1.0, "Pooled"))],
               loc="lower left", bbox_to_anchor=(0.05, 0.0), ncol=3, handlelength=1.2, columnspacing=1.0)
    fs.save(fig, OUT / stem)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    comps4 = [("d_vanished", "Vanished"), ("d_displaced", "Displaced"), ("d_newcomer", "Newcomer"), ("d_rc", "Rank corr.")]
    comps5 = comps4[:3] + [("d_pwio", "Positional")] + comps4[3:]
    response_curves(fs.ORDER4, "response_curves")
    fingerprint(fs.ORDER4, comps4, "fingerprint")
    inversion_rate("error_mix_validation_on_sweep.csv", "inversion_rate")
    inversion_pooled()
    blindspot()
    division_handling()
    switch_geometry()
    response_curves(fs.ORDER6, "response_curves_all")
    fingerprint(fs.ORDER6, comps5, "fingerprint_all")
    inversion_rate("error_mix_low_out_on_sweep.csv", "lowrate_outliers", lim=2.3, ticks=(0, 0.5, 1, 1.5, 2))
    inversion_rate("error_mix_low_knn_on_sweep.csv", "lowrate_knn", lim=2.3, ticks=(0, 0.5, 1, 1.5, 2))
    cross_embryo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
