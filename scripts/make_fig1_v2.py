"""Figure 1, new version (simulation-driven): simulation-free error unmixing.  One file per panel in
figures/fig1_v2/ (the first version, scripts/make_fig1.py, is left as it is).

    python scripts/make_fig1_v2.py [--workers N] [--stage all|figures]

  a_counts     the four counts; each is zero when the tracks are correct
  b_errors     the four tracking errors (the Figure 1 cartoons)
  c_response   each count against the injected rate of each error type
  d_addition   the unmixing: the response matrix measured by adding known errors to the movie itself
  e_recovery   one error type at a time: the rate recovered for it, and what spills into the others
  f_mixes      all four types at once (the fixed mixes of estimate_error_mix.SCENARIOS)
  g_motion     unmixing error across motion types and speeds: Figure 1 unmixing against the new one

Tissue: the persistent-motion movies of scripts/motion_testbed.py at 0.16 spacings per frame (inside
the worm range), two movies of 10 frame pairs, 1000 cells.  Writes data/_results/fig1_v2_counts.csv,
fig1_v2_recovery.csv, fig1_v2_response.csv; g_motion reads motion_testbed_unmixing.csv.
"""
from __future__ import annotations

import argparse
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import make_fig1 as f1  # noqa: E402  (Figure 1 style and the error cartoons)
import motion_testbed as mt  # noqa: E402
import estimate_error_mix as em  # noqa: E402
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from tracking_validation import apply_tracking_errors, realised_error_rates  # noqa: E402
from tracking_validation import unmixing as um  # noqa: E402

RES = ROOT / "data" / "_results"
OUT = ROOT / "figures" / "fig1_v2"
MOTION, SPEED, SEEDS = "persistent", 1.0, (0, 1)
RATES = (0.005, 0.01, 0.02, 0.03, 0.04, 0.05)
ORDER = ["miss", "switch", "frag", "fp"]            # Figure 1 column order
COUNT_LABELS = {"ended": "Labels ending", "begun": "Labels beginning", "relabelled": "Relabelled", "flagged": "Two-way flags"}
MARKERS = {"miss": "o", "switch": "s", "frag": "^", "fp": "D"}   # shape as well as colour (fragmentation and FP
                                                                  # are close in colour for red-blind readers)


# --------------------------------------------------------------------------- data

def _setup(seed):
    gt = mt.movie(MOTION, SPEED, seed).as_frames()
    pairs = [gt[k:k + 2] for k in range(mt.FRAMES - 1)]
    spacing = float(np.median([mt.sweep.median_nn(p[0]["positions"]) for p in pairs]))
    return pairs, spacing


def job(args):
    """One movie, one injection (a single type at one rate, or a fixed mix): per-pair counts, and the
    unmixing of the pooled pairs with its measured response matrix."""
    seed, kind, name, rates = args
    pairs, spacing = _setup(seed)
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])      # stable across runs, unlike hash()
    box = (mt.BOX,) * 3
    counts, obs, real = [], [], []
    for k, p in enumerate(pairs):
        o, log = apply_tracking_errors(p, um.error_config(rates, spacing), box, rng)
        c, n = um.counts(o, mt.CAP)
        r = realised_error_rates(p, o, log, first_frame_clean=True)
        counts.append(dict(seed=seed, kind=kind, name=name, pair=k, **{f"true_{t}": r[t] for t in um.TYPES},
                           **{c_: v / n for c_, v in zip(um.COUNTS, c)}))
        obs.append(o); real.append(r)
    est = um.unmix(obs, box, spacing, mt.CAP, rng)
    A = est.pop("A")
    rec = dict(seed=seed, kind=kind, name=name, **{f"true_{t}": np.mean([r[t] for r in real]) for t in um.TYPES},
               **{f"est_{t}": est[t] for t in um.TYPES})
    resp = [dict(seed=seed, kind=kind, name=name, count=c_, error=t, rise=A[i, j])
            for i, c_ in enumerate(um.COUNTS) for j, t in enumerate(um.TYPES)]
    print(f"seed {seed} {kind:6s} {name}", flush=True)
    return counts, rec, resp


def run(workers: int) -> None:
    jobs =[(s, "single", f"{t}@{r}", {t: r}) for s in SEEDS for t in ORDER for r in RATES]
    jobs += [(s, "mix", name, rates) for s in SEEDS for name, rates in em.SCENARIOS.items()]
    with ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(job, jobs))
    pd.DataFrame([r for c, _, _ in out for r in c]).to_csv(RES / "fig1_v2_counts.csv", index=False)
    pd.DataFrame([rec for _, rec, _ in out]).to_csv(RES / "fig1_v2_recovery.csv", index=False)
    pd.DataFrame([r for _, _, resp in out for r in resp]).to_csv(RES / "fig1_v2_response.csv", index=False)


# --------------------------------------------------------------------------- a: the four counts

def a_counts():
    pal = {"1": "#2a78d6", "2": "#eb6834", "3": "#1baf7a", "4": "#eda100", "5": fs.SERIES["fp"]}
    base = np.array([[0.0, 0.0], [1.0, 0.3], [0.5, -0.9]])
    moved = base + np.array([0.08, 0.05])

    def ring(ax, p):
        ax.scatter(*p, s=420, facecolors="none", edgecolors=fs.INK2, linewidths=0.9, linestyle=(0, (2, 1.5)), zorder=2)

    fig, axes = plt.subplots(2, 4, figsize=(fs.DOUBLE, 3.0), gridspec_kw=dict(left=0.03, right=0.99, top=0.84, bottom=0.17, wspace=0.08, hspace=0.1))
    titles = ("Label ends", "Label begins", "Relabelled", "Two-way flag")
    captions = ("Label 3 is gone at t + 1",
                "Label 5 was not there at t",
                "New label 4 sits where\nended label 3 was predicted",
                "Label 2 predicted forward and\nbackward: both hold label 3")
    for col in range(4):
        ax0, ax1 = axes[0, col], axes[1, col]
        for ax in (ax0, ax1):
            ax.set_aspect("equal"); ax.set_xlim(-0.6, 1.9); ax.set_ylim(-1.5, 1.0); ax.axis("off")
        ax0.set_title(titles[col], loc="left", pad=8)
        ax0.text(-0.55, -1.35, "t", fontsize=7.5, color=fs.MUTED); ax1.text(-0.55, -1.35, "t + 1", fontsize=7.5, color=fs.MUTED)
        ax1.text(-0.55, -1.95, captions[col], fontsize=6.5, color=fs.INK2, va="top")
        f1._cells(ax0, base, ["1", "2", "3"], [pal["1"], pal["2"], pal["3"]])
        if col == 0:
            f1._cells(ax1, moved[:2], ["1", "2"], [pal["1"], pal["2"]])
            ring(ax1, moved[2])
        elif col == 1:
            f1._cells(ax1, list(moved) + [np.array([1.5, -0.4])], ["1", "2", "3", "5"], [pal["1"], pal["2"], pal["3"], pal["5"]])
        elif col == 2:
            f1._cells(ax1, moved, ["1", "2", "4"], [pal["1"], pal["2"], pal["4"]])
            ring(ax1, moved[2])
        else:
            f1._cells(ax1, moved, ["1", "3", "2"], [pal["1"], pal["3"], pal["2"]])
            ring(ax1, moved[1]); ring(ax0, base[2])       # label 2 predicted forward / backward: both hold label 3
    fig.text(0.03, 0.97, "Dashed ring: where the label is predicted to be (its own position plus its neighbours' median step).  "
             "Each count is zero when the tracks are correct.", fontsize=6.5, color=fs.MUTED, va="top")
    fs.save(fig, OUT / "a_counts")


# --------------------------------------------------------------------------- c: response curves

def c_response():
    d = pd.read_csv(RES / "fig1_v2_counts.csv")
    d = d[d.kind == "single"].copy()
    d["error"] = d.name.str.split("@").str[0]; d["rate"] = d.name.str.split("@").str[1].astype(float)
    fig, axes = plt.subplots(4, 4, figsize=(fs.DOUBLE, 5.6), sharex=True, sharey=True,
                             gridspec_kw=dict(left=0.09, right=0.99, top=0.95, bottom=0.09, wspace=0.12, hspace=0.18))
    hi = 0.0
    for r, cnt in enumerate(um.COUNTS):
        for c, t in enumerate(ORDER):
            ax = axes[r, c]
            g = d[d.error == t]
            a = g.groupby("rate").agg(x=(f"true_{t}", "mean"), y=(cnt, "mean"), sd=(cnt, "std"), n=(cnt, "size"))
            ci = 1.96 * a.sd / np.sqrt(a.n)
            x, y = np.r_[0, 100 * a.x], np.r_[0, 100 * a.y]
            ax.errorbar(x[1:], y[1:], yerr=100 * ci.values, fmt="none", ecolor=fs.SERIES[t], elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=3)
            ax.plot(x[1:], y[1:], color=fs.SERIES[t], ls="", marker="x", ms=5.5, mew=1.4, zorder=4)
            ax.plot(x, y, color=fs.SERIES[t], lw=1.2, solid_capstyle="round", zorder=2)
            ax.plot([0, 5.3], [0, 5.3], color=fs.GRID, lw=0.8, zorder=1)          # one count per error
            hi = max(hi, float(np.max(y[1:] + 100 * ci.values)))
            ax.set_xlim(0, 5.3); ax.set_xticks([0, 1, 2, 3, 4, 5])
        axes[r, 0].set_ylabel(f"{COUNT_LABELS[cnt]} (%)")
    axes[0, 0].set_ylim(-0.2, max(hi, 5.3) * 1.04)
    for c, t in enumerate(ORDER):
        axes[0, c].set_title(fs.LABELS[t], loc="left", pad=3)
        axes[-1, c].set_xlabel("Cells affected (%)")
    fig.align_ylabels(axes[:, 0])
    fs.save(fig, OUT / "c_response")


# --------------------------------------------------------------------------- d: the unmixing step

def d_addition():
    resp = pd.read_csv(RES / "fig1_v2_response.csv")
    resp = resp[(resp.kind == "mix") & (resp.name != "clean")]
    A = resp.groupby(["count", "error"]).rise.mean().unstack().loc[list(um.COUNTS), ORDER]
    cmap = LinearSegmentedColormap.from_list("ink", [fs.SURFACE, fs.INK2])
    fig = plt.figure(figsize=(fs.SINGLE, 2.7))
    ax = fig.add_axes([0.36, 0.34, 0.6, 0.46])
    ax.imshow(A.values, cmap=cmap, vmin=0, vmax=1.2, aspect="auto")
    for i in range(4):
        for j in range(4):
            v = A.values[i, j]
            ax.text(j, i, f"{v:.2f}" if abs(v) >= 0.005 else "0", ha="center", va="center", fontsize=7,
                    color="#ffffff" if v > 0.6 else fs.INK)
    ax.set_xticks(range(4)); ax.set_xticklabels(["Missed", "Switch", "Fragm.", "False pos."], fontsize=7)
    ax.set_yticks(range(4)); ax.set_yticklabels([COUNT_LABELS[c] for c in um.COUNTS], fontsize=7)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Rise of each count per added error", loc="left", pad=22, fontsize=8)
    steps = ("1  Count the four signals in the movie.",
             "2  Add a known number of each error type to the same movie\n    and measure how much each count rises: the matrix above.",
             "3  Solve  counts = matrix × rates  for rates ≥ 0.")
    fig.text(0.04, 0.25, "\n".join(steps), fontsize=7, color=fs.INK, va="top", linespacing=1.5)
    fs.save(fig, OUT / "d_addition")


# --------------------------------------------------------------------------- e, f: recovery

def e_recovery(rec: pd.DataFrame | None = None, path: Path | None = None, note: str | None = None):
    """``rec``: single-type rows (name "type@rate", true_*, est_*); default the Figure 1 runs."""
    if rec is None:
        rec = pd.read_csv(RES / "fig1_v2_recovery.csv")
    rec = rec[rec.kind == "single"].copy()
    rec["error"] = rec.name.str.split("@").str[0]
    fig, axes = plt.subplots(1, 4, figsize=(fs.DOUBLE, 2.25), gridspec_kw=dict(left=0.07, right=0.99, top=0.86, bottom=0.3, wspace=0.3))
    if note:
        fig.text(0.07, 0.99, note, fontsize=6.5, color=fs.MUTED, va="top")
    for ax, t in zip(axes, ORDER):
        g = rec[rec.error == t].copy()
        others = [o for o in um.TYPES if o != t]
        g["x"] = 100 * g[f"true_{t}"]; g["y"] = 100 * g[f"est_{t}"]; g["spill"] = 100 * g[[f"est_{o}" for o in others]].sum(axis=1)
        m = g.groupby("name")[["x", "y", "spill"]].mean().sort_values("x")
        ax.plot([0, 5.5], [0, 5.5], color=fs.AXIS, lw=0.8, zorder=1)
        ax.scatter(g.x, g.y, s=16, color=fs.SERIES[t], marker=MARKERS[t], edgecolors=fs.SURFACE, linewidths=0.5, zorder=3)
        ax.plot(np.r_[0, m.x], np.r_[0, m.y], color=fs.SERIES[t], lw=1.6, zorder=2)
        ax.plot(np.r_[0, m.x], np.r_[0, m.spill], color=fs.INK2, lw=1.2, ls=(0, (3, 2)), zorder=2)
        ax.set_xlim(0, 5.5); ax.set_ylim(0, 5.5); ax.set_xticks([0, 1, 2, 3, 4, 5]); ax.set_yticks([0, 1, 2, 3, 4, 5])
        ax.set_title(fs.LABELS[t], loc="left", pad=3); ax.set_xlabel("True rate (%)")
    axes[0].set_ylabel("Estimated rate (%)")
    fig.legend(handles=[Line2D([], [], color=fs.INK2, lw=1.6, label="Injected type"),
                        Line2D([], [], color=fs.INK2, lw=1.2, ls=(0, (3, 2)), label="Attributed to the other types")],
               loc="lower left", bbox_to_anchor=(0.07, 0.0), ncol=2, handletextpad=0.4, columnspacing=1.5)
    fs.save(fig, path or OUT / "e_recovery")


def f_mixes():
    rec = pd.read_csv(RES / "fig1_v2_recovery.csv")
    rec = rec[rec.kind == "mix"]
    fig, ax = plt.subplots(figsize=(fs.SINGLE * 0.85, 2.9), gridspec_kw=dict(left=0.2, right=0.96, top=0.95, bottom=0.36))
    err = []
    for t in ORDER:
        x, y = 100 * rec[f"true_{t}"], 100 * rec[f"est_{t}"]
        err.append(y - x)
        ax.scatter(x, y, s=22, color=fs.SERIES[t], marker=MARKERS[t], edgecolors=fs.SURFACE, linewidths=0.6, zorder=3, label=fs.LABELS[t])
    lim = 4.8
    ax.plot([0, lim], [0, lim], color=fs.AXIS, lw=0.8, zorder=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_xlabel("True rate (%)"); ax.set_ylabel("Estimated rate (%)")
    e = np.concatenate(err)
    ax.text(0.96, 0.05, f"RMSE {np.sqrt(np.mean(e ** 2)):.2f} pt", transform=ax.transAxes, ha="right", fontsize=7.5, color=fs.INK2)
    fig.legend(loc="lower left", bbox_to_anchor=(0.03, 0.0), ncol=2, handletextpad=0.2, columnspacing=0.8)
    fs.save(fig, OUT / "f_mixes")


# --------------------------------------------------------------------------- g: across motion

def g_motion():
    u = pd.read_csv(RES / "motion_testbed_unmixing.csv")
    for m in ("fig1", "self"):
        u[f"sq_{m}"] = sum((100 * (u[f"{m}_{t}"] - u[f"true_{t}"])) ** 2 for t in um.TYPES) / len(um.TYPES)
    g = u.groupby(["motion", "speed"]).agg(resid=("resid", "mean"), fig1=("sq_fig1", "mean"), free=("sq_self", "mean")).reset_index()
    g[["fig1", "free"]] = np.sqrt(g[["fig1", "free"]])
    fig, ax = plt.subplots(figsize=(fs.SINGLE, 2.6), gridspec_kw=dict(left=0.17, right=0.97, top=0.95, bottom=0.36))
    ax.axvspan(0.133, 0.284, color=fs.GRID, alpha=0.6, lw=0, zorder=0)
    ax.scatter(g.resid, g.fig1, s=20, facecolors="none", edgecolors=fs.MUTED, linewidths=1.0, zorder=2, label="Simulation-calibrated (Figure 1)")
    ax.scatter(g.resid, g.free, s=20, color=fs.INK, edgecolors=fs.SURFACE, linewidths=0.5, zorder=3, label="Simulation-free")
    ax.set_yscale("log"); ax.set_ylim(0.03, 30); ax.set_xlim(0, float(g.resid.max()) * 1.05)
    ax.set_xlabel("Motion vs neighbours (spacings per frame)"); ax.set_ylabel("Unmixing error (RMSE, pt)")
    fig.legend(loc="lower left", bbox_to_anchor=(0.03, 0.0), ncol=1, handletextpad=0.3)
    fs.save(fig, OUT / "g_motion")


def figures() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    f1.OUT = OUT
    a_counts(); f1.b_errors(); c_response(); d_addition(); e_recovery(); f_mixes(); g_motion()


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--stage", choices=("all", "figures"), default="all")
    a = p.parse_args()
    if a.stage == "all":
        run(a.workers)
    figures()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
