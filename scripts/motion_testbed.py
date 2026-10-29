"""Controlled ground-truth testing zone, steps 1-2: simulate a range of motions (no errors), inject
known error mixes into the tracks, and test how well each unmixing recovers them.

    python scripts/motion_testbed.py [--workers N] [--stage all|summary]

Step 1, motion (src/tracking_validation/dynamics.py): 1000 cells in a reflective cube at about
the density of the Figure 1 calibration tissue (median spacing ~4.2 µm).  Each motion type at
several speeds, so the motion relative to neighbours per frame spans ~0.02-0.6 spacings
(persistence times in frames):
  random        persistence 0.2 (direction lost within a frame)
  persistent    persistence 5
  straight      persistence 100
  mixed speeds  persistence 5, per-cell speeds lognormal (sigma 0.75, as fitted to real tracks)
  streams       persistence 5, alignment 1 (neighbours move together locally)
  flock         persistence 5, alignment 5 (the whole tissue moves together)

Step 2, unmixing: each fixed mix of estimate_error_mix.SCENARIOS is injected into every frame pair
of the movie, and the four rates (missed, fragmentation, false positive, switch) are recovered
from the pooled pairs in two ways:
  fig1   Figure 1 unmixing: seven metrics, response matrix and error-free level from the simulation
         rows of the fine sweep (also with the movie's own error-free level, "own", an oracle)
  self   simulation-free (tracking_validation.unmixing): four counts that are zero on error-free
         tracks, response matrix measured on the movie itself by standard addition

Truth (the injected, realised rates) is used only to score.  Writes
data/_results/motion_testbed_unmixing.csv and figures/local_metrics/motion_testbed.png.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import ce_error_sweep as sweep  # noqa: E402
import estimate_error_mix as em  # noqa: E402
from tracking_validation import (DynamicsConfig, apply_tracking_errors, realised_error_rates,  # noqa: E402
                                 simulate_dynamics)
from tracking_validation import unmixing as um  # noqa: E402

RES = ROOT / "data" / "_results"
N_CELLS, BOX, SPACING = 1000, 46.0, 4.5
FRAMES = 11                           # 10 frame pairs per movie
CAP = 15.0                            # neighbourhood cap (µm), as in the calibration sweep
SPEEDS = (0.25, 0.5, 1.0, 1.75, 2.5)  # rms speed, µm per frame
SEEDS = (0, 1)
MOTIONS = {
    "random": dict(persistence_time=0.2),
    "persistent": dict(persistence_time=5.0),
    "straight": dict(persistence_time=100.0),
    "mixed speeds": dict(persistence_time=5.0, speed_sigma=0.75),
    "streams": dict(persistence_time=5.0, alignment=1.0),
    "flock": dict(persistence_time=5.0, alignment=5.0),
}
TYPES = list(um.TYPES)
_CAL = None


def calibration():
    """Figure 1 calibration: response matrix, channel noise and error-free level of the simulation sweep."""
    global _CAL
    if _CAL is None:
        _CAL = em.calibrate(RES / "ce_error_sweep_fine_delaunay.csv", em.TYPES_4, source="SIM")
    return _CAL


def movie(motion: str, speed: float, seed: int, frames: int = FRAMES):
    """The error-free movie of one setting (the same one every time; a longer movie starts with the
    same frames)."""
    key = [list(MOTIONS).index(motion), SPEEDS.index(speed), seed]
    return simulate_dynamics(DynamicsConfig(num_cells=N_CELLS, box_size=BOX, spacing=SPACING, speed=speed,
                                            num_frames=frames, seed=int(np.random.default_rng(key).integers(1 << 30)),
                                            **MOTIONS[motion]))


def job(args):
    motion, speed, seed = args
    A, noise, fig1 = calibration()
    key = [list(MOTIONS).index(motion), SPEEDS.index(speed), seed]
    sim = movie(motion, speed, seed)
    gt = sim.as_frames()
    rng = np.random.default_rng(key + [1])
    box = (BOX,) * 3
    pairs = [gt[t:t + 2] for t in range(FRAMES - 1)]
    spacing = float(np.median([sweep.median_nn(p[0]["positions"]) for p in pairs]))
    clean = [sweep.evaluate_pair(p, 8, CAP, None) for p in pairs]
    own = {ch: np.mean([m[ch] for m in clean]) for ch in em.CHANNELS}
    meta = dict(motion=motion, speed=speed, seed=seed, spacing=spacing,
                resid=float(np.mean([m["resid"] for m in clean])),          # motion relative to neighbours / spacing
                neighbour_alignment=sim.metadata["neighbour_alignment"], polarisation=sim.metadata["polarisation"])
    rows = []
    for name, true in em.SCENARIOS.items():
        obs, realised = [], []
        for p in pairs:
            o, log = apply_tracking_errors(p, um.error_config(true, spacing), box, np.random.default_rng(int(rng.integers(1 << 30))))
            obs.append(o); realised.append(realised_error_rates(p, o, log, first_frame_clean=True))
        ms = clean if name == "clean" else [sweep.evaluate_pair(o, 8, CAP, None) for o in obs]
        d = np.array([np.mean([m[ch] for m in ms]) for ch in em.CHANNELS])
        r = pd.DataFrame(realised).mean()
        row = dict(meta, scenario=name, **{f"true_{t}": r[t] for t in TYPES})
        for base_name, base in (("fig1", fig1), ("own", own)):
            est = em.estimate(d - np.array([base[ch] for ch in em.CHANNELS]), A, noise)
            row.update({f"{base_name}_{t}": est[em.TYPES_4.index(t)] for t in TYPES})
        self_est = um.unmix(obs, box, spacing, CAP, rng)
        row.update({f"self_{t}": self_est[t] for t in TYPES})
        rows.append(row)
    print(f"{motion:12s} speed {speed:4.2f} seed {seed}: resid {meta['resid']:.3f}", flush=True)
    return rows


def summary() -> None:
    import matplotlib.pyplot as plt
    import figstyle as fs
    u = pd.read_csv(RES / "motion_testbed_unmixing.csv")
    methods = ("fig1", "own", "self")
    for m in methods:
        u[f"sq_{m}"] = sum((100 * (u[f"{m}_{t}"] - u[f"true_{t}"])) ** 2 for t in TYPES) / len(TYPES)
        for t in TYPES:
            u[f"ae_{m}_{t}"] = 100 * (u[f"{m}_{t}"] - u[f"true_{t}"]).abs()
    agg = {"resid": ("resid", "mean"), **{f"rmse_{m}": (f"sq_{m}", "mean") for m in methods},
           **{f"ae_self_{t}": (f"ae_self_{t}", "mean") for t in TYPES}}
    g = u.groupby(["motion", "speed"]).agg(**agg).reset_index()
    for m in methods:
        g[f"rmse_{m}"] = np.sqrt(g[f"rmse_{m}"])
    pd.set_option("display.width", 220)
    print("unmixing RMSE (points, 4 types x 6 mixes): Figure 1 / Figure 1 with own baseline (oracle) / simulation-free;"
          " then simulation-free mean |error| per type")
    print(g.round(3).to_string(index=False))
    for m in methods:
        print(f"{m:5s} median RMSE {g[f'rmse_{m}'].median():.2f}, worst {g[f'rmse_{m}'].max():.2f} points")

    fs.apply()
    cols = dict(zip(MOTIONS, ("#2a78d6", "#1baf7a", "#eda100", "#eb6834", "#e87ba4", "#5b3a9e")))
    fig, axes = plt.subplots(1, 3, figsize=(fs.DOUBLE, 2.4), gridspec_kw=dict(left=0.07, right=0.86, top=0.84, bottom=0.19, wspace=0.38))
    xmax = float(g.resid.max()) * 1.05
    ymax = float(max(g.rmse_fig1.max(), g.rmse_self.max())) * 1.08
    for ax, m, title in ((axes[0], "fig1", "Figure 1 unmixing"), (axes[1], "self", "Simulation-free unmixing")):
        for mo, c in cols.items():
            k = g[g.motion == mo].sort_values("resid")
            ax.plot(k.resid, k[f"rmse_{m}"], color=c, lw=1.4, marker="o", ms=3.5, mec=fs.SURFACE, mew=0.6, label=mo, zorder=3)
        ax.set_yscale("log"); ax.set_ylim(0.02, ymax); ax.set_title(title, loc="left", pad=3)
    axes[0].set_ylabel("Unmixing error (points, RMSE)")
    # which type the simulation-free unmixing gets wrong, pooled over motion types; ink, not colour,
    # because colour already marks the motion type in the other panels
    ax = axes[2]
    bins = np.quantile(g.resid, np.linspace(0, 1, 7))
    g["bin"] = np.digitize(g.resid, bins[1:-1])
    b = g.groupby("bin").mean(numeric_only=True)
    for t in ("miss", "frag", "fp", "switch"):
        ax.plot(b.resid, b[f"ae_self_{t}"], color=fs.INK if t == "switch" else fs.MUTED, lw=1.4, marker="o", ms=3.5,
                mec=fs.SURFACE, mew=0.6, zorder=3 if t == "switch" else 2)
    end = b.iloc[-1]
    ax.text(end.resid - 0.02, end.ae_self_switch, "Switch", ha="right", va="center", fontsize=6.5, color=fs.INK)
    ax.text(end.resid, 0.024, "Missed, fragmentation,\nfalse positive", ha="right", va="bottom", fontsize=6.5, color=fs.MUTED)
    ax.set_yscale("log"); ax.set_ylim(0.02, ymax)
    ax.set_ylabel("Mean |error| (points)"); ax.set_title("Simulation-free, by error type", loc="left", pad=3)
    for ax in axes:
        ax.axvspan(0.133, 0.284, color=fs.GRID, alpha=0.6, lw=0, zorder=0)    # worm trackers, every to every 4th frame
        ax.set_xlim(0, xmax); ax.set_xlabel("Motion vs neighbours (spacings)")
    fig.legend(*axes[1].get_legend_handles_labels(), fontsize=6.5, loc="upper left", bbox_to_anchor=(0.87, 0.84),
               title="Motion", title_fontsize=6.5, alignment="left")
    fig.text(0.07, 0.97, "Six injected mixes of four error types per movie.  Grey band: the worm trackers (every frame to every 4th).",
             fontsize=6.5, color=fs.MUTED, va="top")
    fs.save(fig, ROOT / "figures" / "local_metrics" / "motion_testbed")


def motion_figure(motion: str = "persistent") -> None:
    """One motion type: simulation-free estimate against the injected rate, one panel per error type,
    coloured by how far cells move relative to their neighbours per frame."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import figstyle as fs
    u = pd.read_csv(RES / "motion_testbed_unmixing.csv")
    u = u[u.motion == motion]
    levels = u.groupby("speed").resid.mean()
    shades = dict(zip(levels.index, ("#9cc1ec", "#6aa2e3", "#3b82d9", "#2563b0", "#173f73")))
    fs.apply()
    fig, axes = plt.subplots(1, 4, figsize=(fs.DOUBLE, 2.15), gridspec_kw=dict(left=0.06, right=0.86, top=0.83, bottom=0.21, wspace=0.38))
    for ax, t in zip(axes, ("miss", "frag", "fp", "switch")):
        x, y = 100 * u[f"true_{t}"], 100 * u[f"self_{t}"]
        lim = float(max(x.max(), y.max())) * 1.08
        ax.plot([0, lim], [0, lim], color=fs.INK2, lw=0.9, ls=(0, (6, 3)), zorder=1)
        for v, c in shades.items():
            k = u.speed == v
            ax.scatter(x[k], y[k], s=14, color=c, edgecolors=fs.SURFACE, linewidths=0.5, zorder=3)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xticks(np.arange(0, lim, 2)); ax.set_yticks(np.arange(0, lim, 2))
        ax.set_title(fs.LABELS[t], loc="left", pad=3); ax.set_xlabel("Injected (%)")
    axes[0].set_ylabel("Estimated (%)")
    handles = [Line2D([], [], ls="", marker="o", ms=4.5, color=c, mec=fs.SURFACE, label=f"{levels[v]:.2f}") for v, c in shades.items()]
    fig.legend(handles=handles, fontsize=6.5, loc="upper left", bbox_to_anchor=(0.87, 0.83), handletextpad=0.2,
               title="Motion vs\nneighbours\n(spacings\nper frame)", title_fontsize=6.5, alignment="left")
    fig.text(0.06, 0.97, f"{motion.capitalize()} motion; simulation-free unmixing of six injected mixes, two movies per speed.  "
             "Dashed line: y = x.", fontsize=6.5, color=fs.MUTED, va="top")
    fs.save(fig, ROOT / "figures" / "local_metrics" / f"motion_testbed_{motion.replace(' ', '_')}")


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--stage", choices=("all", "summary", "motion"), default="all")
    p.add_argument("--motion", default="persistent", choices=list(MOTIONS), help="motion type for --stage motion")
    a = p.parse_args()
    if a.stage == "motion":
        motion_figure(a.motion)
        return 0
    if a.stage == "all":
        jobs = [(m, v, s) for m in MOTIONS for v in SPEEDS for s in SEEDS]
        with ProcessPoolExecutor(a.workers) as ex:
            rows = [r for part in ex.map(job, jobs) for r in part]
        pd.DataFrame(rows).to_csv(RES / "motion_testbed_unmixing.csv", index=False)
    summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
