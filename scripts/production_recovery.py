"""Recovery curves of the production unmixing (tracking_validation.unmixing.unmix_production) over the
whole motion testbed: one error type at a time, injected into the whole movie.

    python scripts/production_recovery.py [--workers N] [--stage all|figures]

Production settings: the WIO components (vanished, newcomer, displaced), PWIO, rank correlation, the
relabelled count and the two-way flag with the occupied-spot gate (0.5 spacings); response matrix by
standard addition on the same movie (2% of each type added); error-free level of displaced, PWIO and
rank correlation from the movie's clean neighbourhoods; non-negative least squares.

Movies: scripts/motion_testbed.py, all six motion types at three speeds, 100 frames, 1000 cells.
Errors: each type alone at 0.5-5% of cells, injected into frames 1-99 (they persist).  Truth (the
injection log) is used only to score.  Rows are appended to the CSV as movies finish, and a rerun
skips the movies already there, so an interrupted run resumes.

Writes data/_results/production_recovery.csv, figures/production/recovery_<speed>.png (all motion
types, one figure per speed) and figures/production/recovery/<motion>_<speed>.png (one per setting).
"""
from __future__ import annotations

import argparse
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import motion_testbed as mt  # noqa: E402
from tracking_validation import apply_tracking_errors  # noqa: E402
from tracking_validation import unmixing as um  # noqa: E402

RES = ROOT / "data" / "_results"
OUT_CSV = RES / "production_recovery.csv"
FIG = ROOT / "figures" / "production"
FRAMES = 100
SPEEDS = (0.5, 1.0, 1.75)
TYPES = ("miss", "switch", "frag", "fp")              # Figure 1 column order
RATES = (0.005, 0.01, 0.02, 0.03, 0.04, 0.05)
KEYS = ["motion", "speed", "error", "rate"]


def job(args):
    motion, speed, t, rate = args
    gt = mt.movie(motion, speed, 0, FRAMES).as_frames()
    spacing = float(np.median([mt.sweep.median_nn(g["positions"]) for g in gt]))
    box = (mt.BOX,) * 3
    rng = np.random.default_rng([list(mt.MOTIONS).index(motion), mt.SPEEDS.index(speed), zlib.crc32(f"{t}@{rate}".encode())])
    obs, log = apply_tracking_errors(gt, um.error_config({t: rate}, spacing), box, rng)
    events = um.logged_events(log)
    n = sum(len(f["assigned_ids"]) for f in obs[:-1])
    est = um.unmix_production(obs, box, spacing, mt.CAP, rng)
    est.pop("A")
    return dict(motion=motion, speed=speed, error=t, rate=rate, n=n,
                **{f"true_{x}": events[x] / n for x in um.TYPES}, **{f"est_{x}": est[x] for x in um.TYPES})


def run(workers: int) -> None:
    done = pd.read_csv(OUT_CSV)[KEYS] if OUT_CSV.exists() else pd.DataFrame(columns=KEYS)
    have = set(map(tuple, done.astype(object).values.tolist()))
    jobs = [(m, v, t, r) for m in mt.MOTIONS for v in SPEEDS for t in TYPES for r in RATES if (m, v, t, r) not in have]
    print(f"{len(jobs)} movies to run ({len(have)} already done)", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        futures = [ex.submit(job, j) for j in jobs]
        for k, f in enumerate(as_completed(futures), 1):
            row = f.result()
            pd.DataFrame([row]).to_csv(OUT_CSV, mode="a", header=not OUT_CSV.exists(), index=False)
            print(f"[{k}/{len(jobs)}] {row['motion']} speed {row['speed']} {row['error']}@{row['rate']}", flush=True)


def figures() -> None:
    import make_fig1_v2 as f1v2         # Figure 1 style and the recovery-panel code
    import matplotlib.pyplot as plt
    import figstyle as fs
    d = pd.read_csv(OUT_CSV)
    resid = pd.read_csv(RES / "motion_testbed_unmixing.csv").groupby(["motion", "speed"]).resid.mean()
    (FIG / "recovery").mkdir(parents=True, exist_ok=True)
    rows = []
    for (motion, speed), g in d.groupby(["motion", "speed"]):
        rec = g.assign(kind="single", name=g.error + "@" + g.rate.astype(str), seed=0)
        f1v2.e_recovery(rec, FIG / "recovery" / f"{motion.replace(' ', '_')}_{speed}",
                        note=f"Production unmixing, {motion} motion, {resid[(motion, speed)]:.2f} spacings per frame, "
                             f"{FRAMES}-frame movies, one error type at a time.")
        for t in TYPES:
            k = g[g.error == t]
            spill = 100 * k[[f"est_{o}" for o in um.TYPES if o != t]].sum(axis=1).mean()
            rows.append(dict(motion=motion, speed=speed, resid=resid[(motion, speed)], error=t,
                             ratio=k[f"est_{t}"].sum() / k[f"true_{t}"].sum(), spill=spill))
    s = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("estimated / true for the injected type, and mean % attributed to the other types")
    print(s.pivot_table(index=["motion", "resid"], columns="error", values=["ratio", "spill"]).round(2).to_string())

    # overview: one figure per speed, motion types as rows, error types as columns
    for speed in SPEEDS:
        motions = [m for m in mt.MOTIONS if ((d.motion == m) & (d.speed == speed)).any()]
        if not motions:
            continue
        fig, axes = plt.subplots(len(motions), 4, figsize=(fs.DOUBLE, 1.55 * len(motions) + 0.6), sharex=True, sharey=True,
                                 gridspec_kw=dict(left=0.13, right=0.99, top=1 - 0.45 / (1.55 * len(motions) + 0.6),
                                                  bottom=0.5 / (1.55 * len(motions) + 0.6), wspace=0.12, hspace=0.22))
        axes = np.atleast_2d(axes)
        for r, motion in enumerate(motions):
            g = d[(d.motion == motion) & (d.speed == speed)]
            for c, t in enumerate(TYPES):
                ax, k = axes[r, c], g[g.error == t].sort_values(f"true_{t}")
                x = 100 * k[f"true_{t}"]
                ax.plot([0, 5.5], [0, 5.5], color=fs.AXIS, lw=0.8, zorder=1)
                ax.plot(np.r_[0, x], np.r_[0, 100 * k[f"est_{t}"]], color=fs.SERIES[t], lw=1.4, marker=f1v2.MARKERS[t], ms=3.5,
                        mec=fs.SURFACE, mew=0.5, zorder=3)
                spill = 100 * k[[f"est_{o}" for o in um.TYPES if o != t]].sum(axis=1)
                ax.plot(np.r_[0, x], np.r_[0, spill], color=fs.INK2, lw=1.0, ls=(0, (3, 2)), zorder=2)
                ax.set_xlim(0, 5.5); ax.set_ylim(0, 5.5); ax.set_xticks([0, 1, 2, 3, 4, 5]); ax.set_yticks([0, 2, 4])
                if r == 0:
                    ax.set_title(fs.LABELS[t], loc="left", pad=3)
                if r == len(motions) - 1:
                    ax.set_xlabel("True rate (%)")
            axes[r, 0].set_ylabel(f"{motion}\n{resid[(motion, speed)]:.2f} sp/frame", fontsize=7)
        fig.text(0.13, 1 - 0.12 / (1.55 * len(motions) + 0.6), "Estimated rate (%), production unmixing, one error type at a time; "
                 "dashed: attributed to the other types; grey line: y = x.", fontsize=6.5, color=fs.MUTED, va="top")
        fs.save(fig, FIG / f"recovery_speed{speed}")


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
