"""Testing zone, step 2b: errors injected once into the whole movie, so they persist over time (a
missed cell comes back in the next frame, fragmentations and switches carry on, and both frames of a
pair can be wrong); unmixed with standard addition on the whole movie.

    python scripts/movie_testbed.py [--workers N] [--stage all|summary]

Movies: the persistent-motion movies of motion_testbed.py (five speeds, two movies of 11 frames).
Mixes: estimate_error_mix.SCENARIOS, injected into frames 1-10 (frame 0 clean).  Truth: the injected
events (injection log) per labelled cell at t, the estimator's own denominator.  Step 2 (each pair
injected separately into a clean copy) is read from motion_testbed_unmixing.csv for comparison.

Writes data/_results/movie_testbed.csv.
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
import estimate_error_mix as em  # noqa: E402
import motion_testbed as mt  # noqa: E402
from tracking_validation import apply_tracking_errors  # noqa: E402
from tracking_validation import unmixing as um  # noqa: E402

RES = ROOT / "data" / "_results"
MOTION, SEEDS = "persistent", (0, 1)


def job(args):
    speed, seed = args
    gt = mt.movie(MOTION, speed, seed).as_frames()
    spacing = float(np.median([mt.sweep.median_nn(g["positions"]) for g in gt]))
    box = (mt.BOX,) * 3
    rng = np.random.default_rng([mt.SPEEDS.index(speed), seed, 5])
    rows = []
    for name, rates in em.SCENARIOS.items():
        obs, log = apply_tracking_errors(gt, um.error_config(rates, spacing), box, rng)
        events = um.logged_events(log)
        c, n = um.counts(obs, mt.CAP)
        est = um.unmix([obs], box, spacing, mt.CAP, rng)
        A = est.pop("A")
        rows.append(dict(speed=speed, seed=seed, scenario=name, n=n,
                         **{f"true_{t}": events[t] / n for t in um.TYPES}, **{f"est_{t}": est[t] for t in um.TYPES},
                         **{f"count_{k}": v / n for k, v in zip(um.COUNTS, c)},
                         **{f"A_{k}_{t}": A[i, j] for i, k in enumerate(um.COUNTS) for j, t in enumerate(um.TYPES)}))
    print(f"speed {speed:4.2f} seed {seed}: done", flush=True)
    return rows


def summary() -> None:
    d = pd.read_csv(RES / "movie_testbed.csv")
    s2 = pd.read_csv(RES / "motion_testbed_unmixing.csv").query("motion == @MOTION")
    resid = s2.groupby("speed").resid.mean().round(2)
    pd.set_option("display.width", 220)
    rows = []
    for label, df, est in (("2: per pair", s2, "self"), ("2b: whole movie", d, "est")):
        for sp, g in df.groupby("speed"):
            r = dict(step=label, motion=resid[sp])
            for t in um.TYPES:
                r[t] = float(np.mean(np.abs(100 * (g[f"{est}_{t}"] - g[f"true_{t}"]))))
            rows.append(r)
    print("mean |error| (points) per type, six mixes x two movies; motion = motion vs neighbours (spacings per frame)")
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    at = d[d.speed == 1.0].groupby("scenario").mean(numeric_only=True)
    print("\nwhole movie, 0.16 spacings per frame: true -> estimated (%)")
    for sc, r in at.iterrows():
        print(f"  {sc:15s} " + "  ".join(f"{t}: {100 * r[f'true_{t}']:4.2f} -> {100 * r[f'est_{t}']:4.2f}" for t in um.TYPES))
    print("\nresponse matrix measured on the whole movie (0.16 spacings per frame, mean over mixes and movies):")
    A = pd.DataFrame([[at[f"A_{k}_{t}"].mean() for t in um.TYPES] for k in um.COUNTS], index=um.COUNTS, columns=um.TYPES)
    print(A.round(2).to_string())


def figure() -> None:
    """Estimated against injected rate per error type: step 2 (top) and step 2b (bottom), coloured by
    how far cells move relative to their neighbours per frame."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import figstyle as fs
    d = pd.read_csv(RES / "movie_testbed.csv")
    s2 = pd.read_csv(RES / "motion_testbed_unmixing.csv").query("motion == @MOTION")
    levels = s2.groupby("speed").resid.mean()
    shades = dict(zip(levels.index, ("#9cc1ec", "#6aa2e3", "#3b82d9", "#2563b0", "#173f73")))
    types = ("miss", "frag", "fp", "switch")
    rows = ((s2, "self", "Step 2: errors injected into each frame pair"),
            (d, "est", "Step 2b: errors injected into the whole movie, so they persist"))
    lims = {t: max(5.0, 1.05 * max(float(100 * df[[f"true_{t}", f"{e}_{t}"]].max().max()) for df, e, _ in rows)) for t in types}
    fs.apply()
    fig, axes = plt.subplots(2, 4, figsize=(fs.DOUBLE, 4.6), gridspec_kw=dict(left=0.07, right=0.86, top=0.86, bottom=0.1, wspace=0.38, hspace=0.78))
    for row, (df, est, label) in enumerate(rows):
        for ax, t in zip(axes[row], types):
            x, y = 100 * df[f"true_{t}"], 100 * df[f"{est}_{t}"]
            lim = lims[t]
            ax.plot([0, lim], [0, lim], color=fs.INK2, lw=0.9, ls=(0, (6, 3)), zorder=1)
            for v, c in shades.items():
                k = df.speed == v
                ax.scatter(x[k], y[k], s=13, color=c, edgecolors=fs.SURFACE, linewidths=0.5, zorder=3)
            ax.set_xlim(0, lim); ax.set_ylim(0, lim)
            ax.set_xticks(np.arange(0, lim, 2)); ax.set_yticks(np.arange(0, lim, 2))
            ax.set_title(fs.LABELS[t], loc="left", pad=3); ax.set_xlabel("Injected (%)")
        axes[row, 0].set_ylabel("Estimated (%)")
        fig.text(0.07, axes[row, 0].get_position().y1 + 0.075, label, fontsize=7.5, color=fs.INK, va="bottom")
    handles = [Line2D([], [], ls="", marker="o", ms=4.5, color=c, mec=fs.SURFACE, label=f"{levels[v]:.2f}") for v, c in shades.items()]
    fig.legend(handles=handles, fontsize=6.5, loc="upper left", bbox_to_anchor=(0.87, 0.86), handletextpad=0.2,
               title="Motion vs\nneighbours\n(spacings\nper frame)", title_fontsize=6.5, alignment="left")
    fig.text(0.07, 0.99, "Persistent motion; simulation-free unmixing of six injected mixes, two movies per speed.  Dashed line: y = x.",
             fontsize=6.5, color=fs.MUTED, va="top")
    fs.save(fig, ROOT / "figures" / "local_metrics" / "movie_testbed")


SINGLE_SPEED = 1.0                     # 0.16 spacings per frame, as Figure 1 panel e


def single_job(args):
    """One error type at one rate, injected into the whole movie, unmixed on the whole movie."""
    seed, t, rate, frames = (*args, mt.FRAMES)[:4]
    gt = mt.movie(MOTION, SINGLE_SPEED, seed, frames).as_frames()
    spacing = float(np.median([mt.sweep.median_nn(g["positions"]) for g in gt]))
    box = (mt.BOX,) * 3
    rng = np.random.default_rng([seed, zlib.crc32(f"{t}@{rate}".encode())])
    obs, log = apply_tracking_errors(gt, um.error_config({t: rate}, spacing), box, rng)
    events = um.logged_events(log)
    _, n = um.counts(obs, mt.CAP)
    est = um.unmix([obs], box, spacing, mt.CAP, rng)
    est.pop("A")
    print(f"seed {seed} {t}@{rate} ({frames} frames): done", flush=True)
    return dict(seed=seed, frames=frames, kind="single", name=f"{t}@{rate}", **{f"true_{x}": events[x] / n for x in um.TYPES},
                **{f"est_{x}": est[x] for x in um.TYPES})


LONG = 100                             # frames in the long-movie test


def mix_job(args):
    """One fixed mix injected into one long movie (one job per mix, so the long movies run in parallel)."""
    seed, name, frames = args
    gt = mt.movie(MOTION, SINGLE_SPEED, seed, frames).as_frames()
    spacing = float(np.median([mt.sweep.median_nn(g["positions"]) for g in gt]))
    box = (mt.BOX,) * 3
    rng = np.random.default_rng([seed, frames, zlib.crc32(name.encode())])
    obs, log = apply_tracking_errors(gt, um.error_config(em.SCENARIOS[name], spacing), box, rng)
    events = um.logged_events(log)
    c, n = um.counts(obs, mt.CAP)
    est = um.unmix([obs], box, spacing, mt.CAP, rng)
    A = est.pop("A")
    print(f"seed {seed} {name} ({frames} frames): done", flush=True)
    return dict(speed=SINGLE_SPEED, seed=seed, frames=frames, scenario=name, n=n,
                **{f"true_{t}": events[t] / n for t in um.TYPES}, **{f"est_{t}": est[t] for t in um.TYPES},
                **{f"count_{k}": v / n for k, v in zip(um.COUNTS, c)},
                **{f"A_{k}_{t}": A[i, j] for i, k in enumerate(um.COUNTS) for j, t in enumerate(um.TYPES)})


def long(workers: int) -> None:
    import make_fig1_v2 as f1v2
    with ProcessPoolExecutor(workers) as ex:
        mixes = ex.map(mix_job, [(s, name, LONG) for s in SEEDS for name in em.SCENARIOS])
        singles = ex.map(single_job, [(s, t, r, LONG) for s in SEEDS for t in f1v2.ORDER for r in f1v2.RATES])
        pd.DataFrame(list(mixes)).to_csv(RES / "movie_testbed_long.csv", index=False)
        pd.DataFrame(list(singles)).to_csv(RES / "movie_testbed_single_long.csv", index=False)
    long_summary()


def long_summary() -> None:
    import make_fig1_v2 as f1v2
    short = pd.read_csv(RES / "movie_testbed.csv").query("speed == @SINGLE_SPEED")
    lng = pd.read_csv(RES / "movie_testbed_long.csv")
    pd.set_option("display.width", 220)
    print("mixes, persistent motion at 0.16 spacings per frame: mean |error| (points) per type")
    for label, df in ((f"{mt.FRAMES} frames", short), (f"{LONG} frames", lng)):
        mixed = df[df.scenario != "clean"]
        e = {t: 100 * (mixed[f"est_{t}"] - mixed[f"true_{t}"]) for t in um.TYPES}
        print(f"  {label:10s} " + "  ".join(f"{t} {e[t].abs().mean():.2f}" for t in um.TYPES)
              + f"   worst {max(e[t].abs().max() for t in um.TYPES):.2f};  r(missed error, FP error) {np.corrcoef(e['miss'], e['fp'])[0, 1]:+.2f}")
    print("\nresponse matrix, missed vs false-positive columns (rise per added error):")
    for label, df in ((f"{mt.FRAMES} frames", short), (f"{LONG} frames", lng)):
        print(f"  {label:10s} " + "  ".join(f"{k}: miss {df[f'A_{k}_miss'].mean():.2f} / fp {df[f'A_{k}_fp'].mean():.2f}"
                                            for k in ("ended", "begun", "relabelled")))
    print("\none type at a time: estimated / true, and spill into the other type")
    for label, f in ((f"{mt.FRAMES} frames", "movie_testbed_single.csv"), (f"{LONG} frames", "movie_testbed_single_long.csv")):
        d = pd.read_csv(RES / f); d["error"] = d.name.str.split("@").str[0]
        parts = []
        for t, o in (("miss", "fp"), ("fp", "miss")):
            g = d[d.error == t]
            parts.append(f"{t}: {g[f'est_{t}'].sum() / g[f'true_{t}'].sum():.2f}, spill into {o} {100 * g[f'est_{o}'].mean():.2f}%")
        print(f"  {label:10s} " + ";  ".join(parts))
    f1v2.e_recovery(pd.read_csv(RES / "movie_testbed_single_long.csv"), ROOT / "figures" / "local_metrics" / "movie_testbed_recovery_long",
                    note=f"Step 2b, {LONG}-frame movies: one error type at a time, injected into the whole movie "
                         "(persistent motion, 0.16 spacings per frame).")


def single(workers: int) -> None:
    import make_fig1_v2 as f1v2          # imported here: it switches matplotlib to the Figure 1 style
    jobs = [(s, t, r) for s in SEEDS for t in f1v2.ORDER for r in f1v2.RATES]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(single_job, jobs))
    pd.DataFrame(rows).to_csv(RES / "movie_testbed_single.csv", index=False)
    single_figure()


def single_figure() -> None:
    import make_fig1_v2 as f1v2
    f1v2.e_recovery(pd.read_csv(RES / "movie_testbed_single.csv"), ROOT / "figures" / "local_metrics" / "movie_testbed_recovery",
                    note="Step 2b: one error type at a time, injected into the whole movie (persistent motion, 0.16 spacings per frame).")


# --------------------------------------------------------------------------- which measurements

ALL = um.NEIGHBOUR + ("relabelled", "flagged")          # every channel, measured once per movie
METHODS = {                                              # each method: a subset of the channels
    "neighbourhood": ALL,
    "wio_rc_2way": ("vanished", "displaced", "newcomer", "rc", "flagged"),
    "wio_rc_2way_pwio": ("vanished", "displaced", "newcomer", "rc", "pwio", "flagged"),
}
METHOD_LABELS = {"counts": "Four counts (labels ending, beginning, relabelled, two-way flag)",
                 "neighbourhood": "WIO components, PWIO, rank correlation, relabelled, two-way flag",
                 "wio_rc_2way": "WIO components (vanished, displaced, newcomer), rank correlation, two-way flag",
                 "wio_rc_2way_pwio": "WIO components, rank correlation, two-way flag, PWIO"}
RECOVERY = ROOT / "figures" / "local_metrics" / "recovery"


def channels_job(args):
    """One error type at one rate (the same injected movie as single_job): every channel and its
    response matrix, so each method can be solved from a subset of rows afterwards."""
    seed, t, rate, frames = args
    gt = mt.movie(MOTION, SINGLE_SPEED, seed, frames).as_frames()
    spacing = float(np.median([mt.sweep.median_nn(g["positions"]) for g in gt]))
    box = (mt.BOX,) * 3
    rng = np.random.default_rng([seed, zlib.crc32(f"{t}@{rate}".encode())])
    obs, log = apply_tracking_errors(gt, um.error_config({t: rate}, spacing), box, rng)
    events = um.logged_events(log)
    n = sum(len(f["assigned_ids"]) for f in obs[:-1])
    base, A = um.neighbour_response(obs, box, spacing, mt.CAP, rng, extra=("relabelled", "flagged"))
    print(f"channels: seed {seed} {t}@{rate} ({frames} frames): done", flush=True)
    return dict(seed=seed, frames=frames, kind="single", name=f"{t}@{rate}", **{f"true_{x}": events[x] / n for x in um.TYPES},
                **{f"base_{c}": base[i] for i, c in enumerate(ALL)},
                **{f"A_{c}_{x}": A[i, j] for i, c in enumerate(ALL) for j, x in enumerate(um.TYPES)})


def channels(workers: int, frames: int) -> None:
    import make_fig1_v2 as f1v2
    jobs = [(s, t, r, frames) for s in SEEDS for t in f1v2.ORDER for r in f1v2.RATES]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(channels_job, jobs))
    pd.DataFrame(rows).to_csv(RES / f"movie_testbed_channels_{frames}f.csv", index=False)
    recovery_figures(frames)


def recovery_figures(frames: int) -> None:
    """One recovery plot (as Figure 1 panel e) per method, on the same injected movies."""
    import make_fig1_v2 as f1v2
    RECOVERY.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(RES / f"movie_testbed_channels_{frames}f.csv")
    counts = pd.read_csv(RES / ("movie_testbed_single.csv" if frames == mt.FRAMES else "movie_testbed_single_long.csv"))
    out = {"counts": counts}
    for method, chans in METHODS.items():
        rows = [ALL.index(c) for c in chans]
        est = []
        for _, r in d.iterrows():
            base = np.array([r[f"base_{c}"] for c in ALL])[rows]
            A = np.array([[r[f"A_{c}_{x}"] for x in um.TYPES] for c in ALL])[rows]
            est.append(um.solve(base, A))
        out[method] = d[["seed", "kind", "name"] + [f"true_{x}" for x in um.TYPES]].assign(
            **{f"est_{x}": [e[x] for e in est] for x in um.TYPES})
    print(f"{frames}-frame movies, one type at a time: estimated / true, and what is attributed to the other types (mean %)")
    for method, rec in out.items():
        rec = rec.assign(error=rec.name.str.split("@").str[0])
        parts = []
        for t in um.TYPES:
            g = rec[rec.error == t]
            spill = 100 * g[[f"est_{o}" for o in um.TYPES if o != t]].sum(axis=1).mean()
            parts.append(f"{t} {g[f'est_{t}'].sum() / g[f'true_{t}'].sum():.2f} (+{spill:.2f})")
        print(f"  {method:17s} " + "  ".join(parts))
        f1v2.e_recovery(rec, RECOVERY / f"{method}_{frames}f",
                        note=f"{METHOD_LABELS[method]}.  {frames}-frame movies, one error type at a time.")


# --------------------------------------------------------------------------- step 2: mixes across motion types

ALL_MIX = ALL + ("ended", "begun")                       # adds the label counts, so the counts method is solvable too
MIX_METHODS = {"counts": ("ended", "begun", "relabelled", "flagged"), **METHODS}
MIX_MOTIONS = ("random", "persistent", "mixed speeds", "streams")
MIX_SPEEDS = (0.5, 1.0, 1.75)                            # 0.08, 0.16, 0.30 spacings per frame for persistent motion


def mix_channels_job(args):
    """One fixed mix injected into one long movie of one motion: every channel and its response matrix."""
    motion, speed, seed, name, frames = args
    gt = mt.movie(motion, speed, seed, frames).as_frames()
    spacing = float(np.median([mt.sweep.median_nn(g["positions"]) for g in gt]))
    box = (mt.BOX,) * 3
    rng = np.random.default_rng([list(mt.MOTIONS).index(motion), mt.SPEEDS.index(speed), seed, zlib.crc32(name.encode())])
    obs, log = apply_tracking_errors(gt, um.error_config(em.SCENARIOS[name], spacing), box, rng)
    events = um.logged_events(log)
    n = sum(len(f["assigned_ids"]) for f in obs[:-1])
    base, A = um.neighbour_response(obs, box, spacing, mt.CAP, rng, extra=ALL_MIX[len(um.NEIGHBOUR):])
    print(f"mix: {motion} speed {speed} seed {seed} {name} ({frames} frames): done", flush=True)
    return dict(motion=motion, speed=speed, seed=seed, frames=frames, scenario=name, **{f"true_{x}": events[x] / n for x in um.TYPES},
                **{f"base_{c}": base[i] for i, c in enumerate(ALL_MIX)},
                **{f"A_{c}_{x}": A[i, j] for i, c in enumerate(ALL_MIX) for j, x in enumerate(um.TYPES)})


def mixes(workers: int, frames: int) -> None:
    jobs = [(m, v, 0, name, frames) for m in MIX_MOTIONS for v in MIX_SPEEDS for name in em.SCENARIOS]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(mix_channels_job, jobs))
    pd.DataFrame(rows).to_csv(RES / f"movie_testbed_mixes_{frames}f.csv", index=False)
    mixes_summary(frames)


def mixes_summary(frames: int) -> None:
    """Unmixing error per method across motion types and speeds; one panel per method."""
    import matplotlib.pyplot as plt
    import figstyle as fs
    d = pd.read_csv(RES / f"movie_testbed_mixes_{frames}f.csv")
    resid = pd.read_csv(RES / "motion_testbed_unmixing.csv").groupby(["motion", "speed"]).resid.mean()
    d["resid"] = [resid[(m, v)] for m, v in zip(d.motion, d.speed)]
    rows = []
    for method, chans in MIX_METHODS.items():
        idx = [ALL_MIX.index(c) for c in chans]
        for _, r in d.iterrows():
            base = np.array([r[f"base_{c}"] for c in ALL_MIX])[idx]
            A = np.array([[r[f"A_{c}_{x}"] for x in um.TYPES] for c in ALL_MIX])[idx]
            est = um.solve(base, A)
            rows.append(dict(method=method, motion=r.motion, speed=r.speed, resid=r.resid, scenario=r.scenario,
                             **{f"err_{x}": 100 * (est[x] - r[f"true_{x}"]) for x in um.TYPES}))
    e = pd.DataFrame(rows)
    e["sq"] = sum(e[f"err_{x}"] ** 2 for x in um.TYPES) / len(um.TYPES)
    g = e.groupby(["method", "motion", "speed"]).agg(resid=("resid", "mean"), rmse=("sq", "mean"),
                                                     **{f"ae_{x}": (f"err_{x}", lambda s: s.abs().mean()) for x in um.TYPES}).reset_index()
    g["rmse"] = np.sqrt(g.rmse)
    pd.set_option("display.width", 220)
    print(f"{frames}-frame movies, six mixes: RMSE (points) and mean |error| per type")
    print(g.round(2).to_string(index=False))
    for method in MIX_METHODS:
        k = g[g.method == method]
        print(f"  {method:17s} median RMSE {k.rmse.median():.2f}, worst {k.rmse.max():.2f}")

    fs.apply()
    cols = dict(zip(mt.MOTIONS, ("#2a78d6", "#1baf7a", "#eda100", "#eb6834", "#e87ba4", "#5b3a9e")))
    fig, axes = plt.subplots(1, len(MIX_METHODS), figsize=(fs.DOUBLE, 2.3), sharey=True,
                             gridspec_kw=dict(left=0.07, right=0.86, top=0.8, bottom=0.2, wspace=0.15))
    titles = {"counts": "Four counts", "neighbourhood": "Full neighbourhood set", "wio_rc_2way": "WIO + RC + two-way",
              "wio_rc_2way_pwio": "WIO + RC + two-way + PWIO"}
    for ax, method in zip(axes, MIX_METHODS):
        ax.axvspan(0.133, 0.284, color=fs.GRID, alpha=0.6, lw=0, zorder=0)
        for m in MIX_MOTIONS:
            k = g[(g.method == method) & (g.motion == m)].sort_values("resid")
            ax.plot(k.resid, k.rmse, color=cols[m], lw=1.4, marker="o", ms=3.5, mec=fs.SURFACE, mew=0.6, label=m)
        ax.set_yscale("log"); ax.set_title(titles[method], loc="left", pad=3, fontsize=7.5)
        ax.set_xlabel("Motion vs neighbours\n(spacings per frame)")
    axes[0].set_ylabel("Unmixing error (RMSE, points)")
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=6.5, loc="upper left", bbox_to_anchor=(0.87, 0.8),
               title="Motion", title_fontsize=6.5, alignment="left")
    fig.text(0.07, 0.97, f"Six mixes of all four error types injected into whole {frames}-frame movies.  Grey band: the worm trackers.",
             fontsize=6.5, color=fs.MUTED, va="top")
    fs.save(fig, ROOT / "figures" / "local_metrics" / f"mixes_{frames}f")


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--stage", choices=("all", "summary", "single", "long", "long-summary", "channels", "recovery",
                                       "mixes", "mixes-summary"), default="all")
    p.add_argument("--frames", type=int, default=mt.FRAMES, help="movie length for --stage channels / recovery / mixes")
    a = p.parse_args()
    if a.stage in ("mixes", "mixes-summary"):
        mixes(a.workers, a.frames) if a.stage == "mixes" else mixes_summary(a.frames)
        return 0
    if a.stage in ("channels", "recovery"):
        channels(a.workers, a.frames) if a.stage == "channels" else recovery_figures(a.frames)
        return 0
    if a.stage == "single":
        single(a.workers)
        return 0
    if a.stage in ("long", "long-summary"):
        long(a.workers) if a.stage == "long" else long_summary()
        return 0
    if a.stage == "all":
        jobs = [(v, s) for v in mt.SPEEDS for s in SEEDS]
        with ProcessPoolExecutor(a.workers) as ex:
            rows = [r for part in ex.map(job, jobs) for r in part]
        pd.DataFrame(rows).to_csv(RES / "movie_testbed.csv", index=False)
    summary()
    figure()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
