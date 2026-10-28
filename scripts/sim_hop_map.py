"""Hop detectability map from simulation only, with no flow model.

    python scripts/sim_hop_map.py [--workers N] [--stage all|summary]

The motion residual subtracts each cell's local flow (median step of its Delaunay neighbours),
so any coherent motion of a system cancels.  What decides whether hops can be seen is the
INCOHERENT part: relative motion of neighbours plus localisation jitter, in spacings.  So:

  frame t     packed tissue from the stage-matched slab simulator (frame 0)
  frame t+1   frame t + independent per-cell displacements, sd = sigma x median spacing per axis,
              Gaussian or Student-t (3 d.o.f., heavy-tailed, rescaled to the same sd)
  x-axis      the MEAN MOTION RESIDUAL measured on the pair (spacings): observable on any
              tracker's output without ground truth, so real data can be placed on the map

A  injected chains (scripts/chain_injection.inject_chains): swap and k = 2, 3, 5, 8 cells,
   2% wrong links on the clean pair.  Fraction of wrong links caught by the window count
   (residual 0.7-1.3 spacings) and the nearest-body test; false hops on the clean pair.
B  LAP-made chains: misses + false positives (1% + 0.5% and 3% + 1%), LAP at 1.0 spacing.
   Window count / true switched cells (the ratio seen on real trackers: 0.63-0.73).

Writes data/_results/sim_hop_map_A.csv, _B.csv and figures/local_metrics/sim_hop_map.png.
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
import chain_injection as ci  # noqa: E402
import estimate_error_mix as em  # noqa: E402
import hop_count_synthetic as hcs  # noqa: E402
import hop_count_variants as hv  # noqa: E402
import run_motile_tracking as rmt  # noqa: E402
from tracking_validation.linking import link_frames  # noqa: E402

RES = ROOT / "data" / "_results"
STAGES = ("50-100", "100-200", "200-360")
SEEDS = range(4)
SIGMAS = (0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20)
NOISES = ("gauss", "t3")
KS = ("swap", 2, 3, 5, 8)


def stage_means() -> pd.DataFrame:
    meas = pd.read_csv(RES / "ce_stage_measurements.csv")
    return meas.groupby("stage")[["nn_med", "nn_p10", "disp_med", "disp_p90", "density", "z_extent"]].mean()


def displace(p0, sp, sigma, noise, rng):
    if noise == "gauss":
        e = rng.normal(size=p0.shape)
    else:                                   # Student-t, 3 d.o.f., unit variance
        e = rng.standard_t(3, size=p0.shape) / np.sqrt(3.0)
    return p0 + sigma * sp * e


def job(args):
    stage, seed = args
    pair, box = em._sim_pair(stage_means(), stage, 70_000 + 100 * STAGES.index(stage) + seed)
    p0 = np.asarray(pair[0]["positions"], float); ids = np.asarray(pair[0]["true_ids"], int)
    sp = sweep.median_nn(p0)
    E = set(int(c) for c in ids)
    rows_a, rows_b = [], []
    for noise in NOISES:
        for sigma in SIGMAS:
            rng = np.random.default_rng(hash((stage, seed, noise, sigma)) % 2**32)
            gt = [dict(positions=p0, true_ids=ids), dict(positions=displace(p0, sp, sigma, noise, rng), true_ids=ids)]
            clean = [dict(positions=g["positions"], true_ids=ids, assigned_ids=ids.copy(), parent_ids=np.full(len(ids), -1)) for g in gt]
            mres = float(sweep.motion_residuals(clean, 15.0, None).mean())
            cc = hv.pair_counts(clean[0], clean[1], E)
            base = dict(stage=stage, seed=seed, noise=noise, sigma=sigma, mean_resid=mres, n=cc["n"],
                        clean_Hwin=cc["Hwin"], clean_Hnb=cc["Hnb_persist"], clean_H05=cc["H05"])
            # ---- A: injected chains ----
            for k in KS:
                f1, wrong = ci.inject_chains(clean, k, 0.02, sorted(E), np.random.default_rng(seed + 31), 15.0)
                c = hv.pair_counts(clean[0], f1, E)
                rows_a.append(dict(**base, k=str(k), wrong=wrong, sw=c["sw"], Hwin=c["Hwin"], Hnb=c["Hnb_persist"], H05=c["H05"],
                                   nb_tp=c["nb_tp"]))
            # ---- B: LAP-made chains ----
            for level, (pm, pf) in hcs.LEVELS.items():
                det = hcs.synth(gt, sp, pm, pf, rng)
                F = link_frames(det, 1.0 * sp, "lap").frames
                c = hv.pair_counts(F[0], F[1], E)
                swp, chn = rmt.switch_shapes(F[0], F[1], E)
                rows_b.append(dict(**base, level=level, sw=c["sw"], swaps=swp, chains=chn, Hwin=c["Hwin"], Hnb=c["Hnb_persist"],
                                   H05=c["H05"], n_trk=c["n"],
                                   trk_resid=float(sweep.motion_residuals(F, 15.0, None).mean())))
    print(f"{stage} seed {seed}: done", flush=True)
    return rows_a, rows_b


def summary() -> None:
    import matplotlib.pyplot as plt
    import figstyle as fs
    a = pd.read_csv(RES / "sim_hop_map_A.csv"); b = pd.read_csv(RES / "sim_hop_map_B.csv")
    pd.set_option("display.width", 220)
    print("### A  injected chains: wrong links caught (window | nearest body), and false hops per 100 clean cells")
    a["win_caught"] = (a.Hwin - a.clean_Hwin) / a.wrong
    a["nb_caught"] = (a.Hnb - a.clean_Hnb) / a.wrong
    for noise, g in a.groupby("noise"):
        t = g.groupby(["sigma", "k"]).agg(mean_resid=("mean_resid", "mean"), win=("win_caught", "mean"), nb=("nb_caught", "mean")).reset_index()
        w = t.pivot(index="sigma", columns="k", values="win").add_prefix("win_")
        nb = t.pivot(index="sigma", columns="k", values="nb").add_prefix("nb_")
        fh = g.groupby("sigma").agg(mean_resid=("mean_resid", "mean"), false_win=("clean_Hwin", "sum"), false_nb=("clean_Hnb", "sum"), n=("n", "sum"))
        fh["false_win_%"] = 100 * fh.false_win / fh.n; fh["false_nb_%"] = 100 * fh.false_nb / fh.n
        print(f"\n[{noise}]"); print(pd.concat([fh[["mean_resid", "false_win_%", "false_nb_%"]], w, nb], axis=1).round(2).to_string())
    print("\n### B  LAP-made chains, link 1.0 spacing: window / true switched cells")
    g = b.groupby(["noise", "sigma"]).agg(trk_resid=("trk_resid", "mean"), sw=("sw", "sum"), n=("n_trk", "sum"), win=("Hwin", "sum"),
                                         nb=("Hnb", "sum"), swaps=("swaps", "sum"), chains=("chains", "sum"))
    g["true_%"] = 100 * g.sw / g.n; g["win/true"] = g.win / g.sw.clip(lower=1); g["nb/true"] = g.nb / g.sw.clip(lower=1)
    print(g[["trk_resid", "true_%", "swaps", "chains", "win/true", "nb/true"]].round(3).to_string())
    real = pd.DataFrame({"interval": [1, 2, 3, 4], "trk_resid": [0.132, 0.189, 0.232, 0.262], "win/true": [0.63, 0.65, 0.72, 0.73]})
    print("\nreal LAP trackers, link 1.0 (for comparison):"); print(real.to_string(index=False))

    fs.apply()
    fig, axes = plt.subplots(1, 3, figsize=(fs.DOUBLE, 2.3), gridspec_kw=dict(left=0.07, right=0.99, top=0.86, bottom=0.21, wspace=0.38))
    cols = {"swap": "#52514e", "2": "#2a78d6", "3": "#1baf7a", "5": "#eda100", "8": "#eb6834"}
    for ax, (col, title) in zip(axes[:2], (("win_caught", "Window count"), ("nb_caught", "Nearest body"))):
        g = a[a.noise == "t3"].groupby(["sigma", "k"]).agg(x=("mean_resid", "mean"), y=(col, "mean")).reset_index()
        for k, c in cols.items():
            h = g[g.k == k]
            ax.plot(h.x, h.y, marker="o", ms=3, color=c, lw=1.2, label="swap" if k == "swap" else f"k = {k}")
        ax.axvspan(0.13, 0.26, color=fs.GRID, zorder=0)
        ax.set_ylim(-0.1, 1.05); ax.set_xlabel("Mean motion residual (spacings)")
        ax.set_title(f"Injected chains caught: {title.lower()}", loc="left", pad=3, fontsize=7.5)
    axes[0].set_ylabel("Fraction of wrong links caught"); axes[0].legend(fontsize=6, loc="lower left")
    ax = axes[2]
    for noise, c in (("gauss", "#2a78d6"), ("t3", "#eb6834")):
        h = g_b = b[b.noise == noise].groupby("sigma").agg(x=("trk_resid", "mean"), sw=("sw", "sum"), win=("Hwin", "sum")).reset_index()
        h = h[h.sw >= 20]
        ax.plot(h.x, h.win / h.sw, marker="o", ms=3, color=c, lw=1.2, label=f"simulation, {'Gaussian' if noise == 'gauss' else 'heavy-tailed'}")
    ax.scatter(real.trk_resid, real["win/true"], s=28, color=fs.INK, zorder=4, label="real LAP trackers")
    ax.axvspan(0.13, 0.26, color=fs.GRID, zorder=0)
    ax.set_xlabel("Mean motion residual (spacings)"); ax.set_ylabel("Window count / true switches")
    ax.set_title("LAP-made chains", loc="left", pad=3, fontsize=7.5); ax.legend(fontsize=6)
    fs.save(fig, ROOT / "figures" / "local_metrics" / "sim_hop_map")


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--stage", choices=("all", "summary"), default="all")
    a = p.parse_args()
    if a.stage == "all":
        em.SWITCH_CAP = 1.1; sweep.MODE = "delaunay"
        jobs = [(st, sd) for st in STAGES for sd in SEEDS]
        ra, rb = [], []
        with ProcessPoolExecutor(a.workers) as ex:
            for x, y in ex.map(job, jobs):
                ra += x; rb += y
        pd.DataFrame(ra).to_csv(RES / "sim_hop_map_A.csv", index=False)
        pd.DataFrame(rb).to_csv(RES / "sim_hop_map_B.csv", index=False)
    summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
