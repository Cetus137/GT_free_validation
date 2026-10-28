"""Hop detectability map, round 2 (simulation only, no flow model), and a simulation-calibrated
switch estimate tested on the real LAP trackers.

    python scripts/sim_hop_map2.py [--workers N] [--stage all|summary]

Changes from sim_hop_map.py:
  * nearest-body test in two variants: any different detection (nb_any) and a different
    label that existed at t (nb_persist); every count is split into caught switches (TP) and
    false flags, so capture and false-flag rate are measured separately
  * injected chains hop to the NEAREST unused body (as an optimal linker's hops do)
  * more switches: detection error levels 2%+1%, 5%+2%, 10%+4% (misses + false positives)
  * each condition is described by two observables of the tracker output: the mean motion
    residual and the share of residuals > 0.5 spacing

Calibrated estimate on real trackers (link 1.0 spacing): for each real tracker, its own two
descriptors pick the k nearest simulated conditions; their capture c and false-flag rate f
give  estimate = (count - f * n) / c.  Truth is used only to score the result.

Writes data/_results/sim_hop_map2_A.csv, _B.csv, sim_hop_calibrated_real.csv and
figures/local_metrics/sim_hop_map2.png.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import ce_error_sweep as sweep  # noqa: E402
import estimate_error_mix as em  # noqa: E402
import hop_count_synthetic as hcs  # noqa: E402
import run_motile_tracking as rmt  # noqa: E402
import sim_hop_map as m1  # noqa: E402
from tracking_validation.linking import link_frames  # noqa: E402
from tracking_validation.local_metrics import predicted_positions  # noqa: E402

RES = ROOT / "data" / "_results"
SEEDS = range(6)
LEVELS = {"2+1": (0.02, 0.01), "5+2": (0.05, 0.02), "10+4": (0.10, 0.04)}
FLAGS = ("win", "nb_any", "nb_persist", "h05")
K_NN = 3


def flags_and_truth(f0, f1, E, cap=15.0):
    """Per-pair counts: n evaluated persisting cells, switched, and for each flag its total and TP;
    plus the residual list (for the descriptors)."""
    l0 = [int(l) for l in f0["assigned_ids"]]; l1 = [int(l) for l in f1["assigned_ids"]]
    out = dict(n=0, sw=0, resid_sum=0.0, **{f: 0 for f in FLAGS}, **{f"{f}_tp": 0 for f in FLAGS})
    if len(l0) < 5 or len(l1) < 5:
        return out
    s0 = set(l0)
    pred, step, sp = predicted_positions(f0, f1, cap)
    p0 = dict(zip(l0, f0["positions"])); tree1 = cKDTree(f1["positions"])
    tid0 = dict(zip(l0, f0["true_ids"].tolist())); tid1 = dict(zip(l1, f1["true_ids"].tolist()))
    for l, s in step.items():
        a, b = tid0[l], tid1[l]
        if a < 0 or b < 0 or a not in E:
            continue
        r = float(np.linalg.norm(p0[l] + s - pred[l])) / sp
        j = int(tree1.query(pred[l])[1])
        fl = dict(win=0.7 <= r <= 1.3, nb_any=l1[j] != l, nb_persist=(l1[j] != l and l1[j] in s0), h05=r > 0.5)
        sw = a != b
        out["n"] += 1; out["sw"] += int(sw); out["resid_sum"] += r
        for f, v in fl.items():
            out[f] += int(v); out[f"{f}_tp"] += int(v and sw)
    return out


def nearest_chain(clean, k, rate, E, rng):
    """Inject chains that hop to the nearest unused body at t+1 (label(c_i) -> body c_{i+1});
    the head body gets a fresh label, the last label is lost; 'swap' exchanges nearest neighbours."""
    f0, f1 = clean
    labels = f1["assigned_ids"].copy(); true1 = f1["true_ids"]; pos1 = f1["positions"]
    idx = {int(t): i for i, t in enumerate(true1)}
    tree = cKDTree(pos1)
    cand = [t for t in sorted(E) if t in idx]
    rng.shuffle(cand)
    target = int(round(rate * len(cand))); used: set[int] = set(); wrong = 0; fresh = int(labels.max()) + 1
    for start in cand:
        if wrong >= target:
            break
        if start in used:
            continue
        chain = [start]
        while len(chain) < (2 if k == "swap" else k):
            _, nn = tree.query(pos1[idx[chain[-1]]], k=min(12, len(pos1)))
            nxt = next((int(true1[j]) for j in nn[1:] if int(true1[j]) not in used and int(true1[j]) not in chain
                        and int(true1[j]) in E), None)
            if nxt is None:
                break
            chain.append(nxt)
        if len(chain) < 2:
            continue
        if k == "swap":
            i, j = idx[chain[0]], idx[chain[1]]
            labels[i], labels[j] = labels[j], labels[i]; wrong += 2
        else:
            old = [labels[idx[c]] for c in chain]
            for n_i, c in enumerate(chain[1:]):
                labels[idx[c]] = old[n_i]
            labels[idx[chain[0]]] = fresh; fresh += 1; wrong += len(chain) - 1
        used.update(chain)
    return {**f1, "assigned_ids": labels}, wrong


def job(args):
    stage, seed = args
    pair, box = em._sim_pair(m1.stage_means(), stage, 90_000 + 100 * m1.STAGES.index(stage) + seed)
    p0 = np.asarray(pair[0]["positions"], float); ids = np.asarray(pair[0]["true_ids"], int)
    sp = sweep.median_nn(p0); E = set(int(c) for c in ids)
    ra, rb = [], []
    for noise in m1.NOISES:
        for sigma in m1.SIGMAS:
            rng = np.random.default_rng(hash((stage, seed, noise, sigma, 2)) % 2**32)
            gt = [dict(positions=p0, true_ids=ids), dict(positions=m1.displace(p0, sp, sigma, noise, rng), true_ids=ids)]
            clean = [dict(positions=g["positions"], true_ids=ids, assigned_ids=ids.copy(), parent_ids=np.full(len(ids), -1)) for g in gt]
            cc = flags_and_truth(clean[0], clean[1], E)
            for k in m1.KS:
                f1, wrong = nearest_chain(clean, k, 0.02, E, np.random.default_rng(seed + 41))
                c = flags_and_truth(clean[0], f1, E)
                ra.append(dict(stage=stage, seed=seed, noise=noise, sigma=sigma, k=str(k), wrong=wrong,
                               clean_mean_resid=cc["resid_sum"] / max(cc["n"], 1), **c,
                               **{f"clean_{f}": cc[f] for f in FLAGS}))
            for level, (pm, pf) in LEVELS.items():
                det = hcs.synth(gt, sp, pm, pf, rng)
                F = link_frames(det, 1.0 * sp, "lap").frames
                c = flags_and_truth(F[0], F[1], E)
                swp, chn = rmt.switch_shapes(F[0], F[1], E)
                rb.append(dict(stage=stage, seed=seed, noise=noise, sigma=sigma, level=level, swaps=swp, chains=chn, **c))
    print(f"{stage} seed {seed}: done", flush=True)
    return ra, rb


def sim_conditions(b: pd.DataFrame) -> pd.DataFrame:
    g = b.groupby(["noise", "sigma", "level"]).sum(numeric_only=True).reset_index()
    g["mean_resid"] = g.resid_sum / g.n
    g["share05"] = g.h05 / g.n
    for f in FLAGS:
        g[f"cap_{f}"] = g[f"{f}_tp"] / g.sw.clip(lower=1)
        g[f"false_{f}"] = (g[f] - g[f"{f}_tp"]) / (g.n - g.sw)
    return g[g.sw >= 30]


def real_trackers() -> pd.DataFrame:
    hv = pd.read_csv(RES / "hop_count_variants.csv", dtype={"seq": str})
    pr = pd.read_csv(RES / "detector_tracking_pairs_resid.csv", dtype={"seq": str})
    keys = ["seq", "img", "op", "stride", "cap"]
    res = pr.groupby(keys).trk_resid.mean().rename("mean_resid").reset_index()
    d = hv.merge(res, on=keys)
    d = d[d.cap == 1.0].copy()
    d["share05"] = d.H05 / d.n
    d = d.rename(columns={"Hwin": "win", "Hnb_any": "nb_any", "Hnb_persist": "nb_persist", "H05": "h05"})
    return d


def calibrate_real(sim: pd.DataFrame, real: pd.DataFrame) -> pd.DataFrame:
    X = sim[["mean_resid", "share05"]].to_numpy(); scale = X.std(0)
    tree = cKDTree(X / scale)
    _, nn = tree.query(real[["mean_resid", "share05"]].to_numpy() / scale, k=K_NN)
    out = real.copy()
    for f in ("win", "nb_any", "nb_persist"):
        c = sim[f"cap_{f}"].to_numpy()[nn].mean(1); fr = sim[f"false_{f}"].to_numpy()[nn].mean(1)
        out[f"cap_{f}"] = c; out[f"false_{f}"] = fr
        out[f"est_{f}"] = 100 * (out[f] - fr * out.n) / (c * out.n)
    out["true"] = 100 * out.sw / out.n
    out["sim_dist"] = tree.query(real[["mean_resid", "share05"]].to_numpy() / scale, k=1)[0]
    return out


def summary() -> None:
    import matplotlib.pyplot as plt
    import figstyle as fs
    a = pd.read_csv(RES / "sim_hop_map2_A.csv"); b = pd.read_csv(RES / "sim_hop_map2_B.csv")
    pd.set_option("display.width", 230)
    print("### A  injected NEAREST-BODY chains: fraction of wrong links caught (TP / wrong), pooled over noise levels <= 0.13")
    lo = a[a.sigma <= 0.13]
    t = lo.groupby(["noise", "k"]).apply(lambda g: pd.Series({f: g[f"{f}_tp"].sum() / g.wrong.sum() for f in FLAGS}), include_groups=False)
    print(t.round(2).to_string())
    print("\n### B  LAP-made chains, link 1.0 spacing")
    sim = sim_conditions(b)
    print(f"switched cells: {int(b.swaps.sum())} in swaps, {int(b.chains.sum())} in chains; {len(sim)} conditions with >= 30 switched cells")
    cols = ["noise", "sigma", "level", "mean_resid", "share05", "sw", "cap_win", "false_win", "cap_nb_any", "false_nb_any", "cap_nb_persist", "false_nb_persist"]
    s = sim[cols].copy(); s["true_%"] = 100 * sim.sw / sim.n
    for c in ("share05", "false_win", "false_nb_any", "false_nb_persist"):
        s[c] = 100 * s[c]
    print(s.round(3).to_string(index=False))

    real = calibrate_real(sim, real_trackers())
    real.to_csv(RES / "sim_hop_calibrated_real.csv", index=False)
    print(f"\n### simulation-calibrated switch estimate on {len(real)} real LAP trackers (link 1.0)")
    print(f"real descriptors: mean residual {real.mean_resid.min():.3f}-{real.mean_resid.max():.3f}, "
          f"share > 0.5 {100 * real.share05.min():.2f}-{100 * real.share05.max():.2f}%;  "
          f"simulated: {sim.mean_resid.min():.3f}-{sim.mean_resid.max():.3f}, {100 * sim.share05.min():.2f}-{100 * sim.share05.max():.2f}%")
    for f in ("win", "nb_any", "nb_persist"):
        x, y = real.true, real[f"est_{f}"]
        print(f"  {f:10s} est mean {y.mean():.2f} vs true {x.mean():.2f}  slope {np.sum(x * y) / np.sum(x * x):.2f}  r {np.corrcoef(x, y)[0, 1]:.2f}  "
              f"median rel err {100 * np.median(np.abs(y - x) / x):.0f}%")
        for st in (1, 2, 3, 4):
            k = real.stride == st
            print(f"      interval {st}: est {y[k].mean():.2f} true {x[k].mean():.2f}  (sim capture {real[f'cap_{f}'][k].mean():.2f}, "
                  f"false {100 * real[f'false_{f}'][k].mean():.3f}%)")

    fs.apply()
    fig, axes = plt.subplots(1, 3, figsize=(fs.DOUBLE, 2.3), gridspec_kw=dict(left=0.07, right=0.99, top=0.86, bottom=0.21, wspace=0.38))
    ax = axes[0]
    kk = ["swap", "2", "3", "5", "8"]
    for f, c in (("win", "#2a78d6"), ("nb_any", "#eb6834"), ("nb_persist", "#1baf7a")):
        g = lo[lo.noise == "t3"].groupby("k").apply(lambda h: h[f"{f}_tp"].sum() / h.wrong.sum(), include_groups=False).reindex(kk)
        ax.plot(range(len(kk)), g.values, marker="o", ms=3, color=c, label={"win": "window", "nb_any": "nearest body, any", "nb_persist": "nearest body, existing"}[f])
    ax.set_xticks(range(len(kk))); ax.set_xticklabels(["swap"] + [f"k={k}" for k in kk[1:]]); ax.set_ylim(0, 1.05)
    ax.set_ylabel("Wrong links caught"); ax.set_title("Injected nearest-body chains", loc="left", pad=3, fontsize=7.5); ax.legend(fontsize=6)
    for ax, f, title in ((axes[1], "win", "Window count, sim-calibrated"), (axes[2], "nb_any", "Nearest body, sim-calibrated")):
        x, y = real.true, real[f"est_{f}"]
        lim = float(max(x.max(), y.max())) * 1.08
        ax.plot([0, lim], [0, lim], color=fs.AXIS, lw=0.8); ax.axhline(0, color=fs.AXIS, lw=0.5)
        for st, c in zip((1, 2, 3, 4), ("#2a78d6", "#1baf7a", "#eda100", "#eb6834")):
            k = real.stride == st
            ax.scatter(x[k], y[k], s=9, color=c, linewidths=0, alpha=0.85, label=f"every {st}")
        ax.set_title(title, loc="left", pad=3, fontsize=7.5); ax.set_xlabel("True switched, real LAP (%)")
        ax.text(0.04, 0.96, f"r = {np.corrcoef(x, y)[0, 1]:.2f}", transform=ax.transAxes, va="top", fontsize=7)
    axes[1].set_ylabel("Estimated (%)"); axes[1].legend(fontsize=6, loc="lower right", title="frame", title_fontsize=6)
    fs.save(fig, ROOT / "figures" / "local_metrics" / "sim_hop_map2")


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--stage", choices=("all", "summary"), default="all")
    a = p.parse_args()
    if a.stage == "all":
        em.SWITCH_CAP = 1.1; sweep.MODE = "delaunay"
        jobs = [(st, sd) for st in m1.STAGES for sd in SEEDS]
        ra, rb = [], []
        with ProcessPoolExecutor(a.workers) as ex:
            for x, y in ex.map(job, jobs):
                ra += x; rb += y
        pd.DataFrame(ra).to_csv(RES / "sim_hop_map2_A.csv", index=False)
        pd.DataFrame(rb).to_csv(RES / "sim_hop_map2_B.csv", index=False)
    summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
