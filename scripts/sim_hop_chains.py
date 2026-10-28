"""Hop chains in simulation: are they picked up?  Stage-matched slab tissue (worm geometry).

    python scripts/sim_hop_chains.py [--workers N] [--stage all|summary]

Motion per frame is scaled x1..x4 (speed of the simulated cells), mimicking linking every
1st..4th frame.  Detections carry 5% spacing jitter.

A  LAP-made chains: random misses and false positives (two levels, as in hop_count_synthetic),
   LAP at 0.5 and 1.0 median spacings.  Truth: switched cells, split into swaps and chains.
   Measured: window count (0.7-1.3 spacings), nearest-body test, raw count (> 0.5), their
   error-free (oracle) rates, and the per-switch response of RC, displaced share and the
   residual-outlier share (tracker minus oracle).  The window ratio (estimate / truth) is
   compared with the real data (0.70 worm detector, 0.73 ascidian synthetic).
B  injected chains of known length on clean jittered pairs (scripts/chain_injection.inject_chains):
   swap, k = 2, 3, 5, 8 cells, 2% wrong links.  Measured: fraction of wrong links caught by
   each count, and per-wrong-link response of RC and displaced share.

Writes data/_results/sim_hop_chains_A.csv, _B.csv and figures/local_metrics/sim_hop_chains.png.
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
import chain_injection as ci  # noqa: E402
import estimate_error_mix as em  # noqa: E402
import hop_count_synthetic as hcs  # noqa: E402
import hop_count_variants as hv  # noqa: E402
import run_motile_tracking as rmt  # noqa: E402
from tracking_validation.linking import link_frames  # noqa: E402

RES = ROOT / "data" / "_results"
STAGES = ("50-100", "100-200", "200-360")
SCALES = (1, 2, 3, 4)
SEEDS = range(6)
KS = ("swap", 2, 3, 5, 8)
CH = ("rc", "displaced", "resid_out")


def stage_means() -> pd.DataFrame:
    meas = pd.read_csv(RES / "ce_stage_measurements.csv")
    if meas.seq.astype(str).str.startswith("Ph").any():
        raise SystemExit("ce_stage_measurements.csv is not the worm's")
    return meas.groupby("stage")[["nn_med", "nn_p10", "disp_med", "disp_p90", "density", "z_extent"]].mean()


def jittered(pair, spacing, rng):
    out = []
    for f in pair:
        p = f["positions"] + rng.normal(0, hcs.JITTER * spacing, f["positions"].shape)
        out.append(dict(positions=p, true_ids=f["true_ids"].astype(int), assigned_ids=f["true_ids"].astype(int).copy(),
                        parent_ids=np.full(len(p), -1)))
    return out


def oracle_of(det, gt):
    out = []
    for f, g in zip(det, gt):
        real = f["true_ids"] >= 0
        have = set(int(c) for c in f["true_ids"][real])
        fill = np.array([int(c) not in have for c in g["true_ids"]], dtype=bool)
        ids = np.concatenate([f["true_ids"][real], g["true_ids"][fill]]).astype(int)
        out.append(dict(positions=np.vstack([f["positions"][real], g["positions"][fill]]), true_ids=ids, assigned_ids=ids))
    return out


def job(args):
    stage, scale, seed = args
    sm = stage_means().copy()
    sm.loc[stage, ["disp_med", "disp_p90"]] *= scale
    pair, box = em._sim_pair(sm, stage, 50_000 + 1000 * scale + 100 * STAGES.index(stage) + seed)
    gt = [dict(positions=f["positions"], true_ids=np.asarray(f["true_ids"], int)) for f in pair]
    spacing = sweep.median_nn(gt[0]["positions"])
    E = set(int(c) for c in gt[0]["true_ids"]) & set(int(c) for c in gt[1]["true_ids"])
    rng = np.random.default_rng(seed + 17 * scale)
    rows_a, rows_b = [], []
    # ---- A: LAP-made chains ----
    for level, (pm, pf) in hcs.LEVELS.items():
        det = hcs.synth(gt, spacing, pm, pf, rng)
        orc = oracle_of(det, gt)
        oc = hv.pair_counts(orc[0], orc[1], E)
        om = sweep.evaluate_pair(orc, 8, 15.0, None)
        for cap in (0.5, 1.0):
            F = link_frames(det, cap * spacing, "lap").frames
            c = hv.pair_counts(F[0], F[1], E)
            sw, chn = rmt.switch_shapes(F[0], F[1], E)
            m = sweep.evaluate_pair(F, 8, 15.0, None)
            rows_a.append(dict(stage=stage, scale=scale, seed=seed, level=level, cap=cap, swaps=sw, chains=chn,
                               **{k: c[k] for k in ["n", "sw", *hv.COUNTS]},
                               **{f"orc_{k}": oc[k] for k in ["n", *hv.COUNTS]},
                               **{f"d_{ch}": m[ch] - om[ch] for ch in CH}))
    # ---- B: injected chains of known length ----
    clean = jittered(gt, spacing, rng)
    cc = hv.pair_counts(clean[0], clean[1], E)
    cm = sweep.evaluate_pair(clean, 8, 15.0, None)
    for k in KS:
        f1, wrong = ci.inject_chains(clean, k, 0.02, sorted(E), np.random.default_rng(seed + 31), 15.0)
        cor = [clean[0], f1]
        c = hv.pair_counts(cor[0], cor[1], E)
        m = sweep.evaluate_pair(cor, 8, 15.0, None)
        rows_b.append(dict(stage=stage, scale=scale, seed=seed, k=str(k), wrong=wrong, n=c["n"], sw=c["sw"],
                           **{f"d_{h}": c[h] - cc[h] for h in hv.COUNTS},
                           **{f"d_{ch}": m[ch] - cm[ch] for ch in CH}))
    print(f"{stage} x{scale} seed {seed}: done", flush=True)
    return rows_a, rows_b


def summary() -> None:
    import matplotlib.pyplot as plt
    import figstyle as fs
    a = pd.read_csv(RES / "sim_hop_chains_A.csv"); b = pd.read_csv(RES / "sim_hop_chains_B.csv")
    pd.set_option("display.width", 220)
    print("### A  LAP on simulated tissue with detection errors (pooled over stages, seeds, error levels)")
    g = a.groupby(["cap", "scale"]).sum(numeric_only=True)
    t = pd.DataFrame({"true_%": 100 * g.sw / g.n, "chains_share": g.chains / (g.swaps + g.chains).clip(lower=1),
                      "win/true": g.Hwin / g.sw, "nb/true": g.Hnb_persist / g.sw, "h05/true": g.H05 / g.sw,
                      "orc_win_%": 100 * g.orc_Hwin / g.orc_n, "orc_h05_%": 100 * g.orc_H05 / g.orc_n})
    print(t.round(3).to_string())
    h = a[a.cap == 1.0]
    x = 100 * h.sw / h.n
    for k, name in (("Hwin", "window"), ("Hnb_persist", "nearest body")):
        y = 100 * h[k] / h.n
        print(f"  link 1.0: {name:12s} ratio {h[k].sum() / h.sw.sum():.2f}  r {np.corrcoef(x, y)[0, 1]:.2f}")
    rate = h.sw / h.n
    for ch in CH:
        s = np.sum(rate * h[f"d_{ch}"]) / np.sum(rate ** 2)
        print(f"  per unit switch rate, {ch:9s}: {s:+.3f}   (swap-calibrated simulation: rc -0.74, displaced +0.68, resid_out +1.00;"
              f" real LAP: rc -0.04, displaced +0.22)" if ch == "rc" else f"  per unit switch rate, {ch:9s}: {s:+.3f}")
    print(f"  switched cells: {int(a.swaps.sum())} in swaps, {int(a.chains.sum())} in chains")

    print("\n### B  injected chains of known length (2% wrong links), per wrong link")
    b["wr"] = b.wrong
    rows = []
    for (k, scale), g in b.groupby(["k", "scale"]):
        w = g.wr.sum()
        rows.append(dict(k=k, scale=scale, window=g.d_Hwin.sum() / w, nearest=g.d_Hnb_persist.sum() / w, h05=g.d_H05.sum() / w,
                         rc=np.sum(g.d_rc) / np.sum(g.wr / g.n), displaced=np.sum(g.d_displaced) / np.sum(g.wr / g.n)))
    bt = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate(KS)}
    bt = bt.sort_values(["scale", "k"], key=lambda s: s.map(lambda v: order.get(v if v == "swap" else int(v), 0)) if s.name == "k" else s)
    print(bt.round(3).to_string(index=False))

    fs.apply()
    fig, axes = plt.subplots(1, 3, figsize=(fs.DOUBLE, 2.2), gridspec_kw=dict(left=0.07, right=0.99, top=0.86, bottom=0.22, wspace=0.4))
    cols = {1: "#2a78d6", 2: "#1baf7a", 3: "#eda100", 4: "#eb6834"}
    ax = axes[0]
    y = 100 * h.Hwin / h.n
    lim = float(max(x.max(), y.max())) * 1.08
    ax.plot([0, lim], [0, lim], color=fs.AXIS, lw=0.8)
    for s, col in cols.items():
        m = h.scale == s
        ax.scatter(x[m], y[m], s=10, color=col, linewidths=0, alpha=0.85, label=f"motion ×{s}")
    ax.set_title("LAP-made chains: window count", loc="left", pad=3); ax.set_xlabel("True switched (%)"); ax.set_ylabel("Estimated (%)")
    ax.legend(fontsize=6, loc="upper left")
    xs = np.arange(len(KS)); labels = [str(k) for k in KS]
    for ax, cols_, title, ylab in ((axes[1], ("window", "nearest"), "Injected chains: caught", "Caught per wrong link"),
                                   (axes[2], ("rc", "displaced"), "Injected chains: response", "Change per unit rate")):
        for c, col in zip(cols_, ("#2a78d6", "#eb6834")):
            g = bt[bt.scale == 1].set_index("k").reindex(labels)
            ax.plot(xs, g[c], marker="o", color=col, label=c)
        ax.set_xticks(xs); ax.set_xticklabels(["swap" if k == "swap" else f"k={k}" for k in KS])
        ax.axhline(0, color=fs.AXIS, lw=0.6); ax.set_title(title, loc="left", pad=3); ax.set_ylabel(ylab)
        ax.legend(fontsize=6.5)
    fs.save(fig, ROOT / "figures" / "local_metrics" / "sim_hop_chains")


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--stage", choices=("all", "summary"), default="all")
    a = p.parse_args()
    if a.stage == "all":
        em.SWITCH_CAP = 1.1; sweep.MODE = "delaunay"
        jobs = [(st, sc, sd) for st in STAGES for sc in SCALES for sd in SEEDS]
        ra, rb = [], []
        with ProcessPoolExecutor(a.workers) as ex:
            for x, y in ex.map(job, jobs):
                ra += x; rb += y
        pd.DataFrame(ra).to_csv(RES / "sim_hop_chains_A.csv", index=False)
        pd.DataFrame(rb).to_csv(RES / "sim_hop_chains_B.csv", index=False)
    summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
