"""Hop-count variants for switches on the real LAP trackers (208 configurations).

    python scripts/hop_count_variants.py [--workers N] [--stage all|summary]

For every persisting, evaluated label (division-free cells, as everywhere) in each frame pair:
  r     motion residual: |own step - median step of retained Delaunay neighbours| / median spacing
  pred  predicted t+1 position = own t position + median neighbour step

Hop counts per tracker (percent of evaluated persisting cells):
  H05         r > 0.5                                   (the raw count tested before)
  Hwin        0.7 <= r <= 1.3                           (window around one spacing)
  H05_excess  H05 minus the count expected from a Maxwell bulk fitted to the pair's own
              residual median (fit to the lower half; 3D residual magnitude)
  Hwin_excess Hwin minus the same bulk expectation in the window
  Hnb_any     the detection nearest to pred is not the label's own body
  Hnb_persist ... and that detection carries a label that already existed at t
Each count is also made on the error-free (oracle) tracks of the same detections: the
false-hop rate from genuine motion.  "oracle-corrected" = tracker count - oracle rate x N
(uses GT: checks whether the bias is entirely baseline).

Writes data/_results/hop_count_variants.csv and figures/local_metrics/hop_count_variants.png.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import maxwell

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import run_detector_tracking as rdt  # noqa: E402
from tracking_validation.linking import link_frames  # noqa: E402
from tracking_validation.local_metrics import predicted_positions  # noqa: E402

RES = ROOT / "data" / "_results"
COUNTS = ("H05", "Hwin", "H05_excess", "Hwin_excess", "Hnb_any", "Hnb_persist")
MAXWELL_MEDIAN = maxwell.median()          # median of the unit-scale Maxwell distribution


def pair_counts(f0, f1, E) -> dict:
    l0 = [int(l) for l in f0["assigned_ids"]]; l1 = [int(l) for l in f1["assigned_ids"]]
    out = dict.fromkeys(COUNTS, 0.0); out.update(n=0, sw=0, nb_tp=0, nb_fp=0)
    if len(l0) < 5 or len(l1) < 5:
        return out
    s0 = set(l0)
    pred, step, sp = predicted_positions(f0, f1, rdt.NEIGH_CAP)
    p0 = dict(zip(l0, f0["positions"]))
    tree1 = cKDTree(f1["positions"])
    tid0 = dict(zip(l0, f0["true_ids"].tolist())); tid1 = dict(zip(l1, f1["true_ids"].tolist()))
    rs, sws, nb_any, nb_per = [], [], [], []
    for l, s in step.items():
        a, b = tid0[l], tid1[l]
        if a < 0 or b < 0 or a not in E:
            continue
        rs.append(float(np.linalg.norm(p0[l] + s - pred[l])) / sp)
        sws.append(a != b)
        j = int(tree1.query(pred[l])[1])
        other = l1[j] != l
        nb_any.append(other); nb_per.append(other and l1[j] in s0)
    if not rs:
        return out
    r = np.array(rs); sw = np.array(sws); nb_any = np.array(nb_any); nb_per = np.array(nb_per)
    n = len(r)
    win = (r >= 0.7) & (r <= 1.3)
    sigma = max(np.median(r), 1e-6) / MAXWELL_MEDIAN
    bulk = maxwell(scale=sigma)
    out.update(n=n, sw=int(sw.sum()), H05=float((r > 0.5).sum()), Hwin=float(win.sum()),
               H05_excess=float((r > 0.5).sum() - n * bulk.sf(0.5)),
               Hwin_excess=float(win.sum() - n * (bulk.cdf(1.3) - bulk.cdf(0.7))),
               Hnb_any=float(nb_any.sum()), Hnb_persist=float(nb_per.sum()),
               nb_tp=int((nb_per & sw).sum()), nb_fp=int((nb_per & ~sw).sum()))
    return out


def job(args):
    seq, img, op, stride = args
    label_map, gt, fidx, keep, spacing = rdt.load_gt(seq, stride)
    det = rdt.detection_frames(seq, img, op, label_map, fidx)
    oracle = []
    for f, g in zip(det, gt):
        real = f["true_ids"] >= 0
        have = set(int(c) for c in f["true_ids"][real])
        fill = np.array([int(c) not in have for c in g["true_ids"]], dtype=bool)
        ids = np.concatenate([f["true_ids"][real], g["true_ids"][fill]])
        oracle.append(dict(positions=np.vstack([f["positions"][real], g["positions"][fill]]), true_ids=ids, assigned_ids=ids))
    orc = pd.DataFrame([pair_counts(oracle[t], oracle[t + 1], keep[t] & set(int(c) for c in gt[t + 1]["true_ids"]))
                        for t in range(len(gt) - 1)]).sum()
    rows = []
    for cap in rdt.CAPS:
        F = link_frames(det, cap * spacing, "lap").frames
        trk = pd.DataFrame([pair_counts(F[t], F[t + 1], keep[t] & set(int(c) for c in gt[t + 1]["true_ids"]))
                            for t in range(len(gt) - 1)]).sum()
        rows.append(dict(seq=seq, img=img.lstrip("_") or "clean", op=op, stride=stride, cap=cap,
                         **{k: trk[k] for k in ["n", "sw", "nb_tp", "nb_fp", *COUNTS]},
                         **{f"orc_{k}": orc[k] for k in ["n", *COUNTS]}))
    print(f"{seq} {img or 'clean':10s} {op:14s} stride {stride}: done", flush=True)
    return rows


def summary() -> None:
    import matplotlib.pyplot as plt
    import figstyle as fs
    d = pd.read_csv(RES / "hop_count_variants.csv", dtype={"seq": str})
    x = 100 * d.sw / d.n
    print(f"{len(d)} trackers; true switched cells {x.mean():.3f}% (mean)\n")
    print("false-hop rate on error-free tracks (percent of cells), by frame interval:")
    o = pd.DataFrame({k: 100 * d[f"orc_{k}"] / d.orc_n for k in COUNTS}); o["stride"] = d.stride
    print(o.groupby("stride").mean().round(3).to_string())
    print("\nestimate vs truth (mean est / mean true / slope / r), overall and per interval:")
    rows = []
    for k in COUNTS:
        for corr in ("raw", "oracle-corrected"):
            y = 100 * d[k] / d.n
            if corr == "oracle-corrected":
                y = y - 100 * d[f"orc_{k}"] / d.orc_n
            r = dict(count=k, ref=corr, est=y.mean(), true=x.mean(), slope=np.sum(x * y) / np.sum(x * x), r=np.corrcoef(x, y)[0, 1])
            for st in (1, 2, 3, 4):
                m = d.stride == st
                r[f"ratio_{st}"] = y[m].mean() / x[m].mean(); r[f"r_{st}"] = np.corrcoef(x[m], y[m])[0, 1]
            rows.append(r)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    tp, fp = d.nb_tp.sum(), d.nb_fp.sum()
    print(f"\nnearest-body test (persist): flags {int(tp + fp)} cells, {100 * tp / max(tp + fp, 1):.0f}% truly switched; "
          f"catches {100 * tp / d.sw.sum():.0f}% of switched cells")

    fs.apply()
    show = [("H05", "raw", "Residual > 0.5"), ("Hwin_excess", "raw", "Window minus fitted bulk"),
            ("Hnb_persist", "raw", "Nearest body to prediction"), ("H05", "oracle-corrected", "Residual > 0.5, error-free\nrate subtracted (uses GT)")]
    fig, axes = plt.subplots(1, 4, figsize=(fs.DOUBLE, 2.15), sharex=True, gridspec_kw=dict(left=0.07, right=0.99, top=0.82, bottom=0.22, wspace=0.35))
    for ax, (k, corr, title) in zip(axes, show):
        y = 100 * d[k] / d.n - (100 * d[f"orc_{k}"] / d.orc_n if corr == "oracle-corrected" else 0)
        lim = max(float(x.max()), float(y.max())) * 1.08
        ax.plot([0, lim], [0, lim], color=fs.AXIS, lw=0.8); ax.axhline(0, color=fs.AXIS, lw=0.5)
        for st, col in zip((1, 2, 3, 4), ("#2a78d6", "#1baf7a", "#eda100", "#eb6834")):
            m = d.stride == st
            ax.scatter(x[m], y[m], s=8, color=col, linewidths=0, alpha=0.8, label=f"every {st}")
        ax.set_title(title, loc="left", pad=3, fontsize=7.5); ax.set_xlabel("True switched (%)")
        ax.text(0.04, 0.96, f"r = {np.corrcoef(x, y)[0, 1]:.2f}", transform=ax.transAxes, va="top", fontsize=7)
    axes[0].set_ylabel("Estimated (%)"); axes[0].legend(fontsize=6, loc="lower right", title="frame", title_fontsize=6)
    fs.save(fig, ROOT / "figures" / "local_metrics" / "hop_count_variants")


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--stage", choices=("all", "summary"), default="all")
    a = p.parse_args()
    if a.stage == "all":
        jobs = [(s, img, op, st) for s in ("01", "02") for img in rdt.IMAGES for op in rdt.OPS for st in rdt.STRIDES
                if not ("_n" in img and op == "high_recall")]
        with ProcessPoolExecutor(a.workers) as ex:
            rows = [r for part in ex.map(job, jobs) for r in part]
        pd.DataFrame(rows).to_csv(RES / "hop_count_variants.csv", index=False)
    summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
