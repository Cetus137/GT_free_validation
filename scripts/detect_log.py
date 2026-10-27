"""LoG nucleus detection on the C. elegans CTC images, tuned on sequence 01, tested on 02.

    python scripts/detect_log.py              # all stages
    python scripts/detect_log.py --stage sweep   # reuse cached candidates

Stages
  cands  multi-scale LoG candidates for every GT frame of 01 and 02 -> data/_cache/log_cands_{seq}.npz
  zoff   constant z offset between image and GT markers, from 01 (detections are shifted by it)
  sweep  threshold x minimum-separation grid on 01 -> data/_results/detection_sweep.csv;
         picks three operating points (balanced = best F1; high recall; high precision)
  run    detections for 01 and 02 at each operating point -> data/_results/detections/ce_{seq}_{op}.csv
         and per-frame counts -> data/_results/detection_quality.csv
  --noise L / --blur-z B  degrade the images first (noise sd = L x signal range; z blur sd = B um);
         files get a _nL / _bzB tag
  plots  figures/detection/: recall/precision vs frame, match distances, slice overlays
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from tracking_validation import detection as D  # noqa: E402
from tracking_validation.ctc import _to_xyz, find_frame_files, label_centroids  # noqa: E402

DATA = ROOT / "data" / "Fluo-N3DH-CE"
CACHE = ROOT / "data" / "_cache"
RES = ROOT / "data" / "_results"
FIG = ROOT / "figures" / "detection"
VOXEL = (0.09, 0.09, 1.0)
SEQS = ("01", "02")
THRESHOLDS = np.round(np.geomspace(0.04, 0.5, 16), 4)
SEPS = (1.2, 1.5, 2.0)
FLOOR = 0.03          # candidates below this response are never kept
MIN_CELLS = 50        # frames with fewer GT cells are outside the analysis window and are skipped
BUDGET = 0.05        # max miss or FP rate (fraction of GT cells) for the skewed operating points


# --------------------------------------------------------------------------- candidates

# Image degradations used to make the detector err more (thresholds stay the clean-tuned ones):
#   NOISE   Gaussian noise, sd as a fraction of the frame's signal range (p99.9 - p50) -> mostly false positives
#   BLUR_Z  Gaussian blur along z, sd in microns -> nuclei stacked in z merge -> mostly misses
NOISE = 0.0
BLUR_Z = 0.0


def SUFFIX() -> str:
    """'' for the clean images, e.g. '_n0.4', '_bz1', '_n0.4_bz0.7' for degraded ones."""
    return (f"_n{NOISE:g}" if NOISE else "") + (f"_bz{BLUR_Z:g}" if BLUR_Z else "")


def tag(seq: str) -> str:
    return seq + SUFFIX()


def degrade(im: np.ndarray, noise: float, blur_z: float, seed: int) -> np.ndarray:
    """Blur along z (sd ``blur_z`` microns), then add Gaussian noise with sd = noise x (p99.9 - p50)."""
    if noise <= 0 and blur_z <= 0:
        return im
    a = im.astype(np.float32)
    if blur_z > 0:
        from scipy import ndimage as ndi
        a = ndi.gaussian_filter(a, (blur_z / VOXEL[2], 0, 0), mode="nearest")
    if noise > 0:
        lo, hi = np.percentile(a, [50, 99.9])
        a += np.random.default_rng(seed).normal(0.0, noise * (hi - lo), a.shape).astype(np.float32)
    return np.clip(a, 0, 65535)


def _frame_job(args):
    seq, t, noise, blur_z = args
    import tifffile
    im = degrade(tifffile.imread(DATA / seq / f"t{t:03d}.tif"), noise, blur_z, seed=1000 * int(seq) + t)
    g = tifffile.imread(DATA / f"{seq}_GT" / "TRA" / f"man_track{t:03d}.tif")
    ids, com = label_centroids(g)
    c = D.candidates(im, VOXEL, floor=FLOOR)
    return t, c, ids, _to_xyz(com, VOXEL)


def stage_cands(workers: int) -> None:
    for seq in SEQS:
        frames = sorted(find_frame_files(DATA / f"{seq}_GT" / "TRA"))
        with ProcessPoolExecutor(workers) as ex:
            out = list(ex.map(_frame_job, [(seq, t, NOISE, BLUR_Z) for t in frames]))
        cf, cp, cr, cs, gf, gi, gp = [], [], [], [], [], [], []
        for t, c, ids, gt in out:
            cf.append(np.full(len(c.pos), t)); cp.append(c.pos); cr.append(c.response); cs.append(c.sigma)
            gf.append(np.full(len(ids), t)); gi.append(ids); gp.append(gt)
        np.savez_compressed(CACHE / f"log_cands_{tag(seq)}.npz", frame=np.concatenate(cf), pos=np.concatenate(cp),
                            response=np.concatenate(cr), sigma=np.concatenate(cs),
                            gt_frame=np.concatenate(gf), gt_id=np.concatenate(gi), gt_pos=np.concatenate(gp))
        print(f"{seq}: {len(frames)} frames, {sum(len(o[1].pos) for o in out)} candidates", flush=True)


def load(seq: str, zoff: float = 0.0):
    z = np.load(CACHE / f"log_cands_{tag(seq)}.npz")
    frames = np.unique(z["gt_frame"])
    per = {}
    for t in frames:
        m = z["frame"] == t; g = z["gt_frame"] == t
        if g.sum() < MIN_CELLS:       # same window as every other analysis (ce_error_sweep --min-cells)
            continue
        pos = z["pos"][m].copy(); pos[:, 2] -= zoff
        per[int(t)] = (D.Candidates(pos=pos, response=z["response"][m], sigma=z["sigma"][m]),
                       z["gt_id"][g], z["gt_pos"][g])
    return per


# --------------------------------------------------------------------------- z offset

def stage_zoff() -> float:
    from scipy.spatial import cKDTree
    per = load("01")
    dz = []
    for t, (c, ids, gt) in per.items():
        k = D.select(c, 0.1, 1.5); det = c.pos[k]
        if len(det) == 0:
            continue
        d, i = cKDTree(det[:, :2]).query(gt[:, :2])
        ok = d < 0.8
        dz.append(det[i[ok], 2] - gt[ok, 2])
    dz = np.concatenate(dz)
    off = float(np.median(dz))
    (RES / "detection_zoffset.json").write_text(json.dumps(dict(z_offset_um=off, n=int(dz.size),
                                                                p10=float(np.percentile(dz, 10)), p90=float(np.percentile(dz, 90)))))
    print(f"z offset (detection - GT), sequence 01: median {off:+.2f} µm over {dz.size} xy-matched cells")
    return off


def zoffset() -> float:
    return json.loads((RES / "detection_zoffset.json").read_text())["z_offset_um"]


# --------------------------------------------------------------------------- sweep

def score(per, thr, sep):
    rows = []
    for t, (c, ids, gt) in per.items():
        k = D.select(c, thr, sep)
        m, _ = D.match_to_gt(c.pos[k], gt, D.match_cap(gt))
        tp = int((m >= 0).sum())
        rows.append(dict(frame=t, n_gt=len(gt), n_det=len(k), tp=tp, fp=len(k) - tp, fn=len(gt) - tp))
    return pd.DataFrame(rows)


def _summ(q: pd.DataFrame) -> dict:
    tp, fp, fn = q.tp.sum(), q.fp.sum(), q.fn.sum()
    rec, prec = tp / (tp + fn), tp / max(tp + fp, 1)
    return dict(recall=rec, precision=prec, f1=2 * rec * prec / (rec + prec),
                miss_rate=fn / (tp + fn), fp_rate=fp / (tp + fn))


def stage_sweep() -> dict:
    per = load("01", zoffset())
    rows = []
    for sep in SEPS:
        for thr in THRESHOLDS:
            rows.append(dict(threshold=thr, min_sep=sep, **_summ(score(per, thr, sep))))
    sw = pd.DataFrame(rows); sw.to_csv(RES / "detection_sweep.csv", index=False)
    best = sw.loc[sw.f1.idxmax()]
    same = sw[sw.min_sep == best.min_sep].sort_values("threshold")
    # error budgets keep every operating point inside the estimator's calibrated range (<= 5%)
    hi_rec = same[same.fp_rate <= BUDGET]
    hi_rec = hi_rec.iloc[0] if len(hi_rec) else best                    # lowest threshold with FP <= budget
    hi_pre = same[same.miss_rate <= BUDGET]
    hi_pre = hi_pre.iloc[-1] if len(hi_pre) else best                   # highest threshold with misses <= budget
    ops = {name: dict(threshold=float(r.threshold), min_sep=float(r.min_sep))
           for name, r in (("balanced", best), ("high_recall", hi_rec), ("high_precision", hi_pre))}
    (RES / "detection_operating_points.json").write_text(json.dumps(ops, indent=1))
    pd.set_option("display.width", 200); pd.set_option("display.precision", 4)
    print("### sweep on 01, best separation:"); print(same.to_string(index=False))
    print("### operating points:", json.dumps(ops))
    return ops


# --------------------------------------------------------------------------- run

def stage_run() -> None:
    ops = json.loads((RES / "detection_operating_points.json").read_text())
    off = zoffset()
    out = RES / "detections"; out.mkdir(exist_ok=True)
    qrows, summ = [], []
    for seq in SEQS:
        per = load(seq, off)
        for name, p in ops.items():
            det_rows = []
            for t, (c, ids, gt) in per.items():
                k = D.select(c, p["threshold"], p["min_sep"])
                m, dist = D.match_to_gt(c.pos[k], gt, D.match_cap(gt))
                lab = np.where(m >= 0, ids[np.maximum(m, 0)], -1)
                for j, i in enumerate(k):
                    det_rows.append(dict(frame=t, x=c.pos[i, 0], y=c.pos[i, 1], z=c.pos[i, 2], sigma=c.sigma[i],
                                         response=c.response[i], true_id=int(lab[j]), match_dist=dist[j]))
                tp = int((m >= 0).sum())
                qrows.append(dict(seq=seq, op=name, frame=t, n_gt=len(gt), n_det=len(k), tp=tp,
                                  fp=len(k) - tp, fn=len(gt) - tp))
            pd.DataFrame(det_rows).to_csv(out / f"ce_{tag(seq)}_{name}.csv", index=False)
            q = pd.DataFrame([r for r in qrows if r["seq"] == seq and r["op"] == name])
            summ.append(dict(seq=seq, op=name, **p, **_summ(q)))
    pd.DataFrame(qrows).to_csv(RES / f"detection_quality{SUFFIX()}.csv", index=False)
    s = pd.DataFrame(summ); s.to_csv(RES / f"detection_summary{SUFFIX()}.csv", index=False)
    print(f"### detection quality (z offset {off:+.2f} µm applied; 01 = tuning, 02 = test)")
    print(s.round(4).to_string(index=False))


# --------------------------------------------------------------------------- plots

def stage_plots() -> None:
    import matplotlib.pyplot as plt
    import tifffile
    import figstyle as fs
    fs.apply()
    FIG.mkdir(parents=True, exist_ok=True)
    q = pd.read_csv(RES / "detection_quality.csv", dtype={"seq": str})
    colors = {"high_recall": "#2a78d6", "balanced": fs.INK, "high_precision": "#eb6834"}
    names = {"high_recall": "High recall", "balanced": "Balanced", "high_precision": "High precision"}

    # miss and false-positive rate per frame (as % of GT cells), both sequences
    fig, axes = plt.subplots(2, 2, figsize=(fs.DOUBLE, 3.6), sharex=True, sharey="row",
                             gridspec_kw=dict(left=0.08, right=0.99, top=0.9, bottom=0.12, hspace=0.15, wspace=0.08))
    for c, seq in enumerate(SEQS):
        for op in colors:
            g = q[(q.seq == seq) & (q.op == op)].sort_values("frame")
            sm = lambda v: pd.Series(v).rolling(5, center=True, min_periods=1).mean()
            axes[0, c].plot(g.frame, sm(100 * g.fn / g.n_gt), color=colors[op], lw=1.2, label=names[op])
            axes[1, c].plot(g.frame, sm(100 * g.fp / g.n_gt), color=colors[op], lw=1.2)
        axes[0, c].set_title(f"Sequence {seq}" + (" (tuning)" if seq == "01" else " (test)"), loc="left")
        axes[1, c].set_xlabel("Frame")
    axes[0, 0].set_ylabel("Missed (% cells)"); axes[1, 0].set_ylabel("False pos. (% cells)")
    axes[0, 1].legend(loc="upper left")
    fs.save(fig, FIG / "rates_vs_frame")

    # match distances by axis, balanced point
    fig, ax = plt.subplots(figsize=(fs.SINGLE, 2.2), gridspec_kw=dict(left=0.16, right=0.97, top=0.95, bottom=0.2))
    off = zoffset()
    for seq, ls in (("01", "-"), ("02", "--")):
        d = pd.read_csv(RES / "detections" / f"ce_{seq}_balanced.csv")
        d = d[d.true_id >= 0]
        z = np.load(CACHE / f"log_cands_{seq}.npz")
        gt = pd.DataFrame(dict(frame=z["gt_frame"], true_id=z["gt_id"], gx=z["gt_pos"][:, 0], gy=z["gt_pos"][:, 1], gz=z["gt_pos"][:, 2]))
        m = d.merge(gt, on=["frame", "true_id"])
        bins = np.linspace(-2, 2, 61)
        ax.hist(m.x - m.gx, bins, histtype="step", color="#2a78d6", ls=ls, label=f"x, seq {seq}")
        ax.hist(m.z - m.gz, bins, histtype="step", color="#eb6834", ls=ls, label=f"z, seq {seq}")
    ax.set_xlabel("Detection − GT (µm)"); ax.set_ylabel("Cells"); ax.legend(fontsize=6.5)
    fs.save(fig, FIG / "match_offsets")

    # overlays on the test sequence: early, middle, late frame
    det = pd.read_csv(RES / "detections" / "ce_02_balanced.csv")
    z = np.load(CACHE / "log_cands_02.npz")
    fig, axes = plt.subplots(1, 3, figsize=(fs.DOUBLE, 2.6), gridspec_kw=dict(left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.03))
    for ax, t in zip(axes, (80, 130, 185)):
        im = tifffile.imread(DATA / "02" / f"t{t:03d}.tif").astype(float)
        gt = z["gt_pos"][z["gt_frame"] == t]; gid = z["gt_id"][z["gt_frame"] == t]
        zc = float(np.median(gt[:, 2])); zi = int(round(zc / VOXEL[2]))
        ax.imshow(im[max(zi - 1, 0):zi + 2].max(0), cmap="gray", vmin=np.percentile(im, 50), vmax=np.percentile(im, 99.8),
                  extent=(0, im.shape[2] * VOXEL[0], im.shape[1] * VOXEL[1], 0))
        d = det[det.frame == t]; near = lambda p: np.abs(p - zc) <= 1.5
        found = set(d.true_id[d.true_id >= 0])
        g_in = near(gt[:, 2])
        hit = np.array([i in found for i in gid])
        ax.scatter(gt[g_in & hit, 0], gt[g_in & hit, 1], s=22, facecolors="none", edgecolors="#1baf7a", linewidths=0.8)
        ax.scatter(gt[g_in & ~hit, 0], gt[g_in & ~hit, 1], s=40, facecolors="none", edgecolors="#e34948", linewidths=1.2)
        fp = d[(d.true_id < 0) & near(d.z)]
        ax.scatter(fp.x, fp.y, s=28, marker="x", color="#eda100", linewidths=1.2)
        ax.set_title(f"Frame {t}, z ≈ {zc:.0f} µm", loc="left"); ax.set_axis_off()
    fs.save(fig, FIG / "overlays_seq02")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("all", "cands", "zoff", "sweep", "run", "plots"), default="all")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--noise", type=float, default=0.0,
                   help="add Gaussian noise (sd = this x frame signal range) before detection; with noise > 0 "
                        "only the cands and run stages make sense (thresholds and z offset stay the clean ones)")
    p.add_argument("--blur-z", type=float, default=0.0, help="blur along z first (sd in microns); same rules as --noise")
    a = p.parse_args()
    global NOISE, BLUR_Z
    NOISE, BLUR_Z = a.noise, a.blur_z
    CACHE.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
    if a.stage in ("all", "cands"):
        stage_cands(a.workers)
    if a.stage in ("all", "zoff"):
        stage_zoff()
    if a.stage in ("all", "sweep"):
        stage_sweep()
    if a.stage in ("all", "run"):
        stage_run()
    if a.stage in ("all", "plots"):
        stage_plots()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
