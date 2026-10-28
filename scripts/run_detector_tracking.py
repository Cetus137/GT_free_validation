"""Real tracker on real data: LoG detections (scripts/detect_log.py) linked by LAP, errors
counted against the CTC ground truth, and the metric-based estimator scored against them.

    python scripts/run_detector_tracking.py                  # link + count + metrics, then estimate
    python scripts/run_detector_tracking.py --stage estimate  # reuse the per-pair table

Trackers (per sequence):
  images      clean, and degraded before detection (detect_log.py --noise / --blur-z):
              n0.4 (noise -> false positives), bz0.7 and bz1 (z blur -> misses), n0.4_bz0.7
  detector    3 operating points (high recall is skipped on noisy images: 40-110% false positives)
  interval    every 1st..4th frame linked (larger steps -> more linking errors)
  LAP cap     max link distance 0.5 or 1.0 median spacings (no gap closing: a bridged track's
              label is still absent from the frame before, so frame-pair metrics cannot see it)

Per frame pair it stores the GT error counts (tracking_eval) and the metric channels for
  tracker  the tracker's output
  oracle   detections with their GT identity, missed cells filled in at GT positions, false
           detections removed (error-free tracks carrying only detector localisation jitter)
  gt       GT positions and identities
all restricted to cells whose neighbourhood touches no GT division (as in the sweeps).

Truth used for scoring (pooled over pairs; a frame-pair metric sees a cell that is missed and
later found again as one broken track, i.e. a fragmentation):
  frag = frag + reappear        miss = miss - reappear (cells lost for good)        fp = fp
  switch = switch

Estimate stage: per tracker, pooled over pairs, the estimator with the response matrix of the
matching frame interval (real-embryo sweeps; simulation sweep at interval 1) is run on
  tracker - oracle     (jitter-matched clean reference; uses GT, so a diagnostic of demixing)
  tracker - baseline   (the calibration's own clean baseline: what a user would have)

Writes data/_results/detector_tracking_pairs.csv and detector_tracking_estimates.csv.
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
from scipy.spatial import KDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from tracking_validation.lineage import division_affected_cells, inherit_parent_labels, subsample_frames  # noqa: E402
from tracking_validation.linking import link_frames  # noqa: E402
from tracking_validation.metrics import compute_neighbour_continuity_observed  # noqa: E402
from tracking_validation.tracking_eval import COUNTS, sequence_errors  # noqa: E402
import ce_error_sweep as sweep  # noqa: E402
import estimate_error_mix as em  # noqa: E402

RES = ROOT / "data" / "_results"
NEIGH_CAP = 15.0
IMAGES = ("", "_n0.4", "_bz0.7", "_bz1", "_n0.4_bz0.7")
OPS = ("balanced", "high_recall", "high_precision")
STRIDES = (1, 2, 3, 4)
CAPS = (0.5, 1.0)
CH = ["survival", "vanished", "displaced", "newcomer", "wio", "rc", "prc", "pwio", "disp_out", "rc_out"]
CALIB = {1: "ce_error_sweep_slab_delaunay.csv", 2: "ce_error_sweep_fs2_delaunay.csv",
         3: "ce_error_sweep_fs3_delaunay.csv", 4: "ce_error_sweep_fs4_delaunay.csv"}


def channels(pair: list[dict], ok_labels: set[int]) -> dict:
    """Mean channel values over the focal labels in ``ok_labels`` for one pair."""
    res = compute_neighbour_continuity_observed(pair, neighbourhood_mode="delaunay",
                                                max_neighbour_distance=NEIGH_CAP, extra_metrics=True)
    cols = [i for i, c in enumerate(res["cell_ids"]) if int(c) in ok_labels]
    out = {}
    for name, key in sweep.METRIC_KEYS.items():
        v = res[key][0][cols]; out[name] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
    set1 = set(pair[1]["assigned_ids"].tolist())
    out["survival"] = float(np.mean([l in set1 for l in ok_labels])) if ok_labels else np.nan
    disp = res["wio_displaced"][0][cols]; rc = res["rank_correlation"][0][cols]
    out["disp_out"] = float(np.nanmean(disp > 0.4)) if np.isfinite(disp).any() else np.nan
    out["rc_out"] = float(np.nanmean(rc < 0.3)) if np.isfinite(rc).any() else np.nan
    # motion residual over the same focal labels (as in ce_error_sweep.motion_residuals)
    excluded = set(int(l) for l in pair[0]["assigned_ids"]) - set(ok_labels)
    r = sweep.motion_residuals(pair, NEIGH_CAP, excluded)
    out["resid"] = float(r.mean()) if r.size else np.nan
    out["resid_out"] = float((r > sweep.RESID_OUT).mean()) if r.size else np.nan
    return out


def seq_channels(frames, keep) -> list[dict]:
    out = []
    for t in range(len(frames) - 1):
        pair = [frames[t], frames[t + 1]]
        ok = {int(l) for l, tid in zip(pair[0]["assigned_ids"], pair[0]["true_ids"]) if int(tid) in keep[t]}
        out.append(channels(pair, ok))
    return out


def load_gt(seq: str, stride: int):
    data = inherit_parent_labels(sweep.load_ce(ROOT / "data", ROOT / "data" / "_cache", seq), prefer="nearest")
    start = int(np.argmax(data.counts() >= 50))
    gt = subsample_frames(data.frames[start:], stride); fidx = list(data.frame_indices[start:])[::stride]
    excl = division_affected_cells(gt, k=8, max_neighbour_distance=NEIGH_CAP, neighbourhood_mode="delaunay")
    keep = [set(int(c) for c in gt[t]["true_ids"]) - excl[t] for t in range(len(gt) - 1)]
    spacing = float(np.median([np.median(KDTree(f["positions"]).query(f["positions"], k=2)[0][:, 1]) for f in gt[::5]]))
    return data.label_map, gt, fidx, keep, spacing


def detection_frames(seq: str, img: str, op: str, label_map: dict, fidx: list[int]) -> list[dict]:
    d = pd.read_csv(RES / "detections" / f"ce_{seq}{img}_{op}.csv")
    frames = []
    for f in fidx:
        g = d[d.frame == f]
        tid = np.array([label_map.get(int(c), int(c)) if c >= 0 else -1 for c in g.true_id], dtype=int)
        frames.append(dict(positions=g[["x", "y", "z"]].to_numpy(float), true_ids=tid,
                           assigned_ids=np.arange(len(g)), parent_ids=np.full(len(g), -1)))
    return frames


def job(args):
    seq, img, op, stride = args
    label_map, gt, fidx, keep, spacing = load_gt(seq, stride)
    det = detection_frames(seq, img, op, label_map, fidx)
    n_cells = [len(f["true_ids"]) for f in gt]
    # oracle: detected cells at their DETECTED positions with their true identity, missed cells
    # filled in at their GT positions, false detections removed
    oracle = []
    for f, g in zip(det, gt):
        real = f["true_ids"] >= 0
        have = set(int(c) for c in f["true_ids"][real])
        fill = np.array([int(c) not in have for c in g["true_ids"]], dtype=bool)
        ids = np.concatenate([f["true_ids"][real], g["true_ids"][fill]])
        oracle.append(dict(positions=np.vstack([f["positions"][real], g["positions"][fill]]), true_ids=ids, assigned_ids=ids))
    ch_oracle = seq_channels(oracle, keep)
    ch_gt = seq_channels(gt, keep)
    rows = []
    for cap in CAPS:
        res = link_frames(det, cap * spacing, "lap")
        errs = sequence_errors(res.frames, gt, keep)
        ch = seq_channels(res.frames, keep)
        for t, e in enumerate(errs):
            rows.append(dict(seq=seq, img=img.lstrip("_") or "clean", op=op, stride=stride, cap=cap, frame=fidx[t],
                             n_cells=n_cells[t], spacing=spacing, **e,
                             **{f"trk_{k}": v for k, v in ch[t].items()},
                             **{f"orc_{k}": v for k, v in ch_oracle[t].items()},
                             **{f"gt_{k}": v for k, v in ch_gt[t].items()}))
        n = sum(e["n_eval"] for e in errs)
        print(f"{seq} {img or 'clean':10s} {op:14s} stride {stride} cap {cap}: " +
              " ".join(f"{k} {100 * sum(e[k] for e in errs) / n:.2f}" for k in COUNTS), flush=True)
    return rows


PAIRS = "detector_tracking_pairs.csv"     # per-pair table; --pairs-out writes a separate one


def stage_link(workers: int) -> None:
    jobs = [(s, img, op, st) for s in ("01", "02") for img in IMAGES for op in OPS for st in STRIDES
            if not ("_n" in img and op == "high_recall")]
    with ProcessPoolExecutor(min(workers, len(jobs))) as ex:
        rows = [r for part in ex.map(job, jobs) for r in part]
    pd.DataFrame(rows).to_csv(RES / PAIRS, index=False)
    print("wrote", RES / PAIRS)


# --------------------------------------------------------------------------- estimate

def truth(g: pd.DataFrame) -> dict:
    n = g.n_eval.sum()
    return dict(switch=100 * g.switch.sum() / n,
                frag=100 * (g.frag.sum() + g.reappear.sum()) / n,
                miss=100 * max(g.miss.sum() - g.reappear.sum(), 0) / n,
                fp=100 * g.fp.sum() / n)


def stage_estimate() -> None:
    df = pd.read_csv(RES / PAIRS, dtype={"seq": str})
    T = em.TYPES_4
    chans = [c for c in em.CHANNELS if c in CH]
    em.CHANNELS[:] = chans
    cals = {s: em.calibrate(RES / f, T) for s, f in CALIB.items()}
    sim = em.calibrate(RES / "ce_error_sweep_fine_delaunay.csv", T, source="SIM")
    rows = []
    for key, g in df.groupby(["seq", "img", "op", "stride", "cap"]):
        row = dict(zip(["seq", "img", "op", "stride", "cap"], key), pairs=len(g),
                   **{f"true_{t}": v for t, v in truth(g).items()},
                   true_miss_raw=100 * g.miss.sum() / g.n_eval.sum(), true_reappear=100 * g.reappear.sum() / g.n_eval.sum(),
                   true_fp_carry=100 * g.fp_carry.sum() / g.n_eval.sum())
        trk = g[[f"trk_{c}" for c in chans]].mean().to_numpy()
        orc = g[[f"orc_{c}" for c in chans]].mean().to_numpy()
        variants = {"real": cals[key[3]]}
        if key[3] == 1:
            variants["sim"] = sim
        for cname, (A, noise, baseline) in variants.items():
            for ref, d in (("oracle", trk - orc), ("baseline", trk - baseline[chans].to_numpy())):
                est = em.estimate(np.nan_to_num(d), A, noise)
                for i, t in enumerate(T):
                    row[f"est_{cname}_{ref}_{t}"] = 100 * est[i]
        rows.append(row)
    out = pd.DataFrame(rows); out.to_csv(RES / "detector_tracking_estimates.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.precision", 2)
    print(f"{len(out)} trackers.  Truth range (percent of evaluated cells):")
    print(out[[f"true_{t}" for t in T]].describe().loc[["min", "50%", "max"]].to_string())
    for cname, ref in (("real", "oracle"), ("real", "baseline"), ("sim", "oracle"), ("sim", "baseline")):
        sub = out.dropna(subset=[f"est_{cname}_{ref}_switch"])
        print(f"\n### {cname} calibration, reference = {ref}: estimated vs true over {len(sub)} trackers")
        stats = []
        for t in T:
            e, tr = sub[f"est_{cname}_{ref}_{t}"], sub[f"true_{t}"]
            stats.append(dict(type=t, true_mean=tr.mean(), est_mean=e.mean(), bias=(e - tr).mean(),
                              mae=(e - tr).abs().mean(), corr=np.corrcoef(e, tr)[0, 1] if e.std() > 0 else np.nan))
        print(pd.DataFrame(stats).round(2).to_string(index=False))
        tot_e = sub[[f"est_{cname}_{ref}_{t}" for t in T]].sum(1); tot_t = sub[[f"true_{t}" for t in T]].sum(1)
        print(f"total error: true mean {tot_t.mean():.2f}, est mean {tot_e.mean():.2f}, corr {np.corrcoef(tot_e, tot_t)[0, 1]:.2f}")
    print("wrote", RES / "detector_tracking_estimates.csv")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("all", "link", "estimate"), default="all")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--pairs-out", default=PAIRS, help="per-pair table file name in data/_results")
    a = p.parse_args()
    globals()["PAIRS"] = a.pairs_out
    if a.stage in ("all", "link"):
        stage_link(a.workers)
    if a.stage in ("all", "estimate"):
        stage_estimate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
