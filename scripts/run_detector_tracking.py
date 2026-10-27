"""Real tracker on real data: LoG detections (scripts/detect_log.py) linked by LAP, errors
counted against the CTC ground truth, and the metric-based estimator scored against them.

    python scripts/run_detector_tracking.py                  # link + count + metrics, then estimate
    python scripts/run_detector_tracking.py --stage estimate  # reuse the per-pair table

Trackers: 3 detector operating points x LAP max link distance {0.5, 1.0} x median spacing
(no gap closing: invisible to frame-pair metrics) = 6 trackers per sequence.

Per frame pair it stores the GT error counts (tracking_eval) and the metric channels for
  tracker  the tracker's output
  oracle   detections with their GT identity, missed cells filled in at GT positions, false
           detections removed (error-free tracks carrying only detector localisation jitter)
  gt       GT positions and identities
all restricted to cells whose neighbourhood touches no GT division (as in the sweeps).

Estimate stage: pooled over all pairs and per stage bin, the estimator (real-embryo or
simulation calibration) is run on  tracker - calibration baseline  and on
tracker - oracle  (jitter-corrected; uses GT, diagnostic only).

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
from tracking_validation.lineage import division_affected_cells, inherit_parent_labels  # noqa: E402
from tracking_validation.linking import close_gaps, link_frames  # noqa: E402
from tracking_validation.metrics import compute_neighbour_continuity_observed  # noqa: E402
from tracking_validation.tracking_eval import COUNTS, sequence_errors  # noqa: E402
import ce_error_sweep as sweep  # noqa: E402
import estimate_error_mix as em  # noqa: E402

RES = ROOT / "data" / "_results"
NEIGH_CAP = 15.0
CAPS = (0.5, 1.0)          # LAP max link distance, in median spacings
GAPS = (0,)                # gap closing is invisible to frame-pair metrics (a bridged track's label
                           # is still absent from the frame before), so it is not varied
CH = ["survival", "vanished", "displaced", "newcomer", "wio", "rc", "prc", "pwio", "disp_out", "rc_out"]
STAGES = ((50, 100), (100, 200), (200, 400))


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
    return out


def seq_channels(frames, keep) -> list[dict]:
    out = []
    for t in range(len(frames) - 1):
        pair = [frames[t], frames[t + 1]]
        ok = {int(l) for l, tid in zip(pair[0]["assigned_ids"], pair[0]["true_ids"]) if int(tid) in keep[t]}
        out.append(channels(pair, ok))
    return out


def load_gt(seq: str):
    data = inherit_parent_labels(sweep.load_ce(ROOT / "data", ROOT / "data" / "_cache", seq), prefer="nearest")
    start = int(np.argmax(data.counts() >= 50))
    gt = data.frames[start:]; fidx = list(data.frame_indices[start:])
    excl = division_affected_cells(gt, k=8, max_neighbour_distance=NEIGH_CAP, neighbourhood_mode="delaunay")
    keep = [set(int(c) for c in gt[t]["true_ids"]) - excl[t] for t in range(len(gt) - 1)]
    spacing = float(np.median([np.median(KDTree(f["positions"]).query(f["positions"], k=2)[0][:, 1]) for f in gt[::10]]))
    return data.label_map, gt, fidx, keep, spacing


def detection_frames(seq: str, op: str, label_map: dict, fidx: list[int]) -> list[dict]:
    d = pd.read_csv(RES / "detections" / f"ce_{seq}_{op}.csv")
    frames = []
    for f in fidx:
        g = d[d.frame == f]
        tid = np.array([label_map.get(int(c), int(c)) if c >= 0 else -1 for c in g.true_id], dtype=int)
        frames.append(dict(positions=g[["x", "y", "z"]].to_numpy(float), true_ids=tid,
                           assigned_ids=np.arange(len(g)), parent_ids=np.full(len(g), -1)))
    return frames


def job(args):
    seq, op = args
    label_map, gt, fidx, keep, spacing = load_gt(seq)
    det = detection_frames(seq, op, label_map, fidx)
    n_cells = [len(f["true_ids"]) for f in gt]
    # oracle: detected cells at their DETECTED positions with their true identity, missed cells
    # filled in at their GT positions, false detections removed -> error-free tracks that carry
    # only the detector's localisation jitter
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
        base = link_frames(det, cap * spacing, "lap")
        for gap in GAPS:
            res = close_gaps(base, gap, cap * spacing) if gap else base
            errs = sequence_errors(res.frames, gt, keep)
            ch = seq_channels(res.frames, keep)
            for t, e in enumerate(errs):
                rows.append(dict(seq=seq, op=op, cap=cap, gap=gap, frame=fidx[t], n_cells=n_cells[t], spacing=spacing, **e,
                                 **{f"trk_{k}": v for k, v in ch[t].items()},
                                 **{f"orc_{k}": v for k, v in ch_oracle[t].items()},
                                 **{f"gt_{k}": v for k, v in ch_gt[t].items()}))
            print(f"{seq} {op:14s} cap {cap} gap {gap}: " + " ".join(
                f"{k} {100 * sum(e[k] for e in errs) / sum(e['n_eval'] for e in errs):.2f}" for k in COUNTS), flush=True)
    return rows


def stage_link(workers: int) -> None:
    ops = list(json.loads((RES / "detection_operating_points.json").read_text()))
    jobs = [(s, o) for s in ("01", "02") for o in ops]
    with ProcessPoolExecutor(min(workers, len(jobs))) as ex:
        rows = [r for part in ex.map(job, jobs) for r in part]
    pd.DataFrame(rows).to_csv(RES / "detector_tracking_pairs.csv", index=False)
    print("wrote", RES / "detector_tracking_pairs.csv")


# --------------------------------------------------------------------------- estimate

TRUE_MAP = {"switch": ["switch"], "frag": ["frag"], "miss": ["miss"], "fp": ["fp", "reappear"]}


def stage_estimate() -> None:
    df = pd.read_csv(RES / "detector_tracking_pairs.csv", dtype={"seq": str})
    T = em.TYPES_4
    chans = [c for c in em.CHANNELS if c in CH]
    cals = {}
    em.CHANNELS[:] = chans
    cals["real"] = em.calibrate(RES / "ce_error_sweep_slab_delaunay.csv", T)
    cals["sim"] = em.calibrate(RES / "ce_error_sweep_fine_delaunay.csv", T, source="SIM")
    df["stage"] = pd.cut(df.n_cells, [s[0] for s in STAGES] + [STAGES[-1][1]], right=False,
                         labels=[f"{a}-{b}" for a, b in STAGES]).astype(str)
    rows = []
    for (seq, op, cap, gap), g0 in df.groupby(["seq", "op", "cap", "gap"]):
        for stage, g in [("all", g0)] + list(g0.groupby("stage")):
            n = g.n_eval.sum()
            true = {t: 100 * sum(g[c].sum() for c in TRUE_MAP[t]) / n for t in T}
            row = dict(seq=seq, op=op, cap=cap, gap=gap, stage=stage, pairs=len(g),
                       **{f"true_{t}": v for t, v in true.items()},
                       true_reappear=100 * g.reappear.sum() / n, true_fp_only=100 * g.fp.sum() / n,
                       true_fp_carry=100 * g.fp_carry.sum() / n)
            for cname, (A, noise, baseline) in cals.items():
                trk = g[[f"trk_{c}" for c in chans]].mean().to_numpy()
                orc = g[[f"orc_{c}" for c in chans]].mean().to_numpy()
                for ref, d in (("baseline", trk - baseline[chans].to_numpy()), ("oracle", trk - orc)):
                    est = em.estimate(np.nan_to_num(d), A, noise)
                    for i, t in enumerate(T):
                        row[f"est_{cname}_{ref}_{t}"] = 100 * est[i]
                row[f"jitter_{cname}"] = json.dumps({c: round(float(v), 4) for c, v in zip(chans, orc - baseline[chans].to_numpy())})
            rows.append(row)
    out = pd.DataFrame(rows); out.to_csv(RES / "detector_tracking_estimates.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.precision", 2)
    a = out[out.stage == "all"]
    for cname in cals:
        for ref in ("baseline", "oracle"):
            cols = ["seq", "op", "cap", "gap"] + [c for t in T for c in (f"true_{t}", f"est_{cname}_{ref}_{t}")]
            print(f"\n### {cname} calibration, reference = {ref}  (percent of evaluated cells; fp true = fp + reappear)")
            print(a[cols].rename(columns=lambda c: c.replace(f"est_{cname}_{ref}_", "est_")).to_string(index=False))
    print("\n### detection jitter: oracle minus calibration baseline (real calibration), mean over runs")
    j = pd.DataFrame([json.loads(s) for s in a.jitter_real]); print(j.mean().round(4).to_string())
    print("wrote", RES / "detector_tracking_estimates.csv")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("all", "link", "estimate"), default="all")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = p.parse_args()
    if a.stage in ("all", "link"):
        stage_link(a.workers)
    if a.stage in ("all", "estimate"):
        stage_estimate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
