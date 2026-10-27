"""Real-linker test: run simple linkers on the embryo detections, label every
link against the truth, and score the metric-based estimator and the per-cell
switch locator against those labels.

    python scripts/run_linker_test.py

Detections: ground-truth centroids (one daughter inherits the mother's label),
with a fraction dropped at random (misses).  Linkers: greedy nearest-neighbour
and frame-to-frame linear assignment, each at several maximum link distances
(in units of the median cell spacing).  Evaluation is restricted to cells
whose neighbourhood touches no division, as in the sweeps.

Per run it reports the TRUE error composition from the link labels, the
estimator's composition from the metrics alone (calibrated on the injected-
error sweeps, outlier channels on the low-rate sweep), and the locator's
precision/recall on the linker's actual switches.

Writes data/_results/linker_test.csv (per run) and prints a summary.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

from tracking_validation import ErrorConfig, apply_tracking_errors  # noqa: E402
from tracking_validation.lineage import division_affected_cells, inherit_parent_labels, subsample_frames  # noqa: E402
from tracking_validation.linking import label_links, link_frames  # noqa: E402
from tracking_validation.metrics import compute_neighbour_continuity_observed  # noqa: E402
import ce_error_sweep as sweep  # noqa: E402
import estimate_error_mix as em  # noqa: E402

RESULTS = ROOT / "data" / "_results"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--caps", type=float, nargs="+", default=[0.5, 1.0, 2.0], help="max link distance / median spacing")
    p.add_argument("--dropout", type=float, nargs="+", default=[0.0],
                   help="fraction of detections dropped per frame (default none: linker errors only)")
    p.add_argument("--methods", nargs="+", default=["greedy", "lap"], help="greedy, lap, metric")
    p.add_argument("--metric-lambda", type=float, default=None,
                   help="displacement weight for the metric-refined linker (default from linking module)")
    p.add_argument("--neigh-cap", type=float, default=15.0)
    p.add_argument("--loc-threshold", type=float, default=0.6, help="kNN locator: displaced share threshold")
    p.add_argument("--frame-stride", type=int, nargs="+", default=[1],
                   help="use every n-th frame (longer interval = harder linking); the clean baseline "
                        "is re-measured at each stride from the ground truth")
    p.add_argument("--calib-pattern", default=None,
                   help="per-stride calibration sweep, e.g. 'data/_results/ce_error_sweep_fs{stride}_delaunay.csv'; "
                        "response matrix, noise AND baseline then all come from that stride's sweep "
                        "(stride 1 uses the default calibration)")
    p.add_argument("--out", type=Path, default=RESULTS / "linker_test.csv")
    a = p.parse_args()
    if a.metric_lambda is not None:
        import tracking_validation.linking as _lk
        _lk.METRIC_DISPLACEMENT_WEIGHT = a.metric_lambda

    # estimator calibration (Delaunay channels + outlier channels from the low-rate sweep)
    em.CHANNELS.extend(c for c in em.OUTLIER_CHANNELS if c not in em.CHANNELS)
    A, noise, baseline = em.calibrate(RESULTS / "ce_error_sweep_out_delaunay.csv", em.TYPES_4)
    A_low, _, _ = em.calibrate(RESULTS / "ce_error_sweep_low_out_delaunay.csv", em.TYPES_4)
    for ch in em.OUTLIER_CHANNELS:
        A.loc[ch] = A_low.loc[ch]
    A_default, noise_default = A.copy(), noise.copy()

    def channel_vector(pair, ok_labels):
        """Mean channel values over the given labels (division-free cells) for one pair."""
        resd = compute_neighbour_continuity_observed(pair, neighbourhood_mode="delaunay",
                                                     max_neighbour_distance=a.neigh_cap, extra_metrics=True)
        cols = [i for i, c in enumerate(resd["cell_ids"]) if int(c) in ok_labels]
        vec = {}
        for name, key in sweep.METRIC_KEYS.items():
            v = resd[key][0][cols]; vec[name] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
        set1 = set(pair[1]["assigned_ids"].tolist())
        vec["survival"] = float(np.mean([l in set1 for l in ok_labels])) if ok_labels else np.nan
        disp = resd["wio_displaced"][0][cols]; rc = resd["rank_correlation"][0][cols]
        vec["disp_out"] = float(np.nanmean(disp > 0.4)) if len(cols) else np.nan
        vec["rc_out"] = float(np.nanmean(rc < 0.3)) if len(cols) else np.nan
        return vec

    base_default = baseline.copy()
    rows = []
    for seq in ("01", "02"):
      data = inherit_parent_labels(sweep.load_ce(ROOT / "data", ROOT / "data" / "_cache", seq), prefer="nearest")
      start = int(np.argmax(data.counts() >= 50)); gt_all = data.frames[start:]
      for stride in a.frame_stride:
        gt = subsample_frames(gt_all, stride)
        excl = division_affected_cells(gt, k=8, max_neighbour_distance=a.neigh_cap, neighbourhood_mode="delaunay")
        keep_sets = [set(gt[t]["true_ids"].tolist()) - excl[t] for t in range(len(gt) - 1)]
        spacing = float(np.median([np.median(KDTree(f["positions"]).query(f["positions"], k=2)[0][:, 1]) for f in gt[::10]]))
        hi = gt[-1]["positions"].max(axis=0); box = tuple(float(v) for v in (hi + 2.0))
        # displacement per step relative to spacing (how hard the linking is)
        disp_ratio = np.median([np.median([np.linalg.norm(p1 - p0) for tid, p0 in zip(gt[t]["true_ids"], gt[t]["positions"])
                                           for p1 in [dict(zip(gt[t + 1]["true_ids"], gt[t + 1]["positions"])).get(tid)] if p1 is not None])
                                for t in range(0, len(gt) - 1, 10)]) / spacing
        # per-stride calibration if provided: A, noise and baseline from the sweep at this interval
        A, noise = A_default, noise_default
        calib_src = "default"
        if a.calib_pattern and stride > 1:
            cf = Path(a.calib_pattern.format(stride=stride))
            if cf.exists():
                A, noise, baseline = em.calibrate(cf, em.TYPES_4)
                calib_src = cf.name
        # stride-matched clean baseline and clean false-flag rate
        if stride > 1 and calib_src == "default":
            clean_vecs, clean_flags, clean_cells = [], 0, 0
            for t in range(len(gt) - 1):
                pair = [gt[t], gt[t + 1]]
                ok = {int(l) for l, tid in zip(pair[0]["assigned_ids"], pair[0]["true_ids"]) if int(tid) in keep_sets[t]}
                clean_vecs.append([channel_vector(pair, ok)[ch] for ch in em.CHANNELS])
                resk = compute_neighbour_continuity_observed(pair, k=8, neighbourhood_mode="knn",
                                                             max_neighbour_distance=a.neigh_cap, extra_metrics=True)
                v = resk["wio_displaced"][0]; sel = [np.isfinite(x) and int(c) in ok for c, x in zip(resk["cell_ids"], v)]
                clean_flags += int(np.sum(v[sel] > a.loc_threshold)); clean_cells += int(np.sum(sel))
            baseline = pd.Series(np.nanmean(np.array(clean_vecs, dtype=float), axis=0), index=em.CHANNELS)
            clean_false_per_pair = clean_flags / (len(gt) - 1)
        elif stride == 1:
            baseline = base_default; clean_false_per_pair = 0.0
        else:
            clean_false_per_pair = float("nan")
        print(f"--- {seq} stride {stride}: {len(gt)} frames, displacement/spacing {disp_ratio:.2f}, "
              f"calibration {calib_src}, baseline displaced {baseline['displaced']:.3f} rc {baseline['rc']:.3f}", flush=True)
        for drop in a.dropout:
            # detections: GT with random misses in every frame (a detector's behaviour)
            det, log = apply_tracking_errors(gt, ErrorConfig(p_miss=drop, lambda_fp=0.0, p_switch=0.0), box,
                                             np.random.default_rng(7))
            missed = {}
            for ev in log["missed"]:
                missed.setdefault(ev["frame"], set()).add(ev["true_id"])
            for method in a.methods:
                for cap_s in a.caps:
                    res = link_frames(det, cap_s * spacing, method)
                    labels = label_links(res, evaluate=keep_sets)
                    # --- truth per pair -------------------------------------------------
                    true_sw = np.array([r["switch_rate"] for r in labels])
                    true_fr = np.array([r["frag_rate"] for r in labels])
                    true_miss = np.array([len(missed.get(t + 1, set()) & keep_sets[t]) / max(r["n1"], 1)
                                          for t, r in enumerate(labels)])
                    # --- metrics on the linker output, division-free cells -------------------
                    d_vecs, flags_tp, flags_n, events_hit, events_n = [], 0, 0, 0, 0
                    for t in range(len(res.frames) - 1):
                        pair = [res.frames[t], res.frames[t + 1]]
                        lab0 = pair[0]["assigned_ids"]; true0 = pair[0]["true_ids"]
                        ok_labels = {int(l) for l, tid in zip(lab0, true0) if int(tid) in keep_sets[t]}
                        vec = channel_vector(pair, ok_labels)
                        d_vecs.append([vec[ch] - baseline[ch] for ch in em.CHANNELS])
                        # --- locator (kNN, every cell) against the linker's real switches ------
                        resk = compute_neighbour_continuity_observed(pair, k=8, neighbourhood_mode="knn",
                                                                     max_neighbour_distance=a.neigh_cap, extra_metrics=True)
                        label_to_true = {int(l): int(tid) for l, tid in zip(lab0, true0)}
                        sw_ids = labels[t]["switched_true_ids"]
                        for c, v in zip(resk["cell_ids"], resk["wio_displaced"][0]):
                            if not np.isfinite(v) or int(c) not in ok_labels:
                                continue
                            if v > a.loc_threshold:
                                flags_n += 1
                                flags_tp += label_to_true.get(int(c), -1) in sw_ids
                        events_n += labels[t]["switch"]
                        # an event is "hit" if any of its switched cells is flagged
                        flagged_true = {label_to_true.get(int(c), -1) for c, v in zip(resk["cell_ids"], resk["wio_displaced"][0])
                                        if np.isfinite(v) and v > a.loc_threshold}
                        events_hit += len(sw_ids & flagged_true)
                    D = np.nanmean(np.array(d_vecs, dtype=float), axis=0)
                    est = em.estimate(np.nan_to_num(D), A, noise)
                    rows.append(dict(seq=seq, stride=stride, disp_over_spacing=disp_ratio, method=method,
                                     cap_spacings=cap_s, dropout=drop, pairs=len(labels), n_tracks=res.n_tracks,
                                     clean_false_flags_per_pair=clean_false_per_pair, calibration=calib_src,
                                     true_switch=100 * true_sw.mean(), true_frag=100 * true_fr.mean(), true_miss=100 * true_miss.mean(),
                                     est_switch=100 * est[0], est_frag=100 * est[1], est_miss=100 * est[2], est_fp=100 * est[3],
                                     loc_flags=flags_n, loc_precision=flags_tp / flags_n if flags_n else np.nan,
                                     loc_recall=events_hit / events_n if events_n else np.nan, switch_events=events_n))
                    r = rows[-1]
                    print(f"{seq} s{stride} {method:6s} cap {cap_s:3.1f} drop {drop:.2f} | true sw {r['true_switch']:4.2f} fr {r['true_frag']:4.2f} "
                          f"miss {r['true_miss']:4.2f} | est sw {r['est_switch']:4.2f} fr {r['est_frag']:4.2f} miss {r['est_miss']:4.2f} "
                          f"fp {r['est_fp']:4.2f} | locator P {r['loc_precision']:.2f} R {r['loc_recall']:.2f} ({events_n} sw)", flush=True)
    df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
