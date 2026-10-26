"""Reference-free parameter selection for a basic tracker, checked against truth.

    python scripts/select_params.py

Tracker: linear-assignment linking with three parameters
    cap       maximum link distance (median spacings)
    max_gap   maximum number of frames a gap may be bridged over (0 = no gap closing)
    gap_dist  maximum bridging distance (median spacings)
run on embryo detections with a fraction dropped at random, at a frame
interval long enough for linking errors to occur.

For every parameter setting:
  truth      identity-discontinuity rate: fraction of real detections whose
             label differs from the label the SAME cell carried at its previous
             detection (counts fragmentations, restarts and switches alike),
             over division-free cells.
  metric     reference-free criterion from the tracker output alone:
             track-end rate (label at t absent at t+1) + new-track rate (label at
             t+1 never seen in any earlier frame) + displaced-outlier fraction,
             over division-free cells.  Uses only frame pairs plus label history.
Reports Spearman correlation between the two over the grid and the settings
each one picks.  Writes data/_results/select_params.csv.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

from tracking_validation import ErrorConfig, apply_tracking_errors  # noqa: E402
from tracking_validation.lineage import division_affected_cells, inherit_parent_labels  # noqa: E402
from tracking_validation.linking import close_gaps, link_frames  # noqa: E402
from tracking_validation.metrics import compute_neighbour_continuity_observed  # noqa: E402
import ce_error_sweep as sweep  # noqa: E402


def truth_rate(frames: list[dict], keep: list[set[int]]) -> float:
    """Identity-discontinuity rate over division-free real cells."""
    last_label: dict[int, int] = {}
    n = 0; bad = 0
    for t, f in enumerate(frames):
        ok = keep[min(t, len(keep) - 1)]
        for tid, lab in zip(f["true_ids"].tolist(), f["assigned_ids"].tolist()):
            if tid < 0:
                continue
            if tid in last_label and tid in ok:
                n += 1; bad += last_label[tid] != lab
            last_label[tid] = lab
    return bad / max(n, 1)


def metric_criterion(frames: list[dict], keep: list[set[int]], cap: float, thr: float = 0.6):
    """Reference-free: track ends + new tracks (with label history) + displaced outliers."""
    seen: set[int] = set(frames[0]["assigned_ids"].tolist())
    ends = starts = outl = n = 0
    for t in range(len(frames) - 1):
        f0, f1 = frames[t], frames[t + 1]
        ok0 = {int(l) for l, tid in zip(f0["assigned_ids"], f0["true_ids"]) if int(tid) in keep[t]}
        set1 = set(f1["assigned_ids"].tolist())
        ends += sum(1 for l in ok0 if l not in set1)
        ok1 = {int(l) for l, tid in zip(f1["assigned_ids"], f1["true_ids"]) if int(tid) in keep[min(t + 1, len(keep) - 1)]}
        starts += sum(1 for l in ok1 if l not in seen)
        seen |= set1
        r = compute_neighbour_continuity_observed([f0, f1], k=8, neighbourhood_mode="knn",
                                                  max_neighbour_distance=cap, extra_metrics=True)
        d = r["wio_displaced"][0]
        outl += sum(1 for c, v in zip(r["cell_ids"], d) if np.isfinite(v) and int(c) in ok0 and v > thr)
        n += len(ok0)
    return dict(ends=ends / n, starts=starts / n, outliers=outl / n, criterion=(ends + starts + outl) / n)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.03)
    p.add_argument("--caps", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.5, 2.0])
    p.add_argument("--gaps", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--gap-dists", type=float, nargs="+", default=[1.0, 2.0])
    p.add_argument("--alphas", type=float, nargs="+", default=[2.0], help="link-cost exponents to grid over")
    p.add_argument("--neigh-cap", type=float, default=15.0)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "_results" / "select_params.csv")
    a = p.parse_args()
    rows = []
    for seq in ("01", "02"):
        data = inherit_parent_labels(sweep.load_ce(ROOT / "data", ROOT / "data" / "_cache", seq), prefer="nearest")
        start = int(np.argmax(data.counts() >= 50)); gt = data.frames[start:][::a.stride]
        excl = division_affected_cells(gt, k=8, max_neighbour_distance=a.neigh_cap, neighbourhood_mode="knn")
        keep = [set(gt[t]["true_ids"].tolist()) - excl[t] for t in range(len(gt) - 1)]
        sp = float(np.median([np.median(KDTree(f["positions"]).query(f["positions"], k=2)[0][:, 1]) for f in gt[::10]]))
        hi = gt[-1]["positions"].max(axis=0); box = tuple(float(v) for v in (hi + 2.0))
        det, _ = apply_tracking_errors(gt, ErrorConfig(p_miss=a.dropout, lambda_fp=0.0, p_switch=0.0), box, np.random.default_rng(11))
        import tracking_validation.linking as lk
        for alpha, cap, gap, gd in itertools.product(a.alphas, a.caps, a.gaps, a.gap_dists):
            if gap == 0 and gd != a.gap_dists[0]:
                continue                      # gap distance is irrelevant without gap closing
            lk.LAP_EXPONENT = alpha
            res = close_gaps(link_frames(det, cap * sp, "lap"), gap, gd * sp)
            tr = truth_rate(res.frames, keep)
            mc = metric_criterion(res.frames, keep, a.neigh_cap)
            rows.append(dict(seq=seq, alpha=alpha, cap=cap, max_gap=gap, gap_dist=gd, n_tracks=res.n_tracks,
                             truth=100 * tr, **{k: 100 * v for k, v in mc.items()}))
            print(f"{seq} a {alpha:3.1f} cap {cap:4.2f} gap {gap} gd {gd:3.1f} | truth {100*tr:5.2f} | metric {100*mc['criterion']:5.2f} "
                  f"(ends {100*mc['ends']:4.2f} starts {100*mc['starts']:4.2f} outl {100*mc['outliers']:4.2f})", flush=True)
    df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
    pd.set_option("display.width", 200); pd.set_option("display.precision", 2)
    keys = ["alpha", "cap", "max_gap", "gap_dist"]
    g = df.groupby(keys)[["truth", "criterion", "ends", "starts", "outliers"]].mean().reset_index()
    rho = spearmanr(g.truth, g.criterion).statistic
    best_t = g.loc[g.truth.idxmin()]; best_m = g.loc[g.criterion.idxmin()]
    print(f"\nSpearman(truth, metric criterion) over {len(g)} settings: {rho:.2f}")
    print(f"truth-optimal : alpha {best_t.alpha} cap {best_t.cap} gap {int(best_t.max_gap)} gd {best_t.gap_dist} -> truth {best_t.truth:.2f}%")
    print(f"metric-chosen : alpha {best_m.alpha} cap {best_m.cap} gap {int(best_m.max_gap)} gd {best_m.gap_dist} -> truth {best_m.truth:.2f}% "
          f"(regret {best_m.truth - best_t.truth:.2f} points)")
    print("\nfull grid (both recordings averaged):"); print(g.to_string(index=False))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
