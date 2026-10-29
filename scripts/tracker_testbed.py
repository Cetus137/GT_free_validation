"""Testing zone, step 3: a tracker that makes its own errors.  The simulation-free unmixing of step 2
is applied to the tracker's output and scored against the tracker's true errors.

    python scripts/tracker_testbed.py [--workers N] [--stage all|summary]

Movies: persistent motion of scripts/motion_testbed.py (five speeds, two movies, 10 frame pairs).
Detections (hop_count_synthetic.synth): missed / false detections at three levels, positions
jittered by 0.05 spacing.  Tracker: frame-to-frame LAP, link distance 1 spacing.

Unmixing of the tracker output (tracking_validation.unmixing; counts per labelled cell at t):
  v1   the step-2 method unchanged: every column of the response matrix measured by injecting
       label errors into the tracker output
  v2   as v1, but the switch column's two-way entry is the capture measured by deleting 3% of the
       detections and re-running the tracker (deletion_calibration.measure)
Truth (tracking_eval.pair_errors, all cells evaluated: no divisions) is used only to score.  A cell
missed at t and found at t+1 gets a new label from the tracker ("reappear"); the counts cannot tell
it from a false positive, so it is reported separately.

Writes data/_results/tracker_testbed.csv.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import deletion_calibration as dc  # noqa: E402
import hop_count_synthetic as hcs  # noqa: E402
import motion_testbed as mt  # noqa: E402
from tracking_validation import unmixing as um  # noqa: E402
from tracking_validation.tracking_eval import pair_errors  # noqa: E402

RES = ROOT / "data" / "_results"
MOTION, SEEDS = "persistent", (0, 1)
LEVELS = {"2+1": (0.02, 0.01), "5+2": (0.05, 0.02), "10+4": (0.10, 0.04)}   # missed, false detections
TRUTH = ("miss", "frag", "fp", "fp_carry", "reappear", "switch")


def job(args):
    speed, seed, level = args
    gt = mt.movie(MOTION, speed, seed).as_frames()
    spacing = float(np.median([mt.sweep.median_nn(g["positions"]) for g in gt]))
    key = [mt.SPEEDS.index(speed), seed, list(LEVELS).index(level)]
    pm, pf = LEVELS[level]
    det = hcs.synth(gt, spacing, pm, pf, np.random.default_rng(key + [1]))
    F = dc.link(det, spacing, "lap", None).frames
    pairs = [F[t:t + 2] for t in range(len(F) - 1)]
    box = (mt.BOX,) * 3
    # truth, as events (all cells evaluated: the simulation has no divisions)
    every = set(int(c) for c in gt[0]["true_ids"])
    truth = pd.DataFrame([pair_errors(F[t], F[t + 1], every, gt[t + 1]["positions"], gt[t + 1]["true_ids"])
                          for t in range(len(F) - 1)]).sum()
    # the unmixing, from the tracker output only
    c = [um.counts(p, mt.CAP) for p in pairs]
    total, n = np.sum([x for x, _ in c], axis=0), sum(m for _, m in c)
    A = um.response_matrix(pairs, box, spacing, mt.CAP, np.random.default_rng(key + [2]))
    est1, _ = nnls(A, total)
    dl = dc.measure(det, [every] * (len(gt) - 1), spacing, np.random.default_rng(key + [3]))
    A2 = A.copy()
    A2[um.COUNTS.index("flagged"), um.TYPES.index("switch")] = dl["changed_two"] / max(dl["changed"], 1)
    est2, _ = nnls(A2, total)
    row = dict(speed=speed, seed=seed, level=level, n=n, spacing=spacing,
               **{f"count_{k}": v / n for k, v in zip(um.COUNTS, total)},
               **{f"true_{k}": truth[k] / n for k in TRUTH},
               **{f"v1_{t}": e / n for t, e in zip(um.TYPES, est1)},
               **{f"v2_{t}": e / n for t, e in zip(um.TYPES, est2)},
               cap_label=A[um.COUNTS.index("flagged"), um.TYPES.index("switch")], cap_deletion=A2[um.COUNTS.index("flagged"), um.TYPES.index("switch")],
               cap_true=dl["two_tp"] / max(dl["sw"], 1))
    print(f"speed {speed:4.2f} seed {seed} level {level:5s}: done", flush=True)
    return row


def summary() -> None:
    d = pd.read_csv(RES / "tracker_testbed.csv")
    pd.set_option("display.width", 240)
    mot = mt.pd.read_csv(RES / "motion_testbed_unmixing.csv")
    resid = mot[mot.motion == MOTION].groupby("speed").resid.mean().round(2)
    d["motion"] = d.speed.map(resid)
    show = d.groupby(["motion", "level"]).mean(numeric_only=True).reset_index()
    cols = ["motion", "level"]
    for t, truth in (("miss", "true_miss"), ("frag", "true_frag"), ("fp", "true_fp"), ("switch", "true_switch")):
        show[f"{t}: true"] = 100 * show[truth]
        show[f"{t}: v1"] = 100 * show[f"v1_{t}"]
        show[f"{t}: v2"] = 100 * show[f"v2_{t}"]
        cols += [f"{t}: true", f"{t}: v1", f"{t}: v2"]
    show["reappear"] = 100 * show.true_reappear; show["fp_carry"] = 100 * show.true_fp_carry
    cols += ["reappear", "fp_carry", "cap_label", "cap_deletion", "cap_true"]
    print("events per labelled cell (%), mean of two movies; motion = motion vs neighbours (spacings per frame)")
    print(show[cols].round(2).to_string(index=False))


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--stage", choices=("all", "summary"), default="all")
    a = p.parse_args()
    if a.stage == "all":
        jobs = [(v, s, lv) for v in mt.SPEEDS for s in SEEDS for lv in LEVELS]
        with ProcessPoolExecutor(a.workers) as ex:
            rows = list(ex.map(job, jobs))
        pd.DataFrame(rows).to_csv(RES / "tracker_testbed.csv", index=False)
    summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
