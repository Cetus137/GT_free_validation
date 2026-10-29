"""Simulation-free switch estimate on the real trackers (link 1.0 spacing): a two-way
nearest-body count, calibrated by deleting detections from the same data and re-linking.

    python scripts/deletion_calibration.py [--workers N] [--stage all|summary]
    /Users/ewheeler/miniforge3/envs/motile_env/bin/python scripts/deletion_calibration.py --tracker motile

Trackers: frame-to-frame LAP (default) or motile (global, no divisions; as in
scripts/run_motile_tracking.py).  Links are compared by detection index, so motile's dropped
detections do not disturb the comparison.

Two-way test: a persisting label is flagged when the detection nearest its forward prediction
(t -> t+1) is not its own body AND the detection nearest its backward prediction (t+1 -> t) is
not its own body either.  A false flag needs a large movement towards a neighbour, and the two
directions rarely point at one, so false flags almost vanish; nothing is subtracted.

Capture by deletion ('standard addition'): delete a random 3% of the detections in every frame,
re-link with the same tracker, and take every link that differs from the tracker's ORIGINAL links
as a known added switch.  The share of them the test flags is its capture, measured in this tissue
with this tracker.  Estimate = two-way count on the original tracking / that capture.  Neither step
uses a simulation or the ground truth; truth is used only to score (and, as everywhere, for the
division exclusion of the evaluated cells).

Writes data/_results/deletion_calibration[_motile].csv and
figures/local_metrics/deletion_calibration[_motile].png.  For LAP the figure compares with the
simulation-calibrated estimate; the simulation was built with LAP, so for motile it compares with
the uncalibrated two-way count.
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
import run_detector_tracking as rdt  # noqa: E402
import run_motile_tracking as rmt  # noqa: E402
from tracking_validation.linking import link_frames  # noqa: E402
from tracking_validation.unmixing import nearest_body_flags  # noqa: E402

RES = ROOT / "data" / "_results"
CAP = 1.0                      # link distance (spacings)
P_DEL, REPEATS = 0.03, 2       # share of detections deleted per frame; independent deletions
KEYS = ["seq", "img", "op", "stride"]


def flags(f0, f1) -> dict:
    """{label: (one-way, two-way)} for every label present in both frames."""
    return nearest_body_flags(f0, f1, rdt.NEIGH_CAP)


def link(frames, spacing, tracker, resp):
    if tracker == "motile":
        from tracking_validation.linking_motile import link_motile
        return link_motile(frames, CAP * spacing, resp)
    return link_frames(frames, CAP * spacing, "lap")


def links_by_detection(R) -> list[set]:
    """Each pair's links as (detection index at t, detection index at t+1)."""
    F = R.frames
    return [{(int(F[t]["det"][i]), int(F[t + 1]["det"][j])) for i, j in m} for t, m in enumerate(R.links)]


def measure(det, evaluated, spacing, rng, tracker="lap", resp=None) -> dict:
    """Two-way counts on the tracker's output (scored against the truth) and the capture
    measured by deletion.  ``det``: detection frames with positions and true_ids (-1 = false
    detection); ``evaluated[t]``: the true ids evaluated in pair (t, t+1)."""
    det = [{**f, "det": np.arange(len(f["positions"]))} for f in det]
    R = link(det, spacing, tracker, resp)
    row = dict(n=0, sw=0, one_tp=0, one_fp=0, two_tp=0, two_fp=0, changed=0, changed_one=0, changed_two=0, changed_true_sw=0)
    # the original tracking: counts on the evaluated cells, scored against the truth
    F = R.frames
    for t in range(len(F) - 1):
        E = evaluated[t]
        tid0 = dict(zip(map(int, F[t]["assigned_ids"]), F[t]["true_ids"].tolist()))
        tid1 = dict(zip(map(int, F[t + 1]["assigned_ids"]), F[t + 1]["true_ids"].tolist()))
        for l, (one, two) in flags(F[t], F[t + 1]).items():
            a, b = tid0[l], tid1[l]
            if a < 0 or b < 0 or a not in E:
                continue
            sw = a != b
            row["n"] += 1; row["sw"] += int(sw)
            row["one_tp"] += int(one and sw); row["one_fp"] += int(one and not sw)
            row["two_tp"] += int(two and sw); row["two_fp"] += int(two and not sw)
    # deletion: links that differ from the original ones are added switches of known position
    orig = links_by_detection(R)
    for _ in range(REPEATS):
        masks = [rng.random(len(f["positions"])) >= P_DEL for f in det]
        sub = [{k: (v[m] if isinstance(v, np.ndarray) and len(v) == len(m) else v) for k, v in f.items()} for f, m in zip(det, masks)]
        P = link(sub, spacing, tracker, [r[m] for r, m in zip(resp, masks)] if resp is not None else None)
        for t, links in enumerate(P.links):
            f0, f1 = P.frames[t], P.frames[t + 1]
            fl = flags(f0, f1)
            for i, j in links:
                l = int(f1["assigned_ids"][j])
                if (int(f0["det"][i]), int(f1["det"][j])) in orig[t] or l not in fl:
                    continue
                row["changed"] += 1; row["changed_one"] += int(fl[l][0]); row["changed_two"] += int(fl[l][1])
                row["changed_true_sw"] += int(f0["true_ids"][i] != f1["true_ids"][j])
    return row


def job(args):
    seq, img, op, stride, tracker = args
    rng = np.random.default_rng([int(seq), rdt.IMAGES.index(img), rdt.OPS.index(op), stride, 7])
    label_map, gt, fidx, keep, spacing = rdt.load_gt(seq, stride)
    det = rdt.detection_frames(seq, img, op, label_map, fidx)
    resp = rmt.responses(seq, img, op, fidx) if tracker == "motile" else None
    evaluated = [keep[t] & set(int(c) for c in gt[t + 1]["true_ids"]) for t in range(len(gt) - 1)]
    row = dict(seq=seq, img=img.lstrip("_") or "clean", op=op, stride=stride,
               **measure(det, evaluated, spacing, rng, tracker, resp))
    print(f"{tracker} {seq} {img or 'clean':10s} {op:14s} stride {stride}: done", flush=True)
    return row


def stem(tracker: str) -> str:
    return "deletion_calibration" + ("" if tracker == "lap" else f"_{tracker}")


def summary(tracker: str) -> None:
    import matplotlib.pyplot as plt
    import figstyle as fs
    d = pd.read_csv(RES / f"{stem(tracker)}.csv", dtype={"seq": str})
    d["true"] = 100 * d.sw / d.n
    d["est_raw"] = 100 * (d.two_tp + d.two_fp) / d.n
    if tracker == "lap":                         # the simulation was built with LAP
        sim = pd.read_csv(RES / "sim_hop_calibrated_real.csv", dtype={"seq": str})
        d = d.merge(sim.loc[sim.cap == CAP, KEYS + ["est_nb_any"]], on=KEYS, how="left")
        ref, ref_label = "est_nb_any", "Simulation-calibrated"
    else:
        ref, ref_label = "est_raw", "Uncalibrated"
    d["f_one"] = 100 * d.one_fp / (d.n - d.sw); d["f_two"] = 100 * d.two_fp / (d.n - d.sw)
    d["c_true"] = d.two_tp / d.sw; d["c_del"] = d.changed_two / d.changed
    d["est_del"] = 100 * (d.two_tp + d.two_fp) / d.n / d.c_del

    pd.set_option("display.width", 200)
    print(f"{tracker}: {len(d)} trackers; added switches per tracker: min {int(d.changed.min())}, median {int(d.changed.median())}; "
          f"{100 * d.changed_true_sw.sum() / d.changed.sum():.0f}% of them truly switched\n")
    g = d.groupby("stride")
    print(pd.DataFrame({"false one-way %": g.f_one.mean(), "false two-way %": g.f_two.mean(),
                        "capture true": g.two_tp.sum() / g.sw.sum(), "capture by deletion": g.changed_two.sum() / g.changed.sum()}).round(3).to_string())
    print()
    shown = [("est_raw", "uncalibrated two-way")] + ([(ref, ref_label.lower())] if ref != "est_raw" else [])
    for c, lab in shown + [("est_del", "deletion-calibrated two-way")]:
        rel = (d[c] - d.true) / d.true
        by = " ".join(f"{100 * np.median((h[c] - h.true) / h.true):+.0f}%" for _, h in g)
        print(f"{lab:31s} r {np.corrcoef(d[c], d.true)[0, 1]:.2f}  median |err| {100 * np.median(rel.abs()):.0f}%  "
              f"bias {100 * np.median(rel):+.0f}%   bias by interval {by}")

    fs.apply()
    cols = {1: "#2a78d6", 2: "#1baf7a", 3: "#eda100", 4: "#eb6834"}
    names = {1: "every frame", 2: "every 2nd", 3: "every 3rd", 4: "every 4th"}
    fig, axes = plt.subplots(1, 4, figsize=(fs.DOUBLE, 2.25),
                             gridspec_kw=dict(left=0.065, right=0.875, top=0.85, bottom=0.2, wspace=0.42))
    smax = float(max(d.true.max(), d[ref].max(), d.est_del.max())) * 1.05
    fmax = float(d.f_one.max()) * 1.05
    panels = ((axes[0], "f_one", "f_two", fmax, "One-way (% of correct cells)", "Two-way (%)", "False flags"),
              (axes[2], "true", ref, smax, "True switched (%)", "Estimated switched (%)", ref_label),
              (axes[3], "true", "est_del", smax, "True switched (%)", "Estimated switched (%)", "Deletion-calibrated"))
    # capture: per-tracker truth rests on few switches, so compare with the truth pooled per interval
    ax, jit = axes[1], np.random.default_rng(0)
    for st, c in cols.items():
        h = d[d.stride == st]
        ax.scatter(st + jit.uniform(-0.2, 0.2, len(h)), h.c_del, s=12, color=c, edgecolors=fs.SURFACE, linewidths=0.5, zorder=3)
        ax.hlines(h.two_tp.sum() / h.sw.sum(), st - 0.32, st + 0.32, color=fs.INK, lw=1.6, zorder=4)
    ax.set_xlim(0.4, 4.6); ax.set_ylim(0, 0.7); ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("Linking every … frames"); ax.set_ylabel("Capture (two-way)"); ax.set_title("Capture", loc="left", pad=3)
    ax.text(0.04, 0.04, "Dots: by deletion\nBars: true", transform=ax.transAxes, va="bottom", fontsize=6.5, color=fs.INK2)
    for ax, x, y, lim, xlabel, ylabel, title in panels:
        ax.plot([0, lim], [0, lim], color=fs.INK2, lw=0.9, ls=(0, (6, 3)), zorder=1)
        for st, c in cols.items():
            h = d[d.stride == st]
            ax.scatter(h[x], h[y], s=12, color=c, edgecolors=fs.SURFACE, linewidths=0.5, zorder=3, label=names[st])
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, loc="left", pad=3)
        if x == "true":
            rel = (d[y] - d.true) / d.true
            ax.text(0.04, 0.96, f"r = {np.corrcoef(d.true, d[y])[0, 1]:.2f}\nmedian error {100 * np.median(rel.abs()):.0f}%",
                    transform=ax.transAxes, va="top", fontsize=6.5, color=fs.INK2)
    fig.text(0.065, 0.97, f"{'LAP' if tracker == 'lap' else 'motile'} trackers.  Dashed line: y = x.  "
             "Ground truth is used only to score.", fontsize=6.5, color=fs.MUTED, va="top")
    axes[3].legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.04, 1.0), title="Linking", title_fontsize=6.5,
                   handletextpad=0.2, alignment="left")
    fs.save(fig, ROOT / "figures" / "local_metrics" / stem(tracker))


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--stage", choices=("all", "summary"), default="all")
    p.add_argument("--tracker", choices=("lap", "motile"), default="lap")
    a = p.parse_args()
    if a.stage == "all":
        jobs = [(s, img, op, st, a.tracker) for s in ("01", "02") for img in rdt.IMAGES for op in rdt.OPS for st in rdt.STRIDES
                if not ("_n" in img and op == "high_recall")]
        with ProcessPoolExecutor(a.workers) as ex:
            rows = list(ex.map(job, jobs))
        pd.DataFrame(rows).to_csv(RES / f"{stem(a.tracker)}.csv", index=False)
    summary(a.tracker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
