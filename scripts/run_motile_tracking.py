"""motile (global, no divisions) on the same 208 detector configurations as the LAP trackers.

    /Users/ewheeler/miniforge3/envs/motile_env/bin/python scripts/run_motile_tracking.py
    ... --stage compare        # LAP vs motile summary + figure (any environment)

Same detections, frame intervals and link distances as scripts/run_detector_tracking.py; the
linker is tracking_validation.linking_motile.link_motile.  Per frame pair it stores the same
GT error counts and metric channels (including the motion residual) as the LAP run, plus the
number of switched cells in two-cell swaps and in hop chains.

Writes data/_results/detector_tracking_pairs_motile.csv (+ lap_switch_shapes.csv for LAP's
swap/chain split) and figures/motile/lap_vs_motile.png.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import run_detector_tracking as rdt  # noqa: E402
from tracking_validation.linking import link_frames  # noqa: E402
from tracking_validation.tracking_eval import sequence_errors  # noqa: E402

RES = ROOT / "data" / "_results"
FIG = ROOT / "figures" / "motile"
KEYS = ["seq", "img", "op", "stride", "cap"]


def switch_shapes(f0: dict, f1: dict, E: set[int]) -> tuple[int, int]:
    """Switched evaluated cells at t+1, split into two-cell swaps and hop chains."""
    tid0 = {int(l): int(c) for l, c in zip(f0["assigned_ids"], f0["true_ids"])}
    tid1 = {int(l): int(c) for l, c in zip(f1["assigned_ids"], f1["true_ids"])}
    move = {tid0[l]: tid1[l] for l in tid0 if l in tid1 and tid0[l] >= 0 and tid1[l] >= 0}
    swap = chain = 0
    for a, b in move.items():
        if a == b or a not in E:
            continue
        if move.get(b) == a:
            swap += 1
        else:
            chain += 1
    return swap, chain


def responses(seq: str, img: str, op: str, fidx: list[int]) -> list[np.ndarray]:
    d = pd.read_csv(RES / "detections" / f"ce_{seq}{img}_{op}.csv")
    return [d[d.frame == f].response.to_numpy() for f in fidx]


def job(args):
    from tracking_validation.linking_motile import link_motile
    seq, img, op, stride = args
    label_map, gt, fidx, keep, spacing = rdt.load_gt(seq, stride)
    det = rdt.detection_frames(seq, img, op, label_map, fidx)
    resp = responses(seq, img, op, fidx)
    rows = []
    for cap in rdt.CAPS:
        res = link_motile(det, cap * spacing, resp)
        errs = sequence_errors(res.frames, gt, keep)
        ch = rdt.seq_channels(res.frames, keep)
        for t, e in enumerate(errs):
            E = keep[t] & set(int(c) for c in gt[t + 1]["true_ids"])
            sw, chn = switch_shapes(res.frames[t], res.frames[t + 1], E)
            rows.append(dict(tracker="motile", seq=seq, img=img.lstrip("_") or "clean", op=op, stride=stride, cap=cap,
                             frame=fidx[t], n_cells=len(gt[t]["true_ids"]), spacing=spacing,
                             dets_kept=len(res.frames[t + 1]["positions"]), dets_in=len(det[t + 1]["positions"]),
                             swap_cells=sw, chain_cells=chn, **e, **{f"trk_{k}": v for k, v in ch[t].items()}))
        n = sum(e["n_eval"] for e in errs)
        print(f"motile {seq} {img or 'clean':10s} {op:14s} stride {stride} cap {cap}: "
              f"switch {100 * sum(e['switch'] for e in errs) / n:.2f} frag {100 * sum(e['frag'] for e in errs) / n:.2f} "
              f"fp {100 * sum(e['fp'] for e in errs) / n:.2f}", flush=True)
    return rows


def lap_shapes_job(args):
    seq, img, op, stride = args
    label_map, gt, fidx, keep, spacing = rdt.load_gt(seq, stride)
    det = rdt.detection_frames(seq, img, op, label_map, fidx)
    out = []
    for cap in rdt.CAPS:
        res = link_frames(det, cap * spacing, "lap")
        sw = ch = 0
        for t in range(len(gt) - 1):
            E = keep[t] & set(int(c) for c in gt[t + 1]["true_ids"])
            a, b = switch_shapes(res.frames[t], res.frames[t + 1], E); sw += a; ch += b
        out.append(dict(seq=seq, img=img.lstrip("_") or "clean", op=op, stride=stride, cap=cap, swap_cells=sw, chain_cells=ch))
    return out


def jobs():
    return [(s, img, op, st) for s in ("01", "02") for img in rdt.IMAGES for op in rdt.OPS for st in rdt.STRIDES
            if not ("_n" in img and op == "high_recall")]


def stage_run(workers: int) -> None:
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for part in ex.map(job, jobs()) for r in part]
    pd.DataFrame(rows).to_csv(RES / "detector_tracking_pairs_motile.csv", index=False)
    with ProcessPoolExecutor(workers) as ex:
        shapes = [r for part in ex.map(lap_shapes_job, jobs()) for r in part]
    pd.DataFrame(shapes).to_csv(RES / "lap_switch_shapes.csv", index=False)
    print("wrote", RES / "detector_tracking_pairs_motile.csv", RES / "lap_switch_shapes.csv")


def stage_compare() -> None:
    import matplotlib.pyplot as plt
    import figstyle as fs
    lap = pd.read_csv(RES / "detector_tracking_pairs_resid.csv", dtype={"seq": str})
    mot = pd.read_csv(RES / "detector_tracking_pairs_motile.csv", dtype={"seq": str})
    shapes_lap = pd.read_csv(RES / "lap_switch_shapes.csv", dtype={"seq": str})
    rows = []
    for name, df in (("LAP", lap), ("motile", mot)):
        for key, g in df.groupby(KEYS):
            t = rdt.truth(g)
            rows.append(dict(tracker=name, **dict(zip(KEYS, key)), **{f"true_{k}": v for k, v in t.items()},
                             true_total=sum(t.values()), wio=100 * (1 - g.trk_wio.mean())))
    s = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("### mean true error rates (percent of evaluated cells), 208 configurations each")
    print(s.groupby("tracker")[["true_switch", "true_frag", "true_miss", "true_fp", "true_total"]].mean().round(2).to_string())
    m = s.pivot_table(index=KEYS, columns="tracker", values="true_total")
    print(f"\nmotile better than LAP in {100 * (m.motile < m.LAP - 1e-9).mean():.0f}% of configurations, "
          f"equal in {100 * np.isclose(m.motile, m.LAP).mean():.0f}%; mean difference {(m.motile - m.LAP).mean():+.3f} points")
    ms = mot.groupby(KEYS)[["swap_cells", "chain_cells"]].sum()
    print(f"\nswitched cells: LAP swaps {int(shapes_lap.swap_cells.sum())} / chains {int(shapes_lap.chain_cells.sum())};  "
          f"motile swaps {int(ms.swap_cells.sum())} / chains {int(ms.chain_cells.sum())}")
    print(f"detections dropped by motile: {100 * (1 - mot.dets_kept.sum() / mot.dets_in.sum()):.2f}%")

    fs.apply(); FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(fs.DOUBLE, 1.95), gridspec_kw=dict(left=0.07, right=0.99, top=0.86, bottom=0.24, wspace=0.4))
    for ax, t in zip(axes, ["switch", "frag", "miss", "fp"]):
        p = s.pivot_table(index=KEYS, columns="tracker", values=f"true_{t}")
        lim = float(p.max().max()) * 1.08
        ax.plot([0, lim], [0, lim], color=fs.AXIS, lw=0.8)
        ax.scatter(p.LAP, p.motile, s=9, color=fs.SERIES[t], alpha=0.7, linewidths=0)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_title(fs.LABELS[t], loc="left", pad=3)
        ax.set_xlabel("LAP, true (%)")
    axes[0].set_ylabel("motile, true (%)")
    fs.save(fig, FIG / "lap_vs_motile")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("all", "run", "compare"), default="all")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = p.parse_args()
    if a.stage in ("all", "run"):
        stage_run(a.workers)
    if a.stage in ("all", "compare"):
        stage_compare()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
