"""Hop count with an internal reference for the false-hop tail.  Real LAP trackers.

    python scripts/hop_reference_test.py [--workers N]

Hop count H: persisting labels whose motion residual (own step minus the median step of
their Delaunay neighbours, in median spacings) exceeds 0.5.  Correct cells also exceed 0.5
sometimes (biology, jitter), so H needs a clean-level correction.  Idea: estimate the false-hop
rate from cells that CANNOT have hopped, in the same frame pair.

Two "cannot hop" criteria, both observable without ground truth:
  far_from_event  no vanished label (at t) or new label (at t+1) within R spacings.  Squared-
                  distance assignment makes no crossing swaps, so wrong links form open chains
                  that end at a detection error, i.e. at a vanished or new label; a cell with no
                  such event nearby cannot be in a chain shorter than R.
  unambiguous     the nearest OTHER body at t+1 is more than twice as far from the cell's t
                  position as the body it was linked to, and more than 0.5 spacing away.
Corrected estimate:  H_hopcapable - rate(H among cannot-hop cells) x N_hopcapable.

Checks: share of true switched cells that fall in the cannot-hop group (should be ~0);
false-hop rate of correct cells in each group (should be equal); corrected estimate vs truth
over all trackers.  Writes data/_results/hop_reference.csv and figures/local_metrics/hop_reference.png.
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
import run_detector_tracking as rdt  # noqa: E402
from tracking_validation.linking import link_frames  # noqa: E402
from tracking_validation.local_metrics import predicted_positions  # noqa: E402

RES = ROOT / "data" / "_results"
THR = 0.5
RADII = (1.5, 2.0, 3.0)


def pair_cells(f0, f1, E):
    """Per evaluated persisting label: residual, distance to nearest event, ambiguity, switched."""
    l0 = [int(l) for l in f0["assigned_ids"]]; l1 = [int(l) for l in f1["assigned_ids"]]
    s0, s1 = set(l0), set(l1)
    if len(l0) < 5 or len(l1) < 5:
        return []
    pred, step, sp = predicted_positions(f0, f1, rdt.NEIGH_CAP)
    p0 = dict(zip(l0, f0["positions"])); p1 = dict(zip(l1, f1["positions"]))
    ev = [p0[l] for l in l0 if l not in s1] + [p1[l] for l in l1 if l not in s0]
    ev_tree = cKDTree(np.array(ev)) if ev else None
    body_tree = cKDTree(f1["positions"])
    idx1 = {l: i for i, l in enumerate(l1)}
    tid0 = dict(zip(l0, f0["true_ids"].tolist())); tid1 = dict(zip(l1, f1["true_ids"].tolist()))
    out = []
    for l, s in step.items():
        a, b = tid0[l], tid1[l]
        if a < 0 or b < 0 or a not in E:
            continue
        r = float(np.linalg.norm(p0[l] + s - pred[l])) / sp
        d_ev = float(ev_tree.query(p0[l])[0]) / sp if ev_tree is not None else np.inf
        dd, ii = body_tree.query(p0[l], k=min(3, len(l1)))
        alt = next((d for d, i in zip(dd, ii) if i != idx1[l]), np.inf)
        own = float(np.linalg.norm(s))
        unamb = (alt > 2 * own) and (alt > 0.5 * sp)
        out.append((r, d_ev, unamb, a != b))
    return out


def job(args):
    seq, img, op, stride = args
    label_map, gt, fidx, keep, spacing = rdt.load_gt(seq, stride)
    det = rdt.detection_frames(seq, img, op, label_map, fidx)
    rows = []
    for cap in rdt.CAPS:
        F = link_frames(det, cap * spacing, "lap").frames
        cells = []
        for t in range(len(F) - 1):
            E = keep[t] & set(int(c) for c in gt[t + 1]["true_ids"])
            cells += pair_cells(F[t], F[t + 1], E)
        c = pd.DataFrame(cells, columns=["r", "d_ev", "unamb", "sw"])
        hop = c.r > THR
        row = dict(seq=seq, img=img.lstrip("_") or "clean", op=op, stride=stride, cap=cap, n=len(c),
                   true_sw=int(c.sw.sum()), H=int(hop.sum()))
        groups = {f"far{R}": c.d_ev >= R for R in RADII}
        groups["unamb"] = c.unamb.astype(bool)
        for g, cannot in groups.items():
            n_cap = int((~cannot).sum()); n_cannot = int(cannot.sum())
            h_cap = int((hop & ~cannot).sum()); h_cannot = int((hop & cannot).sum())
            rate = h_cannot / n_cannot if n_cannot else np.nan
            row.update({f"{g}_n_cannot": n_cannot, f"{g}_sw_cannot": int((c.sw & cannot).sum()),
                        f"{g}_fhop_cannot": int((hop & cannot & ~c.sw).sum()), f"{g}_ok_cannot": int((cannot & ~c.sw).sum()),
                        f"{g}_fhop_can": int((hop & ~cannot & ~c.sw).sum()), f"{g}_ok_can": int((~cannot & ~c.sw).sum()),
                        f"{g}_est": h_cap - rate * n_cap if n_cannot else np.nan})
        rows.append(row)
    print(f"{seq} {img or 'clean':10s} {op:14s} stride {stride}: done", flush=True)
    return rows


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--stage", choices=("all", "summary"), default="all")
    a = p.parse_args()
    if a.stage == "all":
        jobs = [(s, img, op, st) for s in ("01", "02") for img in rdt.IMAGES for op in rdt.OPS for st in rdt.STRIDES
                if not ("_n" in img and op == "high_recall")]
        with ProcessPoolExecutor(a.workers) as ex:
            rows = [r for part in ex.map(job, jobs) for r in part]
        pd.DataFrame(rows).to_csv(RES / "hop_reference.csv", index=False)
    d = pd.read_csv(RES / "hop_reference.csv", dtype={"seq": str})
    tot = d.sum(numeric_only=True)
    print(f"\n{len(d)} trackers, {int(tot.n)} evaluated cells, {int(tot.true_sw)} switched")
    print(f"raw hop count H: {100 * tot.H / tot.n:.3f}% of cells vs true switched {100 * tot.true_sw / tot.n:.3f}%;  "
          f"per-tracker r {np.corrcoef(d.H / d.n, d.true_sw / d.n)[0, 1]:.2f}")
    print("\ncriterion        cannot-hop share | switched cells wrongly in it | false-hop rate of correct cells: cannot-hop vs hop-capable")
    for g in [f"far{R}" for R in RADII] + ["unamb"]:
        print(f"  {g:9s}   {100 * tot[f'{g}_n_cannot'] / tot.n:5.1f}% of cells | {100 * tot[f'{g}_sw_cannot'] / max(tot.true_sw, 1):5.1f}% "
              f"| {100 * tot[f'{g}_fhop_cannot'] / tot[f'{g}_ok_cannot']:.3f}% vs {100 * tot[f'{g}_fhop_can'] / tot[f'{g}_ok_can']:.3f}%")
    print("\ncorrected estimate vs truth (percent of evaluated cells, per tracker):")
    x = 100 * d.true_sw / d.n
    for g in ["raw"] + [f"far{R}" for R in RADII] + ["unamb"]:
        y = 100 * (d.H if g == "raw" else d[f"{g}_est"]) / d.n
        ok = np.isfinite(y)
        print(f"  {g:9s} est mean {y[ok].mean():.3f}  true mean {x[ok].mean():.3f}  slope {np.sum(x[ok] * y[ok]) / np.sum(x[ok] ** 2):.2f}  "
              f"r {np.corrcoef(x[ok], y[ok])[0, 1]:.2f}  mae {np.abs(y[ok] - x[ok]).mean():.3f}")
        for st in (1, 2, 3, 4):
            k = ok & (d.stride == st)
            print(f"      interval {st}: est {y[k].mean():.3f} true {x[k].mean():.3f} r {np.corrcoef(x[k], y[k])[0, 1]:.2f}")

    import matplotlib.pyplot as plt
    import figstyle as fs
    fs.apply()
    fig, axes = plt.subplots(1, 3, figsize=(fs.DOUBLE, 2.1), sharex=True, sharey=True,
                             gridspec_kw=dict(left=0.07, right=0.99, top=0.86, bottom=0.22, wspace=0.12))
    for ax, (g, title) in zip(axes, (("raw", "Raw hop count"), ("far2.0", "Minus far-from-event tail"), ("unamb", "Minus unambiguous-cell tail"))):
        y = 100 * (d.H if g == "raw" else d[f"{g}_est"]) / d.n
        lim = max(float(x.max()), float(np.nanmax(y))) * 1.08
        ax.plot([-1, lim], [-1, lim], color=fs.AXIS, lw=0.8)
        for st, col in zip((1, 2, 3, 4), ("#2a78d6", "#1baf7a", "#eda100", "#eb6834")):
            k = d.stride == st
            ax.scatter(x[k], y[k], s=9, color=col, linewidths=0, alpha=0.8, label=f"every {st}")
        ax.axhline(0, color=fs.AXIS, lw=0.6)
        ax.set_title(title, loc="left", pad=3); ax.set_xlabel("True switched cells (%)")
        ok = np.isfinite(y)
        ax.text(0.04, 0.96, f"r = {np.corrcoef(x[ok], y[ok])[0, 1]:.2f}", transform=ax.transAxes, va="top", fontsize=7)
    axes[0].set_ylabel("Estimated (%)"); axes[0].legend(fontsize=6, loc="lower right", title="frame", title_fontsize=6)
    fs.save(fig, ROOT / "figures" / "local_metrics" / "hop_reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
