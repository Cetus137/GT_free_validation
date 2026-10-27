"""Ground-truth-free estimate of fragmentation, missed detections and false positives on the
real trackers of scripts/run_detector_tracking.py, scored against the truth.

    python scripts/gtfree_estimate.py

Nothing in the calibration comes from ground truth:
  * response matrix and noise: simulation sweep only (ce_error_sweep_fine_delaunay.csv, SIM rows)
  * baselines: the label-counting channels have EXACT clean values (survival 1, vanished 0,
    newcomer 0) whatever the tissue; PWIO (label-free positional overlap) uses the simulation's
    own clean baseline
Switches are not estimated: they leave the counting channels unchanged and their geometric
channels need a tissue baseline (see the notes in the paper plan).

Variants
  counts        survival, vanished, newcomer, solved by NNLS (fully reference-free)
  counts+pwio   the same plus PWIO (PWIO baseline from simulation; it drifts with frame interval)
  parsimonious  frag = min(vanished, newcomer), miss = vanished - frag, fp = newcomer - frag:
                a lost identity and a new identity in the same pair are explained by ONE
                fragmentation rather than a miss plus a false positive (fully reference-free).
                The counting channels alone cannot tell a fragmentation from a miss plus a
                false positive, so some such rule is needed.

Caveat: cells next to divisions are excluded using the GT lineage, as in every analysis so far;
a deployed version would use the tracker's own division calls.

Truth: frag = frag + reappear (a cell missed and found again is one broken track),
miss = miss - reappear (cells lost for good), fp = fp.  Percent of evaluated cells.
Writes data/_results/gtfree_estimates.csv and figures/detection/gtfree_est_vs_true.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import estimate_error_mix as em  # noqa: E402
import figstyle as fs  # noqa: E402

RES = ROOT / "data" / "_results"
TYPES = ["frag", "miss", "fp"]
EXACT = {"survival": 1.0, "vanished": 0.0, "newcomer": 0.0}
VARIANTS = {"counts": ["survival", "vanished", "newcomer"],
            "counts+pwio": ["survival", "vanished", "newcomer", "pwio"]}


def main() -> int:
    df = pd.read_csv(RES / "detector_tracking_pairs.csv", dtype={"seq": str})
    keys = ["seq", "img", "op", "stride", "cap"]
    rows = []
    cal = {}
    for name, chans in VARIANTS.items():
        em.CHANNELS[:] = chans
        A, noise, base = em.calibrate(RES / "ce_error_sweep_fine_delaunay.csv", TYPES, source="SIM")
        b = pd.Series({c: EXACT.get(c, base[c]) for c in chans})
        cal[name] = (A, noise, b, chans)
        print(f"### {name}: response matrix from simulation (per unit rate)"); print(A.round(3).to_string())
        print("    baseline used:", b.round(4).to_dict())
    for key, g in df.groupby(keys):
        n = g.n_eval.sum()
        row = dict(zip(keys, key), true_frag=100 * (g.frag.sum() + g.reappear.sum()) / n,
                   true_miss=100 * max(g.miss.sum() - g.reappear.sum(), 0) / n, true_fp=100 * g.fp.sum() / n,
                   true_switch=100 * g.switch.sum() / n)
        for name, (A, noise, b, chans) in cal.items():
            d = g[[f"trk_{c}" for c in chans]].mean().to_numpy() - b.to_numpy()
            est = em.estimate(np.nan_to_num(d), A, noise)
            for i, t in enumerate(TYPES):
                row[f"est_{name}_{t}"] = 100 * est[i]
        # parsimonious split: a vanished identity and a new identity in the same pair are
        # explained by ONE fragmentation rather than a miss plus a false positive
        A = cal["counts"][0]
        V = 100 * g.trk_vanished.mean() / A.loc["vanished", "miss"]
        N = 100 * g.trk_newcomer.mean() / A.loc["newcomer", "fp"]
        f = min(V, N)
        row.update(est_parsimonious_frag=f, est_parsimonious_miss=V - f, est_parsimonious_fp=N - f,
                   obs_vanished=V, obs_newcomer=N,
                   true_vanished=row["true_frag"] + row["true_miss"], true_newcomer=row["true_frag"] + row["true_fp"])
        rows.append(row)
    out = pd.DataFrame(rows); out.to_csv(RES / "gtfree_estimates.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\n{len(out)} trackers; truth (percent of evaluated cells):")
    print(out[[f"true_{t}" for t in TYPES]].describe().loc[["min", "50%", "max"]].round(2).to_string())
    for obs in ("vanished", "newcomer"):
        x, y = out[f"true_{obs}"], out[f"obs_{obs}"]
        print(f"{obs}: true mean {x.mean():.2f}, observed {y.mean():.2f}, r {np.corrcoef(x, y)[0, 1]:.2f}")
    for name in list(VARIANTS) + ["parsimonious"]:
        stats = []
        for t in TYPES + ["sum"]:
            if t == "sum":
                tr = out[[f"true_{k}" for k in TYPES]].sum(1); e = out[[f"est_{name}_{k}" for k in TYPES]].sum(1)
            else:
                tr, e = out[f"true_{t}"], out[f"est_{name}_{t}"]
            stats.append(dict(type=t, true_mean=tr.mean(), est_mean=e.mean(), bias=(e - tr).mean(),
                              mae=(e - tr).abs().mean(), corr=np.corrcoef(e, tr)[0, 1] if e.std() > 0 else np.nan))
        print(f"\n### {name}: estimated vs true"); print(pd.DataFrame(stats).round(2).to_string(index=False))

    # figure: parsimonious variant, one panel per type plus their sum
    fs.apply()
    name = "parsimonious"
    fig, axes = plt.subplots(1, 4, figsize=(fs.DOUBLE, 1.9), gridspec_kw=dict(left=0.07, right=0.99, top=0.86, bottom=0.24, wspace=0.38))
    for ax, t in zip(axes, ["miss", "frag", "fp", "sum"]):
        if t == "sum":
            x = out[[f"true_{k}" for k in TYPES]].sum(1); y = out[[f"est_{name}_{k}" for k in TYPES]].sum(1)
            c, title = fs.INK, "Sum of the three"
        else:
            x, y, c, title = out[f"true_{t}"], out[f"est_{name}_{t}"], fs.SERIES[t], fs.LABELS[t]
        lim = float(max(x.max(), y.max())) * 1.08
        ax.plot([0, lim], [0, lim], color=fs.AXIS, lw=0.8, zorder=1)
        ax.scatter(x, y, s=9, color=c, alpha=0.7, linewidths=0, zorder=2)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_title(title, loc="left", pad=3); ax.set_xlabel("True (%)")
        ax.text(0.04, 0.96, f"r = {np.corrcoef(x, y)[0, 1]:.2f}", transform=ax.transAxes, va="top", fontsize=7)
    axes[0].set_ylabel("Estimated (%)")
    fs.save(fig, ROOT / "figures" / "detection" / "gtfree_est_vs_true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
