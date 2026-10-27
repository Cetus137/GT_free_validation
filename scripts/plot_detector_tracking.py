"""Estimated vs true error rates for the real-tracker test (scripts/run_detector_tracking.py).

    python scripts/plot_detector_tracking.py [--cal real|sim] [--ref oracle|baseline]

One panel per error type plus total; each point is one tracker (sequence x image condition x
detector setting x frame interval x LAP cap).  Writes figures/detection/tracker_est_vs_true_{cal}_{ref}.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import figstyle as fs  # noqa: E402

T = ["miss", "switch", "frag", "fp"]


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--cal", default="real"); p.add_argument("--ref", default="oracle")
    a = p.parse_args()
    fs.apply()
    df = pd.read_csv(ROOT / "data" / "_results" / "detector_tracking_estimates.csv", dtype={"seq": str})
    df = df.dropna(subset=[f"est_{a.cal}_{a.ref}_switch"])
    fig, axes = plt.subplots(1, 5, figsize=(fs.DOUBLE, 1.8), gridspec_kw=dict(left=0.06, right=0.99, top=0.86, bottom=0.24, wspace=0.42))
    for ax, t in zip(axes, T + ["total"]):
        if t == "total":
            x = df[[f"true_{k}" for k in T]].sum(1); y = df[[f"est_{a.cal}_{a.ref}_{k}" for k in T]].sum(1); c = fs.INK
            title = "All errors"
        else:
            x, y, c, title = df[f"true_{t}"], df[f"est_{a.cal}_{a.ref}_{t}"], fs.SERIES[t], fs.LABELS[t]
        lim = float(max(x.max(), y.max())) * 1.08
        ax.plot([0, lim], [0, lim], color=fs.AXIS, lw=0.8, zorder=1)
        ax.scatter(x, y, s=9, color=c, alpha=0.7, linewidths=0, zorder=2)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        r = np.corrcoef(x, y)[0, 1] if y.std() > 0 else np.nan
        ax.set_title(title, loc="left", pad=3)
        ax.text(0.04, 0.96, f"r = {r:.2f}", transform=ax.transAxes, va="top", fontsize=7)
        ax.set_xlabel("True (%)")
    axes[0].set_ylabel("Estimated (%)")
    out = ROOT / "figures" / "detection" / f"tracker_est_vs_true_{a.cal}_{a.ref}"
    fs.save(fig, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
