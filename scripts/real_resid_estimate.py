"""Demixing with the motion-residual channels on the real trackers, scored against the truth.

    python scripts/real_resid_estimate.py

Input: data/_results/detector_tracking_pairs_resid.csv
       (run_detector_tracking.py --stage link --pairs-out detector_tracking_pairs_resid.csv)
Calibration: simulation only (SIM rows of ce_error_sweep_fine_resid_delaunay.csv).

Channel sets
  current7     survival, vanished, displaced, newcomer, rc, prc, pwio
  min5+resid   vanished, newcomer, displaced, pwio, rc, resid, resid_out

References for the clean level
  sim baseline  the simulation's clean pairs            (reference-free: what a user would run)
  oracle        error-free tracks on the same detections (uses GT: diagnostic of the baseline gap)

Truth as in run_detector_tracking.truth: frag includes dropouts that return, miss = cells lost
for good.  Writes data/_results/real_resid_estimates.csv and
figures/local_metrics/real_resid_switch.png.
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
from run_detector_tracking import truth  # noqa: E402

RES = ROOT / "data" / "_results"
T = em.TYPES_4
SETS = {"current7": ["survival", "vanished", "displaced", "newcomer", "rc", "prc", "pwio"],
        "min5+resid": ["vanished", "newcomer", "displaced", "pwio", "rc", "resid", "resid_out"]}


def main() -> int:
    df = pd.read_csv(RES / "detector_tracking_pairs_resid.csv", dtype={"seq": str})
    cal = {}
    for name, chans in SETS.items():
        em.CHANNELS[:] = chans
        cal[name] = (*em.calibrate(RES / "ce_error_sweep_fine_resid_delaunay.csv", T, source="SIM"), chans)
    keys = ["seq", "img", "op", "stride", "cap"]
    rows = []
    for key, g in df.groupby(keys):
        row = dict(zip(keys, key), **{f"true_{t}": v for t, v in truth(g).items()})
        for name, (A, noise, base, chans) in cal.items():
            trk = g[[f"trk_{c}" for c in chans]].mean().to_numpy()
            refs = {"sim baseline": base[chans].to_numpy(), "oracle": g[[f"orc_{c}" for c in chans]].mean().to_numpy()}
            for ref, b in refs.items():
                est = em.estimate(np.nan_to_num(trk - b), A, noise)
                for i, t in enumerate(T):
                    row[f"est|{name}|{ref}|{t}"] = 100 * est[i]
        row["orc_resid_out"] = g.orc_resid_out.mean(); row["trk_resid_out"] = g.trk_resid_out.mean()
        rows.append(row)
    out = pd.DataFrame(rows); out.to_csv(RES / "real_resid_estimates.csv", index=False)

    pd.set_option("display.width", 220)
    print(f"{len(out)} trackers.  Clean residual-outlier share: simulation {cal['min5+resid'][2]['resid_out']:.4f}, "
          f"error-free tracks on real detections {out.orc_resid_out.mean():.4f} (mean over trackers)")
    print("Truth means (%):", {t: round(out[f'true_{t}'].mean(), 2) for t in T})
    stats = []
    for name in SETS:
        for ref in ("sim baseline", "oracle"):
            r = dict(set=name, ref=ref)
            for t in T:
                e, tr = out[f"est|{name}|{ref}|{t}"], out[f"true_{t}"]
                c = np.corrcoef(e, tr)[0, 1] if e.std() > 0 else np.nan
                r[t] = f"{e.mean():.2f} / {np.abs(e - tr).mean():.2f} / {c:.2f}"
            stats.append(r)
    print("\nper type: estimated mean / mean abs error / correlation (percent of evaluated cells)")
    print(pd.DataFrame(stats).to_string(index=False))

    fs.apply()
    fig, axes = plt.subplots(1, 4, figsize=(fs.DOUBLE, 1.95), sharex=True, sharey=True,
                             gridspec_kw=dict(left=0.07, right=0.99, top=0.84, bottom=0.24, wspace=0.12))
    panels = [("current7", "sim baseline"), ("min5+resid", "sim baseline"), ("current7", "oracle"), ("min5+resid", "oracle")]
    x = out.true_switch
    lim = float(max(x.max(), max(out[f"est|{n}|{r}|switch"].max() for n, r in panels))) * 1.08
    for ax, (n, r) in zip(axes, panels):
        y = out[f"est|{n}|{r}|switch"]
        ax.plot([0, lim], [0, lim], color=fs.AXIS, lw=0.8)
        ax.scatter(x, y, s=9, color=fs.SERIES["switch"], alpha=0.7, linewidths=0)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_title(f"{n}\n{r}", loc="left", pad=3, fontsize=7.5)
        c = np.corrcoef(y, x)[0, 1] if y.std() > 0 else np.nan
        ax.text(0.04, 0.96, f"r = {c:.2f}", transform=ax.transAxes, va="top", fontsize=7)
        ax.set_xlabel("True switches (%)")
    axes[0].set_ylabel("Estimated (%)")
    fs.save(fig, ROOT / "figures" / "local_metrics" / "real_resid_switch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
