"""Cross-organism transfer: worm-calibrated estimator applied to ascidian frames.

    python scripts/cross_organism.py

Three calibrations evaluated on the ascidian single-type injection rows:
  worm matrix + worm baseline       (naive transfer, nothing measured locally)
  worm matrix + ascidian baseline   (response matrix borrowed, baseline measured locally)
  ascidian matrix + ascidian baseline (native reference)
Also reports response slopes per type for both organisms.
Writes data/_results/cross_organism.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import estimate_error_mix as em  # noqa: E402

R = ROOT / "data" / "_results"
WORM, ASC = R / "ce_error_sweep_out_delaunay.csv", R / "astec_error_sweep_delaunay.csv"


def slopes(f):
    df = pd.read_csv(f); real = df[~df.source.str.startswith("SIM")]
    clean = real[real.error == "none"]; err = real[real.error != "none"]
    out = dict(clean_wio=clean.wio.mean(), clean_rc=clean.rc.mean(), clean_displaced=clean.displaced.mean())
    for t in em.TYPES_4:
        g = err[err.error == t]; out[f"slope_{t}"] = -np.sum(g.rate_realised * g.d_wio) / np.sum(g.rate_realised ** 2)
    return out


def main() -> int:
    T = em.TYPES_4
    A_w, n_w, b_w = em.calibrate(WORM, T); A_a, n_a, b_a = em.calibrate(ASC, T)
    rows = []
    for label, A, n, b in (("worm matrix + worm baseline", A_w, n_w, b_w),
                           ("worm matrix + ascidian baseline", A_w, n_w, b_a),
                           ("ascidian matrix + ascidian baseline", A_a, n_a, b_a)):
        on = em.estimate_on_sweep(ASC, A, n, b, T)
        g = on.groupby(["error", "rate"]).agg(true=("rate_realised", "mean"), est=("est_same", "mean"), spill=("est_other", "mean")).reset_index()
        for _, r in g[g.rate <= 0.10].iterrows():
            rows.append(dict(calibration=label, error=r.error, rate=r.rate, true=r.true, est=r.est, recovered=r.est / r.true, spill=r.spill))
    df = pd.DataFrame(rows); df.to_csv(R / "cross_organism.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.precision", 2)
    print("### response slopes and clean baseline"); print(pd.DataFrame({"worm": slopes(WORM), "ascidian": slopes(ASC)}).to_string())
    for label, g in df.groupby("calibration", sort=False):
        print(f"\n### {label}: recovered fraction"); print(g.pivot(index="error", columns="rate", values="recovered").to_string())
    print("\nwrote", R / "cross_organism.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
