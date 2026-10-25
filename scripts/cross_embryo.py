"""Cross-embryo validation of the error-mix estimator: calibrate on one recording,
estimate on the other, both directions, plus the pooled reference.

    python scripts/cross_embryo.py

Writes data/_results/error_mix_cross.csv (one row per setting x error type).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

import ce_error_sweep as sweep  # noqa: E402
import estimate_error_mix as em  # noqa: E402

RESULTS = ROOT / "data" / "_results"


def main() -> int:
    sweep.MODE = "delaunay"
    a = argparse.Namespace(data_root=ROOT / "data", cache=ROOT / "data" / "_cache", cap=15.0, seed=0)
    SW = RESULTS / "ce_error_sweep_slab_delaunay.csv"
    T = em.TYPES_4
    rows = []
    for cal, test, label in (("CE01", "CE02", "01 → 02"), ("CE02", "CE01", "02 → 01"), (None, None, "pooled")):
        A, noise, base = em.calibrate(SW, T, source=cal)
        on = em.estimate_on_sweep(SW, A, noise, base, T, source=test)
        g = on.groupby(["error", "rate"]).agg(true=("rate_realised", "mean"), est=("est_same", "mean")).reset_index()
        g = g[g.rate <= 0.10]
        seqs = (test[-2:],) if test else ("01", "02")
        pooled = em.validate_constant(a, A, noise, base, T, seqs=seqs)
        for t in T:
            e_single = 100 * (g[g.error == t].est - g[g.error == t].true)
            e_pooled = 100 * (pooled[f"est_{t}"] - pooled[f"true_{t}"])
            rows.append(dict(setting=label, error=t,
                             single_pair_rmse=float(np.sqrt((e_single ** 2).mean())),
                             pooled_rmse=float(np.sqrt((e_pooled ** 2).mean())),
                             pooled_max=float(e_pooled.abs().max())))
    df = pd.DataFrame(rows)
    out = RESULTS / "error_mix_cross.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200); pd.set_option("display.precision", 2)
    print(df.pivot(index="error", columns="setting", values="pooled_rmse").to_string())
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
