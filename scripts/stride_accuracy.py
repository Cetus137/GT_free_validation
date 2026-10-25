"""Estimator accuracy per error type at different frame intervals.

    python scripts/stride_accuracy.py

For each frame stride (1 = native 1.5 min), calibrate the estimator on one
recording's injection sweep at that stride and test on the other (both
directions): single-type accuracy (estimate of the injected type and spill
into other types, rates <= 10%) and the fixed-mix pooled-movie test on
frames subsampled at that stride.  Writes data/_results/stride_accuracy.csv.
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
SWEEPS = {1: "ce_error_sweep_out_delaunay.csv", 2: "ce_error_sweep_fs2_delaunay.csv",
          3: "ce_error_sweep_fs3_delaunay.csv", 4: "ce_error_sweep_fs4_delaunay.csv"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--strides", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--pooled", action="store_true", help="also run the pooled fixed-mix test per stride")
    a = p.parse_args()
    sweep.MODE = "delaunay"
    ns = argparse.Namespace(data_root=ROOT / "data", cache=ROOT / "data" / "_cache", cap=15.0, seed=0)
    T = em.TYPES_4
    rows = []
    for s in a.strides:
        sw = RESULTS / SWEEPS[s]
        if not sw.exists():
            continue
        for cal, test in (("CE01", "CE02"), ("CE02", "CE01")):
            A, noise, base = em.calibrate(sw, T, source=cal)
            on = em.estimate_on_sweep(sw, A, noise, base, T, source=test)
            g = on.groupby(["error", "rate"]).agg(true=("rate_realised", "mean"), est=("est_same", "mean"),
                                                  spill=("est_other", "mean")).reset_index()
            g = g[g.rate <= 0.10]
            for t in T:
                gt_ = g[g.error == t]
                e = 100 * (gt_.est - gt_.true)
                rows.append(dict(stride=s, interval_min=1.5 * s, calibrate=cal, test=test, error=t, kind="single",
                                 rmse=float(np.sqrt((e ** 2).mean())), bias=float(e.mean()),
                                 spill_max=float(100 * gt_.spill.max()),
                                 slope=float(np.sum(gt_.true * gt_.est) / np.sum(gt_.true ** 2))))
            if a.pooled:
                pooled = em.validate_constant(ns, A, noise, base, T, seqs=(test[-2:],), frame_stride=s)
                for t in T:
                    e = 100 * (pooled[f"est_{t}"] - pooled[f"true_{t}"])
                    rows.append(dict(stride=s, interval_min=1.5 * s, calibrate=cal, test=test, error=t, kind="pooled",
                                     rmse=float(np.sqrt((e ** 2).mean())), bias=float(e.mean()),
                                     spill_max=np.nan, slope=np.nan))
        print(f"stride {s} done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(RESULTS / "stride_accuracy.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.precision", 2)
    for kind in df.kind.unique():
        d = df[df.kind == kind]
        print(f"\n### {kind}: RMSE (percentage points), cross-embryo, mean of both directions")
        print(d.pivot_table(index="error", columns="interval_min", values="rmse", aggfunc="mean").to_string())
        if kind == "single":
            print("\n    recovered fraction (slope of estimate vs true, 1 = perfect)")
            print(d.pivot_table(index="error", columns="interval_min", values="slope", aggfunc="mean").to_string())
            print("\n    max spill into other types (points)")
            print(d.pivot_table(index="error", columns="interval_min", values="spill_max", aggfunc="mean").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
