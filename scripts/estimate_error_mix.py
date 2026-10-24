"""Estimate the mix of tracking-error types from the metric vector alone.

    python scripts/estimate_error_mix.py            # calibrate + validate on CE mixtures

Calibration: from the single-error sweep on real embryo frames
(data/_results/ce_error_sweep_slab_delaunay.csv), fit the change in each
metric channel per unit of each error type (slope through the origin). This
gives a response matrix A (channels x error types).

Estimation: for a frame pair with metric changes d (channels), solve
    minimise || W (A r - d) ||  subject to r >= 0
for the error rates r, with W = 1 / channel noise (baseline spread over clean
frame pairs, floored).  The weighted identity overlap is not a channel because
it equals 1 - vanished - displaced exactly.

Validation: inject random MIXTURES of errors into held-out embryo frame pairs,
compute the metric changes against (a) that pair's own clean baseline and (b) a
global clean baseline (the realistic case), estimate, and compare to the
realised rates.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tracking_validation import ErrorConfig, apply_tracking_errors, realised_error_rates  # noqa: E402
from tracking_validation.lineage import division_affected_cells, inherit_parent_labels  # noqa: E402
import ce_error_sweep as sweep  # noqa: E402

CHANNELS = ["survival", "vanished", "displaced", "newcomer", "rc", "prc", "pwio"]
TYPES_4 = ["switch", "frag", "miss", "fp"]
TYPES_6 = ["switch", "frag", "miss", "fp", "split", "merge"]
NOISE_FLOOR = 0.003      # metric units; zero-noise channels get this floor


# --------------------------------------------------------------------------- calibration

def calibrate(sweep_csv: Path, types: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Response matrix A[channel, type] (metric change per unit rate), channel
    noise (sd over clean pairs), and global clean baseline per channel."""
    df = pd.read_csv(sweep_csv)
    real = df[df.source.str.startswith("CE")]
    base = real[real.error == "none"].set_index(["source", "pair"])
    err = real[real.error != "none"].copy()
    for ch in CHANNELS:
        b = base[ch].reindex(pd.MultiIndex.from_arrays([err.source, err.pair])).values
        err["d_" + ch] = err[ch].values - b
    A = pd.DataFrame(index=CHANNELS, columns=types, dtype=float)
    for t in types:
        g = err[err.error == t]
        for ch in CHANNELS:
            A.loc[ch, t] = np.sum(g.rate_realised * g["d_" + ch]) / np.sum(g.rate_realised ** 2)
    noise = base[CHANNELS].std().clip(lower=NOISE_FLOOR)
    baseline = base[CHANNELS].mean()
    return A, noise, baseline


def estimate(d: np.ndarray, A: pd.DataFrame, noise: pd.Series) -> np.ndarray:
    w = 1.0 / noise.values
    r, _ = nnls(A.values * w[:, None], d * w)
    return r


# --------------------------------------------------------------------------- validation

def mixture_config(rates: dict[str, float], spacing: float) -> ErrorConfig:
    return ErrorConfig(
        p_miss=rates.get("miss", 0.0), fp_rate=rates.get("fp", 0.0), lambda_fp=0.0,
        p_switch=rates.get("switch", 0.0) / 2.0, max_switch_distance=3.0 * spacing,
        p_frag=rates.get("frag", 0.0), p_merge=rates.get("merge", 0.0) / 2.0,
        p_split=rates.get("split", 0.0), split_distance=0.75 * spacing,
        fp_placement="near_cells", fp_offset=spacing, first_frame_clean=True,
    )


def validate(a, A, noise, baseline, types):
    rng = np.random.default_rng(a.seed)
    rows = []
    for seq in ("01", "02"):
        data = inherit_parent_labels(sweep.load_ce(a.data_root, a.cache, seq), prefer="nearest")
        counts = data.counts()
        start = int(np.argmax(counts >= 50))
        frames = data.frames[start:]
        excluded = division_affected_cells(frames, k=8, max_neighbour_distance=a.cap, neighbourhood_mode="delaunay")
        hi = frames[-1]["positions"].max(axis=0); box = tuple(float(v) for v in (hi + 2.0))
        # held-out pairs: the ones the stride-4 calibration sweep did NOT use
        candidates = [ti for ti in range(len(frames) - 1) if ti % 4 != 0]
        for ti in rng.choice(candidates, size=a.n_mixtures // 2, replace=False):
            pair = [frames[ti], frames[ti + 1]]
            spacing = sweep.median_nn(pair[0]["positions"])
            true = {t: float(rng.uniform(0, a.max_rate)) for t in types}
            if a.sparse and rng.random() < 0.5:        # half the mixtures have one dominant type
                keep = rng.choice(types); true = {t: (v if t == keep else v * 0.2) for t, v in true.items()}
            obs, log = apply_tracking_errors(pair, mixture_config(true, spacing), box, np.random.default_rng(int(rng.integers(1 << 30))))
            realised = realised_error_rates(pair, obs, log, first_frame_clean=True)
            clean = sweep.evaluate_pair(pair, 8, a.cap, excluded[ti])
            corrupt = sweep.evaluate_pair(obs, 8, a.cap, excluded[ti])
            d_own = np.array([corrupt[ch] - clean[ch] for ch in CHANNELS])
            d_glob = np.array([corrupt[ch] - baseline[ch] for ch in CHANNELS])
            est_own = estimate(d_own, A, noise)
            est_glob = estimate(d_glob, A, noise)
            row = dict(seq=seq, frame=start + ti, n0=len(pair[0]["assigned_ids"]))
            for i, t in enumerate(types):
                row[f"true_{t}"] = realised[t]
                row[f"own_{t}"] = est_own[i]
                row[f"glob_{t}"] = est_glob[i]
            rows.append(row)
    return pd.DataFrame(rows)


def report(val: pd.DataFrame, types: list[str], label: str) -> pd.DataFrame:
    out = []
    for t in types:
        for mode in ("own", "glob"):
            e = (val[f"{mode}_{t}"] - val[f"true_{t}"]) * 100
            out.append(dict(error=t, baseline=("same pair" if mode == "own" else "global"),
                            true_mean_pct=100 * val[f"true_{t}"].mean(), bias_pct=e.mean(), rmse_pct=np.sqrt((e ** 2).mean()),
                            corr=np.corrcoef(val[f"{mode}_{t}"], val[f"true_{t}"])[0, 1]))
    t = pd.DataFrame(out)
    print(f"\n### {label}: estimated vs realised rate, {len(val)} held-out mixtures (percent of cells)")
    print(t.round(2).to_string(index=False))
    return t


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", type=Path, default=ROOT / "data" / "_results" / "ce_error_sweep_slab_delaunay.csv")
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--cache", type=Path, default=ROOT / "data" / "_cache")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "_results" / "error_mix_validation.csv")
    p.add_argument("--cap", type=float, default=15.0)
    p.add_argument("--n-mixtures", type=int, default=60)
    p.add_argument("--max-rate", type=float, default=0.06, help="each type's rate ~ U(0, max_rate)")
    p.add_argument("--sparse", action="store_true", help="half the mixtures have one dominant type")
    p.add_argument("--six", action="store_true", help="solve for split and merge as well")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    sweep.MODE = "delaunay"
    types = TYPES_6 if a.six else TYPES_4

    A, noise, baseline = calibrate(a.sweep, types)
    pd.set_option("display.width", 220); pd.set_option("display.precision", 3)
    print("### Response matrix: metric change (x100) per 1% of cells affected")
    print((A * 100 / 100).round(3).to_string())
    print("\n### Channel noise (sd over clean pairs, x100) and global baseline")
    print(pd.DataFrame({"noise_x100": noise * 100, "baseline": baseline}).round(3).to_string())
    print(f"\ncondition number of weighted A: {np.linalg.cond(A.values / noise.values[:, None]):.1f}")

    val = validate(a, A, noise, baseline, types)
    val.to_csv(a.out, index=False)
    report(val, types, "4-type solver" if not a.six else "6-type solver")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
