"""Does adding the motion-residual channels to the demixing improve it?  Simulation only.

    python scripts/demix_resid_test.py

Sweep: data/_results/ce_error_sweep_fine_resid_delaunay.csv (fine sweep re-run with the
channels ``resid`` = mean motion residual / spacing and ``resid_out`` = share of cells with
residual > 0.5 spacing; see ce_error_sweep.motion_residuals).

Channel sets compared
  current7        survival, vanished, displaced, newcomer, rc, prc, pwio   (as in Figure 1)
  current7+resid  the same + resid, resid_out
  min5+resid      vanished, newcomer, displaced, pwio, rc + resid, resid_out

Tests (calibration and test always on DIFFERENT simulated tissues)
  single  calibrate on SIM seeds 100-103, estimate every single-type row of seeds 104-107
  mixture random mixtures of all four types (each U(0, 5%)) on fresh stage-matched slab
          tissues (seeds 30000+), calibration on all sweep SIM rows

Writes data/_results/demix_resid_single.csv, demix_resid_mixture.csv and
figures/local_metrics/demix_resid_mixture.png.
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
import ce_error_sweep as sweep  # noqa: E402
import figstyle as fs  # noqa: E402
from tracking_validation import apply_tracking_errors, realised_error_rates  # noqa: E402

RES = ROOT / "data" / "_results"
SWEEP = RES / "ce_error_sweep_fine_resid_delaunay.csv"
T = em.TYPES_4
SETS = {
    "current7": ["survival", "vanished", "displaced", "newcomer", "rc", "prc", "pwio"],
    "current7+resid": ["survival", "vanished", "displaced", "newcomer", "rc", "prc", "pwio", "resid", "resid_out"],
    "min5+resid": ["vanished", "newcomer", "displaced", "pwio", "rc", "resid", "resid_out"],
}
STAGES = ("50-100", "100-200", "200-360")
N_MIX_PER_STAGE = 25
MAX_RATE = 0.05


def calibrate_subset(df: pd.DataFrame, chans: list[str]):
    """em.calibrate on an in-memory subset of SIM rows."""
    tmp = RES / "_tmp_calib.csv"
    df.to_csv(tmp, index=False)
    em.CHANNELS[:] = chans
    out = em.calibrate(tmp, T, source="SIM")
    tmp.unlink()
    return out


def single_test(df: pd.DataFrame) -> pd.DataFrame:
    sim = df[df.source.str.startswith("SIM")]
    cal, test = sim[sim.pair <= 103], sim[sim.pair >= 104]
    rows = []
    for name, chans in SETS.items():
        A, noise, base = calibrate_subset(cal, chans)
        clean = test[test.error == "none"].set_index(["source", "pair"])
        for _, r in test[test.error != "none"].iterrows():
            b = clean.loc[(r.source, r.pair)]
            d = np.array([r[c] - b[c] for c in chans])        # same-tissue clean reference
            est = em.estimate(np.nan_to_num(d), A, noise)
            rows.append(dict(set=name, error=r.error, rate=r.rate_realised,
                             **{f"est_{t}": est[i] for i, t in enumerate(T)}))
    return pd.DataFrame(rows)


def mixture_test(df: pd.DataFrame) -> pd.DataFrame:
    meas = pd.read_csv(RES / "ce_stage_measurements.csv")
    if meas.seq.astype(str).str.startswith("Ph").any():
        raise SystemExit("ce_stage_measurements.csv is not the worm's; rerun the ce sweep first")
    sm = meas.groupby("stage")[["nn_med", "nn_p10", "disp_med", "disp_p90", "density", "z_extent"]].mean()
    rng = np.random.default_rng(11)
    recs = []
    for st in STAGES:
        for k in range(N_MIX_PER_STAGE):
            pair, box = em._sim_pair(sm, st, 30_000 + 100 * STAGES.index(st) + k)
            sp = sweep.median_nn(pair[0]["positions"])
            clean = sweep.evaluate_pair(pair, 8, 15.0, None)
            true = {t: float(rng.uniform(0, MAX_RATE)) for t in T}
            obs, log = apply_tracking_errors(pair, em.mixture_config(true, sp), box, np.random.default_rng(int(rng.integers(1 << 30))))
            real = realised_error_rates(pair, obs, log, first_frame_clean=True)
            recs.append((st, clean, sweep.evaluate_pair(obs, 8, 15.0, None), real))
        print(f"mixtures, stage {st}: {N_MIX_PER_STAGE} done", flush=True)
    rows = []
    for name, chans in SETS.items():
        A, noise, base = calibrate_subset(df[df.source.str.startswith("SIM")], chans)
        for i, (st, clean, m, real) in enumerate(recs):
            for ref, b in (("same tissue", clean), ("global", base)):
                est = em.estimate(np.nan_to_num(np.array([m[c] - b[c] for c in chans])), A, noise)
                rows.append(dict(set=name, ref=ref, mix=i, stage=st, **{f"true_{t}": real[t] for t in T},
                                 **{f"est_{t}": est[j] for j, t in enumerate(T)}))
    return pd.DataFrame(rows)


def main() -> int:
    em.SWITCH_CAP = 1.1; sweep.MODE = "delaunay"
    df = pd.read_csv(SWEEP)
    pd.set_option("display.width", 200)

    s = single_test(df); s.to_csv(RES / "demix_resid_single.csv", index=False)
    print("### single-type recovery on held-out simulated tissues: estimated / true of the injected type,"
          " and spill into the other three (percent points)")
    s["own"] = [r[f"est_{r.error}"] for _, r in s.iterrows()]
    s["spill"] = s[[f"est_{t}" for t in T]].sum(1) - s.own
    g = s.groupby(["set", "error"]).apply(lambda x: pd.Series(dict(recovered=x.own.sum() / x.rate.sum(),
                                                                   spill_pts=100 * x.spill.mean())), include_groups=False)
    print(g.unstack("error").round(2).to_string())

    m = mixture_test(df); m.to_csv(RES / "demix_resid_mixture.csv", index=False)
    print(f"\n### random mixtures ({len(m.mix.unique())} fresh simulated tissues, each type U(0, {100 * MAX_RATE:.0f}%)):"
          " RMSE in percent points (correlation)")
    out = []
    for (name, ref), g in m.groupby(["set", "ref"], sort=False):
        row = dict(set=name, ref=ref)
        for t in T:
            e, tr = 100 * g[f"est_{t}"], 100 * g[f"true_{t}"]
            row[t] = f"{np.sqrt(((e - tr) ** 2).mean()):.2f} ({np.corrcoef(e, tr)[0, 1]:.2f})"
        out.append(row)
    print(pd.DataFrame(out).to_string(index=False))

    fs.apply()
    g = m[m.ref == "same tissue"]
    fig, axes = plt.subplots(len(SETS), 4, figsize=(fs.DOUBLE, 1.6 * len(SETS) + 0.4), sharex=True, sharey=True,
                             gridspec_kw=dict(left=0.08, right=0.99, top=0.93, bottom=0.1, wspace=0.12, hspace=0.25))
    for r, name in enumerate(SETS):
        for c, t in enumerate(fs.ORDER4):
            ax = axes[r, c]; h = g[g.set == name]
            x, y = 100 * h[f"true_{t}"], 100 * h[f"est_{t}"]
            ax.plot([0, 6], [0, 6], color=fs.AXIS, lw=0.8)
            ax.scatter(x, y, s=8, color=fs.SERIES[t], linewidths=0, alpha=0.8)
            ax.set_xlim(0, 6); ax.set_ylim(0, 6)
            ax.text(0.04, 0.96, f"RMSE {np.sqrt(((y - x) ** 2).mean()):.2f}", transform=ax.transAxes, va="top", fontsize=6.5)
            if r == 0:
                ax.set_title(fs.LABELS[t], loc="left", pad=3)
            if r == len(SETS) - 1:
                ax.set_xlabel("True (%)")
        axes[r, 0].set_ylabel(f"{name}\nEstimated (%)")
    fs.save(fig, ROOT / "figures" / "local_metrics" / "demix_resid_mixture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
