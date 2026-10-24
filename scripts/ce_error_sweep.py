"""Inject tracking errors into Fluo-N3DH-CE ground truth and (optionally) into
stage-matched simulations, and record the neighbourhood-continuity response.

    python scripts/ce_error_sweep.py            # real frames only
    python scripts/ce_error_sweep.py --sim      # plus stage-matched simulations

Real frames: both recordings, frames with >= 50 cells, one daughter inherits the
mother's label, and cells whose neighbourhood touches a division are excluded
from evaluation (division-free sample).  Each sampled consecutive frame pair is
corrupted with ONE error type at ONE nominal rate, several seeds, and the
metrics are computed on the corrupted pair.  The FIRST frame of each pair is
treated as truth and left untouched: all detection errors land in t1, and
switches are wrong links into t1.  Error distances are relative to the pair's
median nearest-neighbour spacing s:

    switch cap = 3 s,  split separation = 0.75 s,  FP offset = s (near cells)

Simulations: for each of three stages (50-100, 100-200, 200-360 cells) the
simulator is configured from the embryo's measured spacing, displacement and
density at that stage, with 1000 cells in a box of matching density, 2 frames,
and the same sweep is run.

Output: a long-format CSV, one row per (source, pair/seed, error type, rate),
plus a summary table on stdout.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, KDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracking_validation import (  # noqa: E402
    ConstraintConfig, ErrorConfig, MotionConfig, SamplingConfig, SimulationConfig,
    apply_tracking_errors, frames_from_array, realised_error_rates, simulate_sequence,
)
from tracking_validation.ctc import read_ctc_tracks  # noqa: E402
from tracking_validation.lineage import division_affected_cells, inherit_parent_labels, mask_cells  # noqa: E402
from tracking_validation.metrics import compute_neighbour_continuity_observed  # noqa: E402

ERROR_TYPES = ("miss", "fp", "switch", "frag", "merge", "split")   # filtered by --types
METRIC_KEYS = {
    "wio": "weighted_identity_overlap", "pwio": "positional_weighted_identity_overlap",
    "rc": "rank_correlation", "prc": "penalised_rank_correlation",
    "vanished": "wio_vanished", "displaced": "wio_displaced", "newcomer": "newcomer_fraction",
}
STAGES = [(50, 100, "50-100"), (100, 200, "100-200"), (200, 400, "200-360")]


def nanmean(a) -> float:
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def median_nn(pos: np.ndarray) -> float:
    d, _ = KDTree(pos).query(pos, k=2)
    return float(np.median(d[:, 1]))


def error_config(kind: str, rate: float, spacing: float) -> ErrorConfig:
    cfg = dict(p_miss=0.0, lambda_fp=0.0, p_switch=0.0, p_merge=0.0, p_split=0.0,
               split_distance=0.75 * spacing, max_switch_distance=3.0 * spacing,
               fp_placement="near_cells", fp_offset=spacing, first_frame_clean=True)
    if kind == "miss":
        cfg["p_miss"] = rate
    elif kind == "fp":
        cfg["fp_rate"] = rate
    elif kind == "switch":
        cfg["p_switch"] = rate / 2.0          # two cells per event -> ~rate of cells affected
    elif kind == "frag":
        cfg["p_frag"] = rate
    elif kind == "merge":
        cfg["p_merge"] = rate / 2.0
    elif kind == "split":
        cfg["p_split"] = rate
    else:
        raise ValueError(kind)
    return ErrorConfig(**cfg)


MODE = "delaunay"   # set from --mode in main()


def evaluate_pair(pair: list[dict], k: int, cap: float, excluded: set[int] | None) -> dict:
    res = compute_neighbour_continuity_observed(pair, k=k, max_neighbour_distance=cap,
                                                neighbourhood_mode=MODE, extra_metrics=True)
    if excluded:
        res = mask_cells(res, [excluded])
    out = {name: nanmean(res[key][0]) for name, key in METRIC_KEYS.items()}
    out["survival"] = float(res["track_survival"][0])
    out["n_eval"] = int(np.isfinite(res["weighted_identity_overlap"][0]).sum())
    return out


def sweep_pair(pair: list[dict], box, k, cap, excluded, rates, seeds, meta: dict) -> list[dict]:
    rows = []
    spacing = median_nn(pair[0]["positions"])
    base = evaluate_pair(pair, k, cap, excluded)
    rows.append({**meta, "error": "none", "rate": 0.0, "rate_realised": 0.0, "seed": -1, "spacing": spacing, **base})
    for kind in ERROR_TYPES:
        for rate in rates:
            for seed in range(seeds):
                rng = np.random.default_rng(hash((meta["source"], meta["pair"], kind, rate, seed)) % 2**32)
                obs, log = apply_tracking_errors(pair, error_config(kind, rate, spacing), box, rng)
                real = realised_error_rates(pair, obs, log, first_frame_clean=True)[kind]
                m = evaluate_pair(obs, k, cap, excluded)
                rows.append({**meta, "error": kind, "rate": rate, "rate_realised": real, "seed": seed,
                             "spacing": spacing, **m, **{f"d_{n}": m[n] - base[n] for n in METRIC_KEYS}})
    return rows


# --------------------------------------------------------------------------- real frames

def load_ce(data_root: Path, cache: Path, seq: str):
    pkl = cache / f"ce{seq}.pkl"
    if pkl.exists():
        return pickle.loads(pkl.read_bytes())
    data = read_ctc_tracks(data_root / "Fluo-N3DH-CE" / f"{seq}_GT" / "TRA", dataset="Fluo-N3DH-CE")
    cache.mkdir(parents=True, exist_ok=True)
    pkl.write_bytes(pickle.dumps(data))
    return data


def stage_of(n: int) -> str:
    for lo, hi, name in STAGES:
        if lo <= n < hi:
            return name
    return "other"


def run_real(a, rates) -> tuple[list[dict], pd.DataFrame]:
    rows, measures = [], []
    for seq in ("01", "02"):
        data = inherit_parent_labels(load_ce(a.data_root, a.cache, seq), prefer="nearest")
        counts = data.counts()
        start = int(np.argmax(counts >= a.min_cells))
        frames = data.frames[start:]
        excluded = division_affected_cells(frames, k=a.k, max_neighbour_distance=a.cap, neighbourhood_mode=MODE)
        lo = frames[0]["positions"].min(axis=0); hi = frames[-1]["positions"].max(axis=0)
        box = tuple(float(v) for v in (hi + 2.0))          # FPs are clipped to this box
        t0 = time.time()
        for ti in range(0, len(frames) - 1, a.stride):
            pair = [frames[ti], frames[ti + 1]]
            n0 = len(pair[0]["assigned_ids"])
            # stage measurements for the simulator
            m0 = dict(zip(pair[0]["assigned_ids"].tolist(), pair[0]["positions"]))
            m1 = dict(zip(pair[1]["assigned_ids"].tolist(), pair[1]["positions"]))
            disp = np.array([np.linalg.norm(m1[c] - m0[c]) for c in m0 if c in m1])
            nn = KDTree(pair[0]["positions"]).query(pair[0]["positions"], k=2)[0][:, 1]
            measures.append(dict(seq=seq, frame=start + ti, n=n0, stage=stage_of(n0),
                                 nn_med=np.median(nn), nn_p10=np.percentile(nn, 10),
                                 disp_med=np.median(disp), disp_p90=np.percentile(disp, 90),
                                 density=n0 / ConvexHull(pair[0]["positions"]).volume))
            meta = dict(source=f"CE{seq}", pair=start + ti, n0=n0, stage=stage_of(n0))
            rows += sweep_pair(pair, box, a.k, a.cap, excluded[ti], rates, a.seeds, meta)
        print(f"CE {seq}: {len([r for r in rows if r['source'] == f'CE{seq}'])} rows in {time.time() - t0:.0f}s", flush=True)
    return rows, pd.DataFrame(measures)


# --------------------------------------------------------------------------- simulations

def sim_config_from_stage(m: pd.Series, n_cells: int, seed: int) -> SimulationConfig:
    side = float((n_cells / m.density) ** (1 / 3))
    floor = float(m.nn_p10)
    sd = float(max(0.1, 0.3 * (m.nn_med - m.nn_p10)))
    return SimulationConfig(
        num_cells=n_cells, num_timepoints=2, seed=seed, box_size=(side, side, side),
        sampling=SamplingConfig(base_min_distance=floor, density_alpha=0.0, min_distance_sd=sd,
                                max_attempts=200_000),
        motion=MotionConfig(dt=1.0, speed_mean=float(m.disp_med), speed_std=float(0.4 * m.disp_med),
                            speed_step_std=float(0.3 * m.disp_med), directional_noise=0.1),
        constraints=ConstraintConfig(min_distance=floor, strength=0.7, iterations=8),
    )


def run_sim(a, rates, measures: pd.DataFrame) -> list[dict]:
    rows = []
    stage_means = measures.groupby("stage")[["nn_med", "nn_p10", "disp_med", "disp_p90", "density"]].mean()
    for _, _, stage in STAGES:
        if stage not in stage_means.index:
            continue
        m = stage_means.loc[stage]
        t0 = time.time()
        for seed in range(a.sim_seeds):
            cfg = sim_config_from_stage(m, a.sim_cells, seed)
            sim = simulate_sequence(cfg)
            pair = frames_from_array(sim.frames)
            # border exclusion: cells within one spacing of a box face at either time
            margin = float(m.nn_med)
            box = cfg.box_size
            inside = lambda p: np.all((p > margin) & (p < np.asarray(box) - margin), axis=1)
            keep = inside(pair[0]["positions"]) & inside(pair[1]["positions"])
            excluded = set(pair[0]["assigned_ids"][~keep].tolist())
            realised_nn = median_nn(pair[0]["positions"])
            meta = dict(source=f"SIM-{stage}", pair=seed, n0=a.sim_cells, stage=stage, sim_nn_med=realised_nn,
                        sim_box=box[0])
            rows += sweep_pair(pair, box, a.k, a.cap, excluded, rates, 1, meta)
        print(f"SIM {stage}: target spacing {m.nn_med:.2f}, realised {realised_nn:.2f} µm; "
              f"box {box[0]:.1f} µm; {time.time() - t0:.0f}s", flush=True)
    return rows


# --------------------------------------------------------------------------- main

def main() -> int:
    global MODE, ERROR_TYPES
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--cache", type=Path, default=ROOT / "data" / "_cache")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "_results" / "ce_error_sweep.csv")
    p.add_argument("--mode", choices=("delaunay", "knn"), default="delaunay",
                   help="neighbourhood definition (default delaunay; k is ignored then)")
    p.add_argument("--k", type=int, default=8, help="neighbours for --mode knn")
    p.add_argument("--cap", type=float, default=15.0, help="neighbour distance cap, microns")
    p.add_argument("--min-cells", type=int, default=50)
    p.add_argument("--stride", type=int, default=4, help="use every n-th frame pair")
    p.add_argument("--seeds", type=int, default=3, help="error seeds per real pair")
    p.add_argument("--rates", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.10, 0.20])
    p.add_argument("--types", nargs="+", default=list(ERROR_TYPES), choices=ERROR_TYPES,
                   help="error types to inject (default all)")
    p.add_argument("--sim", action="store_true", help="also run stage-matched simulations")
    p.add_argument("--sim-cells", type=int, default=1000)
    p.add_argument("--sim-seeds", type=int, default=5)
    a = p.parse_args()
    MODE = a.mode
    ERROR_TYPES = tuple(a.types)

    rows, measures = run_real(a, a.rates)
    if a.sim:
        rows += run_sim(a, a.rates, measures)

    df = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    measures.to_csv(a.out.with_name("ce_stage_measurements.csv"), index=False)
    print(f"\nwrote {a.out} ({len(df)} rows); neighbourhood mode = {MODE}; first frame clean")

    pd.set_option("display.width", 220); pd.set_option("display.precision", 3)
    print("\n### Stage measurements (means): spacing / displacement in µm, density in cells/µm³")
    print(measures.groupby("stage")[["n", "nn_med", "nn_p10", "disp_med", "density"]].mean().to_string())

    err = df[df.error != "none"].copy()
    err["src"] = err.source.str.replace(r"CE0[12]", "CE", regex=True)
    print("\n### Baseline (no error) WIO by source")
    print(df[df.error == "none"].groupby("source")[["wio", "pwio", "rc", "n_eval"]].mean().to_string())
    print("\n### Drop in WIO from the same-pair baseline, mean over pairs/seeds: rows = error type, cols = nominal rate")
    for src, g in err.groupby("src"):
        print(f"\n[{src}]")
        print(g.pivot_table(index="error", columns="rate", values="d_wio", aggfunc="mean").to_string())
    print("\n### Realised vs nominal rate (mean), by error type")
    print(err.pivot_table(index="error", columns="rate", values="rate_realised", aggfunc="mean").to_string())
    print("\n### Fingerprint at nominal rate 0.10: change in decomposition (real CE)")
    f = err[(err.src == "CE") & (err.rate == 0.10)]
    print(f.groupby("error")[["d_wio", "d_pwio", "d_vanished", "d_displaced", "d_newcomer", "d_rc"]].mean().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
