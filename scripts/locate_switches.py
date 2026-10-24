"""Test the per-cell displaced-share flag as a switch LOCATOR.

    python scripts/locate_switches.py

On held-out embryo frame pairs, inject identity switches at low rates (and,
for specificity, mixtures with the other error types), flag every evaluated
cell whose displaced share exceeds a threshold, and score:

  flag precision   fraction of flagged cells that have a switched cell among
                   their Delaunay neighbours (t0 or t1)
  event recall     fraction of switch events with at least one flagged cell
  label precision  fraction of "displaced" neighbour labels of flagged cells
                   that are genuinely switched labels (the locator's guess at
                   WHO switched)
  false alarms     flagged cells per clean frame pair

Writes data/_results/locate_switches.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tracking_validation import ErrorConfig, apply_tracking_errors  # noqa: E402
from tracking_validation.lineage import division_affected_cells, inherit_parent_labels  # noqa: E402
from tracking_validation.metrics import (  # noqa: E402
    compute_neighbour_continuity_observed, _frame_neighbour_map, _frame_neighbour_map_delaunay)
import ce_error_sweep as sweep  # noqa: E402

SCENARIOS = {
    "clean":             dict(),
    "switch 0.5%":       dict(switch=0.005),
    "switch 1%":         dict(switch=0.01),
    "switch 2%":         dict(switch=0.02),
    "switch 1% + others 1% each": dict(switch=0.01, frag=0.01, miss=0.01, fp=0.01),
    "others only, 2% each":       dict(frag=0.02, miss=0.02, fp=0.02),
}


def config(rates: dict, spacing: float) -> ErrorConfig:
    return ErrorConfig(p_miss=rates.get("miss", 0.0), fp_rate=rates.get("fp", 0.0), lambda_fp=0.0,
                       p_switch=rates.get("switch", 0.0) / 2.0, max_switch_distance=3.0 * spacing,
                       p_frag=rates.get("frag", 0.0), fp_placement="near_cells", fp_offset=spacing,
                       first_frame_clean=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.4)
    p.add_argument("--rc-threshold", type=float, default=0.3)
    p.add_argument("--mode", choices=("delaunay", "knn"), default="delaunay",
                   help="knn evaluates every cell (no hull exclusion), delaunay drops surface cells")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--cap", type=float, default=15.0)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "_results" / "locate_switches.csv")
    a = p.parse_args()
    sweep.MODE = a.mode

    def nbmap(pos, ids):
        if a.mode == "knn":
            return _frame_neighbour_map(pos, ids, min(a.k, len(ids) - 1), a.cap)[0]
        return _frame_neighbour_map_delaunay(pos, ids, max_distance=a.cap)[0]

    rows = []
    for seq in ("01", "02"):
        data = inherit_parent_labels(sweep.load_ce(ROOT / "data", ROOT / "data" / "_cache", seq), prefer="nearest")
        start = int(np.argmax(data.counts() >= 50)); frames = data.frames[start:]
        excluded = division_affected_cells(frames, k=a.k, max_neighbour_distance=a.cap, neighbourhood_mode=a.mode)
        hi = frames[-1]["positions"].max(axis=0); box = tuple(float(v) for v in (hi + 2.0))
        held = [ti for ti in range(len(frames) - 1) if ti % 4 != 0]
        for ti in held[::2]:
            pair = [frames[ti], frames[ti + 1]]
            p0 = pair[0]["positions"]; spacing = float(np.median(KDTree(p0).query(p0, k=2)[0][:, 1]))
            nb0 = nbmap(p0, pair[0]["assigned_ids"])
            for name, rates in SCENARIOS.items():
                for seed in range(a.seeds if rates else 1):
                    obs, log = apply_tracking_errors(pair, config(rates, spacing), box,
                                                     np.random.default_rng(10_000 * ti + seed))
                    switched = {ev["true_id_i"] for ev in log["switches"]} | {ev["true_id_j"] for ev in log["switches"]}
                    events = [(ev["true_id_i"], ev["true_id_j"]) for ev in log["switches"]]
                    res = compute_neighbour_continuity_observed(obs, k=a.k, neighbourhood_mode=a.mode,
                                                                max_neighbour_distance=a.cap, extra_metrics=True)
                    ids = res["cell_ids"]; disp = res["wio_displaced"][0]; rc = res["rank_correlation"][0]
                    nb1 = nbmap(obs[1]["positions"], obs[1]["assigned_ids"])
                    set1 = set(obs[1]["assigned_ids"].tolist())
                    per_cell = {int(c): (v, r) for c, v, r in zip(ids, disp, rc)
                                if np.isfinite(v) and c not in excluded[ti]}
                    # two flag rules: displaced share alone, and displaced OR rank-correlation outlier
                    flag_d = [c for c, (v, r) in per_cell.items() if v > a.threshold]
                    flag_dr = [c for c, (v, r) in per_cell.items() if v > a.threshold or (np.isfinite(r) and r < a.rc_threshold)]
                    def score(flagged):
                        self_hits = 0; adj_hits = 0; label_hits = 0; label_total = 0; hit_events: set = set()
                        for c in flagged:
                            n0 = set(nb0.get(c, [])); n1 = set(nb1.get(c, []))
                            if c in switched:                 # the flagged label itself was switched
                                self_hits += 1
                                for (i, j) in events:
                                    if c in (i, j):
                                        hit_events.add((i, j))
                                continue
                            adjacent = (n0 | n1) & switched   # a neighbour was switched
                            if adjacent:
                                adj_hits += 1
                                for (i, j) in events:
                                    if i in adjacent or j in adjacent:
                                        hit_events.add((i, j))
                                # the locator's guess at WHO: t0 neighbours that left but still exist at t1
                                guesses = [nb for nb in n0 if nb not in n1 and nb in set1]
                                label_total += len(guesses); label_hits += sum(g in switched for g in guesses)
                        return self_hits, adj_hits, label_hits, label_total, len(hit_events)
                    sd = score(flag_d); sdr = score(flag_dr)
                    # displaced share of switched cells themselves (how visible is a switch at its own cell?)
                    own = [per_cell[c][0] for c in switched if c in per_cell]
                    rows.append(dict(seq=seq, frame=start + ti, scenario=name, seed=seed, n_eval=len(per_cell),
                                     n_events=len(events), n_switched_evaluated=len(own),
                                     n_switched_over_thr=int(sum(v > a.threshold for v in own)),
                                     d_flagged=len(flag_d), d_self=sd[0], d_adjacent=sd[1], d_label_hits=sd[2],
                                     d_label_guesses=sd[3], d_events_hit=sd[4],
                                     dr_flagged=len(flag_dr), dr_self=sdr[0], dr_adjacent=sdr[1], dr_events_hit=sdr[4]))
    df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
    pd.set_option("display.width", 220)
    s = df.groupby("scenario").sum(numeric_only=True)
    s["pairs"] = df.groupby("scenario").size()
    out = pd.DataFrame({
        "pairs": s.pairs, "events": s.n_events,
        "switched cells over thr": s.n_switched_over_thr / s.n_switched_evaluated.replace(0, np.nan),
        "D: flags/pair": s.d_flagged / s.pairs,
        "D: precision": (s.d_self + s.d_adjacent) / s.d_flagged.replace(0, np.nan),
        "D: self share": s.d_self / s.d_flagged.replace(0, np.nan),
        "D: event recall": s.d_events_hit / s.n_events.replace(0, np.nan),
        "D: who-precision": s.d_label_hits / s.d_label_guesses.replace(0, np.nan),
        "D+RC: flags/pair": s.dr_flagged / s.pairs,
        "D+RC: precision": (s.dr_self + s.dr_adjacent) / s.dr_flagged.replace(0, np.nan),
        "D+RC: event recall": s.dr_events_hit / s.n_events.replace(0, np.nan),
    })
    print(f"Per-cell flags on held-out pairs, {a.mode} neighbourhoods. D = displaced share > {a.threshold}; "
          f"D+RC = that OR rank correlation < {a.rc_threshold}.\n"
          "'self share' = fraction of flags that are the switched cell itself (its whole neighbourhood jumps);\n"
          "'who-precision' = for flags on a NEIGHBOUR of a switch, fraction of departed labels that are the switched ones.\n")
    print(out.round(3).to_string())
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
