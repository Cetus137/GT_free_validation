# tracking_validation

Two things, nothing else:

1. **Simulations** — a 3D tissue of cell centroids over time (`simulate_sequence`), with
   spacing, motion and cell count calibrated to real data, plus injection of known
   tracking errors (`apply_tracking_errors`): missed detections, false positives,
   merges, splits and ID switches (same-frame or cross-frame rule).
2. **Metrics** — neighbourhood-continuity metrics on any track set, simulated or real
   (`compute_neighbour_continuity`, `compute_neighbour_continuity_observed`):
   weighted identity overlap (WIO), rank correlation (RC), track survival, and the
   diagnostic split of WIO loss into vanished / moved-away neighbours.

Positions are in whatever unit you pass in; every distance parameter (neighbour cap,
border margin, split distance, switch cap) is in that same unit.

**Real ground truth.** `read_ctc_tracks` reads a Cell Tracking Challenge tracking
folder (`01_GT/TRA`, `01_ST/TRA` or `01_RES`) into the same per-frame format, with
centroids scaled to microns and a `parent_ids` field for divisions:

```bash
pip install -e ".[ctc]"
python -m tracking_validation.ctc Fluo-N3DH-CE/01_GT/TRA --dataset Fluo-N3DH-CE --out ce01.csv
```

The metrics do not yet account for divisions; see
[docs/division_policy.md](docs/division_policy.md) for the proposed policy.

```bash
pip install -e ".[dev]"
pytest -q
```

Copied from `tracking_validation` on 2026-10-24 (working-tree state, including the
soft-floor sampler, cross-frame ID switches and the extra WIO-loss metrics).

## Changes since the copy (0.3.0)

Behaviour changes — results for a given seed differ from 0.2.0 where noted:

- **Sampler** `exact_min_distance` now defaults to `True`, so `base_min_distance` is
  actually enforced. Previously ~0.3% of cells could be closer than the floor.
  Changes frame 0 for every seed. Pass `exact_min_distance=False` to reproduce old runs.
- **Metrics** `penalise_absence=True` with `border_margin > 0` no longer zeroes WIO for
  cells that are still tracked at t1 but drifted into the border band (they are
  excluded as before, not penalised).
- **Metrics** `penalised_rank_correlation` now uses average ranks for tied penalty
  ranks (correct Spearman with ties). `rank_correlation` is unchanged (no ties there).
- `MotionConfig(speed_distribution=...)` raises on an unknown value instead of silently
  falling back to `"normal"`.

Housekeeping: removed the unused topological-RC code path and its
`successor_distance_factor` parameter, removed duplicated unused helpers, silenced
spurious divide warnings, documented the two ID-switch rules and their effective
rates, and added tests for `errors.py`, the sampler paths and config validation.
`pytest` now always imports this repo's `src/`, even if another copy of the package
is installed in the environment.
