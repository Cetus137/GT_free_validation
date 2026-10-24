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

```bash
pip install -e ".[dev]"
pytest -q
```

Copied from `tracking_validation` on 2026-10-24 (working-tree state, including the
soft-floor sampler, cross-frame ID switches and the extra WIO-loss metrics).
