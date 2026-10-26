"""Baseline on clean embryo tracks under three division treatments.

    python scripts/ce_division_handling.py

Per consecutive frame pair (frames with >= 50 cells, both recordings), the
observed-frames metrics under: raw labels (divisions ignored), one daughter
inheriting the mother's label, and inheritance plus exclusion of cells whose
neighbourhood touches a division. Delaunay neighbourhoods, 15 µm cap.
Writes data/_results/ce_division_handling.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

from tracking_validation.lineage import division_affected_cells, inherit_parent_labels, mask_cells  # noqa: E402
from tracking_validation.metrics import compute_neighbour_continuity_observed  # noqa: E402
import ce_error_sweep as sweep  # noqa: E402

CAP = 15.0


def nanmean(a):
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def run(frames, variant, exclude=False):
    res = compute_neighbour_continuity_observed(frames, neighbourhood_mode="delaunay",
                                                max_neighbour_distance=CAP, extra_metrics=True)
    if exclude:
        res = mask_cells(res, division_affected_cells(frames, max_neighbour_distance=CAP, neighbourhood_mode="delaunay"))
    rows = []
    for ti in range(len(frames) - 1):
        births = int((frames[ti + 1]["parent_ids"] >= 0).sum())
        rows.append(dict(variant=variant, pair=ti, division_pair=births > 0,
                         wio=nanmean(res["weighted_identity_overlap"][ti]), rc=nanmean(res["rank_correlation"][ti]),
                         vanished=nanmean(res["wio_vanished"][ti]), displaced=nanmean(res["wio_displaced"][ti]),
                         newcomer=nanmean(res["newcomer_fraction"][ti]), survival=float(res["track_survival"][ti]),
                         n_eval=int(np.isfinite(res["weighted_identity_overlap"][ti]).sum())))
    return rows


def load_dataset(name: str):
    """Yield (sequence label, CTCTracks) for a dataset: 'ce' or 'astec'."""
    if name == "ce":
        for seq in ("01", "02"):
            yield seq, sweep.load_ce(ROOT / "data", ROOT / "data" / "_cache", seq)
    elif name == "astec":
        from tracking_validation.astec import read_astec_tracks
        for emb in ("Phmamm-1-v2", "Phmamm-16-v2"):
            yield emb, read_astec_tracks(ROOT / "data" / "ASTEC" / emb / f"{emb}_properties.xml", voxel_size=ASTEC_VOXEL)
    else:
        raise ValueError(name)


ASTEC_VOXEL = 0.3   # microns per voxel, isotropic (published ASTEC sampling; to be confirmed)


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("ce", "astec"), default="ce")
    p.add_argument("--min-cells", type=int, default=50)
    p.add_argument("--cap", type=float, default=None, help="neighbour cap in microns (default 15 for ce, 45 for astec)")
    a = p.parse_args()
    global CAP
    CAP = a.cap if a.cap is not None else (15.0 if a.dataset == "ce" else 45.0)
    rows = []
    for seq, data in load_dataset(a.dataset):
        start = int(np.argmax(data.counts() >= a.min_cells))
        raw = data.frames[start:]
        inh = inherit_parent_labels(data, prefer="nearest").frames[start:]
        for r in run(raw, "raw") + run(inh, "inherit") + run(inh, "inherit + exclude", exclude=True):
            rows.append({**r, "seq": seq, "frame": start + r["pair"]})
    df = pd.DataFrame(rows)
    out = ROOT / "data" / "_results" / f"{a.dataset}_division_handling.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200); pd.set_option("display.precision", 3)
    g = df.groupby(["variant", "division_pair"])
    s = g[["wio", "rc", "vanished", "displaced", "newcomer", "survival"]].mean(); s["pairs"] = g.size(); s["cells"] = g.n_eval.mean().round(0)
    print(f"dataset {a.dataset}, cap {CAP} µm"); print(s.to_string())
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
