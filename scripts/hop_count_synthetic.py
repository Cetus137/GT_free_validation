"""Cross-organism check of the hop counts (window count, nearest-body test) with synthetic
detections built from the ground truth of C. elegans (CTC) and Phallusia (ASTEC) embryos.
The ASTEC data have no images, so a real detector cannot run; both organisms are treated the
same way here so the comparison is like for like.

    python scripts/hop_count_synthetic.py [--workers N] [--stage all|summary]

Detections per frame: GT positions + Gaussian jitter (5% of the median spacing per axis),
random misses, and false positives half a spacing from a random cell; two error levels
(1% misses + 0.5% FPs; 3% misses + 1% FPs).  Linking: LAP at 0.5 and 1.0 median spacings.
Frame intervals: worm 1, 2, 3, 4; ascidian 1, 3, 6, 10 (its cells move ~2.5x less per frame
relative to spacing).  Counts as in scripts/hop_count_variants.py.

Writes data/_results/hop_count_synthetic.csv and figures/local_metrics/hop_count_synthetic.png.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import ce_error_sweep as sweep  # noqa: E402
import hop_count_variants as hv  # noqa: E402
import run_detector_tracking as rdt  # noqa: E402
from tracking_validation.lineage import division_affected_cells, inherit_parent_labels, subsample_frames  # noqa: E402
from tracking_validation.linking import link_frames  # noqa: E402

RES = ROOT / "data" / "_results"
DATASETS = {"worm": dict(embryos=("01", "02"), cap=15.0, strides=(1, 2, 3, 4)),
            "ascidian": dict(embryos=("Phmamm-1-v2", "Phmamm-16-v2"), cap=45.0, strides=(1, 3, 6, 10))}
LEVELS = {"low": (0.01, 0.005), "high": (0.03, 0.01)}
JITTER = 0.05


def load(organism: str, emb: str, stride: int):
    cap = DATASETS[organism]["cap"]
    if organism == "worm":
        raw = sweep.load_ce(ROOT / "data", ROOT / "data" / "_cache", emb)
    else:
        from tracking_validation.astec import read_astec_tracks
        raw = read_astec_tracks(ROOT / "data" / "ASTEC" / emb / f"{emb}_properties.xml", voxel_size=sweep.ASTEC_VOXEL)
    data = inherit_parent_labels(raw, prefer="nearest")
    start = int(np.argmax(data.counts() >= 50))
    gt = subsample_frames(data.frames[start:], stride)
    excl = division_affected_cells(gt, k=8, max_neighbour_distance=cap, neighbourhood_mode="delaunay")
    keep = [set(int(c) for c in gt[t]["true_ids"]) - excl[t] for t in range(len(gt) - 1)]
    spacing = float(np.median([np.median(cKDTree(f["positions"]).query(f["positions"], k=2)[0][:, 1]) for f in gt[::5]]))
    return gt, keep, spacing


def synth(gt, spacing, p_miss, p_fp, rng):
    out = []
    for g in gt:
        pos, ids = g["positions"], g["true_ids"]
        keep = rng.random(len(pos)) >= p_miss
        p = pos[keep] + rng.normal(0, JITTER * spacing, (keep.sum(), 3))
        n_fp = rng.poisson(p_fp * len(pos))
        if n_fp:
            v = rng.normal(size=(n_fp, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
            fp = pos[rng.integers(0, len(pos), n_fp)] + 0.5 * spacing * v
            p = np.vstack([p, fp])
        t = np.concatenate([ids[keep], np.full(n_fp, -1)]).astype(int)
        out.append(dict(positions=p, true_ids=t, assigned_ids=np.arange(len(p)), parent_ids=np.full(len(p), -1)))
    return out


def job(args):
    organism, emb, stride = args
    rdt.NEIGH_CAP = DATASETS[organism]["cap"]                 # neighbourhood cap used by the counts
    gt, keep, spacing = load(organism, emb, stride)
    rows = []
    for level, (pm, pf) in LEVELS.items():
        det = synth(gt, spacing, pm, pf, np.random.default_rng(hash((emb, stride, level)) % 2**32))
        for cap in (0.5, 1.0):
            F = link_frames(det, cap * spacing, "lap").frames
            c = pd.DataFrame([hv.pair_counts(F[t], F[t + 1], keep[t] & set(int(x) for x in gt[t + 1]["true_ids"]))
                              for t in range(len(gt) - 1)]).sum()
            rows.append(dict(organism=organism, embryo=emb, stride=stride, level=level, cap=cap,
                             **{k: c[k] for k in ["n", "sw", "nb_tp", "nb_fp", *hv.COUNTS]}))
    print(f"{organism} {emb} stride {stride}: done", flush=True)
    return rows


def summary() -> None:
    import matplotlib.pyplot as plt
    import figstyle as fs
    d = pd.read_csv(RES / "hop_count_synthetic.csv")
    d["true"] = 100 * d.sw / d.n; d["win"] = 100 * d.Hwin / d.n; d["nb"] = 100 * d.Hnb_persist / d.n; d["h05"] = 100 * d.H05 / d.n
    pd.set_option("display.width", 200)
    print("percent of evaluated cells, pooled over embryos and error levels:")
    g = d.groupby(["organism", "cap", "stride"])[["true", "h05", "win", "nb"]].mean()
    g["win/true"] = g.win / g.true; g["nb/true"] = g.nb / g.true
    print(g.round(3).to_string())
    print("\nat link distance 1.0 spacing:")
    for org, h in d[d.cap == 1.0].groupby("organism"):
        for k in ("win", "nb"):
            ok = h.true > 0
            print(f"  {org:8s} {k:3s}: ratio est/true {h[k].sum() / h.true.sum():.2f} (per tracker median {np.median(h[k][ok] / h.true[ok]):.2f}), "
                  f"r {np.corrcoef(h.true, h[k])[0, 1]:.2f}, {len(h)} trackers")
    fs.apply()
    fig, axes = plt.subplots(1, 2, figsize=(fs.DOUBLE * 0.62, 2.2), gridspec_kw=dict(left=0.12, right=0.98, top=0.86, bottom=0.22, wspace=0.35))
    for ax, k, title in zip(axes, ("win", "nb"), ("Window count", "Nearest body to prediction")):
        h = d[d.cap == 1.0]
        lim = float(max(h.true.max(), h[k].max())) * 1.1
        ax.plot([0, lim], [0, lim], color=fs.AXIS, lw=0.8)
        for org, col in (("worm", "#2a78d6"), ("ascidian", "#eb6834")):
            m = h.organism == org
            ax.scatter(h.true[m], h[k][m], s=14, color=col, linewidths=0, alpha=0.85, label=org.capitalize())
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_title(title, loc="left", pad=3)
        ax.set_xlabel("True switched (%)")
    axes[0].set_ylabel("Estimated (%)"); axes[0].legend(fontsize=6.5, loc="upper left")
    fs.save(fig, ROOT / "figures" / "local_metrics" / "hop_count_synthetic")


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--stage", choices=("all", "summary"), default="all")
    a = p.parse_args()
    if a.stage == "all":
        jobs = [(org, emb, st) for org, cfg in DATASETS.items() for emb in cfg["embryos"] for st in cfg["strides"]]
        with ProcessPoolExecutor(a.workers) as ex:
            rows = [r for part in ex.map(job, jobs) for r in part]
        pd.DataFrame(rows).to_csv(RES / "hop_count_synthetic.csv", index=False)
    summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
