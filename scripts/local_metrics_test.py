"""Score the reference-free local metrics (tracking_validation.local_metrics) on the real
trackers of scripts/run_detector_tracking.py (same 208 configurations, relinked here).

    python scripts/local_metrics_test.py              # run + summarise
    python scripts/local_metrics_test.py --stage summary

Per frame pair (cells next to GT divisions excluded, as everywhere):
  events   est: frag / vanished-without-body / new-without-predecessor (classify_events)
           true: frag (same cell, new label) / miss + false detections that ended /
                 false positives + re-detections
  tails    cells beyond the pair's own bulk for displaced share, 1 - RC, motion residual;
           true: switched cells.  Also the tail rate on error-free (oracle) tracks.
Per tracker: per-cell AUC of each score for switched vs correctly linked cells, separately
for swaps (two labels exchanged) and chains (labels shifted along a line of cells).

Writes data/_results/local_metrics_pairs.csv, local_metrics_auc.csv and figures/local_metrics/.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from tracking_validation.linking import link_frames  # noqa: E402
from tracking_validation.local_metrics import bulk_tail, cell_scores, classify_events  # noqa: E402
from tracking_validation.tracking_eval import pair_errors  # noqa: E402
import run_detector_tracking as rdt  # noqa: E402

RES = ROOT / "data" / "_results"
FIG = ROOT / "figures" / "local_metrics"
SCORES = ("disp", "inv_rc", "resid")


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def in_region(positions: np.ndarray, gt_tree, gt_ids: np.ndarray, region: set[int]) -> np.ndarray:
    if len(positions) == 0:
        return np.zeros(0, bool)
    _, k = gt_tree.query(positions)
    return np.array([int(gt_ids[i]) in region for i in k], bool)


def score_pair(f0, f1, g0, g1, E: set[int]):
    """Estimated and true quantities for one pair; per-cell score rows for switched/correct."""
    row = {}
    # ---- truth (counts as in tracking_eval, plus false detections that ended) ----
    tr = pair_errors(f0, f1, E, g1["positions"], g1["true_ids"])
    row.update({f"true_{k}": v for k, v in tr.items() if k != "n_eval"}); row["n_eval"] = tr["n_eval"]
    l0 = f0["assigned_ids"]; l1 = set(int(l) for l in f1["assigned_ids"])
    t0 = cKDTree(g0["positions"]); t1 = cKDTree(g1["positions"])
    reg0 = in_region(f0["positions"], t0, g0["true_ids"], E)
    row["true_fp_end"] = int(sum(1 for i, l in enumerate(l0) if f0["true_ids"][i] < 0 and reg0[i] and int(l) not in l1))
    # ---- events ----
    ev = classify_events(f0, f1, rdt.NEIGH_CAP)
    lab0 = {int(l): i for i, l in enumerate(l0)}; lab1 = {int(l): i for i, l in enumerate(f1["assigned_ids"])}
    def reg_t0(ls): return sum(1 for l in ls if reg0[lab0[l]])
    reg1 = in_region(f1["positions"], t1, g1["true_ids"], E)
    def reg_t1(ls): return sum(1 for l in ls if reg1[lab1[l]])
    row["est_frag"] = sum(1 for v, n in ev["frag"] if reg0[lab0[v]])
    row["est_vanished_nobody"] = reg_t0(ev["vanished_nobody"])
    row["est_new_unmatched"] = reg_t1(ev["new_unmatched"])
    row["body_radius_over_spacing"] = ev["radius"] / ev["spacing"]
    # ---- per-cell scores, tails, switch truth ----
    sc = cell_scores(f0, f1, rdt.NEIGH_CAP)
    labs = [l for l in sc if reg0[lab0[l]]]
    allv = np.array([sc[l] for l in sc], float)
    tails = {s: bulk_tail(allv[:, j]) for j, s in enumerate(SCORES)} if len(allv) else {}
    keyed = list(sc)
    for s in SCORES:
        mask = dict(zip(keyed, tails[s])) if tails else {}
        row[f"est_tail_{s}"] = int(sum(1 for l in labs if mask.get(l, False)))
    # switched cells: label's true id differs between t and t+1 (both real cells)
    tid0 = {int(l): int(c) for l, c in zip(l0, f0["true_ids"])}
    tid1 = {int(l): int(c) for l, c in zip(f1["assigned_ids"], f1["true_ids"])}
    move = {tid0[l]: tid1[l] for l in sc if tid0[l] >= 0 and tid1[l] >= 0}
    cells = []
    for l in labs:
        a, b = tid0[l], tid1[l]
        if a < 0 or b < 0:
            continue
        kind = "ok" if a == b else ("swap" if move.get(b) == a else "chain")
        cells.append((kind, *sc[l]))
    row["true_swap_cells"] = sum(1 for c in cells if c[0] == "swap")
    row["true_chain_cells"] = sum(1 for c in cells if c[0] == "chain")
    return row, cells


def oracle_tail(o0, o1, g0, E):
    sc = cell_scores(o0, o1, rdt.NEIGH_CAP)
    if not sc:
        return {s: 0 for s in SCORES}
    allv = np.array([sc[l] for l in sc], float)
    out = {}
    for j, s in enumerate(SCORES):
        m = bulk_tail(allv[:, j])
        out[s] = int(sum(1 for l, f in zip(sc, m) if f and l in E))
    return out


def job(args):
    seq, img, op, stride = args
    label_map, gt, fidx, keep, spacing = rdt.load_gt(seq, stride)
    det = rdt.detection_frames(seq, img, op, label_map, fidx)
    oracle = []
    for f, g in zip(det, gt):
        real = f["true_ids"] >= 0
        have = set(int(c) for c in f["true_ids"][real])
        fill = np.array([int(c) not in have for c in g["true_ids"]], dtype=bool)
        ids = np.concatenate([f["true_ids"][real], g["true_ids"][fill]])
        oracle.append(dict(positions=np.vstack([f["positions"][real], g["positions"][fill]]), true_ids=ids, assigned_ids=ids))
    orc = [oracle_tail(oracle[t], oracle[t + 1], gt[t], keep[t] & set(int(c) for c in gt[t + 1]["true_ids"]))
           for t in range(len(gt) - 1)]
    rows, aucs = [], []
    for cap in rdt.CAPS:
        res = link_frames(det, cap * spacing, "lap")
        cells_all = []
        for t in range(len(gt) - 1):
            E = keep[t] & set(int(c) for c in gt[t + 1]["true_ids"])
            r, cells = score_pair(res.frames[t], res.frames[t + 1], gt[t], gt[t + 1], E)
            rows.append(dict(seq=seq, img=img.lstrip("_") or "clean", op=op, stride=stride, cap=cap, frame=fidx[t],
                             n_cells=len(gt[t]["true_ids"]), **r, **{f"oracle_tail_{s}": v for s, v in orc[t].items()}))
            cells_all += cells
        c = pd.DataFrame(cells_all, columns=["kind", *SCORES])
        a = dict(seq=seq, img=img.lstrip("_") or "clean", op=op, stride=stride, cap=cap,
                 n_ok=int((c.kind == "ok").sum()), n_swap=int((c.kind == "swap").sum()), n_chain=int((c.kind == "chain").sum()))
        ok = c[c.kind == "ok"]
        for s in SCORES:
            for kind in ("swap", "chain"):
                a[f"auc_{s}_{kind}"] = auc(c[c.kind == kind][s].to_numpy(), ok[s].to_numpy())
            a[f"auc_{s}_all"] = auc(c[c.kind != "ok"][s].to_numpy(), ok[s].to_numpy())
        aucs.append(a)
        print(f"{seq} {img or 'clean':10s} {op:14s} stride {stride} cap {cap}: switched {a['n_swap']}+{a['n_chain']} "
              + " ".join(f"AUC {s} {a[f'auc_{s}_all']:.2f}" for s in SCORES), flush=True)
    return rows, aucs


def stage_run(workers: int) -> None:
    jobs = [(s, img, op, st) for s in ("01", "02") for img in rdt.IMAGES for op in rdt.OPS for st in rdt.STRIDES
            if not ("_n" in img and op == "high_recall")]
    rows, aucs = [], []
    with ProcessPoolExecutor(min(workers, len(jobs))) as ex:
        for r, a in ex.map(job, jobs):
            rows += r; aucs += a
    pd.DataFrame(rows).to_csv(RES / "local_metrics_pairs.csv", index=False)
    pd.DataFrame(aucs).to_csv(RES / "local_metrics_auc.csv", index=False)
    print("wrote", RES / "local_metrics_pairs.csv", RES / "local_metrics_auc.csv")


# --------------------------------------------------------------------------- summary

def stage_summary() -> None:
    import matplotlib.pyplot as plt
    import figstyle as fs
    df = pd.read_csv(RES / "local_metrics_pairs.csv", dtype={"seq": str})
    au = pd.read_csv(RES / "local_metrics_auc.csv", dtype={"seq": str})
    keys = ["seq", "img", "op", "stride", "cap"]
    s = df.groupby(keys).sum(numeric_only=True).reset_index()
    p = lambda c: 100 * s[c] / s.n_eval
    comp = {
        "Fragmentation (same pair)": (p("true_frag"), p("est_frag")),
        "Vanished without a body": (100 * (s.true_miss + s.true_fp_end) / s.n_eval, p("est_vanished_nobody")),
        "New without a predecessor": (100 * (s.true_fp + s.true_reappear) / s.n_eval, p("est_new_unmatched")),
    }
    print(f"{len(s)} trackers.  Events, percent of evaluated cells:")
    for name, (x, y) in comp.items():
        print(f"  {name:28s} true mean {x.mean():5.2f}  est mean {y.mean():5.2f}  slope {np.sum(x * y) / np.sum(x * x):.2f}  "
              f"r {np.corrcoef(x, y)[0, 1]:.2f}  mae {np.abs(y - x).mean():.2f}")
    print("\nSwitch tails, percent of evaluated cells (true = switched cells):")
    xs = p("true_switch")
    for sc in SCORES:
        y = p(f"est_tail_{sc}"); o = 100 * s[f"oracle_tail_{sc}"] / s.n_eval
        print(f"  {sc:7s} tail mean {y.mean():5.2f} (error-free tracks {o.mean():.2f})  true mean {xs.mean():.2f}  "
              f"r {np.corrcoef(xs, y)[0, 1]:.2f}   tail minus error-free r {np.corrcoef(xs, y - o)[0, 1]:.2f}")
    print("\nPer-cell AUC, switched vs correctly linked cells (median over trackers; 0.5 = chance):")
    print(f"  switched cells: {int(au.n_swap.sum())} in swaps, {int(au.n_chain.sum())} in chains")
    tab = pd.DataFrame({k: [au[f"auc_{sc}_{k}"].median() for sc in SCORES] for k in ("all", "swap", "chain")}, index=SCORES)
    print(tab.round(3).to_string())
    print("\nby frame interval (AUC all):"); print(au.groupby("stride")[[f"auc_{sc}_all" for sc in SCORES]].median().round(3).to_string())

    fs.apply(); FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(fs.DOUBLE, 2.0), gridspec_kw=dict(left=0.07, right=0.99, top=0.86, bottom=0.22, wspace=0.35))
    cols = ("frag", "miss", "fp")
    for ax, (name, (x, y)), c in zip(axes, comp.items(), cols):
        lim = float(max(x.max(), y.max())) * 1.08
        ax.plot([0, lim], [0, lim], color=fs.AXIS, lw=0.8)
        ax.scatter(x, y, s=9, color=fs.SERIES[c], alpha=0.7, linewidths=0)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_title(name, loc="left", pad=3); ax.set_xlabel("True (%)")
        ax.text(0.04, 0.96, f"r = {np.corrcoef(x, y)[0, 1]:.2f}", transform=ax.transAxes, va="top", fontsize=7)
    axes[0].set_ylabel("Estimated (%)")
    fs.save(fig, FIG / "events_est_vs_true")

    fig, ax = plt.subplots(figsize=(fs.SINGLE, 2.2), gridspec_kw=dict(left=0.17, right=0.97, top=0.93, bottom=0.2))
    xpos = np.arange(len(SCORES)); w = 0.26
    labels = {"disp": "Displaced share", "inv_rc": "1 − RC", "resid": "Motion residual"}
    for i, (kind, col) in enumerate((("all", fs.INK), ("swap", "#2a78d6"), ("chain", "#eb6834"))):
        ax.bar(xpos + (i - 1) * w, [au[f"auc_{sc}_{kind}"].median() - 0.5 for sc in SCORES], w, bottom=0.5, color=col,
               label={"all": "All switched", "swap": "Swaps", "chain": "Chains"}[kind])
    ax.axhline(0.5, color=fs.AXIS, lw=0.8); ax.set_ylim(0.4, 1.0)
    ax.set_xticks(xpos); ax.set_xticklabels([labels[sc] for sc in SCORES]); ax.set_ylabel("AUC, switched vs correct")
    ax.legend(fontsize=6.5, loc="upper left")
    fs.save(fig, FIG / "switch_auc")
    print("wrote", FIG)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("all", "run", "summary"), default="all")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = p.parse_args()
    if a.stage in ("all", "run"):
        stage_run(a.workers)
    if a.stage in ("all", "summary"):
        stage_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
