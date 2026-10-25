"""Pin the mechanism: inject label-hop CHAINS into clean embryo frame pairs and
measure what each metric channel sees, as a function of chain length and frame
interval.

    python scripts/chain_injection.py

A chain of length k picks k adjacent cells c1..ck (a random walk on the
Delaunay graph at t+1) and, in frame t+1 only, shifts labels one body along:
label(c_i) -> body c_{i+1}, body c_1 gets a fresh label, label(c_k) is lost.
That is k wrong links with one track start and one track end, which is what a
displacement-minimising linker produces.  k = "swap" is the mutual exchange
used in the earlier sweeps (two wrong links, no start or end).

Rates are "cells with a wrong link" per evaluated cell.  Channels are
measured on division-free cells, Delaunay neighbourhoods, against the same
pair's clean values.  Writes data/_results/chain_injection.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

from tracking_validation.lineage import division_affected_cells, inherit_parent_labels  # noqa: E402
from tracking_validation.metrics import compute_neighbour_continuity_observed, _frame_neighbour_map_delaunay  # noqa: E402
import ce_error_sweep as sweep  # noqa: E402

CH = ["survival", "vanished", "displaced", "newcomer", "rc", "disp_out"]


def channels(pair, ok_labels, cap):
    r = compute_neighbour_continuity_observed(pair, neighbourhood_mode="delaunay", max_neighbour_distance=cap, extra_metrics=True)
    cols = [i for i, c in enumerate(r["cell_ids"]) if int(c) in ok_labels]
    set1 = set(pair[1]["assigned_ids"].tolist())
    d = r["wio_displaced"][0][cols]
    return dict(survival=float(np.mean([l in set1 for l in ok_labels])), vanished=float(np.nanmean(r["wio_vanished"][0][cols])),
                displaced=float(np.nanmean(d)), newcomer=float(np.nanmean(r["newcomer_fraction"][0][cols])),
                rc=float(np.nanmean(r["rank_correlation"][0][cols])), disp_out=float(np.nanmean(d > 0.4)))


def inject_chains(pair, k, rate, ok_true, rng, cap):
    """Return a corrupted copy of frame t+1 and the number of wrong links made."""
    f0, f1 = pair
    labels = f1["assigned_ids"].copy()
    true1 = f1["true_ids"]
    idx_of_true = {int(t): i for i, t in enumerate(true1)}
    nb, _ = _frame_neighbour_map_delaunay(f1["positions"], true1, max_distance=cap)   # adjacency by true id at t+1
    present0 = set(f0["true_ids"].tolist())
    candidates = [t for t in ok_true if t in idx_of_true and t in present0 and t in nb]
    n_target = int(round(rate * len(candidates)))
    used: set[int] = set(); wrong = 0; fresh = int(labels.max()) + 1
    rng.shuffle(candidates)
    for start in candidates:
        if wrong >= n_target:
            break
        if start in used:
            continue
        if k == "swap":
            opts = [n for n in nb[start] if n not in used and n in present0 and n in idx_of_true]
            if not opts:
                continue
            b = int(rng.choice(opts)); i, j = idx_of_true[start], idx_of_true[b]
            labels[i], labels[j] = labels[j], labels[i]
            used.update((start, b)); wrong += 2
            continue
        chain = [start]
        while len(chain) < k:
            opts = [n for n in nb[chain[-1]] if n not in used and n not in chain and n in present0 and n in idx_of_true]
            if not opts:
                break
            chain.append(int(rng.choice(opts)))
        if len(chain) < 2:
            continue
        old = [labels[idx_of_true[c]] for c in chain]
        for a, b in zip(chain[:-1], chain[1:]):           # label of a -> body of b
            labels[idx_of_true[b]] = old[chain.index(a)]
        labels[idx_of_true[chain[0]]] = fresh; fresh += 1   # orphan at the head
        # label of the last cell is overwritten (lost): a track end
        used.update(chain); wrong += len(chain) - 1
    return {**f1, "assigned_ids": labels}, wrong


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cap", type=float, default=15.0)
    p.add_argument("--rate", type=float, default=0.05)
    p.add_argument("--strides", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--chains", nargs="+", default=["swap", "2", "3", "7"])
    p.add_argument("--out", type=Path, default=ROOT / "data" / "_results" / "chain_injection.csv")
    a = p.parse_args()
    rows = []
    for seq in ("01", "02"):
        data = inherit_parent_labels(sweep.load_ce(ROOT / "data", ROOT / "data" / "_cache", seq), prefer="nearest")
        start = int(np.argmax(data.counts() >= 50))
        for stride in a.strides:
            gt = data.frames[start:][::stride]
            excl = division_affected_cells(gt, k=8, max_neighbour_distance=a.cap, neighbourhood_mode="delaunay")
            for t in range(0, len(gt) - 1, 2):
                pair = [gt[t], gt[t + 1]]
                ok_true = set(gt[t]["true_ids"].tolist()) - excl[t]
                ok_lab = {int(l) for l, tid in zip(gt[t]["assigned_ids"], gt[t]["true_ids"]) if int(tid) in ok_true}
                base = channels(pair, ok_lab, a.cap)
                n_eval = len(ok_lab)
                for kind in a.chains:
                    k = kind if kind == "swap" else int(kind)
                    for seed in range(2):
                        f1c, wrong = inject_chains(pair, k, a.rate, ok_true, np.random.default_rng(1000 * t + seed), a.cap)
                        v = channels([gt[t], f1c], ok_lab, a.cap)
                        rows.append(dict(seq=seq, stride=stride, frame=t, chain=kind, seed=seed,
                                         wrong_links_pct=100 * wrong / n_eval, **{f"d_{c}": v[c] - base[c] for c in CH}))
            print(f"{seq} stride {stride} done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
    pd.set_option("display.width", 220); pd.set_option("display.precision", 2)
    g = df.groupby(["stride", "chain"]).agg(wrong_pct=("wrong_links_pct", "mean"),
                                            **{f"d_{c}": (f"d_{c}", lambda x: 100 * x.mean()) for c in CH})
    g["visible_pct"] = -g.d_survival + g.d_newcomer           # chain ends seen by the counting channels
    print("\nChannel change (x100) for ~5% of cells with a wrong link, by chain length and frame interval")
    print(g.to_string())
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
