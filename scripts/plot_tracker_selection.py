"""Can the reference-free metrics pick the best tracker?  Uses the 208 real trackers.

    python scripts/plot_tracker_selection.py

A user cannot change their images or frame interval, only the tracker settings.  So trackers
are compared within groups of the same sequence x image condition x frame interval
(40 groups of 4-6 settings: detector operating point x LAP link distance).  In each group the
"best" tracker has the lowest TRUE total error (switch + fragmentation + miss + false positive,
percent of evaluated cells; dropouts that return count as fragmentation).

Reference-free scores (lower = better tracker), none uses ground truth:
  1 - WIO            mean loss of neighbourhood identity overlap
  Label events       vanished + newcomer shares (count channels; exact zero baseline)
  Positional events  fragmentations + vanished-without-body + new-without-predecessor
                     (local_metrics.classify_events)
  Demixed total      sum of the four demixed rates (simulation calibration, current 7 channels)

Reported per score: how often it picks the true best, the extra true error of its pick over
the best (regret), and the within-group Spearman rank correlation with the true error.
Writes figures/local_metrics/tracker_selection.png (+ _groups.png) and
data/_results/tracker_selection.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import figstyle as fs  # noqa: E402
from run_detector_tracking import truth  # noqa: E402

RES = ROOT / "data" / "_results"
KEYS = ["seq", "img", "op", "stride", "cap"]
GROUP = ["seq", "img", "stride"]
SCORES = {"wio": "1 − WIO", "label": "Label events", "pos": "Positional events", "demix": "Demixed total"}


def load() -> pd.DataFrame:
    pairs = pd.read_csv(RES / "detector_tracking_pairs_resid.csv", dtype={"seq": str})
    rows = []
    for key, g in pairs.groupby(KEYS):
        t = truth(g)
        rows.append(dict(zip(KEYS, key), true_total=sum(t.values()),
                         wio=100 * (1 - g.trk_wio.mean()),
                         label=100 * (g.trk_vanished.mean() + g.trk_newcomer.mean())))
    df = pd.DataFrame(rows)
    loc = pd.read_csv(RES / "local_metrics_pairs.csv", dtype={"seq": str}).groupby(KEYS).sum(numeric_only=True).reset_index()
    loc["pos"] = 100 * (loc.est_frag + loc.est_vanished_nobody + loc.est_new_unmatched) / loc.n_eval
    dem = pd.read_csv(RES / "real_resid_estimates.csv", dtype={"seq": str})
    dem["demix"] = dem[[f"est|current7|sim baseline|{t}" for t in ("switch", "frag", "miss", "fp")]].sum(1)
    df = df.merge(loc[KEYS + ["pos"]], on=KEYS).merge(dem[KEYS + ["demix"]], on=KEYS)
    df["setting"] = df.op.str.replace("_", " ") + ", link " + df.cap.astype(str)
    return df


def main() -> int:
    df = load()
    rng = np.random.default_rng(0)
    res = []
    for gk, g in df.groupby(GROUP):
        best = g.true_total.min()
        r = dict(zip(GROUP, gk), n=len(g), best=best, random_regret=(g.true_total - best).mean(),
                 spread=g.true_total.max() - best)
        for s in SCORES:
            pick = g.loc[g[s].idxmin()]
            r[f"hit_{s}"] = bool(np.isclose(pick.true_total, best))
            r[f"regret_{s}"] = pick.true_total - best
            r[f"rho_{s}"] = spearmanr(g[s], g.true_total).statistic
        res.append(r)
    res = pd.DataFrame(res); res.to_csv(RES / "tracker_selection.csv", index=False)

    print(f"{len(df)} trackers in {len(res)} groups (same sequence, image condition, frame interval).")
    print(f"True error spread within a group: median {res.spread.median():.2f} points (best to worst).")
    print(f"Random pick: hits best {100 * (1 / res.n).mean():.0f}% of groups on average, mean regret {res.random_regret.mean():.2f} points.\n")
    tab = pd.DataFrame({SCORES[s]: dict(picks_best=f"{100 * res[f'hit_{s}'].mean():.0f}%",
                                         mean_regret=round(res[f"regret_{s}"].mean(), 2),
                                         worst_regret=round(res[f"regret_{s}"].max(), 2),
                                         median_rank_corr=round(res[f"rho_{s}"].median(), 2)) for s in SCORES}).T
    print(tab.to_string())

    fs.apply()
    cols = {"wio": fs.INK2, "label": "#2a78d6", "pos": "#eb6834", "demix": "#1baf7a"}
    fig, axes = plt.subplots(1, 3, figsize=(fs.DOUBLE, 2.2), gridspec_kw=dict(left=0.07, right=0.99, top=0.88, bottom=0.3, wspace=0.45))
    names = list(SCORES)
    x = np.arange(len(names))
    axes[0].bar(x, [100 * res[f"hit_{s}"].mean() for s in names], color=[cols[s] for s in names], width=0.65)
    axes[0].axhline(100 * (1 / res.n).mean(), color=fs.MUTED, lw=1, ls=(0, (3, 2)))
    axes[0].text(-0.45, 100 * (1 / res.n).mean() + 2, "random", ha="left", fontsize=6.5, color=fs.MUTED)
    axes[0].set_ylim(0, 100); axes[0].set_ylabel("Picks the true best (%)")
    axes[0].set_title("Best tracker found", loc="left")
    axes[1].bar(x, [res[f"regret_{s}"].mean() for s in names], color=[cols[s] for s in names], width=0.65)
    axes[1].axhline(res.random_regret.mean(), color=fs.MUTED, lw=1, ls=(0, (3, 2)))
    axes[1].text(len(names) - 0.5, res.random_regret.mean() * 1.04, "random", ha="right", va="bottom", fontsize=6.5, color=fs.MUTED)
    axes[1].set_ylabel("Extra error of pick (points)"); axes[1].set_title("Cost of the pick", loc="left")
    axes[2].bar(x, [res[f"rho_{s}"].median() for s in names], color=[cols[s] for s in names], width=0.65)
    axes[2].set_ylim(0, 1); axes[2].set_ylabel("Rank correlation"); axes[2].set_title("Ranking within group", loc="left")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels([SCORES[s] for s in names], rotation=35, ha="right")
    fs.save(fig, ROOT / "figures" / "local_metrics" / "tracker_selection")

    # example groups: true error vs positional-event score, one panel per group (clean images)
    ex = df[df.img == "clean"]
    groups = list(ex.groupby(["seq", "stride"]))
    fig, axes = plt.subplots(2, 4, figsize=(fs.DOUBLE, 3.6), gridspec_kw=dict(left=0.08, right=0.99, top=0.92, bottom=0.12, wspace=0.35, hspace=0.5))
    for ax, ((seq, st), g) in zip(axes.ravel(), groups):
        ax.scatter(g.true_total, g.pos, s=22, color="#eb6834", linewidths=0, zorder=3)
        b = g.loc[g.true_total.idxmin()]; p = g.loc[g.pos.idxmin()]
        ax.scatter([b.true_total], [b.pos], s=70, facecolors="none", edgecolors=fs.INK, linewidths=1.0, zorder=4)
        ax.set_title(f"Seq {seq}, every {st}{'st' if st == 1 else 'nd' if st == 2 else 'rd' if st == 3 else 'th'} frame", loc="left", fontsize=7.5)
        lo = min(g.true_total.min(), g.pos.min()) * 0.9; hi = max(g.true_total.max(), g.pos.max()) * 1.1
        ax.plot([lo, hi], [lo, hi], color=fs.AXIS, lw=0.8, zorder=1)
    for ax in axes[1]:
        ax.set_xlabel("True total error (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Positional events (%)")
    fs.save(fig, ROOT / "figures" / "local_metrics" / "tracker_selection_groups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
