"""RC (mean, and share of cells with their own RC < 0.3) and the nearest-body count against the
true switch rate on the real worm LAP trackers (LoG detections, link 1.0 spacing), coloured by
frame interval.  Coloured dashes: the channel's error-free level at that interval (oracle
tracks: the same detections with their true identities), i.e. the baseline a reference-free
reading would have to know.  Grey dashes on the count panels: y = x.

    python scripts/plot_switch_channels_real.py

Reads data/_results/hop_count_variants.csv and detector_tracking_pairs_resid.csv; writes
figures/local_metrics/switch_channels_real.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

RES = ROOT / "data" / "_results"
COLS = {1: "#2a78d6", 2: "#1baf7a", 3: "#eda100", 4: "#eb6834"}
NAMES = {1: "every frame", 2: "every 2nd", 3: "every 3rd", 4: "every 4th"}


def load() -> pd.DataFrame:
    keys = ["seq", "img", "op", "stride", "cap"]
    hv = pd.read_csv(RES / "hop_count_variants.csv", dtype={"seq": str})
    pr = pd.read_csv(RES / "detector_tracking_pairs_resid.csv", dtype={"seq": str})
    pr["img"] = pr.img.fillna("").astype(str).str.lstrip("_").replace("", "clean")
    g = pr.groupby(keys)[["trk_rc", "orc_rc", "trk_rc_out", "orc_rc_out"]].mean().reset_index()
    d = hv.merge(g, on=keys)
    d = d[d.cap == 1.0].copy()
    d["true"] = 100 * d.sw / d.n
    d["rc"] = 100 * d.trk_rc; d["rc_clean"] = 100 * d.orc_rc
    d["rcf"] = 100 * d.trk_rc_out; d["rcf_clean"] = 100 * d.orc_rc_out      # cells with own RC < 0.3
    d["nb"] = 100 * d.Hnb_any / d.n; d["nb_clean"] = 100 * d.orc_Hnb_any / d.orc_n
    return d


def main() -> int:
    d = load()
    print(f"{len(d)} trackers")
    fs.apply()
    fig, axes = plt.subplots(1, 3, figsize=(fs.DOUBLE, 2.5),
                             gridspec_kw=dict(left=0.07, right=0.87, top=0.86, bottom=0.18, wspace=0.36))
    panels = ((axes[0], "rc", "rc_clean", "RC, tracker (×100)", "Rank correlation, mean"),
              (axes[1], "rcf", "rcf_clean", "Cells flagged (%)", "Rank correlation < 0.3"),
              (axes[2], "nb", "nb_clean", "Cells flagged (%)", "Nearest-body test, uncorrected"))
    xmax = d.true.max() * 1.05
    for ax in axes[1:]:                          # count panels: same scale, y = x marks a perfect count
        ax.plot([0, xmax], [0, xmax], color=fs.INK2, lw=0.9, ls=(0, (6, 3)), zorder=2)
        ax.text(xmax * 0.97, xmax * 0.97, "y = x", ha="right", va="bottom", fontsize=6.5, color=fs.INK2)
        ax.set_ylim(0, xmax)
    for ax, col, clean, ylabel, title in panels:
        for st, c in COLS.items():
            h = d[d.stride == st]
            ax.hlines(h[clean].mean(), 0, xmax, color=c, lw=1.0, ls=(0, (3, 2)), zorder=1)
            ax.scatter(h.true, h[col], s=14, color=c, edgecolors=fs.SURFACE, linewidths=0.5, zorder=3, label=NAMES[st])
        r_all = np.corrcoef(d.true, d[col])[0, 1]
        r_in = np.mean([np.corrcoef(h.true, h[col])[0, 1] for _, h in d.groupby("stride")])
        print(f"{col}: r overall {r_all:+.2f}, mean r within interval {r_in:+.2f}")
        rc = col == "rc"        # RC: in the empty gap between the every-frame and every-2nd bands
        ax.text(0.97 if rc else 0.03, 0.78 if rc else 0.96, f"r = {r_all:+.2f}\nwithin interval {r_in:+.2f}",
                transform=ax.transAxes, ha="right" if rc else "left", va="center" if rc else "top", fontsize=6.5, color=fs.INK2)
        ax.set_xlim(0, xmax); ax.set_xlabel("True switched cells (%)"); ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=3)
    axes[2].legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.04, 1.0), title="Linking", title_fontsize=6.5,
                   handletextpad=0.2, alignment="left")
    fig.text(0.07, 0.95, "Coloured dashes: error-free level at that interval (same detections, true identities)",
             fontsize=6.5, color=fs.MUTED, va="top")
    fs.save(fig, ROOT / "figures" / "local_metrics" / "switch_channels_real")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
