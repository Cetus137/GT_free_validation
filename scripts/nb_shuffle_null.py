"""Route 1 to a simulation-free switch estimate: the nearest-body test's false-flag rate from the
data's own motion (a shuffle), on the real LAP trackers (link 1.0 spacing).

    python scripts/nb_shuffle_null.py [--workers N] [--stage all|summary]

Every persisting label l gives a residual vector e_l = own step - median step of its retained
Delaunay neighbours, so its t+1 body sits at pred_l + e_l.  Null: if an evaluated t+1 body x were
correctly linked with a residual drawn from the same pair's pool, its prediction would be x - e,
and the test flags it when the detection nearest x - e is not x.  The mean over bodies and draws is
the expected false-flag count.  It uses only the t+1 detections and the residual pool: no
simulation, and the switches' own residuals enter only through the pool.

Pools: "all" = every persisting label (no truth at all); "eval" = the evaluated cells only (the
division exclusion uses the GT lineage, as everywhere).  The same null is also run on error-free
tracks (same detections, true identities), where no switch contaminates the pool.

Truth is used only to score: the true false-flag rate is flags on unswitched cells.
Writes data/_results/nb_shuffle_null.csv and figures/local_metrics/nb_shuffle_null.png.
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
import run_detector_tracking as rdt  # noqa: E402
from tracking_validation.linking import link_frames  # noqa: E402
from tracking_validation.local_metrics import predicted_positions  # noqa: E402

RES = ROOT / "data" / "_results"
K = 16          # residual draws per body
CAP = 1.0       # link distance (spacings), as in the calibrated estimate


def pair_null(f0, f1, E, rng) -> dict:
    l0 = [int(l) for l in f0["assigned_ids"]]; l1 = [int(l) for l in f1["assigned_ids"]]
    out = dict(n=0, sw=0, flag=0, tp=0, fp=0, null_all=0.0, null_eval=0.0)
    if len(l0) < 5 or len(l1) < 5:
        return out
    pred, step, _ = predicted_positions(f0, f1, rdt.NEIGH_CAP)
    p0 = dict(zip(l0, f0["positions"])); row1 = {l: i for i, l in enumerate(l1)}
    pos1 = np.asarray(f1["positions"], float); tree1 = cKDTree(pos1)
    tid0 = dict(zip(l0, f0["true_ids"].tolist())); tid1 = dict(zip(l1, f1["true_ids"].tolist()))
    ev = [l for l in step if tid0[l] >= 0 and tid1[l] >= 0 and tid0[l] in E]
    if len(ev) < 5:
        return out
    resid = {l: p0[l] + s - pred[l] for l, s in step.items()}
    sw = np.array([tid0[l] != tid1[l] for l in ev])
    flag = np.array([l1[tree1.query(pred[l])[1]] != l for l in ev])
    rows = np.array([row1[l] for l in ev])
    out.update(n=len(ev), sw=int(sw.sum()), flag=int(flag.sum()), tp=int((flag & sw).sum()), fp=int((flag & ~sw).sum()))
    for name, pool in (("all", np.array(list(resid.values()))), ("eval", np.array([resid[l] for l in ev]))):
        e = pool[rng.integers(len(pool), size=(len(ev), K))]
        _, j = tree1.query((pos1[rows][:, None, :] - e).reshape(-1, pos1.shape[1]))
        out[f"null_{name}"] = float((j.reshape(len(ev), K) != rows[:, None]).mean(1).sum())
    return out


def job(args):
    seq, img, op, stride = args
    rng = np.random.default_rng([int(seq), rdt.IMAGES.index(img), rdt.OPS.index(op), stride])
    label_map, gt, fidx, keep, spacing = rdt.load_gt(seq, stride)
    det = rdt.detection_frames(seq, img, op, label_map, fidx)
    oracle = []
    for f, g in zip(det, gt):
        real = f["true_ids"] >= 0
        have = set(int(c) for c in f["true_ids"][real])
        fill = np.array([int(c) not in have for c in g["true_ids"]], dtype=bool)
        ids = np.concatenate([f["true_ids"][real], g["true_ids"][fill]])
        oracle.append(dict(positions=np.vstack([f["positions"][real], g["positions"][fill]]), true_ids=ids, assigned_ids=ids))
    F = link_frames(det, CAP * spacing, "lap").frames
    E = [keep[t] & set(int(c) for c in gt[t + 1]["true_ids"]) for t in range(len(gt) - 1)]
    trk = pd.DataFrame([pair_null(F[t], F[t + 1], E[t], rng) for t in range(len(gt) - 1)]).sum()
    orc = pd.DataFrame([pair_null(oracle[t], oracle[t + 1], E[t], rng) for t in range(len(gt) - 1)]).sum()
    print(f"{seq} {img or 'clean':10s} {op:14s} stride {stride}: done", flush=True)
    return dict(seq=seq, img=img.lstrip("_") or "clean", op=op, stride=stride, cap=CAP,
                **trk.to_dict(), **{f"orc_{k}": orc[k] for k in ("n", "flag", "null_all", "null_eval")})


def summary() -> None:
    import matplotlib.pyplot as plt
    import figstyle as fs
    d = pd.read_csv(RES / "nb_shuffle_null.csv", dtype={"seq": str})
    keys = ["seq", "img", "op", "stride", "cap"]
    sim = pd.read_csv(RES / "sim_hop_calibrated_real.csv", dtype={"seq": str})
    d = d.merge(sim[keys + ["cap_nb_any", "false_nb_any", "est_nb_any"]], on=keys, how="left")
    d["true"] = 100 * d.sw / d.n
    d["f_true"] = 100 * d.fp / (d.n - d.sw); d["f_sim"] = 100 * d.false_nb_any
    for p in ("all", "eval"):
        d[f"f_{p}"] = 100 * d[f"null_{p}"] / d.n
    d["f_orc"] = 100 * d.orc_flag / d.orc_n; d["f_orc_null"] = 100 * d.orc_null_all / d.orc_n
    print(f"{len(d)} trackers; flags match hop_count_variants: "
          f"{(pd.read_csv(RES / 'hop_count_variants.csv', dtype={'seq': str}).query('cap == 1.0').merge(d, on=keys).eval('Hnb_any == flag')).mean():.0%}")
    pd.set_option("display.width", 200)
    print("\nfalse-flag rate (% of cells), mean by frame interval:")
    print(d.groupby("stride")[["f_true", "f_all", "f_eval", "f_sim", "f_orc", "f_orc_null"]].mean().round(3).to_string())
    print("\nper tracker, against the true false-flag rate:")
    for c in ("f_all", "f_eval", "f_sim"):
        within = " ".join(f"{np.corrcoef(h.f_true, h[c])[0, 1]:+.2f}" if h[c].std() > 0 else "  nan" for _, h in d.groupby("stride"))
        print(f"  {c:7s} r {np.corrcoef(d.f_true, d[c])[0, 1]:+.2f}   within interval {within}   "
              f"median ratio {np.median(d[c] / d.f_true.where(d.f_true > 0)):.2f}")

    # switch estimates: capture still from simulation (route 1 replaces only the false flags)
    d["est_shuffle"] = 100 * (d.flag - d.null_all) / (d.cap_nb_any * d.n)
    d["est_sim"] = d.est_nb_any
    d["caught"] = 100 * d.tp / d.n; d["caught_est"] = 100 * (d.flag - d.null_all) / d.n
    print("\nswitch estimate vs truth (r, median |error|, median bias):")
    for c, t, lab in (("est_sim", "true", "simulation false flags + capture"),
                      ("est_shuffle", "true", "shuffle false flags + sim capture"),
                      ("caught_est", "caught", "caught switches, no simulation")):
        e, tr = d[c], d[t]
        rel = (e - tr) / tr
        per = " ".join(f"{100 * np.median(((h[c] - h[t]) / h[t])):+.0f}%" for _, h in d.groupby("stride"))
        print(f"  {lab:36s} r {np.corrcoef(e, tr)[0, 1]:.2f}  |err| {100 * np.median(rel.abs()):.0f}%  "
              f"bias {100 * np.median(rel):+.0f}%   by interval {per}")

    fs.apply()
    cols = {1: "#2a78d6", 2: "#1baf7a", 3: "#eda100", 4: "#eb6834"}
    fig, axes = plt.subplots(1, 3, figsize=(fs.DOUBLE, 2.4), gridspec_kw=dict(left=0.07, right=0.87, top=0.86, bottom=0.2, wspace=0.38))
    panels = ((axes[0], "f_true", "f_sim", "False flags, simulation (%)", "Simulation"),
              (axes[1], "f_true", "f_all", "False flags, shuffle (%)", "Shuffle of own residuals"),
              (axes[2], "true", "est_shuffle", "Estimated switched (%)", "Switch estimate, shuffle"))
    for ax, x, y, ylabel, title in panels:
        lim = float(max(d[x].max(), d[y].max())) * 1.08
        lo = min(0.0, float(d[y].min()) * 1.08)          # the switch estimate can go negative
        ax.plot([0, lim], [0, lim], color=fs.INK2, lw=0.9, ls=(0, (6, 3)), zorder=1)
        ax.axhline(0, color=fs.AXIS, lw=0.6, zorder=1)
        for st, c in cols.items():
            h = d[d.stride == st]
            ax.scatter(h[x], h[y], s=12, color=c, edgecolors=fs.SURFACE, linewidths=0.5, zorder=3, label=f"every {st}" if st > 1 else "every frame")
        ax.set_xlim(0, lim); ax.set_ylim(lo, lim)
        ax.set_xlabel("True false flags (%)" if x == "f_true" else "True switched (%)"); ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=3)
        ax.text(0.04, 0.96, f"r = {np.corrcoef(d[x], d[y])[0, 1]:.2f}", transform=ax.transAxes, va="top", fontsize=6.5, color=fs.INK2)
    axes[2].legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.04, 1.0), title="Linking", title_fontsize=6.5, alignment="left")
    fs.save(fig, ROOT / "figures" / "local_metrics" / "nb_shuffle_null")


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--stage", choices=("all", "summary"), default="all")
    a = p.parse_args()
    if a.stage == "all":
        jobs = [(s, img, op, st) for s in ("01", "02") for img in rdt.IMAGES for op in rdt.OPS for st in rdt.STRIDES
                if not ("_n" in img and op == "high_recall")]
        with ProcessPoolExecutor(a.workers) as ex:
            rows = list(ex.map(job, jobs))
        pd.DataFrame(rows).to_csv(RES / "nb_shuffle_null.csv", index=False)
    summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
