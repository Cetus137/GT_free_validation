"""Error unmixing without simulation: four counts that are exactly zero on error-free tracks, and
a response matrix measured on the data itself by standard addition.

Counts for one frame pair:
  ended       labels at t that are absent at t+1                     (missed detection, fragmentation)
  begun       labels at t+1 that are absent at t                     (false positive, fragmentation)
  relabelled  an ended label whose predicted position holds a begun label   (fragmentation)
  flagged     two-way nearest-body flags: the detection nearest the forward prediction and the one
              nearest the backward prediction both belong to other labels   (switch)

Standard addition: inject a small known extra amount of each error type into the observed pair and
measure how much each count rises per added event; that is the response matrix.  The event counts
then follow from non-negative least squares.  No simulation is used, and no baseline is needed,
because every count is zero without errors (two-way false flags are near zero, not exactly zero).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls
from scipy.spatial import cKDTree

from .errors import ErrorConfig, apply_tracking_errors
from .local_metrics import classify_events, predicted_positions
from .metrics import compute_neighbour_continuity_observed

COUNTS = ("ended", "begun", "relabelled", "flagged")
TYPES = ("miss", "frag", "fp", "switch")
SWITCH_CAP = 1.1      # swap-partner distance (spacings), as in the Figure 1 sweeps


def nearest_body_flags(f0: dict, f1: dict, cap: float, gate: float | None = None) -> dict[int, tuple[bool, bool]]:
    """{label: (one-way, two-way)} for every label present in both frames.  With ``gate`` (spacings),
    the two-way flag also needs the predicted spot to be occupied: the bodies found forward and
    backward must lie within ``gate`` of the predictions.  A switch leaves the bodies where they are
    (only the labels move), so the spot holds a body; a fast but correctly tracked cell has left its
    spot, so the nearest body is an ordinary neighbour about a spacing away."""
    l0 = [int(l) for l in f0["assigned_ids"]]; l1 = [int(l) for l in f1["assigned_ids"]]
    if len(l0) < 5 or len(l1) < 5:
        return {}
    pred, step, sp = predicted_positions(f0, f1, cap)
    back, _, _ = predicted_positions(f1, f0, cap)
    t1, t0 = cKDTree(f1["positions"]), cKDTree(f0["positions"])
    out = {}
    for l in step:
        df, jf = t1.query(pred[l])
        one = l1[jf] != l
        two = False
        if one:
            db, jb = t0.query(back[l])
            two = l0[jb] != l and (gate is None or (df < gate * sp and db < gate * sp))
        out[l] = (one, two)
    return out


def counts(frames: list[dict], cap: float) -> tuple[np.ndarray, int]:
    """The four counts (raw numbers) summed over the consecutive pairs of ``frames`` (a frame pair,
    or a whole movie), and the number of labels at t summed over the same pairs."""
    total, n = np.zeros(len(COUNTS)), 0
    for f0, f1 in zip(frames[:-1], frames[1:]):
        l0, l1 = set(map(int, f0["assigned_ids"])), set(map(int, f1["assigned_ids"]))
        relabelled = len(classify_events(f0, f1, cap)["frag"])
        flagged = sum(two for _, two in nearest_body_flags(f0, f1, cap).values())
        total += [len(l0 - l1), len(l1 - l0), relabelled, flagged]
        n += len(l0)
    return total, n


def error_config(rates: dict[str, float], spacing: float) -> ErrorConfig:
    """Injection settings of the Figure 1 sweeps; ``rates`` per cell, switch counting both cells."""
    return ErrorConfig(p_miss=rates.get("miss", 0.0), fp_rate=rates.get("fp", 0.0), lambda_fp=0.0,
                       p_switch=rates.get("switch", 0.0) / 2.0, max_switch_distance=SWITCH_CAP * spacing,
                       p_frag=rates.get("frag", 0.0), fp_placement="near_cells", fp_offset=spacing,
                       first_frame_clean=True)


def logged_events(log: dict) -> dict[str, int]:
    return dict(miss=len(log["missed"]), frag=len(log.get("fragmentations", [])),
                fp=len(log["false_positives"]), switch=2 * len(log["switches"]))


def response_matrix(movies: list[list[dict]], box, spacing: float, cap: float, rng: np.random.Generator,
                    rate: float = 0.02, repeats: int = 1) -> np.ndarray:
    """A[count, type]: rise of each count per added event, by standard addition on ``movies`` (frame
    pairs or whole movies; errors are added to every frame but the first)."""
    rise, added = np.zeros((len(COUNTS), len(TYPES))), np.zeros(len(TYPES))
    for movie in movies:
        base, _ = counts(movie, cap)
        for k, t in enumerate(TYPES):
            for _ in range(repeats):
                more, log = apply_tracking_errors(movie, error_config({t: rate}, spacing), box, rng)
                rise[:, k] += counts(more, cap)[0] - base
                added[k] += logged_events(log)[t]
    return rise / np.maximum(added, 1)


def unmix(movies: list[list[dict]], box, spacing: float, cap: float, rng: np.random.Generator,
          rate: float = 0.02, repeats: int = 1) -> dict[str, float]:
    """Error rates (events per labelled cell at t) pooled over ``movies`` (frame pairs or whole
    movies), with the response matrix measured on the same movies."""
    c = [counts(m, cap) for m in movies]
    total, n = np.sum([x for x, _ in c], axis=0), sum(m for _, m in c)
    A = response_matrix(movies, box, spacing, cap, rng, rate, repeats)
    events, _ = nnls(A, total)
    return {t: float(e / n) for t, e in zip(TYPES, events)} | {"A": A}


# --------------------------------------------------------------------------- neighbourhood metrics

# The neighbourhood-continuity metrics as unmixing channels.  Vanished and newcomer shares are zero on
# error-free tracks; displaced, PWIO and rank correlation are not (their error-free level depends on
# the motion), so each is taken relative to its mean over the movie's cells whose neighbourhood holds
# no event: no label ending or beginning (vanished = newcomer = 0) and no two-way flagged label within
# FLAG_RADIUS spacings.  Errors are rare, so most neighbourhoods are clean and this is the movie's own
# error-free level.  The two-way flag count can be added as a further channel.
NEIGHBOUR = ("vanished", "newcomer", "displaced", "pwio", "rc")
NEIGHBOUR_KEYS = dict(vanished="wio_vanished", newcomer="newcomer_fraction", displaced="wio_displaced",
                      pwio="positional_weighted_identity_overlap", rc="rank_correlation")
ZERO_BASELINE = ("vanished", "newcomer")
FLAG_RADIUS = 1.5

# Production configuration: the five neighbourhood metrics plus the relabelled count and the two-way
# flag with the occupied-spot gate (0.5 spacings).  The gate keeps fast but correctly tracked cells
# from being flagged, and so also from being dropped from the clean cells that set the error-free level.
PRODUCTION_EXTRA = ("relabelled", "flagged")
GATE = 0.5


def neighbour_channels(frames: list[dict], cap: float, extra: tuple[str, ...] = ("flagged",),
                       channels: tuple[str, ...] = NEIGHBOUR, gate: float | None = None) -> tuple[np.ndarray, int]:
    """The neighbourhood ``channels`` of a movie (or pair), followed by the counts named in ``extra``
    ("relabelled", "flagged", "ended", "begun"; per labelled cell); and the number of labels at t
    summed over the pairs.  ``gate``: occupied-spot gate of the two-way flag (nearest_body_flags)."""
    res = compute_neighbour_continuity_observed(frames, max_neighbour_distance=cap, neighbourhood_mode="delaunay",
                                                extra_metrics=True)
    ids = [int(i) for i in res["cell_ids"]]
    M = {k: np.asarray(res[v], dtype=float) for k, v in NEIGHBOUR_KEYS.items()}
    clean = (M["vanished"] == 0) & (M["newcomer"] == 0)
    flagged_total, relabelled_total, ended_total, begun_total, n = 0, 0, 0, 0, 0
    for t, (f0, f1) in enumerate(zip(frames[:-1], frames[1:])):
        n += len(f0["assigned_ids"])
        l0, l1 = set(map(int, f0["assigned_ids"])), set(map(int, f1["assigned_ids"]))
        ended_total += len(l0 - l1); begun_total += len(l1 - l0)
        if "relabelled" in extra:
            relabelled_total += len(classify_events(f0, f1, cap)["frag"])
        flagged = [l for l, (_, two) in nearest_body_flags(f0, f1, cap, gate).items() if two]
        flagged_total += len(flagged)
        if flagged:
            pos = dict(zip(map(int, f0["assigned_ids"]), f0["positions"]))
            sp = float(np.median(cKDTree(f0["positions"]).query(f0["positions"], k=2)[0][:, 1]))
            tree = cKDTree(np.array([pos[l] for l in flagged]))
            near = np.array([i in pos and tree.query(pos[i])[0] < FLAG_RADIUS * sp for i in ids])
            clean[t] &= ~near
    out = []
    for k in channels:
        v = M[k]; ok = np.isfinite(v)
        value = float(v[ok].mean()) if ok.any() else 0.0
        if k not in ZERO_BASELINE:
            value -= float(v[ok & clean].mean()) if (ok & clean).any() else value
        out.append(value)
    counted = dict(relabelled=relabelled_total, flagged=flagged_total, ended=ended_total, begun=begun_total)
    out += [counted[e] / max(n, 1) for e in extra]
    return np.array(out), n


def neighbour_response(movie: list[dict], box, spacing: float, cap: float, rng: np.random.Generator,
                       rate: float = 0.02, extra: tuple[str, ...] = ("flagged",),
                       channels: tuple[str, ...] = NEIGHBOUR, gate: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The channels of one movie and their response matrix, measured on the same movie by standard
    addition (rise of each channel per added error per cell)."""
    base, _ = neighbour_channels(movie, cap, extra, channels, gate)
    A = np.zeros((len(base), len(TYPES)))
    for k, t in enumerate(TYPES):
        more, log = apply_tracking_errors(movie, error_config({t: rate}, spacing), box, rng)
        x, n = neighbour_channels(more, cap, extra, channels, gate)
        A[:, k] = (x - base) / max(logged_events(log)[t] / n, 1e-9)
    return base, A


def solve(base: np.ndarray, A: np.ndarray) -> dict[str, float]:
    """Rates from channels = A x rates, rates >= 0.  Each channel is scaled by its largest response,
    so a weak but specific channel (PWIO) still counts."""
    scale = np.maximum(np.abs(A).max(axis=1), 1e-9)
    rates, _ = nnls(A / scale[:, None], base / scale)
    return {t: float(r) for t, r in zip(TYPES, rates)}


def unmix_neighbour(movie: list[dict], box, spacing: float, cap: float, rng: np.random.Generator,
                    rate: float = 0.02, extra: tuple[str, ...] = ("flagged",),
                    channels: tuple[str, ...] = NEIGHBOUR, gate: float | None = None) -> dict:
    """Error rates from the neighbourhood channels of one movie (see neighbour_response and solve)."""
    base, A = neighbour_response(movie, box, spacing, cap, rng, rate, extra, channels, gate)
    return solve(base, A) | {"A": A}


def unmix_production(movie: list[dict], box, spacing: float, cap: float, rng: np.random.Generator) -> dict:
    """The production unmixing: neighbourhood metrics, relabelled count and gated two-way flag."""
    return unmix_neighbour(movie, box, spacing, cap, rng, extra=PRODUCTION_EXTRA, gate=GATE)
