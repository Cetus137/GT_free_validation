"""Nucleus detection by multi-scale Laplacian of Gaussian (LoG), and matching to GT markers.

Positions are physical (x, y, z) in microns, the same convention as ``ctc.read_ctc_tracks``,
so detections can be fed straight to ``linking.link_frames``.

Pipeline per frame (array axis order z, y, x):
  1. average x and y over ``bin_xy`` pixels (z unchanged);
  2. normalise: (img - median) / (p99.9 - median), one pair of numbers per frame;
  3. scale-normalised LoG in PHYSICAL units at several widths sigma (microns):
        R_sigma = -sigma^2 * sum_axis d2/dx_axis2 [G_sigma * img] / voxel_axis^2
     so anisotropic voxels are handled exactly; the z width has a floor in voxels;
  4. candidates = local maxima of max_sigma R over a 3x3x3 window;
  5. ``select``: keep candidates above ``threshold``, then greedy non-maximum suppression
     within ``min_sep`` microns; sub-voxel refinement by a parabola along each axis.

Candidates are computed once per frame, so sweeping threshold and separation is cheap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

# microns; blob radius ~ sqrt(3) * sigma.  The smallest width matches the smallest (late) nuclei,
# r ~ 1.4 um; a 0.7 um floor picked up background speckle in the sparse early frames.
DEFAULT_SIGMAS = tuple(np.geomspace(0.85, 2.2, 6))


@dataclass
class Candidates:
    """All LoG local maxima of one frame, before thresholding and suppression."""
    pos: np.ndarray        # (N, 3) refined physical x, y, z (microns)
    response: np.ndarray   # (N,) scale-normalised LoG response at the peak
    sigma: np.ndarray      # (N,) best width (microns)


def downsample_xy(img_zyx: np.ndarray, bin_xy: int) -> np.ndarray:
    """Block-average y and x by ``bin_xy`` (trailing pixels that do not fill a block are dropped)."""
    if bin_xy == 1:
        return img_zyx.astype(np.float32)
    z, y, x = img_zyx.shape
    y2, x2 = y // bin_xy * bin_xy, x // bin_xy * bin_xy
    a = img_zyx[:, :y2, :x2].astype(np.float32)
    return a.reshape(z, y2 // bin_xy, bin_xy, x2 // bin_xy, bin_xy).mean(axis=(2, 4))


def normalise(img: np.ndarray, lo_pct: float = 50.0, hi_pct: float = 99.9) -> np.ndarray:
    lo, hi = np.percentile(img, [lo_pct, hi_pct])
    return (img - lo) / max(hi - lo, 1e-9)


def log_response(img_zyx: np.ndarray, sigma_um: float, voxel_zyx: tuple[float, float, float],
                 min_sigma_vox: float = 0.5) -> np.ndarray:
    """Scale-normalised, sign-flipped LoG (bright blobs positive) in physical units."""
    s_vox = [max(sigma_um / v, min_sigma_vox) for v in voxel_zyx]
    lap = np.zeros_like(img_zyx, dtype=np.float32)
    for ax, v in enumerate(voxel_zyx):
        order = [0, 0, 0]; order[ax] = 2
        lap += ndi.gaussian_filter(img_zyx, s_vox, order=order, mode="nearest") / (v * v)
    return -(sigma_um ** 2) * lap


def _parabola_offset(fm: float, f0: float, fp: float) -> float:
    d = fm - 2 * f0 + fp
    return 0.0 if d >= 0 else float(np.clip(0.5 * (fm - fp) / d, -0.5, 0.5))


def candidates(img_zyx: np.ndarray, voxel_xyz: tuple[float, float, float], bin_xy: int = 4,
               sigmas: tuple[float, ...] = DEFAULT_SIGMAS, floor: float = 0.0) -> Candidates:
    """LoG local maxima of one raw frame.  ``voxel_xyz`` is the ORIGINAL voxel size (x, y, z)."""
    vx, vy, vz = voxel_xyz
    img = normalise(downsample_xy(img_zyx, bin_xy))
    voxel_zyx = (vz, vy * bin_xy, vx * bin_xy)
    stack = np.stack([log_response(img, s, voxel_zyx) for s in sigmas])
    best = stack.max(axis=0)
    which = stack.argmax(axis=0)
    peak = (best == ndi.maximum_filter(best, size=3, mode="nearest")) & (best > floor)
    zz, yy, xx = np.nonzero(peak)
    resp = best[zz, yy, xx]
    off = np.zeros((zz.size, 3))
    shape = best.shape
    for k, (z, y, x) in enumerate(zip(zz, yy, xx)):
        for ax, (i, n) in enumerate(((z, shape[0]), (y, shape[1]), (x, shape[2]))):
            if 0 < i < n - 1:
                idx_m = [z, y, x]; idx_p = [z, y, x]; idx_m[ax] -= 1; idx_p[ax] += 1
                off[k, ax] = _parabola_offset(best[tuple(idx_m)], best[z, y, x], best[tuple(idx_p)])
    zf, yf, xf = zz + off[:, 0], yy + off[:, 1], xx + off[:, 2]
    # binned pixel i covers original pixels i*b .. i*b + b-1, centred at i*b + (b-1)/2
    c = (bin_xy - 1) / 2.0
    pos = np.column_stack([(xf * bin_xy + c) * vx, (yf * bin_xy + c) * vy, zf * vz])
    return Candidates(pos=pos, response=resp.astype(float), sigma=np.asarray(sigmas)[which[zz, yy, xx]])


def select(c: Candidates, threshold: float, min_sep: float, scale_aware: bool = True) -> np.ndarray:
    """Indices into ``c`` kept after thresholding and greedy suppression.

    A weaker peak is suppressed if it lies within ``min_sep`` microns of a stronger kept peak or,
    with ``scale_aware``, inside the stronger peak's blob radius sqrt(3) * sigma (stops large
    early nuclei from producing several detections)."""
    idx = np.nonzero(c.response > threshold)[0]
    idx = idx[np.argsort(-c.response[idx])]
    kept: list[int] = []
    pts = np.empty((0, 3)); rad = np.empty(0)
    for i in idx:
        p = c.pos[i]
        if len(kept):
            d = np.linalg.norm(pts - p, axis=1)
            if np.any(d < (np.maximum(min_sep, rad) if scale_aware else min_sep)):
                continue
        kept.append(int(i))
        pts = np.vstack([pts, p]); rad = np.append(rad, np.sqrt(3.0) * c.sigma[i])
    return np.asarray(kept, dtype=int)


def match_to_gt(det: np.ndarray, gt: np.ndarray, cap: float) -> tuple[np.ndarray, np.ndarray]:
    """Optimal one-to-one matching within ``cap`` microns.

    Returns (gt_index per detection, -1 if unmatched; distance per detection, nan if unmatched)."""
    out = np.full(len(det), -1, dtype=int)
    dist = np.full(len(det), np.nan)
    if len(det) == 0 or len(gt) == 0:
        return out, dist
    d = np.linalg.norm(det[:, None, :] - gt[None, :, :], axis=2)
    big = 1e6
    cost = np.where(d <= cap, d, big)
    r, c = linear_sum_assignment(cost)
    ok = cost[r, c] < big
    out[r[ok]] = c[ok]; dist[r[ok]] = d[r[ok], c[ok]]
    return out, dist


def match_cap(gt: np.ndarray, frac: float = 0.5, floor: float = 1.5) -> float:
    """Matching radius: ``frac`` of the frame's median GT nearest-neighbour spacing, at least ``floor``."""
    if len(gt) < 2:
        return 5.0
    nn = cKDTree(gt).query(gt, k=2)[0][:, 1]
    return max(floor, frac * float(np.median(nn)))
