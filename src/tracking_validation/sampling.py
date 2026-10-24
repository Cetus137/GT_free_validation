from __future__ import annotations

import math

import numpy as np
from scipy.spatial import KDTree


def _within_box(points: np.ndarray, box_size: tuple[float, float, float]) -> np.ndarray:
    lower_ok = points >= 0.0
    upper_ok = points <= np.asarray(box_size)
    return np.all(lower_ok & upper_ok, axis=1)


def sample_centroids_poisson_jittered(
    num_cells: int,
    box_size: tuple[float, float, float],
    base_min_distance: float,
    density_alpha: float,
    influence_radius: float,
    jitter_fraction: float,
    oversample_factor: float,
    max_attempts: int,
    rng: np.random.Generator,
    exact_min_distance: bool = True,
    min_distance_sd: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample cell centroids via jittered Poisson-disk sampling.

    When density_alpha == 0 (uniform packing) the acceptance test is a simple
    minimum-distance check, done efficiently with a KDTree that is rebuilt
    incrementally.  When density_alpha > 0 the adaptive radius rule is applied
    per candidate (slower, but density_alpha > 0 is only used for small N).

    ``exact_min_distance`` (default True): the fast path's KDTree is only rebuilt
    every ``rebuild_interval`` acceptances, so candidates must additionally be
    checked against cells accepted since the last rebuild.  Pass False to reproduce
    pre-0.3 runs, where that check was skipped and roughly 0.3% of cells ended up
    closer than ``base_min_distance`` to a neighbour.

    ``min_distance_sd`` (default 0 = off): soft floor.  Each candidate draws its own
    minimum distance r ~ Normal(base_min_distance, min_distance_sd), truncated below
    at base_min_distance / 2, and two cells must be at least (r_i + r_j) / 2 apart
    (checked exactly against every accepted cell).  Gives a nearest-neighbour
    distribution with a soft lower tail instead of a hard cut-off.
    """
    accepted: list[np.ndarray] = []
    accepted_radii: list[float] = []

    grid_count = max(2, int(math.ceil((num_cells * oversample_factor) ** (1.0 / 3.0))))
    nx, ny, nz = grid_count, grid_count, grid_count
    box = np.asarray(box_size, dtype=float)
    cell = box / np.array([nx, ny, nz], dtype=float)

    gx = (np.arange(nx) + 0.5) * cell[0]
    gy = (np.arange(ny) + 0.5) * cell[1]
    gz = (np.arange(nz) + 0.5) * cell[2]
    lattice    = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1).reshape(-1, 3)
    jitter     = rng.uniform(-1.0, 1.0, size=lattice.shape) * (jitter_fraction * cell)
    candidates = lattice + jitter
    candidates = candidates[_within_box(candidates, box_size)]
    rng.shuffle(candidates)

    if density_alpha == 0.0 and min_distance_sd > 0.0:
        # --- soft-floor path: per-cell minimum distance, exact checking ---
        rebuild_interval = max(50, num_cells // 20)
        r_floor = 0.5 * base_min_distance
        tree: KDTree | None = None
        tree_size = 0
        radii: list[float] = []
        r_max = 0.0

        def _accept_soft(pt: np.ndarray) -> bool:
            nonlocal tree, tree_size, r_max
            r = max(float(rng.normal(base_min_distance, min_distance_sd)), r_floor)
            reach = 0.5 * (r + r_max)
            if tree is not None:
                for j in tree.query_ball_point(pt, reach):
                    if np.linalg.norm(accepted[j] - pt) < 0.5 * (r + radii[j]):
                        return False
            for j in range(tree_size, len(accepted)):
                if np.linalg.norm(accepted[j] - pt) < 0.5 * (r + radii[j]):
                    return False
            accepted.append(pt)
            radii.append(r)
            accepted_radii.append(r)
            r_max = max(r_max, r)
            if len(accepted) % rebuild_interval == 0:
                tree = KDTree(np.asarray(accepted))
                tree_size = len(accepted)
            return True

        for cand in candidates:
            if len(accepted) >= num_cells:
                break
            _accept_soft(cand)
        attempts = 0
        while len(accepted) < num_cells and attempts < max_attempts:
            _accept_soft(rng.uniform([0.0, 0.0, 0.0], box))
            attempts += 1

    elif density_alpha == 0.0:
        # --- fast path: uniform min-distance, use KDTree ---
        # Process all lattice candidates first; rebuild tree periodically.
        rebuild_interval = max(50, num_cells // 20)

        tree: KDTree | None = None
        tree_size = 0

        def _accept_fast(pt: np.ndarray) -> bool:
            nonlocal tree, tree_size
            if not accepted:
                accepted.append(pt)
                accepted_radii.append(base_min_distance)
                return True
            if tree is not None:
                d, _ = tree.query(pt, k=1)
                if d < base_min_distance:
                    return False
                if exact_min_distance and len(accepted) > tree_size:
                    recent = np.asarray(accepted[tree_size:])
                    if np.any(np.linalg.norm(recent - pt, axis=1) < base_min_distance):
                        return False
            else:
                acc = np.asarray(accepted)
                if np.any(np.linalg.norm(acc - pt, axis=1) < base_min_distance):
                    return False
            accepted.append(pt)
            accepted_radii.append(base_min_distance)
            if len(accepted) % rebuild_interval == 0:
                tree = KDTree(np.asarray(accepted))
                tree_size = len(accepted)
            return True

        for cand in candidates:
            if len(accepted) >= num_cells:
                break
            _accept_fast(cand)

        attempts = 0
        while len(accepted) < num_cells and attempts < max_attempts:
            cand = rng.uniform([0.0, 0.0, 0.0], box)
            _accept_fast(cand)
            attempts += 1

    else:
        # --- original adaptive path (density_alpha > 0, typically small N) ---
        def _adaptive_radius(point, acc_arr):
            if acc_arr.size == 0:
                return base_min_distance
            dists = np.linalg.norm(acc_arr - point, axis=1)
            nearby_count = int(np.sum(dists < influence_radius))
            return base_min_distance * (1.0 + density_alpha * nearby_count)

        def _try_accept(candidate: np.ndarray) -> bool:
            if len(accepted) >= num_cells:
                return False
            if not accepted:
                accepted.append(candidate)
                accepted_radii.append(base_min_distance)
                return True
            acc = np.asarray(accepted)
            dists = np.linalg.norm(acc - candidate, axis=1)
            candidate_radius = _adaptive_radius(candidate, acc)
            pairwise_required = np.maximum(candidate_radius, np.asarray(accepted_radii))
            if np.all(dists >= pairwise_required):
                accepted.append(candidate)
                accepted_radii.append(candidate_radius)
                return True
            return False

        for cand in candidates:
            if len(accepted) >= num_cells:
                break
            _try_accept(cand)

        attempts = 0
        while len(accepted) < num_cells and attempts < max_attempts:
            cand = rng.uniform([0.0, 0.0, 0.0], box)
            _try_accept(cand)
            attempts += 1

    if len(accepted) < num_cells:
        raise RuntimeError(
            f"Could not sample {num_cells} centroids under current spacing constraints "
            f"(got {len(accepted)}); try reducing base_min_distance, density_alpha, "
            f"or increasing box size / max_attempts."
        )

    return np.asarray(accepted, dtype=float), np.asarray(accepted_radii, dtype=float)
