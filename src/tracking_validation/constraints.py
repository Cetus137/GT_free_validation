from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree


def constraint_energy(positions: np.ndarray, min_distance: float) -> float:
    """Sum of squared overlaps across all cell pairs."""
    diffs = positions[:, None, :] - positions[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(dists, np.inf)
    overlaps = np.maximum(0.0, min_distance - dists)
    return float(np.sum(overlaps ** 2) / 2)   # /2: each pair counted twice


def apply_soft_repulsion(
    positions: np.ndarray,
    min_distance: float,
    strength: float,
    iterations: int,
    box_size: tuple[float, float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Push overlapping cells apart over several iterations.

    Vectorised implementation using KDTree neighbour lookup — only cell pairs
    within ``min_distance`` of each other are considered, making each iteration
    O(N log N) instead of O(N²).
    """
    box     = np.asarray(box_size, dtype=float)
    updated = positions.copy()

    for _ in range(iterations):
        # Only query pairs within min_distance — much sparser than all pairs
        tree  = KDTree(updated)
        pairs = tree.query_pairs(min_distance, output_type="ndarray")  # (M, 2)

        if len(pairs) == 0:
            break

        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]

        delta = updated[i_idx] - updated[j_idx]          # (M, 3)
        dist  = np.linalg.norm(delta, axis=1)             # (M,)

        # Handle coincident points with a random nudge
        coincident = dist < 1e-12
        if coincident.any():
            rand_dir = rng.normal(size=(coincident.sum(), 3))
            norms = np.linalg.norm(rand_dir, axis=1, keepdims=True)
            delta[coincident] = rand_dir / np.maximum(norms, 1e-12)
            dist[coincident]  = 1e-12

        direction = delta / dist[:, None]                 # (M, 3) unit vectors
        overlap   = min_distance - dist                   # (M,) > 0 by construction
        shift     = (0.5 * strength * overlap)[:, None] * direction   # (M, 3)

        displacements = np.zeros_like(updated)
        np.add.at(displacements, i_idx,  shift)
        np.add.at(displacements, j_idx, -shift)

        updated = np.clip(updated + displacements, 0.0, box)

    return updated
