from __future__ import annotations

import numpy as np


def random_unit_vectors(num: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=(num, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return v / norms


def draw_cell_motion_params(
    num_cells: int,
    speed_mean: float,
    speed_std: float,
    persistence_alpha: float,
    persistence_beta: float,
    rng: np.random.Generator,
    speed_distribution: str = "normal",
    speed_lognorm_sigma: float = 0.751,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample per-cell speed and persistence parameters.

    Parameters
    ----------
    speed_distribution:
        ``"normal"``    — Normal(speed_mean, speed_std), clipped to >0.
        ``"lognormal"`` — LogNormal with median=speed_mean, shape=speed_lognorm_sigma.
                          speed_std is ignored; the shape parameter controls spread.
                          Fitted from real tracking data: sigma=0.751, median≈5.0.
    """
    if speed_distribution == "lognormal":
        # log-normal: median = exp(mu) = speed_mean  →  mu = ln(speed_mean)
        mu = np.log(max(speed_mean, 1e-6))
        speeds = rng.lognormal(mean=mu, sigma=speed_lognorm_sigma, size=num_cells)
    else:
        speeds = rng.normal(loc=speed_mean, scale=speed_std, size=num_cells)
        speeds = np.clip(speeds, 1e-6, None)
    persistence = rng.beta(a=persistence_alpha, b=persistence_beta, size=num_cells)
    persistence = np.clip(persistence, 1e-3, 0.999)
    return speeds, persistence


def sample_frame_speeds(
    speed_means: np.ndarray,
    speed_step_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if speed_step_std <= 0.0:
        return speed_means.copy()
    sampled = rng.normal(loc=speed_means, scale=speed_step_std, size=speed_means.shape)
    return np.clip(sampled, 1e-6, None)


def apply_reflective_boundaries(
    positions: np.ndarray,
    directions: np.ndarray,
    box_size: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    box = np.asarray(box_size, dtype=float)
    for axis in range(3):
        low_mask = positions[:, axis] < 0.0
        high_mask = positions[:, axis] > box[axis]

        positions[low_mask, axis] *= -1.0
        positions[high_mask, axis] = 2.0 * box[axis] - positions[high_mask, axis]

        bounce_mask = low_mask | high_mask
        directions[bounce_mask, axis] *= -1.0

    return positions, directions


def step_correlated_random_walk(
    positions: np.ndarray,
    directions: np.ndarray,
    speeds: np.ndarray,
    persistence: np.ndarray,
    dt: float,
    directional_noise: float,
    box_size: tuple[float, float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    random_dirs = random_unit_vectors(num=positions.shape[0], rng=rng)
    keep = persistence[:, None]
    new_dirs = keep * directions + (1.0 - keep) * random_dirs
    if directional_noise > 0.0:
        new_dirs = new_dirs + rng.normal(scale=directional_noise, size=new_dirs.shape)

    norms = np.linalg.norm(new_dirs, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    new_dirs = new_dirs / norms

    displacement = new_dirs * (speeds[:, None] * dt)
    new_positions = positions + displacement
    new_positions, new_dirs = apply_reflective_boundaries(new_positions, new_dirs, box_size)
    return new_positions, new_dirs
