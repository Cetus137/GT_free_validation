from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import SimulationConfig
from .constraints import apply_soft_repulsion, constraint_energy
from .motion import (
    draw_cell_motion_params,
    random_unit_vectors,
    sample_frame_speeds,
    step_correlated_random_walk,
)
from .sampling import sample_centroids_poisson_jittered


@dataclass(slots=True)
class SimulationResult:
    frames: np.ndarray
    speeds: np.ndarray
    frame_speeds: np.ndarray
    persistence: np.ndarray
    energies: np.ndarray
    metadata: dict[str, Any]


def simulate_sequence(config: SimulationConfig) -> SimulationResult:
    rng = np.random.default_rng(config.seed)

    positions, adaptive_radii = sample_centroids_poisson_jittered(
        num_cells=config.num_cells,
        box_size=config.box_size,
        base_min_distance=config.sampling.base_min_distance,
        density_alpha=config.sampling.density_alpha,
        influence_radius=config.sampling.influence_radius,
        jitter_fraction=config.sampling.jitter_fraction,
        oversample_factor=config.sampling.oversample_factor,
        max_attempts=config.sampling.max_attempts,
        rng=rng,
        exact_min_distance=config.sampling.exact_min_distance,
        min_distance_sd=config.sampling.min_distance_sd,
    )

    speed_means, persistence = draw_cell_motion_params(
        num_cells=config.num_cells,
        speed_mean=config.motion.speed_mean,
        speed_std=config.motion.speed_std,
        persistence_alpha=config.motion.persistence_alpha,
        persistence_beta=config.motion.persistence_beta,
        rng=rng,
        speed_distribution=config.motion.speed_distribution,
        speed_lognorm_sigma=config.motion.speed_lognorm_sigma,
    )

    directions = random_unit_vectors(config.num_cells, rng)
    frame_speeds = [sample_frame_speeds(speed_means, config.motion.speed_step_std, rng)]

    frames = [positions.copy()]
    energies = [constraint_energy(positions, config.constraints.min_distance)]

    for _ in range(1, config.num_timepoints):
        current_speeds = sample_frame_speeds(speed_means, config.motion.speed_step_std, rng)
        positions, directions = step_correlated_random_walk(
            positions=positions,
            directions=directions,
            speeds=current_speeds,
            persistence=persistence,
            dt=config.motion.dt,
            directional_noise=config.motion.directional_noise,
            box_size=config.box_size,
            rng=rng,
        )

        positions = apply_soft_repulsion(
            positions=positions,
            min_distance=config.constraints.min_distance,
            strength=config.constraints.strength,
            iterations=config.constraints.iterations,
            box_size=config.box_size,
            rng=rng,
        )

        frames.append(positions.copy())
        frame_speeds.append(current_speeds)
        energies.append(constraint_energy(positions, config.constraints.min_distance))

    return SimulationResult(
        frames=np.asarray(frames, dtype=float),
        speeds=speed_means,
        frame_speeds=np.asarray(frame_speeds, dtype=float),
        persistence=persistence,
        energies=np.asarray(energies, dtype=float),
        metadata={
            "seed": config.seed,
            "adaptive_radii_mean": float(np.mean(adaptive_radii)),
            "adaptive_radii_std": float(np.std(adaptive_radii)),
            "speed_step_std": config.motion.speed_step_std,
        },
    )
