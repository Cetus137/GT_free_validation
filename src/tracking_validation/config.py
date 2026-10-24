from __future__ import annotations

from dataclasses import dataclass, field

SPEED_DISTRIBUTIONS = ("normal", "lognormal")


@dataclass(slots=True)
class SamplingConfig:
    base_min_distance: float = 4.0
    density_alpha: float = 0.2
    influence_radius: float = 10.0
    jitter_fraction: float = 0.35
    oversample_factor: float = 3.0
    max_attempts: int = 20000
    exact_min_distance: bool = True    # True (default): base_min_distance is enforced
                                       # against every accepted cell. False reproduces
                                       # pre-0.3 runs, where cells accepted since the last
                                       # KDTree rebuild were not checked and ~0.3% of
                                       # pairs ended up closer than base_min_distance.
    min_distance_sd: float = 0.0       # >0: soft floor, per-cell min distance drawn from
                                       # Normal(base_min_distance, sd); 0 = hard floor


@dataclass(slots=True)
class MotionConfig:
    dt: float = 1.0
    speed_mean: float = 5.0
    speed_std: float = 0.25
    speed_step_std: float = 0.15
    persistence_alpha: float = 6.0
    persistence_beta: float = 2.0
    directional_noise: float = 0.1
    # Per-cell speed distribution.
    # "normal"    — Normal(speed_mean, speed_std), clipped to >0 (default)
    # "lognormal" — LogNormal with median=speed_mean, sigma=speed_lognorm_sigma
    #               speed_std is ignored; use speed_lognorm_sigma instead.
    #               Fitted from real data: sigma=0.751, median≈5.0
    speed_distribution: str = "normal"
    speed_lognorm_sigma: float = 0.751  # shape param σ from log-normal fit to real data

    def __post_init__(self) -> None:
        if self.speed_distribution not in SPEED_DISTRIBUTIONS:
            raise ValueError(
                f"Unknown speed_distribution {self.speed_distribution!r}; "
                f"expected one of {SPEED_DISTRIBUTIONS}"
            )


@dataclass(slots=True)
class ConstraintConfig:
    min_distance: float = 4.0
    strength: float = 0.7
    iterations: int = 8


@dataclass(slots=True)
class SimulationConfig:
    num_cells: int = 100
    num_timepoints: int = 3
    seed: int | None = None
    box_size: tuple[float, float, float] = (100.0, 100.0, 100.0)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
