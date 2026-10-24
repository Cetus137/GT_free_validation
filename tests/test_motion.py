import numpy as np

from tracking_validation.motion import random_unit_vectors, step_correlated_random_walk


def test_high_persistence_keeps_direction_more_than_low_persistence() -> None:
    rng = np.random.default_rng(3)
    n = 200
    positions = np.full((n, 3), 50.0)
    directions = random_unit_vectors(n, rng)
    speeds = np.full(n, 1.0)

    high = np.full(n, 0.95)
    low = np.full(n, 0.05)

    _, dirs_high = step_correlated_random_walk(
        positions=positions.copy(),
        directions=directions.copy(),
        speeds=speeds,
        persistence=high,
        dt=1.0,
        directional_noise=0.0,
        box_size=(100.0, 100.0, 100.0),
        rng=np.random.default_rng(4),
    )
    _, dirs_low = step_correlated_random_walk(
        positions=positions.copy(),
        directions=directions.copy(),
        speeds=speeds,
        persistence=low,
        dt=1.0,
        directional_noise=0.0,
        box_size=(100.0, 100.0, 100.0),
        rng=np.random.default_rng(4),
    )

    cos_high = np.sum(directions * dirs_high, axis=1).mean()
    cos_low = np.sum(directions * dirs_low, axis=1).mean()
    assert cos_high > cos_low
