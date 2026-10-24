import numpy as np

from tracking_validation.constraints import apply_soft_repulsion, constraint_energy


def test_soft_repulsion_reduces_constraint_energy() -> None:
    rng = np.random.default_rng(5)
    positions = np.array(
        [
            [10.0, 10.0, 10.0],
            [10.2, 10.0, 10.0],
            [10.0, 10.2, 10.0],
            [10.0, 10.0, 10.2],
        ]
    )
    before = constraint_energy(positions, min_distance=1.0)
    after_positions = apply_soft_repulsion(
        positions,
        min_distance=1.0,
        strength=0.7,
        iterations=10,
        box_size=(20.0, 20.0, 20.0),
        rng=rng,
    )
    after = constraint_energy(after_positions, min_distance=1.0)
    assert after < before
