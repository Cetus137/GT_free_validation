import numpy as np
import pytest
from scipy.spatial import cKDTree

from tracking_validation.dynamics import DynamicsConfig, simulate_dynamics


def test_shapes_bounds_and_reproducible() -> None:
    cfg = DynamicsConfig(num_cells=60, box_size=30.0, num_frames=5, burn_in=2.0, seed=4)
    a, b = simulate_dynamics(cfg), simulate_dynamics(cfg)
    assert a.frames.shape == (5, 60, 3) and a.velocities.shape == (5, 60, 3)
    assert np.array_equal(a.frames, b.frames)
    assert (a.frames >= 0.0).all() and (a.frames <= 30.0).all()
    assert len(a.as_frames()) == 5


def test_springs_keep_cells_apart() -> None:
    res = simulate_dynamics(DynamicsConfig(num_cells=200, box_size=35.0, spacing=5.0, speed=0.3, num_frames=3, seed=1))
    nn = cKDTree(res.frames[-1]).query(res.frames[-1], k=2)[0][:, 1]
    assert np.quantile(nn, 0.1) > 0.8 * 5.0


def test_velocity_keeps_direction_for_persistence_time() -> None:
    # free cells (no springs, walls out of reach): rms speed as set, autocorrelation exp(-lag / tau)
    cfg = DynamicsConfig(num_cells=2000, box_size=1e6, stiffness=0.0, speed=2.0, persistence_time=4.0,
                         frame_interval=4.0, num_frames=2, burn_in=1.0, seed=2)
    v = simulate_dynamics(cfg).velocities
    power = np.mean(np.sum(v[0] ** 2, axis=1))
    assert power == pytest.approx(4.0, rel=0.06)
    assert np.mean(np.sum(v[0] * v[1], axis=1)) / power == pytest.approx(np.exp(-1.0), abs=0.04)


def test_alignment_makes_neighbours_move_together_at_the_same_speed() -> None:
    base = dict(num_cells=400, box_size=45.0, spacing=5.0, speed=1.0, persistence_time=5.0,
                frame_interval=2.0, num_frames=4, seed=3)
    free = simulate_dynamics(DynamicsConfig(**base))
    aligned = simulate_dynamics(DynamicsConfig(alignment=5.0, **base))
    assert abs(free.metadata["neighbour_alignment"]) < 0.1
    assert aligned.metadata["neighbour_alignment"] > 0.5
    rms = [np.sqrt(np.mean(np.sum(r.velocities ** 2, axis=2))) for r in (free, aligned)]
    assert rms[1] == pytest.approx(rms[0], rel=0.06)


def test_unstable_step_rejected() -> None:
    with pytest.raises(ValueError):
        DynamicsConfig(stiffness=10.0, dt=0.05)
