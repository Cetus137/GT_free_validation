import numpy as np
from tracking_validation.config import SimulationConfig
from tracking_validation.simulator import simulate_sequence


def test_simulator_runs_small_case() -> None:
    config = SimulationConfig(
        num_cells=25,
        num_timepoints=3,
        seed=8,
        box_size=(50.0, 50.0, 50.0),
    )
    result = simulate_sequence(config)

    assert result.frames.shape == (3, 25, 3)
    assert result.speeds.shape == (25,)
    assert result.frame_speeds.shape == (3, 25)
    assert result.persistence.shape == (25,)
    assert result.energies.shape == (3,)
    assert np.isfinite(result.frames).all()
    assert np.isfinite(result.energies).all()
    assert np.isfinite(result.frame_speeds).all()
    assert not np.allclose(result.frame_speeds[1], result.frame_speeds[2])
