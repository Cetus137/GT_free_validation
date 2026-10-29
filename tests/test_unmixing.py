import numpy as np
import pytest

from tracking_validation import DynamicsConfig, apply_tracking_errors, realised_error_rates, simulate_dynamics
from tracking_validation import unmixing as um

BOX = 30.0


@pytest.fixture(scope="module")
def pairs():
    gt = simulate_dynamics(DynamicsConfig(num_cells=300, box_size=BOX, spacing=4.5, speed=0.8,
                                          num_frames=5, seed=5)).as_frames()
    return [gt[t:t + 2] for t in range(4)]


def test_counts_are_zero_without_errors(pairs) -> None:
    for p in pairs:
        c, n = um.counts(p, 15.0)
        assert n == 300 and (c == 0).all()


def test_unmixing_recovers_an_injected_mix(pairs) -> None:
    rng = np.random.default_rng(1)
    true = dict(miss=0.02, frag=0.02, fp=0.01, switch=0.02)
    obs, real = [], []
    for p in pairs:
        o, log = apply_tracking_errors(p, um.error_config(true, 4.3), (BOX,) * 3, rng)
        obs.append(o); real.append(realised_error_rates(p, o, log, first_frame_clean=True))
    est = um.unmix(obs, (BOX,) * 3, 4.3, 15.0, rng)
    for t in um.TYPES:
        assert est[t] == pytest.approx(np.mean([r[t] for r in real]), abs=0.006)
