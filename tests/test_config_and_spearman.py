import numpy as np
import pytest

from tracking_validation.config import MotionConfig
from tracking_validation.metrics import _spearman
from tracking_validation.motion import draw_cell_motion_params


def test_motion_config_rejects_unknown_speed_distribution() -> None:
    with pytest.raises(ValueError, match="speed_distribution"):
        MotionConfig(speed_distribution="lognorml")


def test_draw_cell_motion_params_rejects_unknown_speed_distribution() -> None:
    with pytest.raises(ValueError, match="speed_distribution"):
        draw_cell_motion_params(10, 5.0, 0.25, 6.0, 2.0, np.random.default_rng(0),
                                speed_distribution="uniform")


def test_lognormal_speed_distribution_has_requested_median() -> None:
    speeds, _ = draw_cell_motion_params(
        200_000, 5.0, 0.25, 6.0, 2.0, np.random.default_rng(0),
        speed_distribution="lognormal", speed_lognorm_sigma=0.75)
    assert abs(np.median(speeds) - 5.0) < 0.05
    assert np.all(speeds > 0)


def test_spearman_matches_classic_formula_without_ties() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(2, 12))
        x = rng.permutation(n)
        y = rng.permutation(n)
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        classic = 1.0 - 6.0 * np.sum((rx - ry) ** 2) / (n * (n ** 2 - 1))
        assert abs(_spearman(x, y) - classic) < 1e-12


def test_spearman_handles_ties_with_average_ranks() -> None:
    from scipy.stats import spearmanr
    x = [0, 1, 2, 3, 4]
    y = [0, 1, 5, 5, 5]          # three tied penalty ranks
    assert abs(_spearman(x, y) - spearmanr(x, y).statistic) < 1e-12
    # all-tied second vector is undefined, not an arbitrary value
    assert np.isnan(_spearman(x, [7, 7, 7, 7, 7]))
    assert np.isnan(_spearman([1], [2]))
