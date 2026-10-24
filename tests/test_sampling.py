import numpy as np

from tracking_validation.sampling import sample_centroids_poisson_jittered


def test_sampling_returns_expected_shape_and_bounds() -> None:
    rng = np.random.default_rng(1)
    points, radii = sample_centroids_poisson_jittered(
        num_cells=40,
        box_size=(40.0, 40.0, 40.0),
        base_min_distance=3.0,
        density_alpha=0.1,
        influence_radius=8.0,
        jitter_fraction=0.3,
        oversample_factor=3.0,
        max_attempts=20000,
        rng=rng,
    )
    assert points.shape == (40, 3)
    assert radii.shape == (40,)
    assert np.all(points >= 0.0)
    assert np.all(points <= 40.0)


def test_sampling_has_reasonable_min_pairwise_distance() -> None:
    rng = np.random.default_rng(2)
    points, _ = sample_centroids_poisson_jittered(
        num_cells=30,
        box_size=(30.0, 30.0, 30.0),
        base_min_distance=2.5,
        density_alpha=0.0,
        influence_radius=6.0,
        jitter_fraction=0.4,
        oversample_factor=2.0,
        max_attempts=20000,
        rng=rng,
    )

    min_dist = float("inf")
    for i in range(points.shape[0]):
        for j in range(i + 1, points.shape[0]):
            d = float(np.linalg.norm(points[i] - points[j]))
            min_dist = min(min_dist, d)
    assert min_dist >= 2.5
