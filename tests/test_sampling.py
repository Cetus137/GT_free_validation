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


def _min_nn_distance(points: np.ndarray) -> float:
    from scipy.spatial import KDTree
    d, _ = KDTree(points).query(points, k=2)
    return float(d[:, 1].min())


def test_fast_path_enforces_min_distance_beyond_tree_rebuild_interval() -> None:
    """N > 50 triggers the periodic KDTree rebuild; the default must still be exact."""
    common = dict(
        num_cells=800, box_size=(100.0, 100.0, 100.0), base_min_distance=4.0,
        density_alpha=0.0, influence_radius=10.0, jitter_fraction=0.35,
        oversample_factor=3.0, max_attempts=20000,
    )
    for seed in range(3):
        points, _ = sample_centroids_poisson_jittered(rng=np.random.default_rng(seed), **common)
        assert _min_nn_distance(points) >= 4.0


def test_legacy_inexact_path_is_still_available() -> None:
    common = dict(
        num_cells=800, box_size=(100.0, 100.0, 100.0), base_min_distance=4.0,
        density_alpha=0.0, influence_radius=10.0, jitter_fraction=0.35,
        oversample_factor=3.0, max_attempts=20000,
    )
    points, _ = sample_centroids_poisson_jittered(
        rng=np.random.default_rng(3), exact_min_distance=False, **common)
    assert points.shape == (800, 3)   # runs; spacing not guaranteed on this path


def test_soft_floor_path_respects_per_cell_radii() -> None:
    rng = np.random.default_rng(0)
    points, radii = sample_centroids_poisson_jittered(
        num_cells=300, box_size=(60.0, 60.0, 60.0), base_min_distance=4.0,
        density_alpha=0.0, influence_radius=10.0, jitter_fraction=0.35,
        oversample_factor=3.0, max_attempts=20000, rng=rng, min_distance_sd=0.5,
    )
    assert points.shape == (300, 3) and radii.shape == (300,)
    assert np.all(radii >= 2.0)                       # truncated at base/2
    assert radii.std() > 0.0                          # radii actually vary
    # pair threshold is the mean of the two radii
    diffs = points[:, None, :] - points[None, :, :]
    d = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(d, np.inf)
    thresh = 0.5 * (radii[:, None] + radii[None, :])
    assert np.all(d >= thresh - 1e-9)
