from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay, KDTree
from scipy.spatial import QhullError
from scipy.stats import rankdata

# vertex-pair indices for all 6 edges of a tetrahedron
_SIMPLEX_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def border_interior_mask(
    positions: np.ndarray,
    box_size: tuple[float, float, float],
    margin: float,
) -> np.ndarray:
    """Boolean mask — True for cells strictly more than *margin* from every face.

    Parameters
    ----------
    positions : (N, 3)
    box_size  : (Lx, Ly, Lz)
    margin    : exclusion distance from each face (same units as positions)

    Returns
    -------
    (N,) bool — True = interior cell, False = exclude
    """
    box = np.asarray(box_size, dtype=float)
    return np.all((positions > margin) & (positions < box - margin), axis=1)


def compute_neighbour_ranks(positions: np.ndarray, k: int) -> np.ndarray:
    """For each cell, return indices of its k nearest neighbours sorted by distance.

    Returns shape (n, k).
    """
    ranks, _ = compute_neighbour_ranks_and_distances(positions, k)
    return ranks


def compute_neighbour_ranks_and_distances(
    positions: np.ndarray,
    k: int,
    max_distance: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """For each cell, return indices and distances of k nearest neighbours.

    Uses scipy KDTree — O(N log N) vs the previous O(N²) pairwise matrix.
    At N=3324, k=15 this is ~150× faster than np.argsort on the full matrix.

    Parameters
    ----------
    max_distance:
        Hard distance cutoff.  Neighbours beyond this distance are masked out:
        their index is set to ``-1`` (sentinel) and distance to ``np.inf``.
        The arrays remain shape (n, k) so that cells in sparse regions simply
        have fewer valid neighbours — metric functions skip ``-1`` entries.

    Returns
    -------
    ranks : (n, k)  — neighbour cell indices; -1 = beyond max_distance
    dists : (n, k)  — corresponding Euclidean distances; inf = beyond max_distance
    """
    n = positions.shape[0]
    k = min(k, n - 1)
    if k < 1:
        return np.zeros((n, 0), dtype=int), np.zeros((n, 0))
    tree = KDTree(positions)
    # query k+1 because the first result is always the point itself (distance 0)
    dists, ranks = tree.query(positions, k=k + 1, workers=-1)
    ranks = ranks[:, 1:].astype(int)
    dists = dists[:, 1:]
    if max_distance is not None:
        out_of_range = dists > max_distance
        ranks[out_of_range] = -1
        dists[out_of_range] = np.inf
    return ranks, dists


def _delaunay_neighbour_ranks_and_distances(
    positions: np.ndarray,
    max_distance: float | None = None,
    max_degree: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return neighbour indices and distances from the 3-D Delaunay triangulation.

    Two cells are neighbours iff they share a Voronoi face (i.e. are connected
    by a Delaunay edge).  Unlike k-NN, the neighbourhood size is topology-driven
    and varies per cell (~12–20 in dense 3-D packings).

    Parameters
    ----------
    positions : (n, 3)
    max_distance :
        Hard distance cutoff.  Delaunay edges longer than this are masked out
        with -1 / inf sentinels, identical to the kNN convention.
    max_degree :
        Optional cap on the number of Delaunay neighbours retained per cell
        (keeps the closest ``max_degree``).  Pass None to retain all.

    Returns
    -------
    ranks : (n, k_max) int   — neighbour indices; -1 = masked out
    dists : (n, k_max) float — Euclidean distances; inf = masked out
    k_max is the maximum valid-neighbour count across all cells after filtering.
    """
    n = positions.shape[0]
    if n < 4:
        return np.zeros((n, 0), dtype=int), np.zeros((n, 0))

    try:
        tri = Delaunay(positions)
    except QhullError:
        return np.zeros((n, 0), dtype=int), np.zeros((n, 0))

    S = tri.simplices  # (num_simplices, 4)

    # Build all directed edges (both directions for each undirected Delaunay edge)
    fwd_rows = [S[:, a] for a, b in _SIMPLEX_PAIRS]
    fwd_cols = [S[:, b] for a, b in _SIMPLEX_PAIRS]
    row = np.concatenate(fwd_rows + [S[:, b] for a, b in _SIMPLEX_PAIRS])
    col = np.concatenate(fwd_cols + [S[:, a] for a, b in _SIMPLEX_PAIRS])

    # Deduplicate: sort by (row, col) then keep unique pairs
    order = np.lexsort((col, row))
    row = row[order]
    col = col[order]
    edge_key = row.astype(np.int64) * n + col.astype(np.int64)
    unique = np.empty(len(edge_key), dtype=bool)
    unique[0] = True
    unique[1:] = edge_key[1:] != edge_key[:-1]
    row = row[unique]
    col = col[unique]

    # Split into per-vertex neighbour lists (row is already sorted)
    split_at = np.where(np.diff(row))[0] + 1
    unique_rows = row[np.concatenate(([0], split_at))]
    nb_lists = np.split(col, split_at)

    # cells on the convex hull have open Voronoi regions — their Delaunay edges
    # extend toward infinity and are geometrically spurious.  Exclude them so
    # that only interior cells (with fully bounded Voronoi cells) contribute.
    hull_vertices: set[int] = set(tri.convex_hull.ravel().tolist())

    per_vertex: list[tuple[np.ndarray, np.ndarray]] = [(np.empty(0, int), np.empty(0))] * n
    for vertex, neighbours in zip(unique_rows.tolist(), nb_lists):
        if vertex in hull_vertices:
            continue  # leave as empty — hull cell, open Voronoi region
        d = np.linalg.norm(positions[neighbours] - positions[vertex], axis=1)
        order_d = np.argsort(d)
        neighbours = neighbours[order_d]
        d = d[order_d]
        if max_degree is not None:
            neighbours = neighbours[:max_degree]
            d = d[:max_degree]
        if max_distance is not None:
            mask = d <= max_distance
            neighbours = neighbours[mask]
            d = d[mask]
        per_vertex[vertex] = (neighbours, d)

    k_max = max((len(nb) for nb, _ in per_vertex), default=0)
    if k_max == 0:
        return np.zeros((n, 0), dtype=int), np.zeros((n, 0))

    ranks_out = np.full((n, k_max), -1, dtype=int)
    dists_out = np.full((n, k_max), np.inf)
    for i, (nb, d) in enumerate(per_vertex):
        ranks_out[i, :len(nb)] = nb
        dists_out[i, :len(nb)] = d
    return ranks_out, dists_out


def _spearman(x, y) -> float:
    """Spearman rank correlation of two equal-length sequences.

    Ties receive average ranks (so the result is the Pearson correlation of the
    rank vectors, which is the correct Spearman definition in the presence of
    ties).  Without ties this equals the classic ``1 - 6 Σd² / (n(n²-1))``.
    Returns NaN for fewer than 2 entries or when either rank vector is constant.
    """
    n = len(x)
    if n < 2:
        return float("nan")
    rx = rankdata(x) - (n + 1) / 2.0
    ry = rankdata(y) - (n + 1) / 2.0
    denom = float(np.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def neighbour_identity_overlap(
    ranks_t0: np.ndarray,
    ranks_t1: np.ndarray,
) -> np.ndarray:
    """Per-cell fraction of valid k neighbours retained between timepoints.

    Handles the ``-1`` sentinel produced when ``max_neighbour_distance`` is set:
    sentinel slots are excluded from both numerator and denominator so cells in
    sparse regions are not penalised for having fewer neighbours.

    Returns shape (n,), values in [0, 1].  NaN where no valid neighbours at t0.
    """
    n, k = ranks_t0.shape
    if k == 0:
        return np.full(n, np.nan)
    valid_t0 = ranks_t0 != -1                                # (N, k)
    valid_t1 = ranks_t1 != -1                                # (N, k)
    k_eff    = valid_t0.sum(axis=1).astype(float)            # (N,) valid neighbours at t0
    # matches[i,j0,j1]: ranks_t0[i,j0]==ranks_t1[i,j1], only between valid slots
    matches = (ranks_t0[:, :, None] == ranks_t1[:, None, :]) \
              & valid_t0[:, :, None] & valid_t1[:, None, :]  # (N, k, k)
    shared  = matches.any(axis=2).sum(axis=1).astype(float)  # (N,)
    return np.divide(shared, k_eff, out=np.full(n, np.nan), where=k_eff > 0)


def weighted_neighbour_identity_overlap(
    ranks_t0: np.ndarray,
    dists_t0: np.ndarray,
    ranks_t1: np.ndarray,
) -> np.ndarray:
    """Distance-weighted identity overlap: retained neighbours weighted by 1/distance.

    Losing or gaining a close neighbour (small distance, large weight) is penalised
    more heavily than losing or gaining a distant neighbour at the edge of the k set.
    Weights are derived from distances at t=0 so the reference frame is fixed.

    Fully vectorised using (N, k, k) broadcast comparison — no Python loops.

    Parameters
    ----------
    ranks_t0 : (n, k) — neighbour indices at t=0, sorted by distance
    dists_t0 : (n, k) — corresponding distances at t=0
    ranks_t1 : (n, k) — neighbour indices at t=1

    Returns
    -------
    (n,) weighted overlap in [0, 1].
    """
    n, k = ranks_t0.shape
    if k == 0:
        return np.full(n, np.nan)
    valid_t0 = ranks_t0 != -1                                     # (N, k)
    valid_t1 = ranks_t1 != -1                                     # (N, k)
    # weights only over valid t0 neighbours; inf distances → weight 0 after clip
    w = 1.0 / np.clip(dists_t0, 1e-9, None)                      # (N, k)
    w = w * valid_t0                                               # zero out sentinels
    row_sum = w.sum(axis=1, keepdims=True)
    w = np.divide(w, row_sum, out=np.zeros_like(w), where=row_sum > 0)   # normalise
    matches = (ranks_t0[:, :, None] == ranks_t1[:, None, :]) \
              & valid_t0[:, :, None] & valid_t1[:, None, :]       # (N, k, k)
    in_t1   = matches.any(axis=2)                                  # (N, k)
    result  = (w * in_t1).sum(axis=1)
    # cells with no valid t0 neighbours → NaN
    return np.where(valid_t0.any(axis=1), result, np.nan)


def neighbour_rank_correlation(
    ranks_t0: np.ndarray,
    ranks_t1: np.ndarray,
) -> np.ndarray:
    """Per-cell Spearman correlation of neighbour rank order between timepoints.

    Computed only over neighbours shared at both timepoints.
    Uses the (N, k, k) match matrix to find shared neighbour positions without
    Python set operations.
    Returns shape (n,), NaN where fewer than 2 shared neighbours exist.
    """
    n, k = ranks_t0.shape
    correlations = np.full(n, np.nan)
    if k == 0:
        return correlations
    valid_t0 = ranks_t0 != -1                                     # (N, k)
    valid_t1 = ranks_t1 != -1                                     # (N, k)
    matches = (ranks_t0[:, :, None] == ranks_t1[:, None, :]) \
              & valid_t0[:, :, None] & valid_t1[:, None, :]       # (N, k, k)
    for i in range(n):
        j0s, j1s = np.where(matches[i])
        if len(j0s) < 2:
            continue
        correlations[i] = _spearman(j0s, j1s)
    return correlations


def _frame_neighbour_map(
    positions: np.ndarray,
    assigned_ids: np.ndarray,
    k: int,
    max_distance: float | None = None,
) -> tuple[dict[int, list[int]], dict[int, list[float]]]:
    """Return neighbour IDs and distances for each cell, sorted by distance.

    Neighbours beyond ``max_distance`` are excluded from each cell's list,
    so lists may be shorter than k in sparse regions.

    Returns
    -------
    nb_ids  : {assigned_id: [neighbour_id, ...]}   — up to k nearest within cutoff
    nb_dists: {assigned_id: [distance, ...]}        — corresponding distances
    """
    n = positions.shape[0]
    k_use = min(k, n - 1)
    if k_use < 1:
        return {}, {}
    tree = KDTree(positions)
    dists_kd, ranked = tree.query(positions, k=k_use + 1, workers=-1)
    ranked   = ranked[:, 1:]    # drop self
    dists_kd = dists_kd[:, 1:]
    nb_ids: dict[int, list[int]] = {}
    nb_dists: dict[int, list[float]] = {}
    for i in range(n):
        cid = int(assigned_ids[i])
        if max_distance is not None:
            valid = dists_kd[i] <= max_distance
            nb_ids[cid]   = [int(assigned_ids[ranked[i, r]]) for r in range(k_use) if valid[r]]
            nb_dists[cid] = [float(dists_kd[i, r])           for r in range(k_use) if valid[r]]
        else:
            nb_ids[cid]   = [int(assigned_ids[ranked[i, r]]) for r in range(k_use)]
            nb_dists[cid] = [float(dists_kd[i, r])           for r in range(k_use)]
    return nb_ids, nb_dists


def _frame_neighbour_map_delaunay(
    positions: np.ndarray,
    assigned_ids: np.ndarray,
    max_distance: float | None = None,
    max_degree: int | None = None,
) -> tuple[dict[int, list[int]], dict[int, list[float]]]:
    """Delaunay-based equivalent of ``_frame_neighbour_map``.

    Returns neighbour IDs and distances for each cell using Voronoi adjacency
    instead of k-NN.  Lists may be shorter than the maximum degree when
    ``max_distance`` excludes far neighbours.

    Returns
    -------
    nb_ids  : {assigned_id: [neighbour_id, ...]}  — sorted by distance
    nb_dists: {assigned_id: [distance, ...]}
    """
    n = positions.shape[0]
    if n < 4:
        return {}, {}

    try:
        tri = Delaunay(positions)
    except QhullError:
        return {}, {}

    S = tri.simplices

    fwd_rows = [S[:, a] for a, b in _SIMPLEX_PAIRS]
    fwd_cols = [S[:, b] for a, b in _SIMPLEX_PAIRS]
    row = np.concatenate(fwd_rows + [S[:, b] for a, b in _SIMPLEX_PAIRS])
    col = np.concatenate(fwd_cols + [S[:, a] for a, b in _SIMPLEX_PAIRS])

    order = np.lexsort((col, row))
    row = row[order]
    col = col[order]
    edge_key = row.astype(np.int64) * n + col.astype(np.int64)
    unique = np.empty(len(edge_key), dtype=bool)
    unique[0] = True
    unique[1:] = edge_key[1:] != edge_key[:-1]
    row = row[unique]
    col = col[unique]

    split_at = np.where(np.diff(row))[0] + 1
    unique_rows = row[np.concatenate(([0], split_at))]
    nb_lists = np.split(col, split_at)

    # cells on the convex hull have open Voronoi regions — exclude them
    hull_vertices: set[int] = set(tri.convex_hull.ravel().tolist())

    nb_ids: dict[int, list[int]] = {}
    nb_dists: dict[int, list[float]] = {}

    for vertex, neighbours in zip(unique_rows.tolist(), nb_lists):
        if vertex in hull_vertices:
            continue  # open Voronoi region — skip
        cid = int(assigned_ids[vertex])
        d = np.linalg.norm(positions[neighbours] - positions[vertex], axis=1)
        order_d = np.argsort(d)
        neighbours = neighbours[order_d]
        d = d[order_d]
        if max_degree is not None:
            neighbours = neighbours[:max_degree]
            d = d[:max_degree]
        if max_distance is not None:
            mask = d <= max_distance
            neighbours = neighbours[mask]
            d = d[mask]
        nb_ids[cid]   = [int(assigned_ids[j]) for j in neighbours]
        nb_dists[cid] = d.tolist()

    return nb_ids, nb_dists


def compute_neighbour_continuity_observed(
    observed_frames: list[dict],
    k: int = 5,
    box_size: tuple[float, float, float] | None = None,
    border_margin: float = 0.0,
    max_neighbour_distance: float | None = None,
    penalise_absence: bool = False,
    neighbourhood_mode: str = "knn",
    max_delaunay_degree: int | None = None,
    displacement_sigma: float = 5.0,
    extra_metrics: bool = False,
) -> dict[str, object]:
    """Compute neighbour continuity on observed (error-corrupted) frames.

    Uses ``assigned_ids`` to track cell identity, matching the tracker's perspective.

    Parameters
    ----------
    observed_frames:
        List of T frame dicts from ``apply_tracking_errors``, each containing
        ``positions`` (N_obs, 3) and ``assigned_ids`` (N_obs,).
    k:
        Number of nearest neighbours to consider.  Ignored when
        ``neighbourhood_mode="delaunay"``.
    box_size:
        (Lx, Ly, Lz) volume dimensions. Required when border_margin > 0.
    border_margin:
        Exclude cells within this distance of any face. Pass 0 (default) to disable.
    max_neighbour_distance:
        Hard distance cutoff for neighbourhood membership. Cells with no valid
        neighbours within this distance contribute NaN. Pass None to use all k.
    penalise_absence:
        If False (default): cells absent from either frame in a consecutive pair
        contribute NaN and are excluded from the mean.  The metric reflects
        neighbourhood quality *conditional on being tracked*.
        If True: cells present at t0 but absent at t1 (i.e. track ends or breaks)
        contribute 0 to ``weighted_identity_overlap``.  The metric reflects
        neighbourhood quality *including track continuity*, so broken tracks
        and missed detections pull the mean down.  ``rank_correlation`` remains
        NaN for absent cells — rank order is undefined when a cell disappears.
    neighbourhood_mode:
        ``"knn"`` (default) — k nearest neighbours via KDTree.
        ``"delaunay"`` — Voronoi face-sharing neighbours from the 3-D Delaunay
        triangulation; ``k`` is ignored.
    max_delaunay_degree:
        Maximum Delaunay neighbours retained per cell.  Only used when
        ``neighbourhood_mode="delaunay"``.
    displacement_sigma:
        Controls the Gaussian falloff used by ``positional_weighted_identity_overlap``.
        A t0-neighbour j is considered "positionally retained" at t1 if the nearest
        detected cell at t1 is close to j's t0 position.  The retention weight is
        ``exp(−d² / 2σ²)`` where d is that nearest-cell distance and σ is this
        parameter (same units as positions).  Natural default: the expected per-frame
        displacement (~2–3 µm for slow-moving tissue, 5 µm is a conservative choice
        that grades off near zero by ~15 µm).  A missed cell contributes ~0 because
        the nearest t1 cell is many µm away; an ID-switched cell contributes ~1
        because a cell still exists at approximately j's position.
    extra_metrics:
        If True, also return three neighbourhood metrics that split WIO's loss
        by the fate of each lost neighbour, plus its mirror image (off by default):

        ``wio_vanished``   — sum of WIO weights of t0 neighbours that are lost AND
                             whose ID no longer exists anywhere at t1 (misses,
                             merges, true track ends).
        ``wio_displaced``  — sum of WIO weights of t0 neighbours that are lost but
                             whose ID still exists elsewhere at t1 (switches,
                             displacement by FPs/splits, real rearrangement).
                             For every cell: WIO + wio_vanished + wio_displaced = 1.
        ``newcomer_fraction`` — viewed from t1: inverse-distance weighted (t1
                             distances) fraction of the cell's t1 neighbours whose
                             ID did not exist at t0 (FPs, splits, re-detections).

    Returns
    -------
    dict with keys:
        ``weighted_identity_overlap``    (T-1, n_ids) — inverse-distance weighted
        ``rank_correlation``             (T-1, n_ids) — Spearman rho over neighbours
                                             shared at both timepoints
        ``penalised_rank_correlation``   (T-1, n_ids) — PRC: Spearman rho over ALL t0
                                             neighbours, where a neighbour absent at
                                             t1 receives a penalty rank proportional
                                             to how close it was at t0 (ties get
                                             average ranks)
        ``positional_weighted_identity_overlap`` (T-1, n_ids) — PWIO: like WIO but
                                             neighbours matched by position at t1
                                             rather than by ID; immune to switches,
                                             sensitive to misses.  WIO − PWIO isolates
                                             the identity-consistency component.
        ``cell_ids``                         (n_ids,)
        ``track_survival``                   (T-1,) — fraction of eligible t0 cells
                                             present at t1
    """
    if neighbourhood_mode not in ("knn", "delaunay"):
        raise ValueError(
            f"Unknown neighbourhood_mode {neighbourhood_mode!r}. "
            "Expected 'knn' or 'delaunay'."
        )
    if border_margin > 0.0 and box_size is None:
        raise ValueError("box_size must be provided when border_margin > 0")

    t = len(observed_frames)

    all_ids: list[int] = sorted({
        int(aid)
        for frame in observed_frames
        for aid in frame["assigned_ids"]
    })
    id_to_col = {cid: i for i, cid in enumerate(all_ids)}
    n_ids = len(all_ids)

    weighted_identity    = np.full((t - 1, n_ids), np.nan)
    positional_wio       = np.full((t - 1, n_ids), np.nan)
    correlation          = np.full((t - 1, n_ids), np.nan)
    prc_correlation      = np.full((t - 1, n_ids), np.nan)
    track_survival       = np.full(t - 1, np.nan)
    two_sigma_sq         = 2.0 * displacement_sigma ** 2
    if extra_metrics:
        wio_vanished      = np.full((t - 1, n_ids), np.nan)
        wio_displaced     = np.full((t - 1, n_ids), np.nan)
        newcomer_fraction = np.full((t - 1, n_ids), np.nan)

    for ti in range(t - 1):
        f0 = observed_frames[ti]
        f1 = observed_frames[ti + 1]

        ids0 = f0["assigned_ids"]
        ids1 = f1["assigned_ids"]

        # position lookup for t0 cells — used by topological RC successor search
        id_to_pos0: dict[int, np.ndarray] = {
            int(aid): f0["positions"][i] for i, aid in enumerate(ids0)
        }

        set0 = set(ids0.tolist())
        set1 = set(ids1.tolist())

        # build per-cell interior sets for this pair
        if border_margin > 0.0:
            interior0 = set(
                int(ids0[i])
                for i, ok in enumerate(border_interior_mask(f0["positions"], box_size, border_margin))
                if ok
            )
            interior1 = set(
                int(ids1[i])
                for i, ok in enumerate(border_interior_mask(f1["positions"], box_size, border_margin))
                if ok
            )
            interior_both = interior0 & interior1
            # cells eligible for absence penalty: interior at t0 AND absent from t1.
            # (Not ``interior0 - interior1``: that would also penalise cells that
            # are still tracked at t1 but drifted into the border band.)
            absent_eligible = interior0 - set1 if penalise_absence else set()
        else:
            interior_both   = None
            absent_eligible = (set0 - set1) if penalise_absence else set()

        # penalise cells present at t0 but absent at t1
        for cid in absent_eligible:
            weighted_identity[ti, id_to_col[cid]] = 0.0
            # rank_correlation stays NaN — undefined for absent cells

        common = set0 & set1

        # survival: fraction of eligible t0 cells present at t1
        if border_margin > 0.0:
            n_eligible = len(interior0)
            n_survived = len(interior0 & set1)
        else:
            n_eligible = len(set0)
            n_survived = len(common)
        if n_eligible > 0:
            track_survival[ti] = n_survived / n_eligible

        if not common:
            continue

        # KDTree of t1 positions — used by positional WIO to match t0-neighbour
        # positions to the nearest detected cell at t1, regardless of ID
        tree_t1 = KDTree(f1["positions"])

        if neighbourhood_mode == "knn":
            k_eff = min(k, len(ids0) - 1, len(ids1) - 1)
            if k_eff < 1:
                continue
            nb0, nd0 = _frame_neighbour_map(
                f0["positions"], ids0, k_eff, max_distance=max_neighbour_distance)
            nb1, nd1 = _frame_neighbour_map(
                f1["positions"], ids1, k_eff, max_distance=max_neighbour_distance)
        else:  # delaunay
            nb0, nd0 = _frame_neighbour_map_delaunay(
                f0["positions"], ids0,
                max_distance=max_neighbour_distance,
                max_degree=max_delaunay_degree)
            nb1, nd1 = _frame_neighbour_map_delaunay(
                f1["positions"], ids1,
                max_distance=max_neighbour_distance,
                max_degree=max_delaunay_degree)

        for cid in common:
            if cid not in nb0 or cid not in nb1:
                continue
            if interior_both is not None and cid not in interior_both:
                continue

            neighbours0 = nb0[cid]
            neighbours1 = nb1[cid]
            dists0      = nd0[cid]
            n_valid0    = len(neighbours0)

            if n_valid0 == 0:
                continue

            shared = list(set(neighbours0) & set(neighbours1))

            col = id_to_col[cid]

            # inverse-distance weights from i — shared by WIO and PWIO
            w = np.array([1.0 / max(d, 1e-9) for d in dists0])
            w = w / w.sum()

            # WIO: identity-based overlap
            nb1_set = set(neighbours1)
            weighted_identity[ti, col] = float(
                sum(w[r] for r, nb in enumerate(neighbours0) if nb in nb1_set)
            )

            if extra_metrics:
                # split WIO's loss by the fate of each lost t0 neighbour
                vanished = displaced = 0.0
                for r, nb in enumerate(neighbours0):
                    if nb in nb1_set:
                        continue
                    if nb in set1:
                        displaced += w[r]      # ID still exists at t1, elsewhere
                    else:
                        vanished += w[r]       # ID gone from t1 entirely
                wio_vanished[ti, col]  = float(vanished)
                wio_displaced[ti, col] = float(displaced)
                # newcomer fraction: from t1, weighted share of neighbours new at t1
                if neighbours1:
                    w1 = np.array([1.0 / max(d, 1e-9) for d in nd1[cid]])
                    w1 = w1 / w1.sum()
                    newcomer_fraction[ti, col] = float(
                        sum(w1[r] for r, nb in enumerate(neighbours1) if nb not in set0)
                    )

            # PWIO: position-based overlap — find nearest t1 cell to each t0
            # neighbour's position; retention weight decays as a Gaussian with
            # the neighbour-to-nearest-t1-cell distance
            nb_pos_t0 = np.array([id_to_pos0[j] for j in neighbours0])
            match_dists, _ = tree_t1.query(nb_pos_t0, k=1, workers=-1)
            sigma_j = np.exp(-match_dists ** 2 / two_sigma_sq)
            positional_wio[ti, col] = float(np.dot(w, sigma_j))

            if len(shared) >= 2:
                pos_t0 = [neighbours0.index(s) for s in shared]
                pos_t1 = [neighbours1.index(s) for s in shared]
                correlation[ti, col] = _spearman(pos_t0, pos_t1)

            # penalised RC: absent t0-neighbours receive a distance-proportional
            # penalty rank.  A neighbour that was very close at t0 and then
            # disappeared gets the worst possible rank (strong penalty); one
            # that was near the neighbourhood cutoff gets only a mild penalty
            # (it may simply have drifted outside).  The reference distance is
            # max_neighbour_distance if set, otherwise the furthest t0 neighbour.
            if n_valid0 >= 2:
                nb1_rank  = {j: r for r, j in enumerate(neighbours1)}
                n1        = len(neighbours1)
                ref_dist  = (max_neighbour_distance
                             if max_neighbour_distance is not None
                             else max(dists0))
                ranks_t0  = list(range(n_valid0))
                ranks_t1  = []
                for r, (j, d) in enumerate(zip(neighbours0, dists0)):
                    if j in nb1_rank:
                        ranks_t1.append(nb1_rank[j])
                    else:
                        weight  = max(0.0, 1.0 - d / ref_dist)
                        penalty = n_valid0 + round(weight * (max(n_valid0, n1) - n_valid0))
                        ranks_t1.append(penalty)
                prc_correlation[ti, col] = _spearman(ranks_t0, ranks_t1)

    out = {
        "weighted_identity_overlap":             weighted_identity,
        "positional_weighted_identity_overlap":  positional_wio,
        "rank_correlation":                      correlation,
        "penalised_rank_correlation":            prc_correlation,
        "cell_ids":                              np.array(all_ids),
        "track_survival":                        track_survival,
    }
    if extra_metrics:
        out["wio_vanished"]      = wio_vanished
        out["wio_displaced"]     = wio_displaced
        out["newcomer_fraction"] = newcomer_fraction
    return out


def compute_neighbour_continuity(
    frames: np.ndarray,
    k: int = 5,
    box_size: tuple[float, float, float] | None = None,
    border_margin: float = 0.0,
    max_neighbour_distance: float | None = None,
    neighbourhood_mode: str = "knn",
    max_delaunay_degree: int | None = None,
) -> dict[str, np.ndarray]:
    """Compute neighbour continuity metrics between consecutive frame pairs.

    Parameters
    ----------
    frames:
        Shape (T, N, 3). Must have T >= 2.
    k:
        Number of nearest neighbours to consider.  Ignored when
        ``neighbourhood_mode="delaunay"``.
    box_size:
        (Lx, Ly, Lz) volume dimensions. Required when border_margin > 0.
    border_margin:
        Exclude cells within this distance of any face. Their metric values
        are set to NaN in the output arrays. Pass 0 (default) to disable.
    max_neighbour_distance:
        Hard distance cutoff for neighbourhood membership.  Neighbours further
        than this are excluded; the effective k becomes per-cell variable.
        Useful for sparse or noisy regions where forcing k distant neighbours
        would be physically meaningless.  Pass None (default) to use all k.
    neighbourhood_mode:
        ``"knn"`` (default) — k nearest neighbours via KDTree.
        ``"delaunay"`` — Voronoi face-sharing neighbours from the 3-D Delaunay
        triangulation; neighbourhood size is topology-driven and varies per
        cell (~12–20 in dense packings).
    max_delaunay_degree:
        Maximum number of Delaunay neighbours retained per cell (closest first).
        Only used when ``neighbourhood_mode="delaunay"``.  Pass None to keep all.

    Returns
    -------
    dict with keys:
        ``identity_overlap``          (T-1, N) — fraction of valid neighbours retained; NaN for border/sparse cells
        ``weighted_identity_overlap`` (T-1, N) — inverse-distance weighted overlap
        ``rank_correlation``          (T-1, N) — Spearman rho; NaN where < 2 shared neighbours
        ``ranks``                     (T, N, k_max) — -1 sentinel for out-of-range slots
        ``distances``                 (T, N, k_max) — inf for out-of-range slots
        ``interior_mask``             (T, N) bool — True = interior cell at that timepoint
    """
    if neighbourhood_mode not in ("knn", "delaunay"):
        raise ValueError(
            f"Unknown neighbourhood_mode {neighbourhood_mode!r}. "
            "Expected 'knn' or 'delaunay'."
        )

    t, n, _ = frames.shape

    identity          = np.full((t - 1, n), np.nan)
    weighted_identity = np.full((t - 1, n), np.nan)
    correlation       = np.full((t - 1, n), np.nan)
    interior_mask     = np.ones((t, n), dtype=bool)

    if neighbourhood_mode == "knn":
        k_eff = min(k, n - 1)
        if k_eff < 1:
            return {
                "identity_overlap":          identity,
                "weighted_identity_overlap": weighted_identity,
                "rank_correlation":          correlation,
                "ranks":                     np.zeros((t, n, 0), dtype=int),
                "distances":                 np.zeros((t, n, 0)),
                "interior_mask":             interior_mask,
            }

    # build border masks
    if border_margin > 0.0:
        if box_size is None:
            raise ValueError("box_size must be provided when border_margin > 0")
        for i in range(t):
            interior_mask[i] = border_interior_mask(frames[i], box_size, border_margin)

    ranks_list = []
    dists_list = []
    if neighbourhood_mode == "knn":
        for i in range(t):
            r, d = compute_neighbour_ranks_and_distances(
                frames[i], k_eff, max_distance=max_neighbour_distance
            )
            ranks_list.append(r)
            dists_list.append(d)
        all_ranks = np.stack(ranks_list, axis=0)   # (T, N, k_eff)
        all_dists = np.stack(dists_list, axis=0)
    else:  # delaunay
        for i in range(t):
            r, d = _delaunay_neighbour_ranks_and_distances(
                frames[i],
                max_distance=max_neighbour_distance,
                max_degree=max_delaunay_degree,
            )
            ranks_list.append(r)
            dists_list.append(d)
        # each frame may have a different k_max — pad to the global maximum
        k_max = max((r.shape[1] for r in ranks_list), default=0)
        if k_max == 0:
            return {
                "identity_overlap":          identity,
                "weighted_identity_overlap": weighted_identity,
                "rank_correlation":          correlation,
                "ranks":                     np.zeros((t, n, 0), dtype=int),
                "distances":                 np.zeros((t, n, 0)),
                "interior_mask":             interior_mask,
            }
        padded_ranks = []
        padded_dists = []
        for r, d in zip(ranks_list, dists_list):
            pad = k_max - r.shape[1]
            if pad > 0:
                r = np.pad(r, ((0, 0), (0, pad)), constant_values=-1)
                d = np.pad(d, ((0, 0), (0, pad)), constant_values=np.inf)
            padded_ranks.append(r)
            padded_dists.append(d)
        all_ranks = np.stack(padded_ranks, axis=0)  # (T, N, k_max)
        all_dists = np.stack(padded_dists, axis=0)

    for ti in range(t - 1):
        # cells must be interior in BOTH frames to get a valid metric
        interior = interior_mask[ti] & interior_mask[ti + 1]

        id_vals   = neighbour_identity_overlap(all_ranks[ti], all_ranks[ti + 1])
        wid_vals  = weighted_neighbour_identity_overlap(
            all_ranks[ti], all_dists[ti], all_ranks[ti + 1]
        )
        corr_vals = neighbour_rank_correlation(all_ranks[ti], all_ranks[ti + 1])

        identity[ti]          = np.where(interior, id_vals,   np.nan)
        weighted_identity[ti] = np.where(interior, wid_vals,  np.nan)
        correlation[ti]       = np.where(interior, corr_vals, np.nan)

    return {
        "identity_overlap":          identity,
        "weighted_identity_overlap": weighted_identity,
        "rank_correlation":          correlation,
        "ranks":                     all_ranks,
        "distances":                 all_dists,
        "interior_mask":             interior_mask,
    }
