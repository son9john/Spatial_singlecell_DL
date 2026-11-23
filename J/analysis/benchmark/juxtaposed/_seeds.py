"""Seed-centred distance weighting for the juxtaposed value.

Core spots are discovered from the highest spot-level J scores on a fixed
graph. Their count is chosen by the first bend of the sorted score curve, not
by an all-or-none threshold. Every spot and edge then receives a smooth weight
based on its distance to the nearest seed, and the weighted J is computed with
the same ``juxtaposed_value`` kernel.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from ._graph import _moments, diffuse, divide_with_zero
from ._value import energy_scale, juxtaposed_value


def first_elbow(values, min_k=2, max_k=None):
    """Return the number of top observations before the first clear bend.

    The sorted scores are normalised to [0, 1] on both axes. The distance of
    each point to the chord joining (0, 1) and (1, 0) is smoothed, and the
    first point reaching 80% of the largest smoothed distance is taken as the
    first elbow. This avoids the all-or-none endpoint elbow.
    """
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return 0
    if max_k is None:
        max_k = max(2, n // 2)
    max_k = int(np.clip(max_k, 2, n))
    min_k = int(np.clip(min_k, 2, max_k))
    if n <= min_k:
        return n
    ordered = np.sort(values)[::-1][:max_k]
    spread = ordered[0] - ordered[-1]
    if not np.isfinite(spread) or spread <= 0:
        return min_k
    y = (ordered - ordered[-1]) / spread
    x = np.arange(len(ordered), dtype=float) / (len(ordered) - 1)
    distance = np.maximum(0.0, 1.0 - x - y) / np.sqrt(2.0)
    window = max(1, len(distance) // 20)
    smooth = np.convolve(distance, np.ones(window) / window, mode='same')
    peak = smooth.max()
    if not np.isfinite(peak) or peak <= 0:
        return min_k
    index = int(np.argmax(smooth >= 0.8 * peak))
    return int(np.clip(index + 1, min_k, max_k))


def seed_indices(values, min_k=2, max_k=None):
    """Return indices of the top-scoring core spots and their count."""
    values = np.asarray(values, dtype=float).ravel()
    k = first_elbow(values, min_k=min_k, max_k=max_k)
    score = np.nan_to_num(values, nan=-np.inf)
    order = np.argsort(-score, kind='stable')
    return order[:k], k


def node_j(expression, i, j, power=1.0):
    """Mean edge J incident to each spot, using the same power mean as J."""
    expression = np.asarray(expression, dtype=float)
    x, y = expression.T
    dx, dy = x[j] - x[i], y[j] - y[i]
    scale = energy_scale(dx**2, dy**2, power)
    edge = np.divide(-dx * dy, scale, out=np.zeros_like(dx, dtype=float), where=scale > 0)
    edge = np.clip(np.asarray(edge, dtype=float), -1, 1)
    n = len(expression)
    total = (np.bincount(i, weights=edge, minlength=n) +
             np.bincount(j, weights=edge, minlength=n)).astype(float)
    count = (np.bincount(i, minlength=n) + np.bincount(j, minlength=n)).astype(float)
    return np.divide(total, count, out=np.zeros(n, dtype=float), where=count > 0)


def seed_weights(xy, seeds, scale=None, kernel='gaussian'):
    """Spot weights and distances to the nearest seed."""
    xy = np.asarray(xy, dtype=float)
    seeds = np.asarray(seeds, dtype=int)
    if len(seeds) == 0:
        raise ValueError('at least one seed is required')
    distance = cKDTree(xy[seeds]).query(xy)[0]
    if scale is None:
        positive = distance[distance > 0]
        scale = float(np.median(positive)) if positive.size else 1.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
    if kernel == 'gaussian':
        weights = np.exp(-0.5 * (distance / scale)**2)
    elif kernel == 'inverse':
        weights = 1.0 / (1.0 + distance / scale)
    else:
        raise ValueError("kernel must be 'gaussian' or 'inverse'")
    return weights, distance, scale


def edge_weight(weights, i, j):
    """Average the endpoint weights so each undirected edge gets one weight."""
    weights = np.asarray(weights, dtype=float)
    return (weights[i] + weights[j]) * 0.5


def filter_seeds_by_cluster(xy, seeds, i, j, min_size=3, radius=None):
    """Keep only seeds inside a connected component of at least min_size seeds.

    The subgraph is induced by the supplied graph edges among the candidate
    seeds. An optional radius removes distant seed-seed links before the
    connected-component step.
    """
    xy = np.asarray(xy, dtype=float)
    seeds = np.asarray(seeds, dtype=int)
    if len(seeds) == 0 or min_size is None or min_size <= 1:
        return seeds
    n = len(xy)
    keep = np.zeros(n, dtype=bool)
    keep[seeds] = True
    use = keep[i] & keep[j]
    if radius is not None:
        use &= np.linalg.norm(xy[i] - xy[j], axis=1) <= radius
    rows = np.r_[i[use], j[use]]
    cols = np.r_[j[use], i[use]]
    graph = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    size = np.bincount(labels[seeds], minlength=labels.max() + 1)
    valid = np.where(size >= min_size)[0]
    remain = seeds[np.isin(labels[seeds], valid)]
    return remain if len(remain) else seeds[:1]


def panel_seed_map(expression, i, j, pairs=None, walk=None, steps=0,
                   power=1.0, max_pairs=None):
    """Average spot-level J over a gene panel and return its first-elbow seeds.

    The map-level seeds do not depend on any single queried pair, so they can
    be fixed once and reused for null and real pairs without pair-selection
    bias.
    """
    expression = np.asarray(expression, dtype=float)
    n_genes = expression.shape[1]
    if pairs is None:
        pairs = [(a, b) for a in range(n_genes) for b in range(a + 1, n_genes)]
    if max_pairs is not None:
        pairs = list(pairs)[:max_pairs]
    if len(pairs) == 0:
        raise ValueError('at least one gene pair is required')
    z = diffuse(expression, walk, steps) if walk is not None and steps else expression
    total = np.zeros(len(expression), dtype=float)
    for a, b in pairs:
        total += node_j(z[:, [a, b]], i, j, power=power)
    mean_map = total / len(pairs)
    seeds, k = seed_indices(mean_map)
    return mean_map, seeds, k


def _prepare(expression, i, j, xy, walk, steps):
    z = diffuse(expression, walk, steps) if walk is not None and steps else expression
    x, y = z.T
    return z, x[j] - x[i], y[j] - y[i]


def seed_global(expression, i, j, xy, walk=None, steps=0, power=1.0,
                min_k=2, max_k=None, scale=None, kernel='gaussian',
                seeds=None, cluster_min_size=None, cluster_radius=None):
    """Global J after seed-distance weighting, plus the seed diagnostics."""
    xy = np.asarray(xy, dtype=float)
    z, dx, dy = _prepare(expression, i, j, xy, walk, steps)
    scores = node_j(z, i, j, power=power)
    if seeds is None:
        raw, k_raw = seed_indices(scores, min_k=min_k, max_k=max_k)
        chosen = filter_seeds_by_cluster(xy, raw, i, j,
                                         min_size=cluster_min_size,
                                         radius=cluster_radius)
    else:
        raw = np.asarray(seeds, dtype=int)
        k_raw = len(raw)
        chosen = raw
    k = len(chosen)
    weights, distance, scale = seed_weights(xy, chosen, scale=scale, kernel=kernel)
    value = juxtaposed_value(dx, dy, weights=edge_weight(weights, i, j), power=power)
    info = dict(k=k, seeds=chosen, k_raw=k_raw, seeds_raw=raw,
                filtered=k != k_raw, scale=scale, distance=distance,
                weights=weights, node_scores=scores)
    return value, info


def seed_edge_scores(expression, i, j, xy, walk=None, steps=0, power=1.0,
                     ridge=1.0, min_k=2, max_k=None, scale=None, kernel='gaussian',
                     seeds=None, cluster_min_size=None, cluster_radius=None):
    """Local J on every edge after seed-distance weighting."""
    xy = np.asarray(xy, dtype=float)
    z, dx, dy = _prepare(expression, i, j, xy, walk, steps)
    scores = node_j(z, i, j, power=power)
    if seeds is None:
        raw, k_raw = seed_indices(scores, min_k=min_k, max_k=max_k)
        chosen = filter_seeds_by_cluster(xy, raw, i, j,
                                         min_size=cluster_min_size,
                                         radius=cluster_radius)
    else:
        raw = np.asarray(seeds, dtype=int)
        k_raw = len(raw)
        chosen = raw
    k = len(chosen)
    weights, distance, scale = seed_weights(xy, chosen, scale=scale, kernel=kernel)
    edge = edge_weight(weights, i, j)
    _, _, ax, ay, cross, mass, _ = _moments(z, i, j, edge)
    joint = ax + ay
    reference = np.dot(edge, dx**2 + dy**2) / edge.sum()
    result = {
        'j_seed': divide_with_zero(cross, joint),
        'j_seed_regularized': divide_with_zero(cross, joint + ridge * mass * reference),
    }
    info = dict(k=k, seeds=chosen, k_raw=k_raw, seeds_raw=raw,
                filtered=k != k_raw, scale=scale, distance=distance,
                weights=weights, node_scores=scores)
    return result, info
