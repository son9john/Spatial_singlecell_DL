"""Sparse graph extension of J for measured single-cell expression."""
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree


def spatial_graph(xy, neighbors=12, radius=0.060):
    """Symmetric union of distance-limited kNN edges. Coordinates are in mm."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or len(xy) < 2 or xy.shape[1] == 0 or not np.isfinite(xy).all():
        raise ValueError("coordinates must be a finite matrix with at least two rows")
    if not isinstance(neighbors, (int, np.integer)) or neighbors < 1:
        raise ValueError("neighbors must be a positive integer")
    if radius is None:
        radius = np.inf
    if np.isnan(radius) or radius <= 0:
        raise ValueError("radius must be positive")
    distance, index = cKDTree(xy).query(xy, k=min(len(xy), neighbors + 1))
    origin = np.broadcast_to(np.arange(len(xy))[:, None], index.shape)
    use = (origin != index) & (distance <= radius)
    i, j = origin[use], index[use]
    adjacency = sparse.csr_matrix((np.ones(len(i)), (i, j)), shape=(len(xy), len(xy)))
    adjacency = adjacency.maximum(adjacency.T)
    i, j = sparse.triu(adjacency, k=1, format='csr').nonzero()
    return adjacency, i, j


def graph_operators(adjacency):
    """Neighbor mean and lazy random walk, with identity rows at isolates."""
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.divide(1, degree, out=np.zeros_like(degree), where=degree > 0)
    mean = adjacency.multiply(inverse[:, None]).tocsr()
    walk = (sparse.diags(np.where(degree > 0, 0.5, 1.0)) + 0.5 * mean).tocsr()
    return mean, walk, degree


def diffuse(expression, walk, steps):
    """A fixed small polynomial graph filter without an eigendecomposition."""
    if not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError('steps must be a nonnegative integer')
    for _ in range(steps):
        expression = walk @ expression
    return expression


def divide_with_zero(numerator, denominator):
    """Use a neutral zero for windows with no joint variation when ranking edges."""
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def graph_gram(expression, adjacency, degree):
    """G = Zᵀ(D−A)Z. Each undirected edge contributes exactly once."""
    expression = np.asarray(expression, dtype=adjacency.dtype, order='C')
    laplacian_expression = degree[:, None] * expression - adjacency @ expression
    gram = expression.T @ laplacian_expression
    return (gram + gram.T) * 0.5


def juxtaposed_matrix(gram, ridge=0.0):
    """All-pair J with a pair-specific global energy ridge.

    Here the queried set is the full graph, so the ridge adds the same fraction
    of joint energy to every pair and leaves their ranking unchanged. The
    local/selected-edge estimators use the global mean as a fixed reference.
    """
    energy = np.maximum(gram.diagonal(), 0)
    joint = energy[:, None] + energy
    result = divide_with_zero(-2 * gram, (1 + ridge) * joint)
    return np.clip(result, -1, 1)


def _pool(value, i, j, n):
    node_total = (np.bincount(i, weights=value, minlength=n) +
                  np.bincount(j, weights=value, minlength=n))
    return node_total[i] + node_total[j]


def _moments(expression, i, j, weights):
    x, y = expression.T
    dx, dy = x[j] - x[i], y[j] - y[i]
    n = len(expression)
    ax, ay = _pool(weights * dx**2, i, j, n), _pool(weights * dy**2, i, j, n)
    cross = -2 * _pool(weights * dx * dy, i, j, n)
    mass = _pool(weights, i, j, n)
    reference = np.dot(weights, dx**2 + dy**2) / weights.sum() if len(weights) else 0.0
    return dx, dy, ax, ay, cross, mass, reference


def local_moments(expression, i, j, weights):
    """Shared weighted local cross-moment and support used by the package."""
    _, _, ax, ay, cross, mass, reference = _moments(expression, i, j, weights)
    return cross, ax, ay, mass, reference


def edge_scores(expression, mean, i, j, ridge=1.0):
    """Rank edges with shared input features and graph across all comparators."""
    n = len(expression)
    x, y = expression.T
    dx, dy, ax, ay, cross, support, global_mean = _moments(expression, i, j, np.ones(len(i)))
    joint = ax + ay
    x_lag, y_lag = mean @ x, mean @ y
    mxx, myy, mxy = mean @ (x * x), mean @ (y * y), mean @ (x * y)
    cosine_denom = np.sqrt(mxx * myy)
    pearson_denom = np.sqrt(np.maximum(mxx - x_lag*x_lag, 0) *
                             np.maximum(myy - y_lag*y_lag, 0))

    def at_edges(values):
        return (values[i] + values[j]) * 0.5

    # Power-mean-tempered J: p=1 is J, p=0 is signed codispersion. Lower p
    # preserves opposition when one marker's gradient is much weaker.
    root_energy = np.sqrt(np.maximum(ax, 0)) + np.sqrt(np.maximum(ay, 0))
    quarter_energy = (np.maximum(ax, 0)**0.25 + np.maximum(ay, 0)**0.25) / 2

    return {
        'j': divide_with_zero(cross, joint),
        'j_tempered': divide_with_zero(2 * cross, root_energy**2),
        'j_tempered_025': divide_with_zero(cross, 2 * quarter_energy**4),
        'j_regularized': divide_with_zero(cross, joint + ridge * support * global_mean),
        'negative_codispersion': divide_with_zero(0.5 * cross, np.sqrt(ax * ay)),
        'signed_gradient': divide_with_zero(cross, support),
        'unsigned_gradient': divide_with_zero(_pool(np.abs(dx - dy), i, j, n), support),
        'negative_local_lee': at_edges(-x_lag * y_lag),
        'negative_local_moran': at_edges(-0.5 * (x * y_lag + y * x_lag)),
        'negative_local_cosine': at_edges(divide_with_zero(-mxy, cosine_denom)),
        'negative_local_pearson': at_edges(divide_with_zero(x_lag*y_lag - mxy, pearson_denom)),
        'joint_gradient': divide_with_zero(joint, support),
    }
