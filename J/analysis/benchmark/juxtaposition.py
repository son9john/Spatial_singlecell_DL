"""Bivariate association of finite differences on a fixed tissue graph."""

import numpy as np
from scipy import sparse
from scipy.ndimage import sobel
from scipy.spatial import cKDTree

from juxtaposed import juxtaposed_value


def standardize(values):
    """Spot-wise population z scores over a single section."""
    values = np.asarray(values, dtype=float)
    if values.ndim not in (1, 2) or not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected finite, nonempty spot values.')
    scale = values.std(axis=0)
    if np.any(scale == 0):
        raise ValueError('Constant features have undefined spot z scores.')
    return (values - values.mean(axis=0)) / scale


def visium_edges(positions):
    """Undirected first-ring pairs on Visium's doubled-column hex lattice."""
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] != 2 or not np.isfinite(positions).all():
        raise ValueError('Expected finite row/column coordinates.')
    if not np.equal(positions, np.floor(positions)).all():
        raise ValueError('Visium coordinates must be integers.')
    index = {tuple(rc): i for i, rc in enumerate(positions.astype(int))}
    if len(index) != len(positions):
        raise ValueError('Visium coordinates must be unique.')
    edges = [
        (i, j) for (r, c), i in index.items() for dr, dc in ((0, 2), (1, -1), (1, 1))
        if (j := index.get((r + dr, c + dc))) is not None
    ]
    return np.asarray(edges, dtype=int).reshape(-1, 2)


def spatial_weights(n, edges):
    """Row-standardized binary first-ring graph with empty isolate rows."""
    i, j = np.asarray(edges).T
    row, col = np.r_[i, j], np.r_[j, i]
    adjacency = sparse.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.divide(1, degree, where=degree > 0, out=np.zeros_like(degree))
    return adjacency.multiply(inverse[:, None]).tocsr()


def edge_differences(x, y, edges):
    i, j = np.asarray(edges).T
    return x[j] - x[i], y[j] - y[i]


def lee_l(x, y, weights):
    """Lee's L from jointly lagged, spot-centered vectors and row-sum correction."""
    x = x - x.mean()
    y = y - y.mean()
    mass = np.sum(np.asarray(weights.sum(axis=1)).ravel()**2)
    denominator = np.linalg.norm(x) * np.linalg.norm(y) * mass
    return len(x) * np.dot(weights @ x, weights @ y) / denominator if denominator else np.nan


def legacy_bsc(x, y, weights):
    """Latest notebook all-gene BSC, equivalent to its lagged Pearson factorization."""
    xs, ys = weights @ x, weights @ y
    xs, ys = xs - xs.mean(), ys - ys.mean()
    norm = np.linalg.norm(x - x.mean()) * np.linalg.norm(y - y.mean())
    return np.dot(xs, ys) / norm if norm and np.linalg.norm(xs) and np.linalg.norm(ys) else np.nan


def legacy_sobel(x_map, y_map):
    """Literal unsigned contrast surface and Sobel magnitude, notebook cells 194/199."""
    contrast = np.abs(x_map - y_map)
    return np.hypot(sobel(contrast, axis=0), sobel(contrast, axis=1))


def edge_neighborhoods(positions, edges, radius):
    """Neighbors of edge midpoints, including each midpoint itself."""
    midpoints = np.asarray(positions, dtype=float)[edges].mean(axis=1)
    return cKDTree(midpoints).query_ball_point(midpoints, radius, return_sorted=True)


def local_profiles(dx, dy, neighborhoods):
    """Pool signed cross-increment, energy and unsigned contrast over each neighborhood."""
    joint_energy = dx**2 + dy**2
    energy = np.fromiter((joint_energy[ix].sum() for ix in neighborhoods), dtype=float)
    opposition_edge = -2 * dx * dy
    q = np.fromiter((opposition_edge[ix].mean() for ix in neighborhoods), dtype=float)
    unsigned = np.fromiter((np.abs(dx[ix] - dy[ix]).mean() for ix in neighborhoods), dtype=float)
    return {
        'local_j': np.divide(q, energy / np.asarray(list(map(len, neighborhoods))),
                             where=energy > 0, out=np.zeros_like(q)),
        'opposition': q, 'unsigned': unsigned,
    }
