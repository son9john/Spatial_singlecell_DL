"""Spatial graphs, Laplacians and the smoothing operator shared by every statistic."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

__all__ = ["spatial_graph", "visium_graph", "laplacian", "normalized_laplacian", "smoother",
           "lazy_walk", "smooth", "check_graph"]

LAPLACIANS = ("combinatorial", "normalized")
SMOOTHERS = ("heat", "lazy")


def check_graph(graph, n: int | None = None) -> sparse.csr_matrix:
    """Validate and return a symmetric, nonnegative, zero-diagonal CSR adjacency matrix."""
    graph = sparse.csr_matrix(graph, dtype=float)
    if graph.shape[0] != graph.shape[1]:
        raise ValueError("graph must be square")
    if n is not None and graph.shape[0] != n:
        raise ValueError(f"graph has {graph.shape[0]} nodes but the data have {n} observations")
    if graph.nnz and (np.any(graph.data < 0) or not np.isfinite(graph.data).all()):
        raise ValueError("graph weights must be finite and nonnegative")
    if abs(graph - graph.T).max() > 1e-12 if graph.nnz else False:
        raise ValueError("graph must be symmetric")
    if graph.diagonal().any():
        raise ValueError("graph must have a zero diagonal")
    graph.eliminate_zeros()
    return graph


def spatial_graph(coordinates, neighbors: int = 6, radius: float | None = None) -> sparse.csr_matrix:
    """Symmetric binary k-nearest-neighbour graph (union of directed kNN edges).

    Parameters
    ----------
    coordinates : (n, d) array of spot or cell positions.
    neighbors : number of nearest neighbours per observation (default 6).
    radius : optional maximum edge length, in the units of ``coordinates``; edges longer
        than this are dropped so the graph does not bridge tissue gaps.
    """
    xy = np.asarray(coordinates, dtype=float)
    if xy.ndim != 2 or len(xy) < 2 or not np.isfinite(xy).all():
        raise ValueError("coordinates must be a finite (n, d) matrix with n >= 2")
    if not isinstance(neighbors, (int, np.integer)) or neighbors < 1:
        raise ValueError("neighbors must be a positive integer")
    k = min(len(xy) - 1, int(neighbors))
    dist, idx = cKDTree(xy).query(xy, k=k + 1)
    rows = np.repeat(np.arange(len(xy)), k + 1).reshape(len(xy), k + 1)
    use = rows != idx
    if radius is not None:
        if not (np.isfinite(radius) and radius > 0):
            raise ValueError("radius must be positive")
        use &= dist <= radius
    i, j = rows[use], idx[use]
    a = sparse.csr_matrix((np.ones(len(i)), (i, j)), shape=(len(xy), len(xy)))
    a = a.maximum(a.T).tocsr()
    a.setdiag(0)
    a.eliminate_zeros()
    return a


def visium_graph(array_row, array_col) -> sparse.csr_matrix:
    """First-ring hexagonal adjacency (up to 6 neighbours) from Visium array coordinates.

    Visium spots in the same row are two ``array_col`` units apart; diagonal neighbours
    differ by one row and one column.
    """
    r = np.asarray(array_row, dtype=int)
    c = np.asarray(array_col, dtype=int)
    if r.shape != c.shape or r.ndim != 1:
        raise ValueError("array_row and array_col must be 1-D arrays of equal length")
    index = {(a, b): k for k, (a, b) in enumerate(zip(r.tolist(), c.tolist()))}
    if len(index) != len(r):
        raise ValueError("Visium array coordinates must be unique")
    ii, jj = [], []
    for (a, b), k in index.items():
        for da, db in ((0, 2), (1, -1), (1, 1)):
            m = index.get((a + da, b + db))
            if m is not None:
                ii.append(k)
                jj.append(m)
    ii, jj = np.asarray(ii, int), np.asarray(jj, int)
    n = len(r)
    a = sparse.csr_matrix((np.ones(2 * len(ii)), (np.r_[ii, jj], np.r_[jj, ii])), shape=(n, n))
    a.data[:] = 1.0
    return a


def laplacian(graph) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Combinatorial Laplacian L = D - A and the degree vector."""
    a = check_graph(graph)
    degree = np.asarray(a.sum(axis=1)).ravel()
    lap = (sparse.diags(degree) - a).tocsr()
    return lap, degree


def normalized_laplacian(graph) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Symmetric normalized Laplacian I - D^-1/2 A D^-1/2 (zero rows at isolated nodes)."""
    a = check_graph(graph)
    degree = np.asarray(a.sum(axis=1)).ravel()
    inv = np.divide(1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0)
    lap = (sparse.diags((degree > 0).astype(float)) - sparse.diags(inv) @ a @ sparse.diags(inv)).tocsr()
    return lap, degree


def _laplacian(graph, kind: str):
    if kind == "combinatorial":
        return laplacian(graph)
    if kind == "normalized":
        return normalized_laplacian(graph)
    raise ValueError(f"laplacian must be one of {LAPLACIANS}")


def smoother(graph, laplacian_kind: str = "combinatorial") -> sparse.csr_matrix:
    """Uniform-rate heat-diffusion step S = I - L / (2 d_max).

    S is symmetric, doubly stochastic and positive semidefinite, and it is a polynomial in
    the Laplacian, so every smoothed statistic remains an exact spectral filter. On a
    regular lattice (e.g. the interior of a Visium array) it equals the lazy random walk.
    With ``laplacian_kind='normalized'`` the step is I - L_sym / 2, the symmetric form of
    the lazy random walk.
    """
    lap, degree = _laplacian(graph, laplacian_kind)
    n = lap.shape[0]
    if laplacian_kind == "normalized":
        return (sparse.identity(n, format="csr") - 0.5 * lap).tocsr()
    dmax = float(degree.max()) if len(degree) else 0.0
    if dmax == 0:
        return sparse.identity(n, format="csr")
    return (sparse.identity(n, format="csr") - lap / (2.0 * dmax)).tocsr()


def lazy_walk(graph) -> sparse.csr_matrix:
    """Lazy random walk P = (I + D^-1 A) / 2, identity rows at isolated nodes.

    This is the diffusion used by the local benchmark of the juxtaposed preprint (v0.1).
    P is row-stochastic but not symmetric, so it is offered for local edge maps only.
    """
    a = check_graph(graph)
    degree = np.asarray(a.sum(axis=1)).ravel()
    inv = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    mean = sparse.diags(inv) @ a
    return (sparse.diags(np.where(degree > 0, 0.5, 1.0)) + 0.5 * mean).tocsr()


def smooth(values, graph_or_smoother, steps: int, *, is_smoother: bool = False,
           method: str = "heat"):
    """Apply ``steps`` smoothing steps (``method`` 'heat' or 'lazy') to the columns of ``values``."""
    if not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("scale (number of smoothing steps) must be a nonnegative integer")
    if steps == 0:
        return np.asarray(values, dtype=float)
    if is_smoother:
        s = graph_or_smoother
    elif method == "heat":
        s = smoother(graph_or_smoother)
    elif method == "lazy":
        s = lazy_walk(graph_or_smoother)
    else:
        raise ValueError(f"method must be one of {SMOOTHERS}")
    out = np.asarray(values, dtype=float)
    for _ in range(int(steps)):
        out = s @ out
    return out
