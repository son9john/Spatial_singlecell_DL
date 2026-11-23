"""Edge-level (local) juxtaposed values and the scalar increment statistic."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from ._core import power_mean, standardize
from ._graph import check_graph, lazy_walk, smoother

__all__ = ["juxtaposed_value", "EdgeResult", "score_edges"]


def juxtaposed_value(dx, dy, weights=None, axis=None, power: float = 1.0):
    """J for paired increments: -C / M_p(A, B), A = Σw dx², B = Σw dy², C = Σw dx dy.

    ``power=1`` (default) gives the original J = -2C/(A+B); ``power=0`` gives signed
    codispersion -C/sqrt(AB). Returns NaN when either energy is zero.
    """
    dx, dy = np.asarray(dx, dtype=float), np.asarray(dy, dtype=float)
    if dx.shape != dy.shape or not (np.isfinite(dx).all() and np.isfinite(dy).all()):
        raise ValueError("expected matching, finite increment arrays")
    if weights is None:
        weights = 1.0
    else:
        weights = np.broadcast_to(np.asarray(weights, dtype=float), dx.shape)
        if np.any(weights < 0) or not np.isfinite(weights).all():
            raise ValueError("weights must be nonnegative and finite")
    ex = np.sum(weights * dx ** 2, axis=axis)
    ey = np.sum(weights * dy ** 2, axis=axis)
    cross = np.sum(weights * dx * dy, axis=axis)
    scale = np.asarray(power_mean(ex, ey, power), dtype=float)
    value = np.divide(-np.asarray(cross, float), scale, where=scale > 0,
                      out=np.full_like(scale, np.nan))
    value = np.clip(value, -1, 1)
    return value if value.ndim else float(value)


@dataclass(frozen=True)
class EdgeResult:
    """Local scores per undirected edge (pooled over edges incident to either endpoint).

    Attributes
    ----------
    endpoints : (m, 2) node indices of each edge.
    j : bounded local J in [-1, 1] (shrunk toward 0 in low-energy windows when ``ridge > 0``).
    signed_evidence : mean signed cross-gradient energy -2 Δx Δy over the window (Q). It is
        not normalized, so it ranks strong opposing transitions first; in known-truth
        benchmarks it localized interfaces better than the bounded J.
    energy : mean joint gradient energy Δx² + Δy² over the window.
    """

    endpoints: np.ndarray
    j: np.ndarray
    signed_evidence: np.ndarray
    energy: np.ndarray

    def to_nodes(self, n: int, field: str = "j") -> np.ndarray:
        """Average an edge field over the edges incident to each node (for plotting maps)."""
        val = np.asarray(getattr(self, field), dtype=float)
        i, j = self.endpoints[:, 0], self.endpoints[:, 1]
        ok = np.isfinite(val)
        tot = np.bincount(i[ok], weights=val[ok], minlength=n) + np.bincount(j[ok], weights=val[ok], minlength=n)
        cnt = np.bincount(i[ok], minlength=n) + np.bincount(j[ok], minlength=n)
        return np.divide(tot, cnt, out=np.full(n, np.nan), where=cnt > 0)


def _pool(value, i, j, n):
    node = np.bincount(i, weights=value, minlength=n) + np.bincount(j, weights=value, minlength=n)
    return node[i] + node[j]


def score_edges(values, graph, *, scale: int = 0, ridge: float = 0.0, power: float = 1.0,
                smoother_kind: str = "heat", standardized: bool = False,
                steps: int | None = None, standardize_values: bool | None = None) -> EdgeResult:
    """Local J for one pair of features on every edge of the graph.

    Each edge pools the cross-increments and energies of all edges incident to its two
    endpoints: J_e = -C_e / (M_p(A_e, B_e) + ridge * w_e * m / 2), where m is the
    whole-graph mean joint gradient energy and w_e the pooled edge weight. ``ridge`` shrinks
    low-energy windows toward zero without using any boundary labels. ``smoother_kind``
    selects the diffusion applied ``scale`` times before differencing: 'heat'
    (I - L / 2 d_max, as for the global J) or 'lazy' ((I + D^-1 A) / 2, the setting of the
    juxtaposed preprint's known-truth benchmark: scale=4, ridge=1).
    """
    if steps is not None:  # v0.1 keyword
        scale = steps
    if standardize_values is not None:  # v0.1 keyword: standardize=True
        standardized = not standardize_values
    z = np.asarray(values, float) if standardized else standardize(values)
    if z.ndim != 2 or z.shape[1] != 2:
        raise ValueError("score_edges requires exactly two features")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    a = check_graph(graph, z.shape[0])
    if scale:
        if smoother_kind == "heat":
            s = smoother(a)
        elif smoother_kind == "lazy":
            s = lazy_walk(a)
        else:
            raise ValueError("smoother_kind must be 'heat' or 'lazy'")
        for _ in range(int(scale)):
            z = s @ z
    up = sparse.triu(a, k=1).tocoo()
    i, j, w = up.row, up.col, up.data
    dx, dy = z[j, 0] - z[i, 0], z[j, 1] - z[i, 1]
    n = z.shape[0]
    cross = -2 * _pool(w * dx * dy, i, j, n)
    ea, eb = _pool(w * dx ** 2, i, j, n), _pool(w * dy ** 2, i, j, n)
    mass = _pool(w, i, j, n)
    ref = float(np.dot(w, dx ** 2 + dy ** 2) / w.sum()) if len(w) else 0.0
    base = power_mean(ea, eb, power)
    val = np.divide(0.5 * cross, base + 0.5 * ridge * mass * ref, where=base > 0,
                    out=np.full_like(cross, np.nan))
    return EdgeResult(np.column_stack([i, j]), np.clip(val, -1, 1), cross / mass, (ea + eb) / mass)
