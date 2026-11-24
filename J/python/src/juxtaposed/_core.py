"""Pair-level spatial association indices on a shared graph.

Every function takes an observation-by-feature matrix ``X`` (spots or cells by genes) and a
symmetric adjacency matrix, standardizes the features within the sample, and returns a dense
feature-by-feature matrix. Signs follow the original convention of J:

* ``juxtaposition_matrix`` (J): +1 = the two features change in opposite directions across the
  same edges (complementary domains that touch); -1 = they change in the same direction.
* ``lee_l_matrix``, ``moran_bv_matrix``, ``pearson_matrix``: the usual sign (+ = co-occurrence).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from ._backends import accelerated
from ._graph import _laplacian, check_graph, laplacian, smoother

__all__ = [
    "standardize", "juxtaposition_matrix", "juxtaposition_score", "codispersion_matrix",
    "gradient_coenergy_matrix", "lee_l_matrix", "moran_bv_matrix", "pearson_matrix",
    "row_standardize", "graph_gram", "power_mean",
]


def power_mean(a, b, power: float = 1.0):
    """Power mean M_p(a, b) of two nonnegative energies, stable for every p >= 0.

    p = 1 is the arithmetic mean (J), p = 0 the geometric mean (negative codispersion).
    For p >= 0, M_p >= sqrt(ab), so |C| <= M_p and J_p stays in [-1, 1].
    """
    if not np.isfinite(power) or power < 0:
        raise ValueError("power must be finite and nonnegative")
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if power == 1:
        return 0.5 * (a + b)
    if power == 0:
        return np.sqrt(a) * np.sqrt(b)
    hi, lo = np.maximum(a, b), np.minimum(a, b)
    rel = np.divide(lo, hi, out=np.zeros_like(hi), where=hi > 0)
    with np.errstate(divide="ignore"):
        factor = np.log1p(0.5 * np.expm1(power * np.log(rel))) / power
    return hi * np.exp(factor)


def standardize(values, *, return_mask: bool = False):
    """Centre each column and divide by its population SD.

    Columns with zero variance are returned as zeros (and flagged in the mask) so that the
    corresponding statistics become NaN rather than raising. Two passes over the data, no
    temporaries beyond the output.
    """
    x = np.asarray(values)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("values must be an observation-by-feature matrix with >= 2 rows")
    z = np.array(x, dtype=float, order="C", copy=True)
    if not np.isfinite(z).all():
        raise ValueError("values must be finite")
    z -= z[0].copy()
    z -= z.mean(axis=0)
    sd = np.sqrt(np.einsum("ij,ij->j", z, z) / z.shape[0])
    ok = sd > 0
    z /= np.where(ok, sd, 1.0)
    z[:, ~ok] = 0.0
    return (z, ok) if return_mask else z


def row_standardize(graph) -> sparse.csr_matrix:
    a = check_graph(graph)
    d = np.asarray(a.sum(axis=1)).ravel()
    inv = np.divide(1.0, d, out=np.zeros_like(d), where=d > 0)
    return sparse.diags(inv) @ a


def _prep(values, graph, scale, standardized, laplacian_kind="combinatorial"):
    z, ok = (standardize(values, return_mask=True) if not standardized else
             (np.asarray(values, float), np.ones(np.asarray(values).shape[1], bool)))
    a = check_graph(graph, z.shape[0])
    if not isinstance(scale, (int, np.integer)) or scale < 0:
        raise ValueError("scale must be a nonnegative integer (number of smoothing steps)")
    if scale:
        s = smoother(a, laplacian_kind)
        for _ in range(int(scale)):
            z = s @ z
    return z, a, ok


def graph_gram(z, graph, laplacian_kind: str = "combinatorial", *, dtype=None,
               method: str = "auto") -> np.ndarray:
    """G = Zᵀ L Z, where each undirected edge e = (i, j) contributes w_e Δz_i Δz_j once.

    With ``laplacian_kind='normalized'`` the increments are taken after dividing each
    observation by the square root of its degree (symmetric normalized Laplacian). For many
    features G is formed as a symmetric rank-k update after a sparse Cholesky-type transform
    (see ``juxtaposed._fast``), the same cost as Lee's L.
    """
    from ._fast import gram
    lap, _ = _laplacian(graph, laplacian_kind)
    return gram(z, lap, dtype=dtype, method=method)


def _mask(mat, ok):
    if not ok.all():
        mat = mat.copy()
        mat[~ok, :] = np.nan
        mat[:, ~ok] = np.nan
    return mat


def juxtaposition_matrix(values, graph, *, scale: int = 1, standardized: bool = False,
                         power: float = 1.0, laplacian: str = "combinatorial", dtype=None,
                         backend: str = "cpu") -> np.ndarray:
    """All-pairs J at a given smoothing scale.

    J(x, y) = -2 Σ_e w_e Δx_e Δy_e / Σ_e w_e (Δx_e² + Δy_e²), computed after ``scale`` steps
    of uniform heat diffusion. J lies in [-1, 1]; J(x, x) = -1 and J(x, -x) = +1.
    Equivalently J = [E(x - y) - E(x + y)] / [E(x - y) + E(x + y)] with E the gradient energy.

    ``power`` replaces the arithmetic mean of the two energies by their power mean
    (``power=0`` gives negative codispersion); lower powers keep pairs with unequal gradient
    amplitudes on a comparable scale. The spectral test does not depend on ``power``.
    ``laplacian='normalized'`` uses I - D^-1/2 A D^-1/2, for which
    J = 2 (I_xy - r) / ((1 - I_x) + (1 - I_y)) exactly on any graph, with Moran-type
    coefficients I defined on the symmetric normalized weights.

    ``dtype=np.float32`` halves memory and roughly halves run time (relative error ~1e-6);
    For ``power=1`` and a combinatorial Laplacian, ``backend='auto'`` tries an available
    float32 accelerator and falls back to the CPU. Explicit accelerators are ``'mlx'``
    on Apple Silicon and ``'torch'`` with CUDA, ROCm or Intel XPU.
    """
    if backend not in ("cpu", "numpy", "auto", "mlx", "torch"):
        raise ValueError("backend must be 'cpu', 'numpy', 'auto', 'mlx' or 'torch'")
    can_accelerate = power == 1 and laplacian == "combinatorial"
    if backend in ("mlx", "torch") and not can_accelerate:
        raise ValueError(f"backend='{backend}' supports power=1 and laplacian='combinatorial'")
    if not isinstance(scale, (int, np.integer)) or scale < 0:
        raise ValueError("scale must be a nonnegative integer (number of smoothing steps)")

    def run_mlx():
        from ._metal import score_mlx
        z0 = np.asarray(values, dtype=float)
        z0 = z0[:, None] if z0.ndim == 1 else z0
        if z0.ndim != 2 or z0.shape[0] < 2:
            raise ValueError("values must be an observation-by-feature matrix with >= 2 rows")
        return score_mlx(z0, check_graph(graph, z0.shape[0]), int(scale), standardized)

    def run_torch(torch, device):
        from ._torch import juxtaposition
        z, a, valid = _prep(values, graph, scale, standardized)
        lap, _ = _laplacian(a, "combinatorial")
        return juxtaposition(torch, device, z, lap @ z, valid)

    if can_accelerate and backend not in ("cpu", "numpy"):
        if backend != "auto" or dtype is None or np.dtype(dtype) == np.dtype("float32"):
            result = accelerated(backend, run_mlx, run_torch)
            if result is not None:
                return result
    z, ok = (standardize(values, return_mask=True) if not standardized else
             (np.asarray(values, float), np.ones(np.asarray(values).shape[1], bool)))
    a = check_graph(graph, z.shape[0])
    if dtype is not None:  # cast before the sparse products, which are memory bound
        z = z.astype(dtype, copy=False)
    if scale:
        s = smoother(a, laplacian)
        s = s.astype(z.dtype) if s.dtype != z.dtype else s
        for _ in range(int(scale)):
            z = s @ z
    g = graph_gram(z, a, laplacian, dtype=dtype)
    e = np.clip(np.diag(g), 0, None).astype(float)
    if power == 1:
        # in place: J = -2 G / (e_i + e_j), without extra p x p temporaries
        g = np.asarray(g, dtype=float)
        den = e[:, None] + e[None, :]
        np.divide(g, den, out=g, where=den > 0)
        g[den <= 0] = np.nan
        g *= -2.0
        del den
        j = g
    else:
        g = np.asarray(g, dtype=float)
        denom = power_mean(e[:, None], e[None, :], power)
        j = np.divide(-g, denom, out=np.full_like(g, np.nan), where=denom > 0)
    np.clip(j, -1.0, 1.0, out=j)
    return _mask(j, ok)


def juxtaposition_score(x, y, graph, *, scale: int = 1, power: float = 1.0,
                        laplacian: str = "combinatorial") -> float:
    """J for a single pair of features."""
    return float(juxtaposition_matrix(np.column_stack([x, y]), graph, scale=scale, power=power,
                                      laplacian=laplacian)[0, 1])


def codispersion_matrix(values, graph, *, scale: int = 1, standardized: bool = False) -> np.ndarray:
    """Negative graph codispersion -Σ w ΔxΔy / sqrt(Σ w Δx² Σ w Δy²) (same sign as J)."""
    z, a, ok = _prep(values, graph, scale, standardized)
    g = graph_gram(z, a)
    e = np.clip(np.diag(g), 0, None)
    denom = np.sqrt(np.outer(e, e))
    c = np.divide(-g, denom, out=np.full_like(g, np.nan), where=denom > 0)
    return _mask(np.clip(c, -1.0, 1.0), ok)


def gradient_coenergy_matrix(values, graph, *, scale: int = 1, standardized: bool = False) -> np.ndarray:
    """Sign-free overlap of gradient magnitudes Σ w|Δx||Δy| / sqrt(Σ wΔx² Σ wΔy²)."""
    z, a, ok = _prep(values, graph, scale, standardized)
    up = sparse.triu(a, k=1).tocoo()
    sw = np.sqrt(up.data)[:, None]
    d = np.abs(z[up.col] - z[up.row]) * sw
    g = d.T @ d
    e = np.diag(g).copy()
    denom = np.sqrt(np.outer(e, e))
    c = np.divide(g, denom, out=np.full_like(g, np.nan), where=denom > 0)
    return _mask(c, ok)


def lee_l_matrix(values, graph, *, standardized: bool = False, backend: str = "cpu") -> np.ndarray:
    """Lee's L (Lee 2001) with a row-standardized weight matrix.

    L = [n / Σ_i (Σ_j w_ij)²] Σ_i (Σ_j w_ij x̃_j)(Σ_j w_ij ỹ_j) / (||x̃|| ||ỹ||).
    """
    def run_mlx():
        from ._metal import lee_mlx
        z0 = np.asarray(values, dtype=float)
        z0 = z0[:, None] if z0.ndim == 1 else z0
        if z0.ndim != 2 or z0.shape[0] < 2:
            raise ValueError("values must be an observation-by-feature matrix with >= 2 rows")
        a0 = check_graph(graph, z0.shape[0])
        return lee_mlx(z0, row_standardize(a0), standardized)

    def run_torch(torch, device):
        from ._torch import lee
        z, a, valid = _prep(values, graph, 0, standardized)
        w = row_standardize(a)
        mass = np.sum(np.asarray(w.sum(axis=1)).ravel() ** 2)
        return lee(torch, device, w @ z, valid, mass)

    if backend not in ("cpu", "numpy"):
        result = accelerated(backend, run_mlx, run_torch)
        if result is not None:
            return result
    z, a, ok = _prep(values, graph, 0, standardized)
    w = row_standardize(a)
    rows = np.asarray(w.sum(axis=1)).ravel()
    wz = w @ z
    lee = wz.T @ wz
    lee /= np.sum(rows ** 2)
    return _mask(lee, ok)


def moran_bv_matrix(values, graph, *, standardized: bool = False) -> np.ndarray:
    """Symmetrized bivariate Moran's I, (xᵀWy + yᵀWx) / (2 S0) for standardized x, y."""
    z, a, ok = _prep(values, graph, 0, standardized)
    w = row_standardize(a)
    s0 = float(w.sum())
    m = z.T @ (w @ z)
    return _mask(0.5 * (m + m.T) / s0, ok)


def pearson_matrix(values) -> np.ndarray:
    z, ok = standardize(values, return_mask=True)
    return _mask(z.T @ z / z.shape[0], ok)
