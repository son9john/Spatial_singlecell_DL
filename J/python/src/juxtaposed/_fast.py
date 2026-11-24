"""Fast all-pairs Gram matrices for J.

J needs G = Zᵀ L Z for a graph Laplacian L. A general matrix product costs n·p² multiply-adds.
Because L is positive semidefinite and sparse, a sparse Cholesky-type factor L + εI = Rᵀ R
(R = D^1/2 Lᵀ P from a symmetric sparse LU) turns G into Mᵀ M with M = R Z, which BLAS evaluates
as a symmetric rank-k update in n·p²/2 multiply-adds: the same cost as Lee's L, whose Gram is
(WZ)ᵀ(WZ). ε = 1e-13·d_max changes G by ε·ZᵀZ (relative error ~1e-13). The Apple-GPU path
lives in ``_metal.py`` (custom Metal kernels, float32).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from ._backends import _require_apple_silicon

__all__ = ["LaplacianFactor", "gram", "available_backends", "syrk"]


def available_backends():
    """Return installed scoring backends with compatible hardware."""
    out = ["cpu"]
    try:
        _require_apple_silicon()
        import mlx.core as mx
        if hasattr(getattr(mx, "fast", None), "metal_kernel") and mx.metal.is_available():
            out.append("mlx")
    except (ImportError, OSError, RuntimeError):
        pass
    try:
        from ._torch import gpu
        gpu()
        out.append("torch")
    except (ImportError, OSError, RuntimeError):
        pass
    return out


def syrk(m):
    """mᵀ m for a dense (n, p) matrix; numpy dispatches A.T @ A to BLAS syrk."""
    m = np.ascontiguousarray(m)
    return m.T @ m


class LaplacianFactor:
    """Symmetric sparse factor R with Rᵀ R = L + εI, built once per graph (milliseconds)."""

    def __init__(self, lap, eps_rel: float = 1e-13):
        from scipy.sparse.linalg import splu
        lap = sparse.csc_matrix(lap, dtype=float)
        n = lap.shape[0]
        dmax = float(lap.diagonal().max()) if n else 0.0
        self.eps = eps_rel * max(dmax, 1.0)
        self.ok = False
        self.lap = lap
        try:
            lu = splu((lap + self.eps * sparse.identity(n, format="csc")).tocsc(),
                      permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                      options=dict(SymmetricMode=True))
            d = lu.U.diagonal()
            if np.array_equal(lu.perm_r, lu.perm_c) and np.all(d > 0):
                self.R = (sparse.diags(np.sqrt(d)) @ lu.L.T).tocsr()
                self.inv = np.argsort(lu.perm_r)
                self.ok = True
        except Exception:  # noqa: BLE001 - fall back to the general product
            self.ok = False

    def transform(self, z):
        """M = R z, so that Mᵀ M = zᵀ (L + εI) z."""
        return self.R @ np.asarray(z)[self.inv]


def gram(z, lap, *, factor: LaplacianFactor | None = None, dtype=None, method: str = "auto"):
    """G = zᵀ L z (symmetric). ``method``: 'factor' (Cholesky + syrk), 'gemm', or 'auto'."""
    z = np.asarray(z)
    if dtype is not None:
        z = z.astype(dtype, copy=False)
    use_factor = method == "factor" or (method == "auto" and z.shape[1] >= 64)
    if use_factor:
        factor = factor if factor is not None else LaplacianFactor(lap)
        if factor.ok:
            r = factor.R if dtype is None else factor.R.astype(dtype)
            m = r @ z[factor.inv]
            g = syrk(m)
            return np.asarray(g, dtype=z.dtype if dtype is None else dtype)
    y = lap @ z
    if dtype is not None:
        y = np.asarray(y, dtype=dtype)
    g = z.T @ y
    return 0.5 * (g + g.T)
