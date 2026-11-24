"""Calibrated inference for pair-level spatial statistics.

Spatially autocorrelated genes violate the independence that ordinary correlation tests and
spot-permutation tests assume, which inflates false positives. ``SpectralNull`` implements
Moran spectral randomization (Wagner & Dray 2015): the graph-Fourier coefficients of a feature
are sign-flipped, which keeps the feature's graph power spectrum (and therefore its Moran's I,
Geary's c and variogram on the graph) exactly unchanged while breaking any association with a
second feature. For statistics that are quadratic forms xᵀMy the null variance has a closed
form, so p-values for every gene pair need one eigendecomposition and two matrix products,
with no permutations.

The closed-form p-value is a normal tail area. It is accurate when many spectral modes carry the
null variance, which every test reports as ``modes`` (the participation ratio). When few modes
carry it, as for Pearson's r or Lee's L on smooth maps, the null is far from normal and the normal
tail is too conservative (a warning says so); ``method="monte_carlo"`` samples the same null
directly and ``method="auto"`` does so only for the pairs with fewer than ``min_modes`` modes.
Sampling costs ``n_draws`` Gram products, so the default stays ``method="normal"``.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import sparse
from scipy.special import ndtr

from ._core import power_mean, row_standardize, standardize
from ._graph import _laplacian, check_graph, laplacian, smoother

__all__ = ["SpectralNull", "permutation_test", "bh_qvalues"]


def _two_sided(z):
    return 2.0 * ndtr(-np.abs(z))


def bh_qvalues(p):
    """Benjamini–Hochberg q-values for a 1-D array (NaN preserved)."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = ok.sum()
    if m == 0:
        return q
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


class SpectralNull:
    """Exact spectral sign-flip null for a fixed graph.

    Parameters
    ----------
    graph : symmetric adjacency matrix (n x n).
    max_nodes : refuse dense eigendecomposition above this size (memory ~ 8 n² bytes).
    laplacian : 'combinatorial' (D - A, default) or 'normalized' (I - D^-1/2 A D^-1/2); must
        match the ``laplacian`` used for J.
    """

    def __init__(self, graph, *, max_nodes: int = 20000, laplacian: str = "combinatorial"):
        a = check_graph(graph)
        n = a.shape[0]
        if n > max_nodes:
            raise ValueError(
                f"{n} nodes exceeds max_nodes={max_nodes} for the exact spectral null; "
                "aggregate cells into spatial bins, or raise max_nodes if memory allows "
                f"(about {8 * n * n * 3 / 1e9:.1f} GB needed)")
        lap, degree = _laplacian(a, laplacian)
        self.laplacian = laplacian
        mu, u = np.linalg.eigh(lap.toarray())
        self.mu = np.clip(mu, 0.0, None)
        self.U = u
        self.graph = a
        self.degree = degree
        # the heat step is I - L/(2 d_max); for the normalized Laplacian it is I - L_sym/2
        self.dmax = (1.0 if laplacian == "normalized" else float(degree.max())) if n else 0.0
        self._mhat = {}
        # Group (numerically) repeated eigenvalues. Within a repeated eigenspace the basis is
        # arbitrary, so the null randomizes by rotation within the eigenspace; its variance
        # uses the eigenspace-averaged squared coefficients and does not depend on the basis.
        tol = 1e-8 * max(1.0, float(self.mu.max())) if n else 0.0
        self._starts = np.r_[0, np.where(np.diff(self.mu) > tol)[0] + 1] if n else np.zeros(0, int)
        self._sizes = np.diff(np.r_[self._starts, n])
        self._group = np.repeat(np.arange(len(self._starts)), self._sizes)
        self._degenerate = bool((self._sizes > 1).any())

    def squared_coefficients(self, xh):
        """Squared spectral coefficients, averaged within repeated eigenspaces."""
        a2 = xh * xh
        if not self._degenerate:
            return a2
        sums = np.add.reduceat(a2, self._starts, axis=0) / self._sizes[:, None]
        return sums[self._group]

    def _rotated_blocks(self, weight):
        """Repeated eigenspaces (start, size) that carry weight in the statistic. A space whose weight is
        below 1e-8 of the largest (e.g. the constant vectors of each graph component for J) does not
        change the statistic and is not rotated."""
        if not self._degenerate:
            return []
        w = np.abs(np.asarray(weight, float))
        top = float(w.max()) if w.size else 0.0
        return [(st, sz) for st, sz in zip(self._starts[self._sizes > 1], self._sizes[self._sizes > 1])
                if float(w[st:st + sz].max()) > 1e-8 * top]

    def _randomize(self, xh, rng, blocks=None):
        """One draw of the null for every feature at once: independent signs per mode and feature,
        and an independent Haar rotation inside each repeated eigenspace (``blocks``; all by default).
        A Haar rotation applied to one vector gives a uniformly random direction of the same length,
        so each feature's block is replaced by a normalized Gaussian direction scaled to its norm."""
        xb = xh * rng.choice((-1.0, 1.0), size=xh.shape)
        if self._degenerate:
            if blocks is None:
                blocks = list(zip(self._starts[self._sizes > 1], self._sizes[self._sizes > 1]))
            for st, sz in blocks:
                blk = xb[st:st + sz]
                g = rng.normal(size=blk.shape)
                gn = np.sqrt(np.einsum("ij,ij->j", g, g))
                xb[st:st + sz] = g * (np.sqrt(np.einsum("ij,ij->j", blk, blk)) / np.where(gn > 0, gn, 1.0))
        return xb

    def _monte_carlo(self, xh, weight, obs, n_draws, seed, mask=None):
        """p = (1 + b) / (B + 1); b counts draws with |null statistic| >= |observed|.
        ``weight``: per-mode filter (1-D, diagonal statistics) or projected matrix (2-D).
        ``mask``: optional symmetric boolean matrix of the pairs that need p-values. When few pairs are
        needed, only their null statistics are formed; the draws, and so the p-values, are the same as
        for the full matrix."""
        rng = np.random.default_rng(seed)
        dense = np.ndim(weight) == 2
        tol = np.abs(obs) * (1 - 1e-10)
        blocks = self._rotated_blocks(weight)
        p = xh.shape[1]
        if mask is not None:
            ia, ib = np.nonzero(np.triu(mask))
            if len(ia) < p * (p + 1) // 4:
                count = np.zeros(len(ia))
                tp = tol[ia, ib]
                for _ in range(int(n_draws)):
                    xb = self._randomize(xh, rng, blocks)
                    left = weight @ xb[:, ib] if dense else xb[:, ib] * weight[:, None]
                    count += np.abs(np.einsum("ij,ij->j", xb[:, ia], left)) >= tp
                out = np.ones(obs.shape)
                out[ia, ib] = out[ib, ia] = (1.0 + count) / (int(n_draws) + 1.0)
                return out
        count = np.zeros(obs.shape)
        for _ in range(int(n_draws)):
            xb = self._randomize(xh, rng, blocks)
            null = xb.T @ (weight @ xb) if dense else (xb * weight[:, None]).T @ xb
            count += np.abs(null) >= tol
        return (1.0 + count) / (int(n_draws) + 1.0)

    @staticmethod
    def _modes(a2, weight):
        """Participation ratio (sum a^2)^2 / sum a^4 of the per-mode null terms a = w x_k y_l."""
        if np.ndim(weight) == 2:
            s2 = a2.T @ (weight ** 2) @ a2
            s4 = (a2 ** 2).T @ (weight ** 4) @ (a2 ** 2)
        else:
            s2 = (a2 * (weight ** 2)[:, None]).T @ a2
            s4 = (a2 ** 2 * (weight ** 4)[:, None]).T @ (a2 ** 2)
        return np.divide(s2 ** 2, s4, out=np.full_like(s2, np.nan), where=s4 > 0)

    def _pvalues(self, out, xh, weight, obs, method, n_draws, seed, min_modes):
        """Normal tail areas, or Monte Carlo p-values: for every pair ('monte_carlo') or only for
        pairs whose null rests on fewer than ``min_modes`` modes ('auto')."""
        if method not in ("auto", "normal", "monte_carlo"):
            raise ValueError("method must be 'auto', 'normal' or 'monte_carlo'")
        ok = np.isfinite(out["p"])
        few = ok & (np.nan_to_num(out["modes"], nan=np.inf) < min_modes)
        if method == "normal":
            sampled = np.zeros_like(ok)
            if ok.any() and few[ok].mean() > 0.05:
                warnings.warn(f"{100 * few[ok].mean():.0f}% of pairs have fewer than {min_modes:g} effective "
                              "spectral modes, where normal p-values are too conservative; use "
                              "method='monte_carlo' or 'auto'", RuntimeWarning, stacklevel=3)
        elif method == "monte_carlo":
            sampled = ok
        else:
            sampled = few
        if sampled.any():
            mc = self._monte_carlo(xh, weight, np.asarray(obs, float), n_draws, seed,
                                   mask=None if method == "monte_carlo" else sampled)
            out["p"] = np.where(sampled, mc, out["p"])
        out["sampled"] = sampled
        out["method"] = ("normal" if not sampled.any() else
                         "monte_carlo" if sampled[ok].all() else "mixed")
        return out

    @property
    def n(self) -> int:
        return self.U.shape[0]

    def spectral_filter(self, scale: int = 1) -> np.ndarray:
        """h(μ) = μ (1 - μ / (2 d_max))^(2·scale): the filter that defines J at ``scale``."""
        if self.dmax == 0:
            return np.zeros_like(self.mu)
        return self.mu * (1.0 - self.mu / (2.0 * self.dmax)) ** (2 * int(scale))

    def coefficients(self, values, standardized: bool = False, dtype=None):
        z, ok = (standardize(values, return_mask=True) if not standardized else
                 (np.asarray(values, float), np.ones(np.asarray(values).shape[1], bool)))
        if z.shape[0] != self.n:
            raise ValueError("values and graph have different numbers of observations")
        if dtype is not None and np.dtype(dtype) != np.float64:
            if getattr(self, "_U32", None) is None:
                self._U32 = np.ascontiguousarray(self.U, dtype=dtype)
            return self._U32.T @ np.asarray(z, dtype=dtype), ok
        return self.U.T @ z, ok

    # ------------------------------------------------------------------ J
    def test_j(self, values, *, scale: int = 1, standardized: bool = False, power: float = 1.0,
               dtype=None, method: str = "normal", n_draws: int = 4000, seed: int | None = 0,
               min_modes: float = 20.0):
        """Return dict(stat, sd, z, p, modes, sampled, method) for all feature pairs.

        ``modes`` is the effective number of spectral modes behind each null variance (participation
        ratio). ``method="normal"`` (default) gives normal tail areas; ``method="auto"`` replaces the normal tail area by a Monte Carlo p-value
        (``n_draws`` randomizations) for pairs with fewer than ``min_modes`` modes; ``sampled``
        marks those pairs and ``method`` reports 'normal', 'mixed' or 'monte_carlo'.

        The energies in J's denominator are invariant under the sign-flip null, so z and p
        are those of the cross-energy C and do not depend on ``power``. Because the filter h is
        nonnegative, both Gram matrices are symmetric rank-k updates (half the cost of a
        general product); ``dtype=np.float32`` halves it again.
        """
        xh, ok = self.coefficients(values, standardized, dtype=dtype)
        h = self.spectral_filter(scale).astype(xh.dtype)
        m = np.sqrt(h)[:, None] * xh
        g = np.asarray(m.T @ m, dtype=float)
        e = np.clip(np.diag(g), 0, None)
        denom = power_mean(e[:, None], e[None, :], power)
        jmat = np.divide(-g, denom, out=np.full_like(g, np.nan), where=denom > 0)
        a2 = h[:, None] * self.squared_coefficients(xh)
        var = np.asarray(a2.T @ a2, dtype=float)
        sd = np.divide(np.sqrt(np.clip(var, 0, None)), denom,
                       out=np.full_like(g, np.nan), where=denom > 0)
        sq = self.squared_coefficients(np.asarray(xh, float))
        out = self._finish(jmat, sd, ok, self._modes(sq, h.astype(float)))
        return self._pvalues(out, np.asarray(xh, float), h.astype(float), g, method, n_draws, seed, min_modes)

    # --------------------------------------------- generic quadratic forms
    def _projected(self, key, m):
        if key not in self._mhat:
            mu = m @ self.U if sparse.issparse(m) else np.asarray(m) @ self.U
            mh = self.U.T @ mu
            self._mhat[key] = 0.5 * (mh + mh.T)
        return self._mhat[key]

    def _quadratic(self, values, key, m, norm, standardized, method, n_draws, seed, min_modes):
        xh, ok = self.coefficients(values, standardized)
        mh = self._projected(key, m)
        num = xh.T @ mh @ xh
        a2 = self.squared_coefficients(xh)
        var = a2.T @ (mh * mh) @ a2
        sd = np.sqrt(np.clip(var, 0, None)) / norm
        out = self._finish(num / norm, sd, ok, self._modes(a2, mh))
        return self._pvalues(out, xh, mh, num, method, n_draws, seed, min_modes)

    def test_lee(self, values, *, standardized: bool = False, method: str = "normal", n_draws: int = 4000,
                 seed: int | None = 0, min_modes: float = 20.0):
        """Lee's L under the spectral null; arguments as in :meth:`test_j`."""
        w = row_standardize(self.graph)
        rows = np.asarray(w.sum(axis=1)).ravel()
        return self._quadratic(values, "lee", (w.T @ w).tocsr(), float(np.sum(rows ** 2)), standardized,
                               method, n_draws, seed, min_modes)

    def test_moran(self, values, *, standardized: bool = False, method: str = "normal", n_draws: int = 4000,
                   seed: int | None = 0, min_modes: float = 20.0):
        """Bivariate Moran's I under the spectral null; arguments as in :meth:`test_j`."""
        w = row_standardize(self.graph)
        return self._quadratic(values, "moran", (0.5 * (w + w.T)).tocsr(), float(w.sum()), standardized,
                               method, n_draws, seed, min_modes)

    def test_pearson(self, values, *, standardized: bool = False, method: str = "normal", n_draws: int = 4000,
                     seed: int | None = 0, min_modes: float = 20.0):
        """Pearson's r under the spectral null; arguments as in :meth:`test_j`."""
        xh, ok = self.coefficients(values, standardized)
        n = self.n
        num = xh.T @ xh
        a2 = self.squared_coefficients(xh)
        sd = np.sqrt(np.clip(a2.T @ a2, 0, None)) / n
        ones = np.ones(xh.shape[0])
        out = self._finish(num / n, sd, ok, self._modes(a2, ones))
        return self._pvalues(out, xh, ones, num, method, n_draws, seed, min_modes)

    @staticmethod
    def _finish(stat, sd, ok, modes):
        z = np.divide(stat, sd, out=np.full_like(stat, np.nan), where=sd > 0)
        p = _two_sided(z)
        for arr in (stat, sd, z, p, modes):
            arr[~ok, :] = np.nan
            arr[:, ~ok] = np.nan
            np.fill_diagonal(arr, np.nan)
        return {"stat": stat, "sd": sd, "z": z, "p": p, "modes": modes}


def permutation_test(x, y, graph, *, statistic: str = "J", scale: int = 1, n_perm: int = 999,
                     seed: int | None = 0):
    """Classical spot-permutation test for one pair (shown for comparison; not calibrated
    for autocorrelated features). Returns (observed statistic, two-sided p-value)."""
    a = check_graph(graph)
    zx = standardize(x)[:, 0]
    zy = standardize(y)[:, 0]
    n = len(zx)
    rng = np.random.default_rng(seed)
    perms = np.stack([rng.permutation(n) for _ in range(int(n_perm))])
    yp = zy[perms].T  # n x B
    statistic = statistic.upper()
    if statistic == "J":
        s = smoother(a)
        lap, _ = laplacian(a)
        for _ in range(int(scale)):
            zx, zy, yp = s @ zx, s @ zy, s @ yp
        lzx = lap @ zx
        ex = zx @ lzx
        obs = -2 * (zy @ lzx) / (ex + zy @ (lap @ zy))
        null = -2 * (yp.T @ lzx) / (ex + np.einsum("ib,ib->b", yp, lap @ yp))
    elif statistic in ("L", "LEE"):
        w = row_standardize(a)
        rows = np.asarray(w.sum(axis=1)).ravel()
        v = w.T @ (w @ zx) / np.sum(rows ** 2)
        obs, null = zy @ v, yp.T @ v
    elif statistic in ("I", "MORAN"):
        w = row_standardize(a)
        v = 0.5 * (w @ zx + w.T @ zx) / w.sum()
        obs, null = zy @ v, yp.T @ v
    elif statistic in ("R", "PEARSON"):
        obs, null = zy @ zx / n, yp.T @ zx / n
    else:
        raise ValueError("statistic must be one of 'J', 'L', 'I', 'r'")
    pval = (1 + np.sum(np.abs(null) >= abs(obs) - 1e-15)) / (len(null) + 1)
    return float(obs), float(pval)
