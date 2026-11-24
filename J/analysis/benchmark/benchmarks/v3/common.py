"""Shared helpers for the benchmark scripts (graph, statistics on paired columns)."""
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy import sparse, stats

from datasets import ROOT  # noqa: E402

sys.path.insert(0, str(ROOT / "packages" / "python_v0.3" / "src"))
import juxtaposed as jx  # noqa: E402
from juxtaposed._core import row_standardize  # noqa: E402

RESULTS = ROOT / "results" / "v3"
RESULTS.mkdir(exist_ok=True)


def hex_xy(array_row, array_col):
    """Euclidean coordinates in units of the 100 µm centre-to-centre spot spacing."""
    return np.column_stack([np.asarray(array_col) / 2.0, np.asarray(array_row) * np.sqrt(3) / 2.0])


def zcols(m):
    m = np.asarray(m, float)
    sd = m.std(axis=0)
    sd[sd == 0] = np.nan
    return (m - m.mean(axis=0)) / sd


class PairedStats:
    """Statistics and spectral-null tests for column-paired (x_k, y_k) replicates."""

    def __init__(self, graph, null=None):
        self.graph = jx._graph.check_graph(graph)
        self.n = self.graph.shape[0]
        self.lap, self.deg = jx.laplacian(self.graph)
        self.S = jx.smoother(self.graph)
        self.W = row_standardize(self.graph)
        rows = np.asarray(self.W.sum(axis=1)).ravel()
        self.lee_norm = float(np.sum(rows ** 2))
        self.s0 = float(self.W.sum())
        self.M_lee = (self.W.T @ self.W).tocsr()
        self.M_mor = (0.5 * (self.W + self.W.T)).tocsr()
        self.null = null
        self._mh = {}

    # ---------- observed statistics for paired columns
    def smooth(self, z, s):
        for _ in range(s):
            z = self.S @ z
        return z

    def j(self, zx, zy, s):
        ax, ay = self.smooth(zx, s), self.smooth(zy, s)
        lx, ly = self.lap @ ax, self.lap @ ay
        num = np.einsum("ik,ik->k", ax, ly)
        ex, ey = np.einsum("ik,ik->k", ax, lx), np.einsum("ik,ik->k", ay, ly)
        return -2 * num / (ex + ey)

    def lee(self, zx, zy):
        return np.einsum("ik,ik->k", self.W @ zx, self.W @ zy) / self.lee_norm

    def moran(self, zx, zy):
        return np.einsum("ik,ik->k", zx, self.M_mor @ zy) / self.s0

    def pearson(self, zx, zy):
        return np.einsum("ik,ik->k", zx, zy) / self.n

    # ---------- spectral null (paired columns)
    def mhat(self, key):
        if key not in self._mh:
            m = {"lee": self.M_lee, "moran": self.M_mor}[key]
            u = self.null.U
            mh = u.T @ (m @ u)
            self._mh[key] = 0.5 * (mh + mh.T)
        return self._mh[key]

    def spectral(self, zx, zy, scales=(0, 1)):
        """Return dict name -> (stat, z) using the exact sign-flip null."""
        u = self.null.U
        xh, yh = u.T @ zx, u.T @ zy
        out = {}
        for s in scales:
            h = self.null.spectral_filter(s)
            num = np.einsum("ik,ik->k", xh * h[:, None], yh)
            ex = np.einsum("ik,ik->k", xh * h[:, None], xh)
            ey = np.einsum("ik,ik->k", yh * h[:, None], yh)
            var = np.einsum("ik,ik->k", self.null.squared_coefficients(xh) * (h ** 2)[:, None],
                            self.null.squared_coefficients(yh))
            jv = -2 * num / (ex + ey)
            out[f"J{s}"] = (jv, jv / (2 * np.sqrt(var) / (ex + ey)))
        for key, norm in (("lee", self.lee_norm), ("moran", self.s0)):
            mh = self.mhat(key)
            st = np.einsum("ik,ik->k", xh, mh @ yh) / norm
            var = np.einsum("ik,ik->k", self.null.squared_coefficients(xh),
                            (mh * mh) @ self.null.squared_coefficients(yh)) / norm ** 2
            out[key] = (st, st / np.sqrt(var))
        st = np.einsum("ik,ik->k", xh, yh) / self.n
        var = np.einsum("ik,ik->k", self.null.squared_coefficients(xh),
                        self.null.squared_coefficients(yh)) / self.n ** 2
        out["pearson"] = (st, st / np.sqrt(var))
        return out

    # ---------- permutation null for one pair (y permuted), all statistics at once
    def permutation_pvalues(self, zx, zy, perms, scales=(0, 1)):
        """zx, zy: 1-D standardized vectors. perms: (B, n) permutation indices."""
        yp = zy[perms].T  # n x B
        res = {}
        # Lee / Moran / Pearson: linear in y_perm
        v_lee = self.W.T @ (self.W @ zx) / self.lee_norm
        v_mor = self.M_mor @ zx / self.s0
        for key, v in (("lee", v_lee), ("moran", v_mor), ("pearson", zx / self.n)):
            obs = zy @ v
            null = yp.T @ v
            res[key] = (1 + np.sum(np.abs(null) >= abs(obs) - 1e-15)) / (len(null) + 1)
        for s in scales:
            ax = self.smooth(zx, s)
            ay = self.smooth(zy, s)
            ap = self.smooth(yp, s)
            lx = self.lap @ ax
            ex = ax @ lx
            obs = -2 * (ay @ lx) / (ex + ay @ (self.lap @ ay))
            null = -2 * (ap.T @ lx) / (ex + np.einsum("ib,ib->b", ap, self.lap @ ap))
            res[f"J{s}"] = (1 + np.sum(np.abs(null) >= abs(obs) - 1e-15)) / (len(null) + 1)
        return res


def pearson_t_p(r, n):
    t = r * np.sqrt((n - 2) / np.clip(1 - r ** 2, 1e-300, None))
    return 2 * stats.t.sf(np.abs(t), n - 2)


def two_sided(z):
    return 2 * stats.norm.sf(np.abs(z))


def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h
