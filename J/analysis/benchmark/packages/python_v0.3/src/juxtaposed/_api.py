"""High-level entry points: one call from a matrix (or AnnData) to a ranked pair table."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._core import (
    codispersion_matrix, juxtaposition_matrix, lee_l_matrix, moran_bv_matrix, pearson_matrix,
    power_mean, standardize as _standardize,
)
from ._graph import check_graph, laplacian, lazy_walk, spatial_graph, visium_graph
from ._inference import SpectralNull, bh_qvalues

__all__ = ["JResult", "juxtaposition", "compare", "from_anndata", "score_pairs"]


@dataclass
class JResult:
    """All-pairs J with optional calibrated statistics.

    Attributes
    ----------
    J : (p, p) matrix of juxtaposition scores (+1 opposing, -1 concordant changes).
    features : feature names in matrix order.
    scale : number of smoothing steps used.
    z, p, q : spectral-null z-scores, two-sided p-values and Benjamini–Hochberg q-values
        (q computed over the p(p-1)/2 unique pairs); ``None`` when ``test=False``.
    """

    J: np.ndarray
    features: list
    scale: int
    z: np.ndarray | None = None
    p: np.ndarray | None = None
    q: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    def _index(self, name):
        return self.features.index(name) if not isinstance(name, (int, np.integer)) else int(name)

    def __getitem__(self, pair):
        a, b = pair
        return float(self.J[self._index(a), self._index(b)])

    def table(self, sort: str = "J", ascending: bool = False):
        """Unique pairs as a pandas DataFrame (or a dict of arrays without pandas)."""
        iu, ju = np.triu_indices(len(self.features), k=1)
        cols = {"feature_x": np.asarray(self.features, dtype=object)[iu],
                "feature_y": np.asarray(self.features, dtype=object)[ju],
                "J": self.J[iu, ju]}
        for key in ("z", "p", "q"):
            val = getattr(self, key)
            if val is not None:
                cols[key] = val[iu, ju]
        for key, mat in self.extra.items():
            cols[key] = mat[iu, ju]
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover
            order = np.argsort(cols[sort])
            order = order if ascending else order[::-1]
            return {k: v[order] for k, v in cols.items()}
        df = pd.DataFrame(cols)
        return df.sort_values(sort, ascending=ascending, na_position="last").reset_index(drop=True)

    def top(self, n: int = 20, direction: str = "opposing", max_q: float | None = None):
        """Top pairs: ``direction='opposing'`` (largest J) or ``'concordant'`` (smallest J)."""
        df = self.table(sort="J", ascending=(direction == "concordant"))
        if max_q is not None and "q" in df:
            df = df[df["q"] <= max_q]
        return df.head(n)


def _names(features, p):
    if features is None:
        return [f"f{k}" for k in range(p)]
    features = list(map(str, features))
    if len(features) != p:
        raise ValueError("features must have one name per column")
    return features


def juxtaposition(values, graph, *, features=None, scale: int = 1, test: bool = True,
                  null: SpectralNull | None = None, max_nodes: int = 20000,
                  power: float = 1.0, laplacian: str = "combinatorial", dtype=None,
                  backend: str = "cpu") -> JResult:
    """Compute J for every pair of features, with approximate spectral-null p-values by default.

    Parameters
    ----------
    values : (n, p) expression matrix (spots/cells by genes), e.g. log-normalized counts.
    graph : (n, n) symmetric adjacency, e.g. from :func:`spatial_graph` or :func:`visium_graph`.
    features : optional list of p feature names.
    scale : smoothing steps before differencing (0 = raw increments; default 1).
    test : if True, compute spectral-null z, p and BH q-values (needs n <= max_nodes).
    null : reuse a precomputed :class:`SpectralNull` for the same graph.
    power : power mean of the two gradient energies in the denominator (1 = J; 0 = negative
        codispersion). It changes the effect-size scale only; z, p and q are unchanged.
    laplacian : 'combinatorial' (default) or 'normalized'.
    dtype : ``np.float32`` for about twice the speed (relative error ~1e-6).
    backend : 'cpu' (default), 'auto', 'mlx' or 'torch' for the J matrix when
        ``test=False``; spectral inference runs on the CPU.
    """
    values = np.asarray(values, dtype=float)
    names = _names(features, values.shape[1])
    if not test:
        return JResult(juxtaposition_matrix(values, graph, scale=scale, power=power, laplacian=laplacian,
                                            dtype=dtype, backend=backend), names, scale)
    if backend not in ("cpu", "numpy", "auto"):
        raise ValueError("Spectral inference uses the CPU; use test=False for accelerator scores")
    if null is not None and getattr(null, "laplacian", "combinatorial") != laplacian:
        raise ValueError("the supplied SpectralNull uses a different laplacian")
    null = null or SpectralNull(graph, max_nodes=max_nodes, laplacian=laplacian)
    out = null.test_j(values, scale=scale, power=power, dtype=dtype)
    # J itself comes from the same spectral Gram (identical to the edge form up to rounding)
    jmat = out["stat"].copy()
    ok = np.isfinite(out["sd"]).any(axis=0)  # features with nonzero variance
    np.fill_diagonal(jmat, np.where(ok, -1.0, np.nan))
    iu = np.triu_indices(len(names), k=1)
    q = np.full_like(out["p"], np.nan)
    q[iu] = bh_qvalues(out["p"][iu])
    q = np.where(np.isnan(q), q.T, q)
    return JResult(jmat, names, scale, out["z"], out["p"], q)


def compare(values, graph, *, features=None, scale: int = 1):
    """J alongside Lee's L, bivariate Moran's I, Pearson r and codispersion (the L–J map)."""
    names = _names(features, np.asarray(values).shape[1])
    res = JResult(juxtaposition_matrix(values, graph, scale=scale), names, scale)
    res.extra = {
        "lee_L": lee_l_matrix(values, graph),
        "moran_I": moran_bv_matrix(values, graph),
        "pearson_r": pearson_matrix(values),
        "neg_codispersion": codispersion_matrix(values, graph, scale=scale),
    }
    return res.table(sort="J")


def from_anndata(adata, *, genes=None, layer: str | None = None, spatial_key: str = "spatial",
                 neighbors: int = 6, radius: float | None = None, visium: bool | None = None):
    """Extract (values, graph, gene names) from an AnnData object without importing anndata.

    Uses Visium hexagonal adjacency when ``adata.obs`` has ``array_row``/``array_col`` (or
    ``visium=True``); otherwise a k-nearest-neighbour graph on ``adata.obsm[spatial_key]``.
    """
    names = list(adata.var_names) if genes is None else list(genes)
    sub = adata[:, names]
    mat = sub.layers[layer] if layer is not None else sub.X
    values = mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)
    obs = adata.obs
    use_visium = visium if visium is not None else ("array_row" in obs and "array_col" in obs)
    if use_visium:
        graph = visium_graph(obs["array_row"].to_numpy(), obs["array_col"].to_numpy())
    else:
        graph = spatial_graph(np.asarray(adata.obsm[spatial_key])[:, :2], neighbors=neighbors,
                              radius=radius)
    return np.asarray(values, dtype=float), graph, names


def score_pairs(values, graph, *, steps: int = 0, ridge: float = 0.0, standardize: bool = True,
                power: float = 1.0):
    """v0.1 compatibility: dense all-pairs J after ``steps`` lazy-random-walk steps.

    J = -G_xy / (M_p(G_xx, G_yy) + ridge (G_xx + G_yy) / 2) with G = Zᵀ (D - A) Z. With the
    whole graph as the window the ridge rescales every pair by the same factor.
    """
    x = np.asarray(values, dtype=float)
    if standardize:
        x = _standardize(x)
    a = check_graph(graph, x.shape[0])
    walk = lazy_walk(a)
    for _ in range(int(steps)):
        x = walk @ x
    lap, _ = laplacian(a)
    g = x.T @ (lap @ x)
    g = 0.5 * (g + g.T)
    e = np.clip(np.diag(g), 0, None)
    base = power_mean(e[:, None], e[None, :], power)
    out = np.divide(-g, base + ridge * 0.5 * (e[:, None] + e[None, :]), where=base > 0,
                    out=np.full_like(g, np.nan))
    return np.clip(out, -1, 1)
