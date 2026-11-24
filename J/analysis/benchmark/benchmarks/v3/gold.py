"""Re-implementations of the bivariate statistics shipped by the Scanpy/scverse and Seurat/Bioconductor
ecosystems, with each package's DEFAULT significance test.

Sources (formulas checked against the package code/docs, November 2026):
* Pearson / Spearman correlation with t-test: what Scanpy and Seurat users compute for two genes
  (scipy.stats.pearsonr / spearmanr; Seurat FeatureScatter / cor.test). No spatial term.
* SpatialDM 0.3 (`weight_matrix` + `spatialdm_global`, default method='z-score'): RBF kernel
  w = exp(-d^2 / (2 l^2)) on the 186 nearest neighbours (incl. self), entries < cutoff set to 0,
  diagonal 1 for spot data, weights rescaled to sum N; global R = sum_i (W x)_i y_i / S0 on
  standardized x, y; analytical z with Var(R) = [N^2 sum w_ij w_ji - 2N sum r_i c_i + S0^2] /
  [N^2 (N-1)^2]; one-sided p = norm.sf(z).
* LIANA+ 2.x (`li.pp.spatial_neighbors(bandwidth, cutoff=0.1, kernel='gaussian', set_diag=True)` +
  `li.mt.bivariate(global_name='morans', n_perms=100)`): Gaussian kernel exp(-d^2/(2 bw^2)),
  cutoff 0.1, max 100 neighbours, diagonal 1; global bivariate Moran's R on standardized values;
  p-value by shuffling spot labels (n_perms = 100, as in the official tutorial).
* Hotspot (`create_knn_graph(n_neighbors=30)`, `compute_local_correlations`): H = x^T (A + A^T) y on a
  binary kNN graph; conditional second moment EG2(x) = ||(A + A^T) x||^2; Z computed in both
  orientations and the smaller |Z| kept (centered, conditional, symmetric variant).
* spdep `lee.mc`: Lee's L, row-standardized binary neighbours, 999 permutations.
* MERINGUE spatialCrossCorTest: bivariate Moran's I on binary neighbours, permutations.
* Giotto `detectSpatialCorFeats(method = 'network')`: correlation of neighbour-averaged expression
  (score only; no p-value).
* LIANA+ local 'cosine' (tutorial default local metric): spatially weighted cosine per spot; we
  use its mean over spots as the pair score (score only).
"""
import numpy as np
from scipy import sparse, stats
from scipy.spatial import cKDTree


def zcols(m):
    m = np.asarray(m, float)
    sd = m.std(axis=0)
    sd[sd == 0] = np.nan
    return (m - m.mean(axis=0)) / sd


def rank_cols(m):
    return np.column_stack([stats.rankdata(c) for c in np.asarray(m, float).T])


# ----------------------------------------------------------------------------- weight matrices
def spatialdm_weights(coords, l, cutoff, n_neighbors=186, single_cell=False):
    n = len(coords)
    k = min(n_neighbors, n)
    d, idx = cKDTree(coords).query(coords, k=k)
    rows = np.repeat(np.arange(n), k)
    w = np.exp(-d.ravel() ** 2 / (2 * l ** 2))
    w[w < cutoff] = 0
    W = sparse.csr_matrix((w, (rows, idx.ravel())), shape=(n, n))
    W.setdiag(0.0 if single_cell else 1.0)
    W.eliminate_zeros()
    return (W * (n / W.sum())).tocsr()


def liana_weights(coords, bandwidth, cutoff=0.1, max_neighbours=100, set_diag=True):
    n = len(coords)
    k = min(max_neighbours, n)
    d, idx = cKDTree(coords).query(coords, k=k)
    rows = np.repeat(np.arange(n), k)
    w = np.exp(-d.ravel() ** 2 / (2 * bandwidth ** 2))
    w[w < cutoff] = 0
    W = sparse.csr_matrix((w, (rows, idx.ravel())), shape=(n, n))
    W.setdiag(1.0 if set_diag else 0.0)
    W.eliminate_zeros()
    return W.tocsr()


def hotspot_weights(coords, n_neighbors=30):
    n = len(coords)
    k = min(n_neighbors, n - 1)
    _, idx = cKDTree(coords).query(coords, k=k + 1)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return (A + A.T).tocsr()  # non-redundant weights moved onto one orientation: x^T (A + A^T) y


def row_normalize(A):
    d = np.asarray(A.sum(axis=1)).ravel()
    inv = np.divide(1.0, d, out=np.zeros_like(d), where=d > 0)
    return (sparse.diags(inv) @ A).tocsr()


# ----------------------------------------------------------------------------- tests (paired columns)
class GoldStandards:
    """Default tests of the ecosystem methods for column-paired (x_k, y_k) replicates."""

    def __init__(self, coords_units, hex_graph, px_per_unit=137.0, spatialdm_l=1.2, spatialdm_cutoff=0.2,
                 liana_bandwidth_px=200.0, single_cell=False):
        self.n = len(coords_units)
        self.W_sdm = spatialdm_weights(coords_units, spatialdm_l, spatialdm_cutoff, single_cell=single_cell)
        Wt = self.W_sdm.T.tocsr()
        N = self.n
        s0 = self.W_sdm.sum()
        r = np.asarray(self.W_sdm.sum(axis=1)).ravel()
        c = np.asarray(self.W_sdm.sum(axis=0)).ravel()
        num = N ** 2 * self.W_sdm.multiply(Wt).sum() - 2 * N * np.sum(r * c) + s0 ** 2
        self.sdm_s0 = s0
        self.sdm_sd = np.sqrt(num / (N ** 2 * (N - 1) ** 2))
        self.W_liana = liana_weights(coords_units, liana_bandwidth_px / px_per_unit,
                                     set_diag=not single_cell)
        self.liana_s0 = self.W_liana.sum()
        self.S_hot = hotspot_weights(coords_units)
        self.G_avg = row_normalize(hex_graph)

    # -- statistics
    def spatialdm_R(self, zx, zy):
        return np.einsum("ik,ik->k", self.W_sdm @ zx, zy) / self.sdm_s0

    def liana_R(self, zx, zy):
        return np.einsum("ik,ik->k", zx, self.W_liana @ zy) / self.liana_s0

    def hotspot(self, zx, zy):
        H = np.einsum("ik,ik->k", zx, self.S_hot @ zy)
        eg2x = np.einsum("ik,ik->k", self.S_hot @ zx, self.S_hot @ zx)
        eg2y = np.einsum("ik,ik->k", self.S_hot @ zy, self.S_hot @ zy)
        zx_, zy_ = H / np.sqrt(eg2x), H / np.sqrt(eg2y)
        return H, np.where(np.abs(zx_) < np.abs(zy_), zx_, zy_)

    # -- default p-values
    def pvalues(self, X, Y, rng, liana_perms=100):
        """X, Y: raw (log-normalized) paired columns. Returns dict method -> (score, p)."""
        n = self.n
        zx, zy = zcols(X), zcols(Y)
        out = {}
        r = np.einsum("ik,ik->k", zx, zy) / n
        t = r * np.sqrt((n - 2) / np.clip(1 - r ** 2, 1e-300, None))
        out["pearson_ttest"] = (r, 2 * stats.t.sf(np.abs(t), n - 2))
        rx, ry = zcols(rank_cols(X)), zcols(rank_cols(Y))
        rho = np.einsum("ik,ik->k", rx, ry) / n
        t = rho * np.sqrt((n - 2) / np.clip(1 - rho ** 2, 1e-300, None))
        out["spearman_ttest"] = (rho, 2 * stats.t.sf(np.abs(t), n - 2))
        R = self.spatialdm_R(zx, zy)
        out["spatialdm_ztest"] = (R, stats.norm.sf(R / self.sdm_sd))
        Rl = self.liana_R(zx, zy)
        exceed = np.zeros(len(Rl))
        for _ in range(liana_perms):
            perm = rng.permutation(n)
            Rp = self.liana_R(zx[perm], zy[perm])
            exceed += np.abs(Rp) >= np.abs(Rl)
        out["liana_morans_perm100"] = (Rl, exceed / liana_perms)
        H, Z = self.hotspot(zx, zy)
        out["hotspot_z"] = (H, 2 * stats.norm.sf(np.abs(Z)))
        return out

    # -- all-pairs score matrices (for accuracy tasks)
    def score_matrices(self, X):
        n = self.n
        Z = zcols(X)
        Z = np.nan_to_num(Z)
        sc = {}
        sc["pearson"] = Z.T @ Z / n
        R = zcols(rank_cols(X))
        sc["spearman"] = np.nan_to_num(R).T @ np.nan_to_num(R) / n
        M = Z.T @ (self.W_sdm @ Z) / self.sdm_s0
        sc["spatialdm_R"] = 0.5 * (M + M.T)
        M = Z.T @ (self.W_liana @ Z) / self.liana_s0
        sc["liana_morans_R"] = 0.5 * (M + M.T)
        SZ = self.S_hot @ Z
        H = Z.T @ SZ
        eg2 = np.einsum("ik,ik->k", SZ, SZ)
        zx_ = H / np.sqrt(eg2)[:, None]
        zy_ = H / np.sqrt(eg2)[None, :]
        sc["hotspot_Z"] = np.where(np.abs(zx_) < np.abs(zy_), zx_, zy_)
        A = self.G_avg @ np.asarray(X, float)
        sc["giotto_spatial_cor"] = np.nan_to_num(zcols(A)).T @ np.nan_to_num(zcols(A)) / n
        Xp = np.clip(np.asarray(X, float), 0, None)
        Wl = self.W_liana
        num = np.zeros((X.shape[1], X.shape[1]))
        # mean over spots of local cosine: sum_j w_ij x_j y_j / sqrt(sum_j w_ij x_j^2 sum_j w_ij y_j^2)
        Wx2 = Wl @ (Xp ** 2)
        for a in range(Xp.shape[1]):
            prod = Wl @ (Xp[:, [a]] * Xp)
            den = np.sqrt(Wx2[:, [a]] * Wx2)
            num[a] = np.mean(np.divide(prod, den, out=np.zeros_like(prod), where=den > 0), axis=0)
        sc["liana_cosine_mean"] = num
        return sc
