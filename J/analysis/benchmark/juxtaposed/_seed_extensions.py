"""Experimental seed-weighted extensions for the juxtaposition index J (post hoc; not part of the released packages).

Implements the two seed-weighted modes tested post hoc (Supplementary Results):
  1. Coherence-Gated Clustered Seed-J (`seed_soft_j` / `SeedJ_soft_clust`):
     Automatic curvature-based First-Elbow seed detection on ridge-regularized
     spot-level J, 2-hop spatial contiguity filtering, and coherence-gated
     blending with uniform edge weights (graceful fallback to standard J when
     no spatially coherent seed cluster is present).
  2. Pair-Independent Map-Level Fixed Seed-J (`MapFixedSpectralNull`):
     Fixes a single anatomical boundary weight field from a reference gene
     panel (independent of tested pairs), builds a fixed weighted Laplacian
     L_W, and computes exact analytical Spectral Sign-Flip Null variances in
     the unweighted tissue Laplacian eigenbasis U_std via the quadratic-form
     identity Var(x^T S^s L_W S^s y) = sum_{k,l} x_hat_k^2 y_hat_l^2 (u_k^T M u_l)^2.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter1d
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.stats import norm

from ._value import juxtaposed_value


# =============================================================================
# Shared Utilities
# =============================================================================
def bh_qvalues(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values for a 1D array of p-values."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1, dtype=float)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def build_heat_smoother(adjacency: sparse.csr_matrix) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, float]:
    """Return (L, S, degree, d_max) for uniform heat diffusion S = I - L/(2*d_max)."""
    A = adjacency.tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel().astype(float)
    L = sparse.diags(deg) - A
    d_max = float(deg.max()) if len(deg) else 1.0
    S = sparse.eye(A.shape[0], format="csr") - L / (2.0 * max(d_max, 1e-12))
    return L.tocsr(), S.tocsr(), deg, d_max


def diffuse_heat(X: np.ndarray, S: sparse.csr_matrix, scale: int) -> np.ndarray:
    """Apply `scale` steps of uniform heat diffusion S^scale X."""
    out = np.asarray(X, dtype=float).copy()
    for _ in range(int(scale)):
        out = S @ out
    return out


def regularized_node_j(
    x: np.ndarray,
    y: np.ndarray,
    edges: np.ndarray,
    degree: np.ndarray,
    ridge: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ridge-regularized spot-level average local J over incident edges.

    Returns (j_spot, dx, dy).
    """
    u, v = np.asarray(edges, dtype=int).T
    n = len(x)
    dx = x[v] - x[u]
    dy = y[v] - y[u]
    num_e = -2.0 * dx * dy
    den_e = dx * dx + dy * dy
    mean_energy = float(np.mean(den_e)) if len(den_e) else 0.0

    num_spot = np.bincount(u, weights=num_e, minlength=n) + np.bincount(v, weights=num_e, minlength=n)
    den_spot = np.bincount(u, weights=den_e, minlength=n) + np.bincount(v, weights=den_e, minlength=n)
    denom = den_spot + float(ridge) * degree * mean_energy
    j_spot = np.divide(num_spot, denom, out=np.zeros(n, dtype=float), where=denom > 0)
    return np.clip(j_spot, -1.0, 1.0), dx, dy


# =============================================================================
# Curvature-Based Automatic First-Elbow & 2-Hop Spatial Cluster Filter
# =============================================================================
def first_elbow_curvature(
    spot_scores: np.ndarray,
    min_seeds: int = 5,
    max_frac: float = 0.15,
) -> tuple[np.ndarray, dict]:
    """Automatically detect the first positive-curvature bend (First-Elbow) on
    the descending positive prefix of spot_scores, avoiding the global
    all-or-none tail split.
    """
    scores = np.asarray(spot_scores, dtype=float).ravel()
    n = len(scores)
    order = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="stable")
    sorted_vals = scores[order]

    pos_count = int(np.sum(sorted_vals > 0))
    max_seeds = max(min_seeds + 2, int(round(n * max_frac)))

    if pos_count <= min_seeds:
        k = min(min_seeds, n)
        return order[:k], dict(k_first=k, pos_count=pos_count)

    M = min(pos_count, max(max_seeds * 2, int(0.5 * n)))
    y_raw = sorted_vals[:M]
    y_span = float(y_raw[0] - y_raw[-1])
    if not np.isfinite(y_span) or y_span <= 1e-12:
        k = min(min_seeds, n)
        return order[:k], dict(k_first=k, pos_count=pos_count)

    u = np.linspace(0.0, 1.0, M)
    y = (y_raw - y_raw[-1]) / y_span
    y_smooth = gaussian_filter1d(y, sigma=max(1.5, 0.01 * M), mode="nearest")

    kneedle_d = (1.0 - u) - y_smooth
    du = 1.0 / max(M - 1, 1)
    dy = np.gradient(y_smooth, du)
    d2y = np.gradient(dy, du)
    kappa = d2y / np.power(1.0 + dy * dy, 1.5)

    hi = min(max_seeds, M - 2)
    lo = max(min_seeds - 1, 2)
    search_k = kappa[lo:hi]

    k_first = None
    if len(search_k) >= 3:
        is_peak = (search_k[1:-1] > search_k[:-2]) & (search_k[1:-1] >= search_k[2:])
        peak_pos = np.where(is_peak)[0] + 1 + lo
        if len(peak_pos) > 0:
            max_kappa = float(np.max(search_k))
            prominent = peak_pos[kappa[peak_pos] >= 0.25 * max(max_kappa, 1e-9)]
            if len(prominent) > 0:
                k_first = int(prominent[0]) + 1

    if k_first is None:
        sub_d = kneedle_d[lo:hi]
        if len(sub_d) > 0 and np.max(sub_d) > 0:
            k_first = int(np.argmax(sub_d >= 0.75 * np.max(sub_d))) + lo + 1
        else:
            k_first = int(np.clip(int(np.argmax(kneedle_d)) + 1, min_seeds, max_seeds))

    k_first = int(np.clip(k_first, min_seeds, max_seeds))
    return order[:k_first], dict(k_first=k_first, pos_count=pos_count)


def filter_clustered_seeds_2hop(
    seeds: np.ndarray,
    adjacency: sparse.csr_matrix,
    min_cluster_size: int = 5,
    two_hop: bool = True,
) -> tuple[np.ndarray, float, int]:
    """Retain only spatially contiguous seed clusters of size >= min_cluster_size.

    Returns (kept_seeds, coherence_fraction, n_clusters). When no cluster meets
    min_cluster_size, returns an empty array and coherence_fraction = 0.0 so
    callers can gracefully fall back to uniform J.
    """
    seeds = np.asarray(seeds, dtype=int)
    if len(seeds) == 0:
        return np.array([], dtype=int), 0.0, 0
    A = adjacency.tocsr()
    sub_A1 = A[seeds][:, seeds]
    if two_hop:
        sub_A2 = A[seeds, :] @ A[:, seeds]
        sub_adj = ((sub_A1 + sub_A2) > 0).astype(int)
    else:
        sub_adj = sub_A1

    n_comp, labels = connected_components(sub_adj, directed=False)
    counts = np.bincount(labels, minlength=n_comp)
    keep_comp = np.where(counts >= int(min_cluster_size))[0]
    if len(keep_comp) == 0:
        return np.array([], dtype=int), 0.0, 0

    kept = seeds[np.isin(labels, keep_comp)]
    coherence = float(len(kept)) / float(len(seeds))
    return kept, coherence, len(keep_comp)


# =============================================================================
# Mode 1: Coherence-Gated Clustered Seed-J (SeedJ_soft_clust)
# =============================================================================
def seed_soft_j(
    x: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    edges: np.ndarray,
    adjacency: sparse.csr_matrix,
    degree: np.ndarray,
    sigma: float = 4.0,
    ridge: float = 1.0,
    min_cluster_size: int = 5,
    two_hop: bool = True,
    power: float = 1.0,
    spacing: float | None = None,
) -> tuple[float, dict]:
    """Compute Mode 1: Coherence-Gated Clustered Seed-J (`SeedJ_soft_clust`).

    Inputs `x` and `y` should already be standardized (and diffused to the
    desired scale). Edge weights blend uniform weight 1.0 with Gaussian
    distance decay from clustered First-Elbow seeds using alpha = coherence^2:
        w_e = (1 - alpha) * 1.0 + alpha * exp(-d_e^2 / (2 * (sigma * spacing)^2)).
    If no seed cluster reaches `min_cluster_size`, alpha = 0 and the score
    equals standard uniform Boundary J.
    """
    coords = np.asarray(coords, dtype=float)
    edges = np.asarray(edges, dtype=int)
    midpoints = coords[edges].mean(axis=1)
    if spacing is None:
        edge_lens = np.linalg.norm(coords[edges[:, 1]] - coords[edges[:, 0]], axis=1)
        spacing = float(np.median(edge_lens)) if len(edge_lens) else 1.0

    j_spot, dx, dy = regularized_node_j(x, y, edges, degree, ridge=ridge)
    raw_seeds, elbow_info = first_elbow_curvature(j_spot)
    kept_seeds, coherence, n_clusters = filter_clustered_seeds_2hop(
        raw_seeds, adjacency, min_cluster_size=min_cluster_size, two_hop=two_hop
    )

    if len(kept_seeds) > 0:
        d_edge = cKDTree(coords[kept_seeds]).query(midpoints)[0]
        w_clust = np.exp(-0.5 * (d_edge / (sigma * spacing)) ** 2)
        alpha = coherence ** 2
        w_soft = (1.0 - alpha) * 1.0 + alpha * w_clust
        score = float(juxtaposed_value(dx, dy, weights=w_soft, power=power))
    else:
        alpha = 0.0
        score = float(juxtaposed_value(dx, dy, power=power))

    info = dict(
        k_raw=len(raw_seeds),
        k_kept=len(kept_seeds),
        coherence=coherence,
        alpha=alpha,
        n_clusters=n_clusters,
        seeds_kept=kept_seeds,
        j_std=float(juxtaposed_value(dx, dy, power=power)),
    )
    return score, info


# =============================================================================
# Mode 2: Map-Level Fixed Seed-J + Exact U_std Quadratic-Form Spectral Null
# =============================================================================
def build_map_fixed_weights(
    Z_ref_a: np.ndarray,
    Z_ref_b: np.ndarray | None,
    coords: np.ndarray,
    edges: np.ndarray,
    adjacency: sparse.csr_matrix,
    degree: np.ndarray,
    sigma: float = 4.0,
    floor: float = 0.05,
    ridge: float = 1.0,
    min_cluster_size: int = 5,
    max_pairs: int = 400,
    spacing: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Extract fixed anatomical boundary edge weights from a reference gene panel.

    If `Z_ref_b` is provided, averages positive regularized spot J over all
    cross-pairs between columns of `Z_ref_a` and `Z_ref_b` (ideal for a known
    interface such as tumor vs stroma or L6 vs WM). Otherwise averages over
    unique pairs within `Z_ref_a`.

    Returns (w_edge, selected_seeds, mean_spot_j, diagnostics).
    """
    coords = np.asarray(coords, dtype=float)
    edges = np.asarray(edges, dtype=int)
    midpoints = coords[edges].mean(axis=1)
    if spacing is None:
        edge_lens = np.linalg.norm(coords[edges[:, 1]] - coords[edges[:, 0]], axis=1)
        spacing = float(np.median(edge_lens)) if len(edge_lens) else 1.0

    n = coords.shape[0]
    acc = np.zeros(n, dtype=float)
    n_pairs = 0
    if Z_ref_b is not None:
        for a in range(Z_ref_a.shape[1]):
            for b in range(Z_ref_b.shape[1]):
                jsp, _, _ = regularized_node_j(Z_ref_a[:, a], Z_ref_b[:, b], edges, degree, ridge=ridge)
                acc += np.maximum(jsp, 0.0)
                n_pairs += 1
    else:
        p = Z_ref_a.shape[1]
        iu, ju = np.triu_indices(p, k=1)
        if len(iu) > max_pairs:
            rng = np.random.default_rng(20261122)
            pick = rng.choice(len(iu), size=max_pairs, replace=False)
            iu, ju = iu[pick], ju[pick]
        for a, b in zip(iu, ju):
            jsp, _, _ = regularized_node_j(Z_ref_a[:, a], Z_ref_a[:, b], edges, degree, ridge=ridge)
            acc += np.maximum(jsp, 0.0)
            n_pairs += 1

    mean_spot_j = acc / max(n_pairs, 1)
    raw_seeds, _ = first_elbow_curvature(mean_spot_j)
    kept_seeds, coherence, n_clusters = filter_clustered_seeds_2hop(
        raw_seeds, adjacency, min_cluster_size=min_cluster_size, two_hop=True
    )
    use_seeds = kept_seeds if len(kept_seeds) > 0 else raw_seeds
    d_edge = cKDTree(coords[use_seeds]).query(midpoints)[0]
    w_edge = float(floor) + (1.0 - float(floor)) * np.exp(-0.5 * (d_edge / (sigma * spacing)) ** 2)
    diag = dict(k_raw=len(raw_seeds), k_used=len(use_seeds), coherence=coherence, n_clusters=n_clusters)
    return w_edge, use_seeds, mean_spot_j, diag


def build_weighted_laplacian(n: int, edges: np.ndarray, w_edge: np.ndarray) -> sparse.csr_matrix:
    """Build symmetric positive-semidefinite weighted graph Laplacian L_W."""
    u, v = np.asarray(edges, dtype=int).T
    w = np.asarray(w_edge, dtype=float)
    row = np.r_[u, v]
    col = np.r_[v, u]
    data = np.r_[w, w]
    A_w = sparse.csr_matrix((data, (row, col)), shape=(n, n))
    deg_w = np.asarray(A_w.sum(axis=1)).ravel()
    return (sparse.diags(deg_w) - A_w).tocsr()


class MapFixedSpectralNull:
    """Exact analytical Spectral Sign-Flip Null for fixed edge-weighted J.

    Randomizes graph-Fourier coefficient signs in the *unweighted* tissue
    Laplacian eigenbasis U_std so each gene's stationary spatial autocorrelation
    is preserved, while evaluating the exact variance of the fixed weighted
    quadratic form x^T (S^s L_W S^s) y for all gene pairs in a single BLAS
    matrix multiplication.
    """

    def __init__(
        self,
        adjacency: sparse.csr_matrix,
        edges: np.ndarray,
        w_edge: np.ndarray,
        eigendecomp: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        self.adjacency = adjacency.tocsr()
        self.edges = np.asarray(edges, dtype=int)
        self.w_edge = np.asarray(w_edge, dtype=float)
        self.n = self.adjacency.shape[0]
        self.L_std, self.S, self.deg, self.d_max = build_heat_smoother(self.adjacency)
        self.L_w = build_weighted_laplacian(self.n, self.edges, self.w_edge)

        if eigendecomp is not None:
            self.mu_std, self.U_std = eigendecomp
        else:
            self.mu_std, self.U_std = eigh(self.L_std.toarray())

        # Base transformed weighted Laplacian in unweighted eigenbasis: U^T L_W U
        self._base_M = self.U_std.T @ (self.L_w @ self.U_std)
        self._scale_cache: dict[int, np.ndarray] = {}

    def _m_sq_for_scale(self, scale: int) -> np.ndarray:
        scale = int(scale)
        if scale not in self._scale_cache:
            s_eig = (1.0 - self.mu_std / (2.0 * self.d_max)) ** scale
            M_s = self._base_M * np.outer(s_eig, s_eig)
            self._scale_cache[scale] = M_s * M_s
        return self._scale_cache[scale]

    def test_all_pairs(self, Z: np.ndarray, scale: int = 1) -> dict[str, np.ndarray]:
        """Compute all-pairs Map-Fixed J, exact spectral z, two-sided p, and BH q.

        `Z` should be spot-standardized (mean 0, unit variance per gene column).
        """
        Z = np.asarray(Z, dtype=float)
        Zs = diffuse_heat(Z, self.S, scale) if scale > 0 else Z
        C = Zs.T @ (self.L_w @ Zs)
        E = np.diag(C)
        denom = E[:, None] + E[None, :]
        J = np.clip(np.divide(-2.0 * C, denom, out=np.zeros_like(C), where=denom > 0), -1.0, 1.0)

        XH2 = (self.U_std.T @ Z) ** 2
        M_sq = self._m_sq_for_scale(scale)
        Var = XH2.T @ (M_sq @ XH2)
        z = np.divide(-C, np.sqrt(np.maximum(Var, 1e-18)))
        np.fill_diagonal(z, 0.0)
        p = 2.0 * norm.sf(np.abs(z))
        np.fill_diagonal(p, 1.0)

        iu, ju = np.triu_indices(Z.shape[1], k=1)
        q_flat = bh_qvalues(p[iu, ju])
        q = np.ones_like(p)
        q[iu, ju] = q_flat
        q[ju, iu] = q_flat
        return dict(J=J, z=z, p=p, q=q)

    def test_cross_panels(self, Za: np.ndarray, Zb: np.ndarray, scale: int = 1) -> dict[str, np.ndarray]:
        """Compute exact spectral z and p for all cross-pairs between Za and Zb."""
        Za_s = diffuse_heat(Za, self.S, scale) if scale > 0 else Za
        Zb_s = diffuse_heat(Zb, self.S, scale) if scale > 0 else Zb
        C = Za_s.T @ (self.L_w @ Zb_s)
        Ea = np.sum(Za_s * (self.L_w @ Za_s), axis=0)
        Eb = np.sum(Zb_s * (self.L_w @ Zb_s), axis=0)
        denom = Ea[:, None] + Eb[None, :]
        J = np.clip(np.divide(-2.0 * C, denom, out=np.zeros_like(C), where=denom > 0), -1.0, 1.0)

        Ha2 = (self.U_std.T @ Za) ** 2
        Hb2 = (self.U_std.T @ Zb) ** 2
        M_sq = self._m_sq_for_scale(scale)
        Var = Ha2.T @ (M_sq @ Hb2)
        z = -C / np.sqrt(np.maximum(Var, 1e-18))
        p = 2.0 * norm.sf(np.abs(z))
        return dict(J=J, z=z, p=p)
