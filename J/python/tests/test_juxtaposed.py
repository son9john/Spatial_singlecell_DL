import numpy as np
import pytest
from scipy import sparse

import juxtaposed as jx
from juxtaposed._core import graph_gram


def torus_triangular(nr=12, nc=12):
    """Periodic triangular lattice (every node has 6 neighbours): a regular graph."""
    idx = np.arange(nr * nc).reshape(nr, nc)
    rows, cols = [], []
    for r in range(nr):
        for c in range(nc):
            for dr, dc in ((0, 1), (1, 0), (1, -1 if r % 2 == 0 else 1)):
                rows.append(idx[r, c])
                cols.append(idx[(r + dr) % nr, (c + dc) % nc])
    a = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(nr * nc, nr * nc))
    a = ((a + a.T) > 0).astype(float).tocsr()
    a.setdiag(0)
    a.eliminate_zeros()
    return a


@pytest.fixture(scope="module")
def grid():
    rng = np.random.default_rng(1)
    xy = np.column_stack(np.unravel_index(np.arange(400), (20, 20))).astype(float)
    g = jx.spatial_graph(xy, neighbors=4)
    x = np.sin(xy[:, 0] / 3) + 0.3 * rng.normal(size=400)
    y = np.cos(xy[:, 1] / 4) + 0.3 * rng.normal(size=400)
    return xy, g, np.column_stack([x, y, -x, 2 * x + 5, rng.normal(size=400)])


def test_basic_properties(grid):
    _, g, X = grid
    J = jx.juxtaposition_matrix(X, g, scale=0)
    assert np.allclose(np.diag(J), -1)
    assert np.isclose(J[0, 2], 1)            # x vs -x
    assert np.isclose(J[0, 3], -1)           # affine invariance
    assert np.allclose(J, J.T, equal_nan=True)
    assert np.nanmax(np.abs(J)) <= 1 + 1e-12


def test_edge_sum_definition_and_v01_parity(grid):
    _, g, X = grid
    z = jx.standardize(X[:, :2])
    up = sparse.triu(g, k=1).tocoo()
    dx = z[up.col, 0] - z[up.row, 0]
    dy = z[up.col, 1] - z[up.row, 1]
    expected = -2 * np.sum(dx * dy) / np.sum(dx ** 2 + dy ** 2)
    assert np.isclose(jx.juxtaposition_score(X[:, 0], X[:, 1], g, scale=0), expected)
    assert np.isclose(jx.juxtaposed_value(dx, dy), expected)
    # v0.1 graph_gram: G = Z^T (D z - A z)
    deg = np.asarray(g.sum(axis=1)).ravel()
    g01 = z.T @ (deg[:, None] * z - g @ z)
    assert np.allclose(graph_gram(z, g), 0.5 * (g01 + g01.T))
    assert np.allclose(jx.score_pairs(X[:, :2], g, steps=0), jx.juxtaposition_matrix(X[:, :2], g, scale=0))


def test_regular_graph_identity():
    """On a regular graph J = (I_xy - r) / (1 - (I_x + I_y)/2)."""
    a = torus_triangular()
    assert np.all(np.asarray(a.sum(axis=1)).ravel() == 6)
    rng = np.random.default_rng(3)
    s = jx.smoother(a)
    X = rng.normal(size=(a.shape[0], 3))
    X = s @ (s @ X) + 0.2 * X  # spatially structured
    J = jx.juxtaposition_matrix(X, a, scale=0)
    I = jx.moran_bv_matrix(X, a)
    r = jx.pearson_matrix(X)
    for i in range(3):
        for j in range(3):
            pred = (I[i, j] - r[i, j]) / (1 - 0.5 * (I[i, i] + I[j, j]))
            assert np.isclose(J[i, j], pred)


def test_spectral_statistic_matches_direct(grid):
    _, g, X = grid
    null = jx.SpectralNull(g)
    for s in (0, 1, 3):
        out = null.test_j(X, scale=s)
        J = jx.juxtaposition_matrix(X, g, scale=s)
        off = ~np.eye(X.shape[1], dtype=bool)
        assert np.allclose(out["stat"][off], J[off], atol=1e-10)
    lee = null.test_lee(X)["stat"]
    assert np.allclose(lee[off], jx.lee_l_matrix(X, g)[off], atol=1e-10)
    mor = null.test_moran(X)["stat"]
    assert np.allclose(mor[off], jx.moran_bv_matrix(X, g)[off], atol=1e-10)
    pr = null.test_pearson(X)["stat"]
    assert np.allclose(pr[off], jx.pearson_matrix(X)[off], atol=1e-10)


def _haar(m, rng):
    q, r = np.linalg.qr(rng.normal(size=(m, m)))
    return q * np.sign(np.diag(r))


def test_closed_form_variance_matches_monte_carlo():
    """Closed-form null SD equals the SD of explicit spectral randomizations (sign flips,
    plus Haar rotations inside repeated eigenspaces, which kNN graphs often have)."""
    rng = np.random.default_rng(11)
    xy = rng.uniform(0, 20, size=(300, 2))
    g = jx.spatial_graph(xy, neighbors=5)
    X = np.column_stack([np.sin(xy[:, 0] / 3) + 0.3 * rng.normal(size=300),
                         np.cos(xy[:, 1] / 4) + 0.3 * rng.normal(size=300)])
    null = jx.SpectralNull(g)
    z = jx.standardize(X)
    xh, yh = null.U.T @ z[:, 0], null.U.T @ z[:, 1]
    h = null.spectral_filter(1)
    ex, ey = np.sum(h * xh ** 2), np.sum(h * yh ** 2)
    w = jx._core.row_standardize(g)
    mh = null.U.T @ (w.T @ w).toarray() @ null.U
    norm_l = np.sum(np.asarray(w.sum(1)).ravel() ** 2)
    sims_j, sims_l = [], []
    for _ in range(6000):
        a = xh.copy()
        b = yh.copy()
        for st, sz in zip(null._starts, null._sizes):
            sl = slice(st, st + sz)
            if sz == 1:
                a[sl] *= rng.choice([-1.0, 1.0])
                b[sl] *= rng.choice([-1.0, 1.0])
            else:
                a[sl] = _haar(sz, rng) @ a[sl]
                b[sl] = _haar(sz, rng) @ b[sl]
        sims_j.append(-2 * np.sum(h * a * yh) / (ex + ey))
        sims_l.append(a @ mh @ b / norm_l)
    sd_j = null.test_j(X, scale=1)["sd"][0, 1]
    sd_l = null.test_lee(X)["sd"][0, 1]
    assert abs(np.std(sims_j) / sd_j - 1) < 0.05
    assert abs(np.std(sims_l) / sd_l - 1) < 0.05


def test_lee_l_formula(grid):
    _, g, X = grid
    x, y = X[:, 0], X[:, 1]
    w = jx._core.row_standardize(g).toarray()
    xt, yt = x - x.mean(), y - y.mean()
    n = len(x)
    expected = n / np.sum(w.sum(1) ** 2) * np.sum((w @ xt) * (w @ yt)) / (
        np.sqrt(np.sum(xt ** 2)) * np.sqrt(np.sum(yt ** 2)))
    assert np.isclose(jx.lee_l_matrix(X[:, :2], g)[0, 1], expected)


def test_visium_graph():
    g = jx.visium_graph([0, 0, 1, 0, 7], [0, 2, 1, 4, 1])
    edges = {tuple(sorted(e)) for e in zip(*sparse.triu(g, 1).nonzero())}
    assert edges == {(0, 1), (0, 2), (1, 2), (1, 3)}


def test_permutation_and_api(grid):
    _, g, X = grid
    obs, p = jx.permutation_test(X[:, 0], X[:, 1], g, statistic="J", n_perm=99)
    assert 0 < p <= 1 and np.isclose(obs, jx.juxtaposition_score(X[:, 0], X[:, 1], g, scale=1))
    for stat in ("L", "I", "r"):
        _, p = jx.permutation_test(X[:, 0], X[:, 1], g, statistic=stat, n_perm=99)
        assert 0 < p <= 1
    res = jx.juxtaposition(X, g, features=list("abcde"))
    tab = res.table()
    assert len(tab) == 10 and {"J", "z", "p", "q"} <= set(tab.columns)
    assert np.isclose(res["a", "c"], 1)
    top = res.top(3)
    assert top["J"].is_monotonic_decreasing
    cmp = jx.compare(X, g, features=list("abcde"))
    assert {"lee_L", "moran_I", "pearson_r", "neg_codispersion"} <= set(cmp.columns)


def test_zero_variance_feature(grid):
    _, g, X = grid
    X2 = np.column_stack([X[:, :2], np.ones(len(X))])
    J = jx.juxtaposition_matrix(X2, g)
    assert np.isnan(J[0, 2]) and np.isfinite(J[0, 1])


def test_local_edges(grid):
    _, g, X = grid
    res = jx.score_edges(X[:, :2], g, scale=0, ridge=1.0)
    assert res.endpoints.shape[1] == 2 and len(res.j) == sparse.triu(g, 1).nnz
    assert np.nanmax(np.abs(res.j)) <= 1


def test_degenerate_eigenspaces_are_basis_invariant():
    """On a symmetric lattice, the null SD must not depend on the eigenvector basis."""
    a = torus_triangular(8, 8)
    rng = np.random.default_rng(5)
    X = jx.smoother(a) @ rng.normal(size=(64, 3))
    null = jx.SpectralNull(a)
    assert null._degenerate
    sd1 = null.test_j(X, scale=1)["sd"]
    # rotate the basis inside every repeated eigenspace
    u = null.U.copy()
    for st, sz in zip(null._starts, null._sizes):
        if sz > 1:
            q, _ = np.linalg.qr(rng.normal(size=(sz, sz)))
            u[:, st:st + sz] = u[:, st:st + sz] @ q
    null.U = u
    sd2 = null.test_j(X, scale=1)["sd"]
    off = ~np.eye(3, dtype=bool)
    assert np.allclose(sd1[off], sd2[off], rtol=1e-8)


# ----------------------------------------------------------------------------- v0.3 additions
def _v01_reference_edges(values, graph, steps, ridge, power):
    """Independent copy of the v0.1 (juxtaposed preprint) local estimator."""
    x = np.asarray(values, float)
    x = (x - x.mean(0)) / x.std(0)
    a = sparse.csr_matrix(graph, dtype=float)
    deg = np.asarray(a.sum(1)).ravel()
    inv = np.divide(1, deg, out=np.zeros_like(deg), where=deg > 0)
    walk = (sparse.diags(np.where(deg > 0, 0.5, 1.0)) + 0.5 * a.multiply(inv[:, None])).tocsr()
    for _ in range(steps):
        x = walk @ x
    i, j = sparse.triu(a, k=1, format="csr").nonzero()
    w = np.asarray(a[i, j]).ravel()
    n = len(x)

    def pool(v):
        t = np.bincount(i, weights=v, minlength=n) + np.bincount(j, weights=v, minlength=n)
        return t[i] + t[j]

    dx, dy = x[j, 0] - x[i, 0], x[j, 1] - x[i, 1]
    ax, ay = pool(w * dx ** 2), pool(w * dy ** 2)
    cross, mass = -2 * pool(w * dx * dy), pool(w)
    ref = np.dot(w, dx ** 2 + dy ** 2) / w.sum()
    if power == 1:
        base = 0.5 * (ax + ay)
    elif power == 0:
        base = np.sqrt(ax * ay)
    else:
        base = (0.5 * (ax ** power + ay ** power)) ** (1 / power)
    val = np.divide(0.5 * cross, base + 0.5 * ridge * mass * ref, where=base > 0,
                    out=np.full_like(cross, np.nan))
    return np.column_stack([i, j]), np.clip(val, -1, 1), cross / mass


@pytest.mark.parametrize("power", [1.0, 0.5, 0.0])
def test_local_parity_with_preprint_v01(grid, power):
    _, g, X = grid
    ends, ref, q = _v01_reference_edges(X[:, [0, 1]], g, steps=4, ridge=1.0, power=power)
    res = jx.score_edges(X[:, [0, 1]], g, scale=4, ridge=1.0, power=power, smoother_kind="lazy")
    assert np.array_equal(res.endpoints, ends)
    assert np.allclose(res.j, ref, atol=1e-12, equal_nan=True)
    assert np.allclose(res.signed_evidence, q, atol=1e-12)
    nodes = res.to_nodes(g.shape[0])
    assert nodes.shape == (g.shape[0],) and np.isfinite(nodes).all()


def test_power_mean_family(grid):
    _, g, X = grid
    j1 = jx.juxtaposition_matrix(X, g, scale=1)
    j0 = jx.juxtaposition_matrix(X, g, scale=1, power=0.0)
    cod = jx.codispersion_matrix(X, g, scale=1)
    assert np.allclose(j0, cod, equal_nan=True)
    jh = jx.juxtaposition_matrix(X, g, scale=1, power=0.5)
    ok = np.isfinite(j1)
    # M_0 <= M_0.5 <= M_1, so |J_1| <= |J_0.5| <= |J_0|
    assert np.all(np.abs(j1[ok]) <= np.abs(jh[ok]) + 1e-12)
    assert np.all(np.abs(jh[ok]) <= np.abs(j0[ok]) + 1e-12)
    null = jx.SpectralNull(g)
    z1 = null.test_j(X, scale=1)["z"]
    zh = null.test_j(X, scale=1, power=0.5)["z"]
    assert np.allclose(z1, zh, equal_nan=True)
    m = jx.power_mean(np.array([1.0, 4.0, 0.0]), np.array([9.0, 4.0, 2.0]), 0.5)
    assert np.allclose(m, [((1 + 3) / 2) ** 2, 4.0, 0.5])


def test_normalized_laplacian_identity_on_irregular_graph():
    rng = np.random.default_rng(5)
    xy = rng.uniform(0, 30, size=(500, 2))
    g = jx.spatial_graph(xy, neighbors=7)  # irregular degrees
    X = np.column_stack([np.sin(xy[:, 0] / 4) + 0.4 * rng.normal(size=500),
                         np.cos(xy[:, 1] / 5) + 0.4 * rng.normal(size=500)])
    z = jx.standardize(X)
    deg = np.asarray(g.sum(1)).ravel()
    dinv = sparse.diags(1 / np.sqrt(deg))
    wsym = dinv @ g @ dinv
    n = len(z)
    ixy = z[:, 0] @ (wsym @ z[:, 1]) / n
    ixx = z[:, 0] @ (wsym @ z[:, 0]) / n
    iyy = z[:, 1] @ (wsym @ z[:, 1]) / n
    r = z[:, 0] @ z[:, 1] / n
    expected = 2 * (ixy - r) / ((1 - ixx) + (1 - iyy))
    got = jx.juxtaposition_score(X[:, 0], X[:, 1], g, scale=0, laplacian="normalized")
    assert abs(got - expected) < 1e-12
    # the combinatorial J differs on irregular graphs, and the null must match the Laplacian
    null = jx.SpectralNull(g, laplacian="normalized")
    res = jx.juxtaposition(X, g, scale=1, laplacian="normalized", null=null)
    assert abs(res.J[0, 1] - jx.juxtaposition_score(X[:, 0], X[:, 1], g, scale=1,
                                                    laplacian="normalized")) < 1e-12
    with pytest.raises(ValueError):
        jx.juxtaposition(X, g, scale=1, null=null)


def test_normalized_null_variance_matches_sign_flips():
    rng = np.random.default_rng(9)
    xy = rng.uniform(0, 15, size=(220, 2))
    g = jx.spatial_graph(xy, neighbors=6)
    X = np.column_stack([np.sin(xy[:, 0] / 2) + 0.3 * rng.normal(size=220),
                         np.cos(xy[:, 1] / 3) + 0.3 * rng.normal(size=220)])
    null = jx.SpectralNull(g, laplacian="normalized")
    z = jx.standardize(X)
    xh, yh = null.U.T @ z[:, 0], null.U.T @ z[:, 1]
    h = null.spectral_filter(2)
    ex, ey = np.sum(h * xh ** 2), np.sum(h * yh ** 2)
    sims = []
    for _ in range(5000):
        a = xh.copy()
        for st, sz in zip(null._starts, null._sizes):
            sl = slice(st, st + sz)
            a[sl] = a[sl] * rng.choice([-1.0, 1.0]) if sz == 1 else _haar(sz, rng) @ a[sl]
        sims.append(-2 * np.sum(h * a * yh) / (ex + ey))
    assert abs(np.std(sims) / null.test_j(X, scale=2)["sd"][0, 1] - 1) < 0.05


def test_score_pairs_v01_parity(grid):
    _, g, X = grid
    x = (X - X.mean(0)) / X.std(0)
    walk = jx.lazy_walk(g)
    for _ in range(2):
        x = walk @ x
    lap, _ = jx.laplacian(g)
    gram = x.T @ (lap @ x)
    gram = 0.5 * (gram + gram.T)
    e = np.diag(gram)
    expected = np.clip(-gram / (0.5 * (e[:, None] + e[None, :]) * (1 + 0.5)), -1, 1)
    assert np.allclose(jx.score_pairs(X, g, steps=2, ridge=0.5), expected)


def test_fast_gram_paths_match_direct_product():
    """Cholesky + syrk, float32 and the spectral Gram all reproduce Zᵀ L Z."""
    from juxtaposed._fast import LaplacianFactor, gram
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 25, size=(400, 2))
    g = jx.spatial_graph(xy, neighbors=6, radius=2.5)  # radius leaves some isolated nodes
    Z = jx.standardize(rng.normal(size=(400, 120)) + np.sin(xy[:, :1] / 3))
    for kind in ("combinatorial", "normalized"):
        lap, _ = (jx.laplacian(g) if kind == "combinatorial" else jx.normalized_laplacian(g))
        ref = Z.T @ (lap @ Z)
        fac = LaplacianFactor(lap)
        assert fac.ok
        got = gram(Z, lap, factor=fac, method="factor")
        assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-8
        got32 = gram(Z, lap, method="factor", dtype=np.float32)
        assert np.abs(got32 - ref).max() / np.abs(ref).max() < 1e-4
    j_fast = jx.juxtaposition_matrix(Z, g, scale=1)
    lap, _ = jx.laplacian(g)
    zs = jx.smoother(g) @ Z
    gg = zs.T @ (lap @ zs)
    e = np.diag(gg)
    j_ref = np.clip(-2 * gg / (e[:, None] + e[None, :]), -1, 1)
    assert np.allclose(j_fast, j_ref, atol=1e-8)
    res = jx.juxtaposition(Z, g, scale=1)
    off = ~np.eye(Z.shape[1], dtype=bool)
    assert np.allclose(res.J[off], j_ref[off], atol=1e-8)
    res32 = jx.juxtaposition(Z, g, scale=1, dtype=np.float32)
    assert np.allclose(res32.J[off], j_ref[off], atol=1e-4)
    assert np.allclose(res32.z[off], res.z[off], atol=1e-2)
