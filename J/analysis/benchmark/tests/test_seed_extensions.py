"""Unit tests for juxtaposed._seed_extensions (Mode 1 & Mode 2)."""
import unittest

import numpy as np
from scipy import sparse

import juxtaposed as jx


def _grid(rows=14, cols=16):
    r, c = np.indices((rows, cols))
    coords = np.column_stack([r.ravel(), c.ravel()]).astype(float)
    keys = np.arange(r.size).reshape((rows, cols))
    edges = np.vstack((
        np.column_stack([keys[:, :-1].ravel(), keys[:, 1:].ravel()]),
        np.column_stack([keys[:-1].ravel(), keys[1:].ravel()]),
    ))
    u, v = edges.T
    n = len(coords)
    A = sparse.csr_matrix((np.ones(2 * len(u)), (np.r_[u, v], np.r_[v, u])), shape=(n, n))
    deg = np.asarray(A.sum(axis=1)).ravel()
    return coords, edges, A, deg


class SeedExtensionsTests(unittest.TestCase):
    def test_seed_soft_j_bounds_and_fallback(self):
        coords, edges, A, deg = _grid()
        rng = np.random.default_rng(42)
        x = rng.normal(size=len(coords))
        y = rng.normal(size=len(coords))
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()

        score, info = jx.seed_soft_j(x, y, coords, edges, A, deg, min_cluster_size=50)
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)
        self.assertEqual(info["k_kept"], 0)
        self.assertEqual(info["alpha"], 0.0)
        self.assertTrue(np.isclose(score, info["j_std"], atol=1e-12))

    def test_map_fixed_spectral_null_matches_monte_carlo(self):
        coords, edges, A, deg = _grid(10, 12)
        n = len(coords)
        rng = np.random.default_rng(7)
        w_edge = rng.uniform(0.1, 1.0, size=len(edges))
        null = jx.MapFixedSpectralNull(A, edges, w_edge)

        Z = rng.normal(size=(n, 3))
        Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
        out = null.test_all_pairs(Z, scale=1)
        self.assertEqual(out["J"].shape, (3, 3))
        self.assertTrue(np.all((out["J"] >= -1.0) & (out["J"] <= 1.0)))
        self.assertTrue(np.allclose(out["J"], out["J"].T))

        # Verify exact analytical variance against 3,000 Monte Carlo sign flips in U_std
        L_w = jx.build_weighted_laplacian(n, edges, w_edge)
        Zs = jx.diffuse_heat(Z, null.S, 1)
        C_obs = float(Zs[:, 0] @ (L_w @ Zs[:, 1]))
        var_ana = (C_obs / out["z"][0, 1]) ** 2

        xh = null.U_std.T @ Z[:, 0]
        yh = null.U_std.T @ Z[:, 1]
        s_eig = 1.0 - null.mu_std / (2.0 * null.d_max)
        M_s = null._base_M * np.outer(s_eig, s_eig)
        mc_vals = [
            float((rng.choice([-1.0, 1.0], size=n) * xh) @ M_s @ (rng.choice([-1.0, 1.0], size=n) * yh))
            for _ in range(3000)
        ]
        var_mc = float(np.var(mc_vals))
        rel_err = abs(var_ana - var_mc) / var_ana
        self.assertLess(rel_err, 0.08, f"Analytical var {var_ana} vs MC {var_mc} (rel_err={rel_err:.3f})")


if __name__ == "__main__":
    unittest.main()
