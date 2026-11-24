"""Protocol amendment v3, axis A: false-positive rates of the ecosystem gold-standard tests.

A1: the same independent Gaussian random fields as PROTOCOL_v2 (same seeds, same fields).
A2: real genes from two donors on the shared Visium grid, aligned (v2) and with section B rotated
    by 180 degrees on the array (removes shared array-position effects).

Each comparator is tested with its package's default procedure (see gold.py and the amendment).
J uses the exact spectral sign-flip null shipped in `juxtaposed`. For A2 the spectral null is also
applied to every comparator statistic, to separate the effect of the statistic from that of the null.
"""
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

from bench_calibration import (A1_SECTIONS, A2_PAIRS, NPAIRS, SEED, SETTINGS, B as B_V2,
                               gaussian_fields, top_moran_genes)
from common import PairedStats, RESULTS, hex_xy, jx, zcols
from datasets import load_dlpfc, lognorm
from gold import GoldStandards, rank_cols

B = 1000          # SpatialDM and MERINGUE permutations
B_LIANA = 100     # LIANA+ tutorial
B_LEE = 999       # number of permutations for the Lee reference
GOLD_SEED = 777


def two_sided(z):
    return 2 * stats.norm.sf(np.abs(z))


def t_two_sided(r, n):
    t = r * np.sqrt((n - 2) / np.clip(1 - r ** 2, 1e-300, None))
    return 2 * stats.t.sf(np.abs(t), n - 2)


class Setup:
    """Weights, coefficient maps and permutation machinery for one tissue layout."""

    def __init__(self, graph, xy, spectral=True, extra_spectral=True):
        self.n = graph.shape[0]
        self.ps = PairedStats(graph, jx.SpectralNull(graph) if spectral else None)
        self.gs = GoldStandards(xy, graph)
        self.lin = {  # linear-in-y coefficient maps: statistic = sum_i (C @ zx)_i * zy_i
            "spatialdm": (self.gs.W_sdm, self.gs.sdm_s0),
            "liana": (self.gs.W_liana.T.tocsr(), self.gs.liana_s0),
            "meringue": (self.ps.M_mor, self.ps.s0),
            "lee": (self.ps.M_lee, self.ps.lee_norm),
        }
        self._mh = {}
        if spectral and extra_spectral:
            u = self.ps.null.U
            for key, (c, _) in self.lin.items():
                cs = 0.5 * (c + c.T)
                mh = u.T @ (cs @ u)
                self._mh[key] = 0.5 * (mh + mh.T)

    def coef(self, zx):
        return {k: (c @ zx) / s for k, (c, s) in self.lin.items()}

    # ------------------------------------------------------------------ default tests
    def default_tests(self, X, Y, obs, null_ge, null_abs):
        """obs: dict stat -> observed; null_ge/null_abs: counts from permutations."""
        n = self.n
        zx, zy = zcols(X), zcols(Y)
        out = {}
        r = np.einsum("ik,ik->k", zx, zy) / n
        out["pearson_t"] = t_two_sided(r, n)
        rx, ry = zcols(rank_cols(X)), zcols(rank_cols(Y))
        rho = np.einsum("ik,ik->k", rx, ry) / n
        out["spearman_t"] = t_two_sided(rho, n)
        out["spatialdm_z"] = stats.norm.sf(obs["spatialdm"] / self.gs.sdm_sd)
        out["spatialdm_perm"] = null_ge["spatialdm"] / B
        out["liana_perm"] = (1 + null_abs["liana"]) / (B_LIANA + 1)
        out["meringue_perm"] = (1 + null_abs["meringue"]) / (B + 1)
        out["lee_perm"] = (1 + null_ge["lee"]) / (B_LEE + 1)
        _, zh = self.gs.hotspot(zx, zy)
        out["hotspot_z"] = stats.norm.sf(zh)
        stats_ = {"pearson_r": r, "spearman_rho": rho, "hotspot_Z": zh,
                  **{f"{k}_stat": v for k, v in obs.items()}}
        return out, stats_


def perm_counts_paired(coefs, zy, perms):
    """Permutation exceedance counts for paired columns (y permuted, x fixed)."""
    obs = {k: np.einsum("ik,ik->k", v, zy) for k, v in coefs.items()}
    k = zy.shape[1]
    ge = {key: np.zeros(k) for key in coefs}
    ab = {key: np.zeros(k) for key in coefs}
    lim = {"spatialdm": B, "meringue": B, "lee": B_LEE, "liana": B_LIANA}
    for b, p in enumerate(perms):
        yp = zy[p]
        for key, v in coefs.items():
            if b >= lim[key]:
                continue
            nv = np.einsum("ik,ik->k", v, yp)
            ge[key] += nv >= obs[key] - 1e-12
            ab[key] += np.abs(nv) >= np.abs(obs[key]) - 1e-12
    return obs, ge, ab


def perm_counts_cross(coefs, zy, perms):
    """Same, for all (x gene, y gene) combinations: coefficient maps n x Kx, zy n x Ky."""
    obs = {k: v.T @ zy for k, v in coefs.items()}
    ge = {k: np.zeros_like(o) for k, o in obs.items()}
    ab = {k: np.zeros_like(o) for k, o in obs.items()}
    lim = {"spatialdm": B, "meringue": B, "lee": B_LEE, "liana": B_LIANA}
    for b, p in enumerate(perms):
        yp = zy[p]
        for key, v in coefs.items():
            if b >= lim[key]:
                continue
            nv = v.T @ yp
            ge[key] += nv >= obs[key] - 1e-12
            ab[key] += np.abs(nv) >= np.abs(obs[key]) - 1e-12
    return obs, ge, ab


# ---------------------------------------------------------------------------------------- A1
def a1(section):
    t0 = time.time()
    counts, genes, meta = load_dlpfc(section)
    g = jx.visium_graph(meta.array_row.values, meta.array_col.values)
    xy = hex_xy(meta.array_row.values, meta.array_col.values)
    n = g.shape[0]
    st = Setup(g, xy, spectral=True, extra_spectral=False)
    rng = np.random.default_rng([SEED, int(section)])
    _ = np.stack([rng.permutation(n) for _ in range(B_V2)])  # replay v2's RNG stream exactly
    grng = np.random.default_rng([GOLD_SEED, int(section)])
    perms = np.stack([grng.permutation(n) for _ in range(B)])
    frames = []
    for name, nu, rho in SETTINGS:
        f = gaussian_fields(xy, nu, rho, 2 * NPAIRS, rng)
        X, Y = f[:, :NPAIRS], f[:, NPAIRS:]
        zx, zy = zcols(X), zcols(Y)
        obs, ge, ab = perm_counts_paired(st.coef(zx), zy, perms)
        out, sts = st.default_tests(X, Y, obs, ge, ab)
        spec = st.ps.spectral(zx, zy, scales=(1,))
        out["J1_spectral"] = two_sided(spec["J1"][1])
        sts["J1_stat"] = spec["J1"][0]
        df = pd.DataFrame({**out, **sts})
        df.insert(0, "pair", np.arange(NPAIRS))
        df.insert(0, "setting", name)
        df.insert(0, "unit", section)
        df.insert(0, "axis", "A1")
        df["n"] = n
        frames.append(df)
        print(f"A1 {section} {name} {time.time() - t0:.0f}s", flush=True)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------------------- A2
def a2(task):
    (sa, sb), mode = task
    t0 = time.time()
    ca, _, ma = load_dlpfc(sa)
    cb, _, mb = load_dlpfc(sb)
    ia, na, _ = top_moran_genes(sa)
    ib, nb, _ = top_moran_genes(sb)
    rb, cbcol = mb.array_row.values, mb.array_col.values
    if mode == "rotated":
        rb, cbcol = 77 - rb, 127 - cbcol
    ka = pd.Index(list(zip(ma.array_row, ma.array_col)))
    kb = pd.Index(list(zip(rb, cbcol)))
    com = ka.intersection(kb)
    pa, pb = ka.get_indexer(com), kb.get_indexer(com)
    xa = lognorm(ca)[:, ia].toarray()[pa]
    xb = lognorm(cb)[:, ib].toarray()[pb]
    rr = np.array([c[0] for c in com])
    cc = np.array([c[1] for c in com])
    g = jx.visium_graph(rr, cc)
    n = g.shape[0]
    st = Setup(g, hex_xy(rr, cc), spectral=True, extra_spectral=True)
    okx = np.isfinite(zcols(xa)).all(axis=0)
    oky = np.isfinite(zcols(xb)).all(axis=0)
    xa, xb, gx, gy = xa[:, okx], xb[:, oky], na[okx], nb[oky]
    zx, zy = zcols(xa), zcols(xb)
    grng = np.random.default_rng([GOLD_SEED, int(sa), int(sb), mode == "rotated"])
    perms = np.stack([grng.permutation(n) for _ in range(B)])
    obs, ge, ab = perm_counts_cross(st.coef(zx), zy, perms)
    ix, iy = np.meshgrid(np.arange(zx.shape[1]), np.arange(zy.shape[1]), indexing="ij")
    ix, iy = ix.ravel(), iy.ravel()
    flat = lambda d: {k: v[ix, iy] for k, v in d.items()}  # noqa: E731
    out, sts = st.default_tests(xa[:, ix], xb[:, iy], flat(obs), flat(ge), flat(ab))
    # spectral null: J (diagonal) and every comparator statistic (dense projected weights)
    u = st.ps.null.U
    xh, yh = u.T @ zx, u.T @ zy
    sqx, sqy = st.ps.null.squared_coefficients(xh), st.ps.null.squared_coefficients(yh)
    h = st.ps.null.spectral_filter(1)
    num = (xh * h[:, None]).T @ yh
    ex = np.einsum("ik,ik->k", xh * h[:, None], xh)
    ey = np.einsum("ik,ik->k", yh * h[:, None], yh)
    var = (sqx * (h ** 2)[:, None]).T @ sqy
    jv = -2 * num / (ex[:, None] + ey[None, :])
    jz = jv / (2 * np.sqrt(var) / (ex[:, None] + ey[None, :]))
    out["J1_spectral"] = two_sided(jz[ix, iy])
    sts["J1_stat"] = jv[ix, iy]
    for key, mh in st._mh.items():
        s = xh.T @ (mh @ yh)
        v = sqx.T @ ((mh * mh) @ sqy)
        out[f"{key}_spectral"] = two_sided((s / np.sqrt(v))[ix, iy])
    s = xh.T @ yh
    v = sqx.T @ sqy
    out["pearson_spectral"] = two_sided((s / np.sqrt(v))[ix, iy])
    df = pd.DataFrame({**out, **sts})
    df.insert(0, "gene_y", gy[iy])
    df.insert(0, "gene_x", gx[ix])
    df.insert(0, "pair", np.arange(len(ix)))
    df.insert(0, "setting", mode)
    df.insert(0, "unit", f"{sa}x{sb}")
    df.insert(0, "axis", "A2")
    df["n"] = n
    print(f"A2 {mode} {sa}x{sb} n={n} pairs={len(df)} {time.time() - t0:.0f}s", flush=True)
    return df


def run(task):
    kind, arg = task
    return a1(arg) if kind == "A1" else a2(arg)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    tasks = []
    if which in ("all", "A2"):
        tasks += [("A2", (p, m)) for m in ("aligned", "rotated") for p in A2_PAIRS]
    if which in ("all", "A1"):
        tasks += [("A1", s) for s in A1_SECTIONS]
    if which == "smoke":
        tasks = [("A2", (A2_PAIRS[0], "rotated"))]
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 2) as pool:
        frames = pool.map(run, tasks, chunksize=1)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(RESULTS / f"gold_calibration_{which}.csv.gz", index=False)
    print("saved", len(df))
