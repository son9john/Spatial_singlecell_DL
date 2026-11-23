"""Axis A of PROTOCOL_v2: false-positive rates under spatial nulls.

A1: independent Gaussian random fields on real Visium spot layouts.
A2: real genes from two different donors aligned on the shared Visium array grid.
"""
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from scipy.spatial.distance import cdist

from common import PairedStats, RESULTS, hex_xy, jx, pearson_t_p, two_sided, zcols
from datasets import load_dlpfc, lognorm

SEED = 20261122
NPAIRS = 1000
B = 999
A1_SECTIONS = ("151507", "151669", "151673")
SETTINGS = [("white", None, None)] + [(f"matern{nu}_range{rho}", nu, rho)
                                       for nu in (0.5, 2.5) for rho in (1, 3, 10)]
A2_PAIRS = [("151507", "151673"), ("151508", "151674"), ("151509", "151669"),
            ("151510", "151670"), ("151671", "151675"), ("151672", "151676")]
METHODS = ["pearson", "lee", "moran", "J0", "J1"]


def matern(d, nu, rho):
    if nu == 0.5:
        return np.exp(-d / rho)
    if nu == 1.5:
        a = np.sqrt(3) * d / rho
        return (1 + a) * np.exp(-a)
    if nu == 2.5:
        a = np.sqrt(5) * d / rho
        return (1 + a + a * a / 3) * np.exp(-a)
    raise ValueError(nu)


def gaussian_fields(xy, nu, rho, k, rng):
    n = len(xy)
    if nu is None:
        return rng.normal(size=(n, k))
    c = matern(cdist(xy, xy), nu, rho)
    jitter = 1e-6
    while True:
        try:
            lc = cholesky(c + jitter * np.eye(n), lower=True, check_finite=False)
            break
        except np.linalg.LinAlgError:
            jitter *= 10
    return lc @ rng.normal(size=(n, k))


def a1(section):
    t0 = time.time()
    counts, genes, meta = load_dlpfc(section)
    g = jx.visium_graph(meta.array_row.values, meta.array_col.values)
    xy = hex_xy(meta.array_row.values, meta.array_col.values)
    n = g.shape[0]
    ps = PairedStats(g, jx.SpectralNull(g))
    rng = np.random.default_rng([SEED, int(section)])
    perms = np.stack([rng.permutation(n) for _ in range(B)])
    rows = []
    for name, nu, rho in SETTINGS:
        f = gaussian_fields(xy, nu, rho, 2 * NPAIRS, rng)
        zx, zy = zcols(f[:, :NPAIRS]), zcols(f[:, NPAIRS:])
        moran_x = np.einsum("ik,ik->k", zx, ps.M_mor @ zx) / ps.s0
        spec = ps.spectral(zx, zy)
        r = ps.pearson(zx, zy)
        p_t = pearson_t_p(r, n)
        for k in range(NPAIRS):
            pp = ps.permutation_pvalues(zx[:, k], zy[:, k], perms)
            row = dict(axis="A1", unit=section, setting=name, pair=k, n=n,
                       moran_x=moran_x[k], pearson_t=p_t[k])
            for m in METHODS:
                row[f"{m}_perm"] = pp[m]
                row[f"{m}_spectral"] = two_sided(spec[m][1][k])
                row[f"{m}_stat"] = spec[m][0][k]
            rows.append(row)
        print(f"A1 {section} {name} done {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def top_moran_genes(section, k=50, min_detect=0.10):
    counts, genes, meta = load_dlpfc(section)
    detect = np.asarray((counts > 0).mean(axis=0)).ravel()
    keep = np.where(detect >= min_detect)[0]
    x = lognorm(counts)[:, keep].toarray()
    g = jx.visium_graph(meta.array_row.values, meta.array_col.values)
    z = zcols(x)
    w = jx._core.row_standardize(g)
    mi = np.einsum("ik,ik->k", z, w @ z) / w.sum()
    order = np.argsort(-mi)[:k]
    return keep[order], genes[keep[order]], mi[order]


def a2(pair):
    t0 = time.time()
    sa, sb = pair
    ca, ga, ma = load_dlpfc(sa)
    cb, gb, mb = load_dlpfc(sb)
    ia, na, _ = top_moran_genes(sa)
    ib, nb, _ = top_moran_genes(sb)
    key_a = pd.Index(list(zip(ma.array_row, ma.array_col)))
    key_b = pd.Index(list(zip(mb.array_row, mb.array_col)))
    common = key_a.intersection(key_b)
    pa, pb = key_a.get_indexer(common), key_b.get_indexer(common)
    xa = lognorm(ca)[:, ia].toarray()[pa]
    xb = lognorm(cb)[:, ib].toarray()[pb]
    rr = np.array([c[0] for c in common])
    cc = np.array([c[1] for c in common])
    g = jx.visium_graph(rr, cc)
    n = g.shape[0]
    ps = PairedStats(g, jx.SpectralNull(g))
    zx_all, zy_all = zcols(xa), zcols(xb)
    okx = np.isfinite(zx_all).all(axis=0)
    oky = np.isfinite(zy_all).all(axis=0)
    zx_all, zy_all = zx_all[:, okx], zy_all[:, oky]
    gx, gy = na[okx], nb[oky]
    rng = np.random.default_rng([SEED, int(sa), int(sb)])
    perms = np.stack([rng.permutation(n) for _ in range(B)])
    # paired layout of all cross pairs for the spectral null
    ix, iy = np.meshgrid(np.arange(zx_all.shape[1]), np.arange(zy_all.shape[1]), indexing="ij")
    ix, iy = ix.ravel(), iy.ravel()
    spec = ps.spectral(zx_all[:, ix], zy_all[:, iy])
    r = ps.pearson(zx_all[:, ix], zy_all[:, iy])
    p_t = pearson_t_p(r, n)
    # permutation: share the smoothing of each permuted y across all x genes
    perm = {m: np.zeros((zx_all.shape[1], zy_all.shape[1])) for m in METHODS}
    v = {"lee": ps.W.T @ (ps.W @ zx_all) / ps.lee_norm,
         "moran": ps.M_mor @ zx_all / ps.s0, "pearson": zx_all / n}
    ax = {s: ps.smooth(zx_all, s) for s in (0, 1)}
    lx = {s: ps.lap @ ax[s] for s in (0, 1)}
    ex = {s: np.einsum("ik,ik->k", ax[s], lx[s]) for s in (0, 1)}
    for b in range(zy_all.shape[1]):
        zy = zy_all[:, b]
        yp = zy[perms].T
        for m in ("lee", "moran", "pearson"):
            obs = zy @ v[m]
            null = yp.T @ v[m]
            perm[m][:, b] = (1 + np.sum(np.abs(null) >= np.abs(obs)[None, :] - 1e-15, axis=0)) / (B + 1)
        for s in (0, 1):
            ay, ap = ps.smooth(zy, s), ps.smooth(yp, s)
            ey_obs = ay @ (ps.lap @ ay)
            ey_null = np.einsum("ib,ib->b", ap, ps.lap @ ap)
            obs = -2 * (lx[s].T @ ay) / (ex[s] + ey_obs)
            null = -2 * (ap.T @ lx[s]) / (ex[s][None, :] + ey_null[:, None])
            perm[f"J{s}"][:, b] = (1 + np.sum(np.abs(null) >= np.abs(obs)[None, :] - 1e-15, axis=0)) / (B + 1)
    rows = []
    for k in range(len(ix)):
        row = dict(axis="A2", unit=f"{sa}x{sb}", setting="cross_donor", pair=k, n=n,
                   gene_x=gx[ix[k]], gene_y=gy[iy[k]], pearson_t=p_t[k])
        for m in METHODS:
            row[f"{m}_perm"] = perm[m][ix[k], iy[k]]
            row[f"{m}_spectral"] = two_sided(spec[m][1][k])
            row[f"{m}_stat"] = spec[m][0][k]
        rows.append(row)
    print(f"A2 {sa}x{sb} n={n} pairs={len(rows)} done {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def run(task):
    kind, arg = task
    return a1(arg) if kind == "A1" else a2(arg)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    tasks = []
    if which in ("all", "A2"):
        tasks += [("A2", p) for p in A2_PAIRS]
    if which in ("all", "A1"):
        tasks += [("A1", s) for s in A1_SECTIONS]
    with Pool(2) as pool:
        frames = pool.map(run, tasks, chunksize=1)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(RESULTS / f"calibration_{which}.csv.gz", index=False)
    print("saved", len(df))
