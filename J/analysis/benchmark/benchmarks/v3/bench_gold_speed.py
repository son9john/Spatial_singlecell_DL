"""Amendment v3, axis D: cost of each ecosystem statistic for all pairs of 4,000 genes
(section 151673, 3,611 spots, one BLAS thread). A permutation test costs B such evaluations."""
import time

import numpy as np
import pandas as pd
from scipy import stats

from common import RESULTS, hex_xy, jx, zcols
from datasets import load_dlpfc, lognorm
from gold import GoldStandards, rank_cols


def best_of(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main(ng=4000):
    counts, genes, meta = load_dlpfc("151673")
    x = lognorm(counts)
    top = np.argsort(-np.asarray(x.mean(axis=0)).ravel())[:ng]
    X = x[:, top].toarray()
    g = jx.visium_graph(meta.array_row.values, meta.array_col.values)
    gs = GoldStandards(hex_xy(meta.array_row.values, meta.array_col.values), g)
    n = X.shape[0]
    Z = np.nan_to_num(zcols(X))
    rows = []

    def pearson_t():
        r = Z.T @ Z / n
        t = r * np.sqrt((n - 2) / np.clip(1 - r ** 2, 1e-12, None))
        return 2 * stats.t.sf(np.abs(t), n - 2)

    def spearman_t():
        R = np.nan_to_num(zcols(rank_cols(X)))
        r = R.T @ R / n
        t = r * np.sqrt((n - 2) / np.clip(1 - r ** 2, 1e-12, None))
        return 2 * stats.t.sf(np.abs(t), n - 2)

    def sdm_z():
        R = Z.T @ (gs.W_sdm @ Z) / gs.sdm_s0
        return stats.norm.sf(R / gs.sdm_sd)

    def liana_once():
        return Z.T @ (gs.W_liana @ Z) / gs.liana_s0

    def hotspot_z():
        sz = gs.S_hot @ Z
        h = Z.T @ sz
        e = np.sqrt(np.einsum("ik,ik->k", sz, sz))
        za, zb = h / e[:, None], h / e[None, :]
        return stats.norm.sf(np.where(np.abs(za) < np.abs(zb), za, zb))

    w = jx._core.row_standardize(g)
    def moran_once():
        m = Z.T @ (w @ Z)
        return 0.5 * (m + m.T) / w.sum()

    def lee_once():
        wz = w @ Z
        return wz.T @ wz

    for name, fn, perms in (("Pearson t-test (Scanpy/Seurat)", pearson_t, 0),
                            ("Spearman t-test (Scanpy/Seurat)", spearman_t, 0),
                            ("SpatialDM z-test", sdm_z, 0),
                            ("SpatialDM permutation (1,000)", sdm_z, 1000),
                            ("LIANA+ Moran's R, 100 permutations", liana_once, 100),
                            ("Hotspot Z", hotspot_z, 0),
                            ("MERINGUE, 1,000 permutations", moran_once, 1000),
                            ("Lee's L, 999 permutations", lee_once, 999)):
        sec = best_of(fn)
        rows.append(dict(method=name, genes=ng, spots=n, seconds_one_evaluation=sec, permutations=perms,
                         seconds_total=sec * max(perms, 1) if perms else sec,
                         extrapolated=bool(perms)))
        print(rows[-1], flush=True)
    sp = pd.read_csv(RESULTS / "speed.csv")
    eig = float(sp[sp.task == "spectral_eigendecomposition_once"].seconds.iloc[0])
    jp = sp[(sp.task == "J1_spectral_pvalues_all_pairs") & (sp.genes == ng)].seconds.iloc[0]
    rows.append(dict(method="J, spectral null (this work)", genes=ng, spots=n, seconds_one_evaluation=jp,
                     permutations=0, seconds_total=jp + eig, extrapolated=False))
    pd.DataFrame(rows).to_csv(RESULTS / "gold_speed.csv", index=False)


if __name__ == "__main__":
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
