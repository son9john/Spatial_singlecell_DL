"""Application: public colorectal cancer Visium section GSM8594561 (the original specimen).

Descriptive analysis (no ground truth): all pairs among the 300 most spatially autocorrelated
genes; J and Lee's L with naive and spectral-null p-values; ranked juxtaposed pairs.
"""
import time

import numpy as np
import pandas as pd

from common import RESULTS, jx, pearson_t_p
from datasets import load_crc, lognorm

TOP = 300
B = 999


def main():
    t0 = time.time()
    counts, genes, meta = load_crc()
    lib = np.asarray(counts.sum(axis=1)).ravel()
    keep_spots = lib > 0
    counts, meta = counts[keep_spots], meta[keep_spots]
    graph = jx.visium_graph(meta.array_row.values, meta.array_col.values)
    detect = np.asarray((counts > 0).mean(axis=0)).ravel()
    cand = np.where((detect >= 0.10) & ~pd.Series(genes).str.upper().str.startswith("MT-").values)[0]
    x = lognorm(counts)[:, cand].toarray()
    z = jx.standardize(x)
    w = jx._core.row_standardize(graph)
    moran = np.einsum("ik,ik->k", z, w @ z) / w.sum()
    sel = np.argsort(-moran)[:TOP]
    X = x[:, sel]
    names = list(genes[cand][sel])
    n = X.shape[0]
    null = jx.SpectralNull(graph)
    tj = null.test_j(X, scale=1)
    tl = null.test_lee(X)
    tr = null.test_pearson(X)
    L = jx.lee_l_matrix(X, graph)
    R = jx.pearson_matrix(X)
    J1 = jx.juxtaposition_matrix(X, graph, scale=1)
    J0 = jx.juxtaposition_matrix(X, graph, scale=0)
    # naive tests: Pearson t-test; Lee permutation (each permuted gene reused for all partners)
    zs = jx.standardize(X)
    rng = np.random.default_rng(20261122)
    perms = np.stack([rng.permutation(n) for _ in range(B)])
    rows_w = np.asarray(w.sum(axis=1)).ravel()
    v = w.T @ (w @ zs) / np.sum(rows_w ** 2)
    lee_perm = np.zeros((TOP, TOP))
    s = jx.smoother(graph)
    lap, _ = jx.laplacian(graph)
    ax = s @ zs
    lx = lap @ ax
    ex = np.einsum("ik,ik->k", ax, lx)
    j_perm = np.zeros((TOP, TOP))
    for b in range(TOP):
        yp = zs[perms, b].T
        null_l = yp.T @ v
        lee_perm[:, b] = (1 + np.sum(np.abs(null_l) >= np.abs(zs[:, b] @ v)[None, :], axis=0)) / (B + 1)
        ap = s @ yp
        ey = np.einsum("ib,ib->b", ap, lap @ ap)
        null_j = -2 * (ap.T @ lx) / (ex[None, :] + ey[:, None])
        obs_j = -2 * (lx.T @ ax[:, b]) / (ex + ex[b])
        j_perm[:, b] = (1 + np.sum(np.abs(null_j) >= np.abs(obs_j)[None, :], axis=0)) / (B + 1)
    iu, ju = np.triu_indices(TOP, k=1)
    df = pd.DataFrame(dict(gene_x=np.array(names)[iu], gene_y=np.array(names)[ju],
                           J1=J1[iu, ju], J0=J0[iu, ju], lee_L=L[iu, ju], pearson_r=R[iu, ju],
                           J1_z=tj["z"][iu, ju], J1_p_spectral=tj["p"][iu, ju],
                           J1_p_perm=0.5 * (j_perm[iu, ju] + j_perm[ju, iu]),
                           lee_z=tl["z"][iu, ju], lee_p_spectral=tl["p"][iu, ju],
                           lee_p_perm=0.5 * (lee_perm[iu, ju] + lee_perm[ju, iu]),
                           pearson_p_t=pearson_t_p(R[iu, ju], n),
                           pearson_p_spectral=tr["p"][iu, ju]))
    for col in ("J1_p_spectral", "lee_p_spectral", "lee_p_perm", "pearson_p_t", "J1_p_perm",
                "pearson_p_spectral"):
        df[col.replace("_p_", "_q_")] = jx.bh_qvalues(df[col].to_numpy())
    df.to_csv(RESULTS / "crc_pairs_top300.csv.gz", index=False)
    pd.DataFrame(dict(gene=names, moran_I=moran[sel], detection=detect[cand][sel])).to_csv(
        RESULTS / "crc_genes_top300.csv", index=False)
    summ = {c: float((df[c] < 0.05).mean()) for c in df.columns if "_q_" in c}
    summ.update(n_spots=n, n_edges=graph.nnz // 2, n_pairs=len(df), seconds=time.time() - t0)
    pd.Series(summ).to_csv(RESULTS / "crc_summary.csv")
    print(pd.Series(summ))


if __name__ == "__main__":
    main()
