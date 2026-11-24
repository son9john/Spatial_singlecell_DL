"""The two seed remedies on the full observed-anatomy benchmark (12 DLPFC sections, 3 seqFISH embryos).

  soft_clust  seeds of each pair (first knee, as in seed_j.py) kept only in spatial clusters: components of the
              seed graph within 2 hops with >= 5 seeds; alpha = (kept / all seeds)^2; edge weight
              (1 - alpha) * 1 + alpha * w_clust, so a pair without clustered seeds falls back to J.
  mapfixed    one seed set per section, independent of the pair: first knee of the panel-mean local J map;
              the same weights for every pair (a fixed quadratic form, so the spectral null applies).
Settings as specified in the remedy description; nothing tuned here.

    python seed_j_remedies.py [processes]   -> results_remedies.csv
"""
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from seed_j import (MIN_SEEDS, OUT, RIDGE, SCALES, SIGMA, V5, evaluate, first_knee, hex_xy, jx, load_dlpfc,
                    load_seqfish, lognorm, SECTIONS)


def weights_from(seeds, xy, spacing):
    d, _ = cKDTree(xy[seeds]).query(xy, k=1)
    return np.exp(-(d / spacing) ** 2 / (2 * SIGMA ** 2))


def seeds_of(r):
    order = np.argsort(-r)
    pos = order[r[order] > 0]
    if len(pos) < MIN_SEEDS:
        pos = order[:MIN_SEEDS]
    return pos[:first_knee(r[pos]) if len(pos) > MIN_SEEDS else len(pos)]


def run_unit(task):
    kind, unit = task
    t0 = time.time()
    if kind == "B1":
        pf = pd.read_csv(V5 / f"B1_{unit}_pairs_2026-11-22_104504_EST.csv.gz")
        c, genes, meta = load_dlpfc(unit)
        names = sorted(set(pf.gene_x) | set(pf.gene_y))
        gi = {g: i for i, g in enumerate(genes)}
        x = lognorm(c)[:, [gi[g] for g in names]].toarray()
        graph = jx.visium_graph(meta.array_row.values, meta.array_col.values)
        xy = hex_xy(meta.array_row.values, meta.array_col.values)
    else:
        pf = pd.read_csv(V5 / f"C1_{unit}_pairs_2026-11-22_104504_EST.csv.gz")
        ex, genes, meta = load_seqfish()
        m = (meta.embryo == unit).to_numpy()
        xy = meta.loc[m, ["x_global_affine", "y_global_affine"]].to_numpy()
        graph = jx.spatial_graph(xy, neighbors=6)
        names = sorted(set(pf.gene_x) | set(pf.gene_y))
        gi = {g: i for i, g in enumerate(genes)}
        x = ex[m][:, [gi[g] for g in names]].toarray()
    graph = sparse.csr_matrix(graph)
    a = sparse.triu(graph, k=1).tocoo()
    ei, ej = a.row, a.col
    n, ne = x.shape[0], len(ei)
    inc = sparse.csr_matrix((np.ones(2 * ne), (np.r_[ei, ej], np.r_[np.arange(ne), np.arange(ne)])), shape=(n, ne))
    deg = np.asarray(inc.sum(axis=1)).ravel()
    hop2 = ((graph + graph @ graph) > 0).astype(np.int8)
    spacing = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))
    col = {g: k for k, g in enumerate(names)}
    ia = np.array([col[g] for g in pf.gene_x])
    ib = np.array([col[g] for g in pf.gene_y])
    zs = jx.standardize(x)
    sm = jx.smoother(graph)
    done = 0
    for s in SCALES:
        while done < s:
            zs = sm @ zs
            done += 1
        dz = zs[ei] - zs[ej]
        m_all = np.mean(dz * dz, axis=0)
        C, E, R = [], [], []
        for k in range(len(pf)):
            da, db = dz[:, ia[k]], dz[:, ib[k]]
            c, e = -2 * da * db, da * da + db * db
            C.append(c)
            E.append(e)
            R.append((inc @ c) / np.maximum(inc @ e + 0.5 * RIDGE * deg * (m_all[ia[k]] + m_all[ib[k]]), 1e-300))
        R = np.array(R)
        # remedy 2: one seed set per section from the panel-mean local J map
        wfix = weights_from(seeds_of(R.mean(axis=0)), xy, spacing)
        wfe = 0.5 * (wfix[ei] + wfix[ej])
        soft, fixed = np.empty(len(pf)), np.empty(len(pf))
        for k in range(len(pf)):
            c, e = C[k], E[k]
            fixed[k] = (wfe * c).sum() / (wfe * e).sum()
            seeds = seeds_of(R[k])
            sub = hop2[seeds][:, seeds]
            ncomp, lab = connected_components(sub, directed=False)
            sizes = np.bincount(lab)
            keep = seeds[np.isin(lab, np.where(sizes >= 5)[0])]
            alpha = (len(keep) / len(seeds)) ** 2 if len(seeds) else 0.0
            w = (1 - alpha) + alpha * (weights_from(keep, xy, spacing) if len(keep) else np.ones(n))
            we = 0.5 * (w[ei] + w[ej])
            soft[k] = (we * c).sum() / (we * e).sum()
        pf[f"softclust_J{s}"] = soft
        pf[f"mapfixed_J{s}"] = fixed
    print(f"{kind} {unit}: {len(pf)} pairs, {time.time() - t0:.0f}s", flush=True)
    return pf


if __name__ == "__main__":
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    tasks = [("B1", s) for s in SECTIONS] + [("C1", e) for e in ("embryo1", "embryo2", "embryo3")]
    with Pool(procs) as pool:
        df = pd.concat(pool.map(run_unit, tasks, chunksize=1), ignore_index=True)
    import seed_j
    seed_j_methods = ["J1", "J4", "softclust_J1", "softclust_J4", "mapfixed_J1", "mapfixed_J4", "lee_l", "hotspot_Z",
                      "J_region"]
    rows = []
    from sklearn.metrics import roc_auc_score
    for (design, unit), g in df.groupby(["design", "unit"]):
        for task, negs in (("touch_apart", ["apart"]), ("touch_same", ["same"]),
                           ("touch_apart_or_same", ["apart", "same"])):
            s_ = g[g.pair_class.isin(["touch", *negs])]
            yv = (s_.pair_class == "touch").to_numpy()
            for mth in seed_j_methods:
                v = s_[mth].to_numpy(float)
                ok = np.isfinite(v)
                rows.append(dict(design=design, unit=unit, task=task, method=mth, auroc=roc_auc_score(yv[ok], v[ok])))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "results_remedies.csv", index=False)
    print(res.groupby(["design", "task", "method"]).auroc.mean().unstack("method")[seed_j_methods].round(3).to_string())
