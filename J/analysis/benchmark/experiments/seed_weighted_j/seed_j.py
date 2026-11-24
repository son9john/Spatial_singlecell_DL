"""Seed-weighted J: score each pair around its own strongest opposing spots.

For every gene pair:
  1. local J at each spot = sum of edge cross-terms / (sum of edge energies + ridge * degree * mean edge energy)
     over the edges incident to the spot (ridge 1), at the same diffusion scale as the score;
  2. seeds = spots above the first knee of the descending local-J curve (Kneedle, online mode: first local
     maximum of the difference curve that is confirmed by a drop below its threshold), positive values only,
     at least 3 seeds;
  3. spot weight w = exp(-d^2 / (2 sigma^2)), d = distance to the nearest seed, sigma = 3 spot spacings;
     edge weight = mean of its two spot weights;
  4. seed-weighted J = -2 sum_e w_e dx_e dy_e / sum_e w_e (dx_e^2 + dy_e^2);
     seed-weighted product = -sum_i w_i x_i y_i / sum_i w_i (standardized expression, opposing first).

Pairs, labels and every comparator score are those of the v5 benchmark (revisions/v5_.../results).
All settings are fixed above; no setting was tuned on the outcome.

    python seed_j.py [processes]      -> results_seed_j.csv (per unit), summary printed
"""
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

ROOT = Path(os.environ.get("JBENCH_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "benchmarks" / "v3"))
from common import hex_xy, jx  # noqa: E402
from datasets import load_dlpfc, load_seqfish, lognorm, SECTIONS  # noqa: E402

V5 = ROOT / "revisions" / "v5_2026-11-22_104348_EST" / "results"
SIGMA, RIDGE, MIN_SEEDS, SCALES = 3.0, 1.0, 3, (1, 4)
OUT = Path(__file__).resolve().parent


def first_knee(v, smooth_frac=0.01, sensitivity=1.0):
    """Number of points above the first knee of a descending curve (Kneedle, online detection)."""
    n = len(v)
    if n <= MIN_SEEDS:
        return n
    x = np.linspace(0, 1, n)
    y = (v - v[-1]) / max(v[0] - v[-1], 1e-12)          # 1 -> 0
    w = max(1, int(round(smooth_frac * n)))
    if w > 1:
        y = np.convolve(np.r_[np.full(w, y[0]), y, np.full(w, y[-1])], np.ones(2 * w + 1) / (2 * w + 1), "valid")
    d = (1 - x) - y                                      # difference curve for a convex decreasing curve
    step = 1.0 / (n - 1)
    lmx, thr = None, None
    for i in range(1, n - 1):
        if d[i] >= d[i - 1] and d[i] >= d[i + 1]:
            lmx, thr = i, d[i] - sensitivity * step
        elif lmx is not None and d[i] < thr:
            return max(MIN_SEEDS, lmx + 1)
    return max(MIN_SEEDS, int(np.argmax(d)) + 1)


def scores_for_unit(x, xy, graph, pairs, names):
    """x: spots x genes (normalized expression); returns dict of score arrays, one value per pair."""
    a = sparse.triu(sparse.csr_matrix(graph), k=1).tocoo()
    ei, ej, we = a.row, a.col, a.data.astype(float)
    n, ne = x.shape[0], len(ei)
    inc = sparse.csr_matrix((np.ones(2 * ne), (np.r_[ei, ej], np.r_[np.arange(ne), np.arange(ne)])), shape=(n, ne))
    deg = np.asarray(inc.sum(axis=1)).ravel()
    tree_nn = cKDTree(xy)
    spacing = float(np.median(tree_nn.query(xy, k=2)[0][:, 1]))
    z = jx.standardize(x)
    sm = jx.smoother(sparse.csr_matrix(graph))
    col = {g: k for k, g in enumerate(names)}
    ia = np.array([col[g] for g in pairs.gene_x]); ib = np.array([col[g] for g in pairs.gene_y])
    out = {}
    zs = z.copy()
    done = 0
    for s in SCALES:
        while done < s:
            zs = sm @ zs
            done += 1
        dz = (zs[ei] - zs[ej]) * np.sqrt(we)[:, None]       # weighted edge increments, edges x genes
        m_all = np.mean(dz * dz, axis=0)                     # mean edge energy per gene
        sj, sjp, glob = (np.empty(len(pairs)) for _ in range(3))
        for k in range(len(pairs)):
            da, db = dz[:, ia[k]], dz[:, ib[k]]
            c = -2 * da * db
            e = da * da + db * db
            glob[k] = c.sum() / e.sum()
            num = inc @ c
            den = inc @ e + 0.5 * RIDGE * deg * (m_all[ia[k]] + m_all[ib[k]])
            r = num / np.maximum(den, 1e-300)
            order = np.argsort(-r)
            pos = order[r[order] > 0]
            if len(pos) < MIN_SEEDS:
                pos = order[:MIN_SEEDS]
            kseeds = first_knee(r[pos]) if len(pos) > MIN_SEEDS else len(pos)
            seeds = pos[:kseeds]
            d, _ = cKDTree(xy[seeds]).query(xy, k=1)
            w = np.exp(-(d / spacing) ** 2 / (2 * SIGMA ** 2))
            we_ = 0.5 * (w[ei] + w[ej])
            sj[k] = (we_ * c).sum() / max((we_ * e).sum(), 1e-300)
            if s == SCALES[0]:
                sjp[k] = -np.sum(w * z[:, ia[k]] * z[:, ib[k]]) / w.sum()
                out.setdefault("n_seeds", np.empty(len(pairs)))[k] = kseeds
        out[f"J{s}_recomputed"] = glob
        out[f"seedJ{s}"] = sj
        if s == SCALES[0]:
            out["seed_product"] = sjp
    return out


def unit_task(task):
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
    sc = scores_for_unit(x, xy, graph, pf, names)
    for k, v in sc.items():
        pf[k] = v
    print(f"{kind} {unit}: {len(pf)} pairs, median seeds {np.median(sc['n_seeds']):.0f}, {time.time() - t0:.0f}s",
          flush=True)
    return pf


def evaluate(df):
    tasks = {"touch_apart": ("touch", ["apart"]), "touch_same": ("touch", ["same"]),
             "touch_apart_or_same": ("touch", ["apart", "same"])}
    methods = ["J1", "J4", "J1_recomputed", "seedJ1", "seedJ4", "seed_product", "pearson", "hotspot_Z",
               "lee_l", "J_region", "smooth_pearson_s16"]
    rows = []
    for (design, unit), g in df.groupby(["design", "unit"]):
        for task, (pos, negs) in tasks.items():
            s = g[g.pair_class.isin([pos, *negs])]
            yv = (s.pair_class == pos).to_numpy()
            if yv.all() or not yv.any():
                continue
            for mth in methods:
                v = s[mth].to_numpy(float)
                ok = np.isfinite(v)
                rows.append(dict(design=design, unit=unit, task=task, method=mth,
                                 auroc=roc_auc_score(yv[ok], v[ok]), auroc_reversed=roc_auc_score(yv[ok], -v[ok])))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    tasks = [("B1", s) for s in SECTIONS] + [("C1", e) for e in ("embryo1", "embryo2", "embryo3")]
    if len(sys.argv) > 2 and sys.argv[2] == "smoke":
        tasks = [("B1", SECTIONS[0])]
    with Pool(procs) as pool:
        frames = pool.map(unit_task, tasks, chunksize=1)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT / "pairs_seed_j.csv.gz", index=False)
    res = evaluate(df)
    res.to_csv(OUT / "results_seed_j.csv", index=False)
    mean = res.groupby(["design", "task", "method"])[["auroc", "auroc_reversed"]].mean().round(4)
    print(mean.to_string())
