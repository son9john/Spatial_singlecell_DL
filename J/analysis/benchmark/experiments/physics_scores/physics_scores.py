"""Pair scores from physics, mathematics and chemistry, on the observed-anatomy benchmark of v5.

Fixed before scoring (no setting tuned on the outcome). Higher score = predicted closer / touching.
  sliced_W     optimal transport: minus the sliced Wasserstein-1 distance between the two genes' spatial mass
               distributions (expression / total), 64 fixed random directions
  fidelity     quantum fidelity (Bhattacharyya coefficient) of the two distributions after heat diffusion,
               mean over 1, 2, 4, 8 and 16 steps
  flory        Flory-Huggins contact energy: contacts between "x-only" (phi_x = u(1-v)) and "y-only"
               (phi_y = v(1-u)) spots across graph edges, relative to random placement; u, v are the ranks of the
               once-smoothed expression scaled to [0, 1]
  lagged_MI    spatial mutual information, I(x; Wy) + I(y; Wx), 8 quantile bins, W the row-standardized graph
Pairs, labels and comparator scores are those of revisions/v5_.../results.

    python physics_scores.py [processes]   -> pairs_physics.csv.gz, results_physics.csv
"""
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score

ROOT = Path(os.environ.get("JBENCH_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "benchmarks" / "v3"))
from common import hex_xy, jx  # noqa: E402
from datasets import SECTIONS, load_dlpfc, load_seqfish, lognorm  # noqa: E402

V5 = ROOT / "revisions" / "v5_2026-11-22_104348_EST" / "results"
OUT = Path(__file__).resolve().parent
N_DIR, TIMES, BINS = 64, (1, 2, 4, 8, 16), 8


def sliced_w(p, xy, seed=20261122):
    """Mean over directions of the 1-D Wasserstein-1 distance between columns of p (spots x genes, columns sum to 1)."""
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, np.pi, N_DIR)
    g = p.shape[1]
    out = np.zeros((g, g))
    for a in ang:
        t = xy @ np.array([np.cos(a), np.sin(a)])
        o = np.argsort(t)
        dt = np.diff(t[o])
        c = np.cumsum(p[o], axis=0)[:-1]            # CDF at each gap
        for i in range(g):
            out[i] += np.abs(c[:, [i]] - c).T @ dt
    return out / N_DIR


def fidelity(p, sm):
    g = p.shape[1]
    f = np.zeros((g, g))
    q = p.copy()
    done = 0
    for t in TIMES:
        while done < t:
            q = sm @ q
            done += 1
        r = np.sqrt(np.clip(q, 0, None))
        f += r.T @ r
    return f / len(TIMES)


def flory(u, ei, ej, ia, ib):
    out = np.empty(len(ia))
    for k, (a, b) in enumerate(zip(ia, ib)):
        px = u[:, a] * (1 - u[:, b])
        py = u[:, b] * (1 - u[:, a])
        obs = np.sum(px[ei] * py[ej] + py[ei] * px[ej])
        exp = 2 * len(ei) * px.mean() * py.mean()
        out[k] = obs / exp if exp > 0 else np.nan
    return out


def onehot_quantiles(v):
    q = np.argsort(np.argsort(v, axis=0, kind="stable"), axis=0) * BINS // v.shape[0]
    n, g = v.shape
    m = np.zeros((n, g * BINS))
    m[np.arange(n)[:, None], np.arange(g)[None, :] * BINS + q] = 1.0
    return m


def lagged_mi(x, w):
    n, g = x.shape
    bx, bl = onehot_quantiles(x), onehot_quantiles(w @ x)
    joint = (bx.T @ bl) / n                         # (g*BINS) x (g*BINS)
    px, pl = bx.mean(axis=0), bl.mean(axis=0)
    mi = np.zeros((g, g))
    for a in range(g):
        for b in range(g):
            pj = joint[a * BINS:(a + 1) * BINS, b * BINS:(b + 1) * BINS]
            e = np.outer(px[a * BINS:(a + 1) * BINS], pl[b * BINS:(b + 1) * BINS])
            ok = pj > 0
            mi[a, b] = np.sum(pj[ok] * np.log(pj[ok] / e[ok]))
    return mi + mi.T                                # I(x; Wy) + I(y; Wx)


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
    x = np.clip(x, 0, None)
    graph = sparse.csr_matrix(graph)
    sm = jx.smoother(graph)
    col = {g: k for k, g in enumerate(names)}
    ia = np.array([col[g] for g in pf.gene_x])
    ib = np.array([col[g] for g in pf.gene_y])
    tot = x.sum(axis=0)
    p = x / np.where(tot > 0, tot, 1)
    sw = sliced_w(p, xy)
    fid = fidelity(p, sm)
    xs = sm @ x
    u = (np.argsort(np.argsort(xs, axis=0, kind="stable"), axis=0) + 0.5) / xs.shape[0]
    tri = sparse.triu(graph, k=1).tocoo()
    fl = flory(u, tri.row, tri.col, ia, ib)
    w = jx._core.row_standardize(graph)
    mi = lagged_mi(x, w)
    pf["sliced_W"] = -sw[ia, ib]
    pf["fidelity"] = fid[ia, ib]
    pf["flory"] = fl
    pf["lagged_MI"] = mi[ia, ib]
    print(f"{kind} {unit}: {len(pf)} pairs, {time.time() - t0:.0f}s", flush=True)
    return pf


def evaluate(df):
    tasks = {"touch_apart": ["apart"], "touch_same": ["same"], "touch_apart_or_same": ["apart", "same"]}
    methods = ["sliced_W", "fidelity", "flory", "lagged_MI", "lee_l", "pearson", "hotspot_Z", "J_region",
               "giotto_spatial_cor", "J1", "J4"]
    rows = []
    for (design, unit), g in df.groupby(["design", "unit"]):
        for task, negs in tasks.items():
            s = g[g.pair_class.isin(["touch", *negs])]
            yv = (s.pair_class == "touch").to_numpy()
            if yv.all() or not yv.any():
                continue
            for mth in methods:
                if mth not in s:
                    continue
                v = s[mth].to_numpy(float)
                ok = np.isfinite(v)
                rows.append(dict(design=design, unit=unit, task=task, method=mth, auroc=roc_auc_score(yv[ok], v[ok])))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    tasks = [("B1", s) for s in SECTIONS] + [("C1", e) for e in ("embryo1", "embryo2", "embryo3")]
    if len(sys.argv) > 2 and sys.argv[2] == "smoke":
        tasks = [("B1", SECTIONS[0])]
    with Pool(procs) as pool:
        frames = pool.map(unit_task, tasks, chunksize=1)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT / "pairs_physics.csv.gz", index=False)
    res = evaluate(df)
    res.to_csv(OUT / "results_physics.csv", index=False)
    print(res.groupby(["design", "task", "method"]).auroc.mean().unstack("method").round(3).to_string())
