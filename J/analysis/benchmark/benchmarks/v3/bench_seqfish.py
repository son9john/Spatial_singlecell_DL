"""Axis C of PROTOCOL_v2: single-cell downstream test with expert cell-type labels
(seqFISH mouse embryo, Lohoff et al. 2022)."""
import time

import numpy as np
import pandas as pd
from scipy import sparse

from bench_dlpfc import all_scores, auc
from common import RESULTS, jx
from datasets import load_seqfish

SEED = 20261122
K = 3
MIN_CELLS = 50
N_PERM = 1000


def contact_z(graph, labels, types, rng):
    """Label-permutation z-scores for edges between cell types (no expression used)."""
    up = sparse.triu(graph, k=1).tocoo()
    code = pd.Categorical(labels, categories=types).codes
    keep = code >= 0

    def counts(c):
        a, b = c[up.row], c[up.col]
        ok = (a >= 0) & (b >= 0)
        m = np.zeros((len(types), len(types)))
        np.add.at(m, (a[ok], b[ok]), 1)
        return m + m.T - np.diag(np.diag(m))

    obs = counts(code)
    sims = np.empty((N_PERM, len(types), len(types)))
    for b in range(N_PERM):
        c = code.copy()
        c[keep] = rng.permutation(c[keep])
        sims[b] = counts(c)
    sd = sims.std(axis=0)
    return np.divide(obs - sims.mean(axis=0), sd, out=np.zeros_like(obs), where=sd > 0), obs


def markers_loeo(expr, genes, meta, test, types):
    train = [e for e in sorted(meta.embryo.unique()) if e != test]
    margin_sum = np.zeros((len(genes), len(types)))
    mean_sum = np.zeros((len(genes), len(types)))
    cnt = np.zeros(len(types))
    for e in train:
        m = (meta.embryo == e).values
        lab = meta.celltype_mapped_refined.values[m]
        x = expr[m]
        avail = [t for t in types if (lab == t).sum() >= MIN_CELLS]
        mu = {t: np.asarray(x[lab == t].mean(axis=0)).ravel() for t in avail}
        for t in avail:
            others = np.column_stack([mu[o] for o in avail if o != t])
            k = types.index(t)
            margin_sum[:, k] += mu[t] - others.max(axis=1)
            mean_sum[:, k] += mu[t]
            cnt[k] += 1
    ok_t = cnt > 0
    margin = np.where(ok_t, margin_sum / np.maximum(cnt, 1), -np.inf)
    meanv = np.where(ok_t, mean_sum / np.maximum(cnt, 1), 0)
    best = margin.argmax(axis=1)
    bm = margin[np.arange(len(genes)), best]
    bmean = meanv[np.arange(len(genes)), best]
    ok = (bm > 0) & (bmean >= 0.1)
    out = {}
    for k, t in enumerate(types):
        idx = np.where(ok & (best == k))[0]
        out[t] = idx[np.argsort(-bm[idx])][:K]
    return out


def main():
    t0 = time.time()
    expr, genes, meta = load_seqfish()
    expr = expr.tocsr()
    types = sorted(meta.celltype_mapped_refined.unique())
    rng = np.random.default_rng(SEED)
    rows, pair_rows, contact_rows = [], [], []
    for emb in sorted(meta.embryo.unique()):
        m = (meta.embryo == emb).values
        sub = meta[m]
        xy = sub[["x_global_affine", "y_global_affine"]].to_numpy()
        graph = jx.spatial_graph(xy, neighbors=6)
        lab = sub.celltype_mapped_refined.values
        present = [t for t in types if (lab == t).sum() >= MIN_CELLS]
        z, obs = contact_z(graph, lab, present, rng)
        for a in range(len(present)):
            for b in range(a + 1, len(present)):
                contact_rows.append(dict(embryo=emb, type_a=present[a], type_b=present[b],
                                         edges=obs[a, b], z=z[a, b]))
        mk = markers_loeo(expr, genes, meta, emb, types)
        idx = np.concatenate([mk[t] for t in present])
        typ = np.concatenate([[t] * len(mk[t]) for t in present])
        x = expr[m][:, idx].toarray()
        keep = x.std(axis=0) > 0
        x, idx, typ = x[:, keep], idx[keep], typ[keep]
        sc = all_scores(x, graph)
        iu, ju = np.triu_indices(len(idx), k=1)
        ti = np.array([present.index(t) for t in typ])
        zz = z[ti[iu], ti[ju]]
        same = typ[iu] == typ[ju]
        contact = ~same & (zz >= 3)
        apart = ~same & (zz <= -3)
        flat = {n: v[iu, ju] for n, v in sc.items()}
        for name, v in flat.items():
            rows.append(dict(embryo=emb, method=name, n_cells=int(m.sum()), n_genes=len(idx),
                             n_contact=int(contact.sum()), n_apart=int(apart.sum()), n_same=int(same.sum()),
                             auc_contact_vs_apart=auc(v, contact, apart),
                             auc_contact_vs_same=auc(v, contact, same),
                             auc_apart_vs_same=auc(v, apart, same)))
        pf = pd.DataFrame(dict(embryo=emb, gene_x=genes[idx[iu]], gene_y=genes[idx[ju]], type_x=typ[iu],
                               type_y=typ[ju], contact_z=np.where(same, np.nan, zz),
                               pair_class=np.where(same, "same", np.where(contact, "contact",
                                                                           np.where(apart, "apart", "neutral")))))
        for name, v in flat.items():
            pf[name] = v
        pair_rows.append(pf)
        print(f"C {emb} n={m.sum()} genes={len(idx)} done {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "seqfish_c_auroc.csv", index=False)
    pd.concat(pair_rows).to_csv(RESULTS / "seqfish_c_pairs.csv.gz", index=False)
    pd.DataFrame(contact_rows).to_csv(RESULTS / "seqfish_contact_z.csv", index=False)


if __name__ == "__main__":
    main()
