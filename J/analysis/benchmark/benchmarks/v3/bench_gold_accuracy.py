"""Protocol amendment v3, axes B, C and CRC: accuracy of the ecosystem statistics against expert
labels and known truth, each used in its natural orientation (higher = more co-localized, which
predicts touching domains), next to J (higher = more opposing change, which predicts touching).

B1  DLPFC layer adjacency (v2 markers and pairs, all comparators).
B2  Semi-synthetic exclusive layer markers (v2 simulation, identical seeds).
C1  seqFISH, v2 design: label-permutation contact z >= 3 versus <= -3, held-out markers.
C2  seqFISH, the parallel analysis's design: contact rate >= 0.15 versus <= 0.01 on a Gaussian
    kNN-8 graph, top-3 within-embryo markers (mean z difference > 1), plus the contact-rate rank
    correlation of the mean marker-pair score.
CRC Fraction of the 44,850 pairs of the top 300 spatially variable genes called significant
    (BH q < 0.05) by each default test and by the spectral null.
"""
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.linalg import cholesky
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score

from bench_dlpfc import (KS, MIN_SPOTS, RANK, SEED, auc, layer_means, pair_frame,
                         select_markers)
from bench_gold_calibration import B, B_LIANA, B_LEE, t_two_sided
from bench_seqfish import K as K_SEQ, MIN_CELLS, contact_z, markers_loeo
from common import RESULTS, hex_xy, jx, zcols
from datasets import DONOR, LAYER_ORDER, SECTIONS, load_crc, load_dlpfc, load_seqfish, lognorm
from gold import GoldStandards, rank_cols

PX_PER_UNIT = 137.0


def method_scores(x, graph, gs):
    """All-pairs score matrices, natural orientation for every method."""
    sc = gs.score_matrices(x)
    sc["meringue_moran"] = jx.moran_bv_matrix(x, graph)
    sc["lee_l"] = jx.lee_l_matrix(x, graph)
    sc["J1"] = jx.juxtaposition_matrix(x, graph, scale=1)
    sc["J0"] = jx.juxtaposition_matrix(x, graph, scale=0)
    z = jx.standardize(x)
    s = jx.smoother(graph)
    lap, _ = jx.laplacian(graph)
    zs = s @ z
    sc["Q1_signed_cross_energy"] = -2 * (zs.T @ (lap @ zs)) / len(z)
    return sc


def tri(sc, iu, ju):
    return {k: np.asarray(v)[iu, ju] for k, v in sc.items()}


def three_aucs(v, pos, neg, same):
    return dict(auc_touch_vs_apart=auc(v, pos, neg), auc_touch_vs_same=auc(v, pos, same),
                auc_touch_vs_apart_or_same=auc(v, pos, neg | same))


# --------------------------------------------------------------------------------------- B1
def b1():
    t0 = time.time()
    means, genes = layer_means()
    markers = {(d, k): select_markers(means, genes, d, k) for d in sorted(set(DONOR.values())) for k in KS}
    rows, pair_rows = [], []
    for s in SECTIONS:
        counts, _, meta = load_dlpfc(s)
        graph = jx.visium_graph(meta.array_row.values, meta.array_col.values)
        gs = GoldStandards(hex_xy(meta.array_row.values, meta.array_col.values), graph)
        lay_counts = meta.layer.value_counts()
        present = [l for l in LAYER_ORDER if lay_counts.get(l, 0) >= MIN_SPOTS]
        xall = lognorm(counts)
        for k in KS:
            mk = markers[(DONOR[s], k)]
            mp = {l: mk[l] for l in present if l in mk and len(mk[l])}
            idx, lay = pair_frame(mp, genes)
            x = xall[:, idx].toarray()
            keep = x.std(axis=0) > 0
            x, idx, lay = x[:, keep], idx[keep], lay[keep]
            iu, ju = np.triu_indices(len(idx), k=1)
            rk = np.array([RANK[l] for l in lay])
            dist = np.abs(rk[iu] - rk[ju])
            adj, far, same = dist == 1, dist >= 2, dist == 0
            flat = tri(method_scores(x, graph, gs), iu, ju)
            for name, v in flat.items():
                rows.append(dict(section=s, donor=DONOR[s], K=k, method=name, **three_aucs(v, adj, far, same)))
            if k == 10:
                pf = pd.DataFrame(dict(section=s, gene_x=genes[idx[iu]], gene_y=genes[idx[ju]],
                                       layer_distance=dist))
                for name, v in flat.items():
                    pf[name] = v
                pair_rows.append(pf)
        print(f"B1 {s} {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "gold_b1_auroc.csv", index=False)
    pd.concat(pair_rows).to_csv(RESULTS / "gold_b1_pairs_K10.csv.gz", index=False)


# --------------------------------------------------------------------------------------- B2
def b2(n_rep=5, betas=(0.5, 1.0, 2.0), per_layer=10, mean_count=2.0):
    t0 = time.time()
    rows = []
    for s in SECTIONS:
        counts, _, meta = load_dlpfc(s)
        graph = jx.visium_graph(meta.array_row.values, meta.array_col.values)
        sm = jx.smoother(graph)
        xy = hex_xy(meta.array_row.values, meta.array_col.values)
        gs = GoldStandards(xy, graph)
        a = np.sqrt(3) * cdist(xy, xy) / 3.0
        lc = cholesky((1 + a) * np.exp(-a) + 1e-6 * np.eye(len(xy)), lower=True, check_finite=False)
        del a
        lib = np.asarray(counts.sum(axis=1)).ravel().astype(float)
        rel = lib / lib.mean()
        lay_counts = meta.layer.value_counts()
        present = [l for l in LAYER_ORDER if lay_counts.get(l, 0) >= MIN_SPOTS]
        lay_of_gene = np.repeat(present, per_layer)
        p = len(lay_of_gene)
        ind = np.column_stack([(meta.layer.values == l).astype(float) for l in lay_of_gene])
        ind = sm @ ind
        rng = np.random.default_rng([SEED, int(s)])  # identical stream to PROTOCOL_v2 B2
        iu, ju = np.triu_indices(p, k=1)
        rk = np.array([RANK[l] for l in lay_of_gene])
        dist = np.abs(rk[iu] - rk[ju])
        adj, far, same = dist == 1, dist >= 2, dist == 0
        for beta in betas:
            for rep in range(n_rep):
                bg = 0.3 * (lc @ rng.normal(size=(len(xy), p)))
                lam = rel[:, None] * np.exp(beta * ind + bg)
                lam *= mean_count / lam.mean(axis=0, keepdims=True)
                y = rng.poisson(lam)
                x = np.log1p(1e4 * y / lib[:, None])
                flat = tri(method_scores(x, graph, gs), iu, ju)
                for name, v in flat.items():
                    rows.append(dict(section=s, beta=beta, rep=rep, method=name,
                                     **three_aucs(v, adj, far, same)))
        print(f"B2 {s} {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "gold_b2_semisynthetic.csv", index=False)


# --------------------------------------------------------------------------------------- C
def seq_gold(xy, graph):
    d, _ = cKDTree(xy).query(xy, k=2)
    unit = float(np.median(d[:, 1]))
    return GoldStandards(xy / unit, graph, px_per_unit=PX_PER_UNIT, single_cell=True), unit


def gaussian_knn8(coords, k=8):
    """The parallel analysis's graph: Gaussian-weighted kNN (sigma = median kNN distance), symmetrized."""
    n = len(coords)
    dists, idx = cKDTree(coords).query(coords, k=k + 1)
    sigma = float(np.median(dists[:, 1:]))
    w = np.exp(-0.5 * (dists[:, 1:].ravel() / sigma) ** 2)
    a = sparse.csr_matrix((w, (np.repeat(np.arange(n), k), idx[:, 1:].ravel())), shape=(n, n))
    a = 0.5 * (a + a.T)
    a.setdiag(0)
    a.eliminate_zeros()
    return a.tocsr()


def c_seqfish():
    t0 = time.time()
    expr, genes, meta = load_seqfish()
    expr = expr.tocsr()
    types = sorted(meta.celltype_mapped_refined.unique())
    rng = np.random.default_rng(SEED)  # same stream as v2 axis C
    rows1, rows2, rho_rows = [], [], []
    for emb in sorted(meta.embryo.unique()):
        m = (meta.embryo == emb).values
        sub = meta[m]
        xy = sub[["x_global_affine", "y_global_affine"]].to_numpy()
        lab = sub.celltype_mapped_refined.values
        # ---------------- C1: v2 design
        graph = jx.spatial_graph(xy, neighbors=6)
        gs, unit = seq_gold(xy, graph)
        present = [t for t in types if (lab == t).sum() >= MIN_CELLS]
        z, _ = contact_z(graph, lab, present, rng)
        mk = markers_loeo(expr, genes, meta, emb, types)
        idx = np.concatenate([mk[t] for t in present])
        typ = np.concatenate([[t] * len(mk[t]) for t in present])
        x = expr[m][:, idx].toarray()
        keep = x.std(axis=0) > 0
        x, idx, typ = x[:, keep], idx[keep], typ[keep]
        iu, ju = np.triu_indices(len(idx), k=1)
        ti = np.array([present.index(t) for t in typ])
        zz = z[ti[iu], ti[ju]]
        same = typ[iu] == typ[ju]
        touch, apart = ~same & (zz >= 3), ~same & (zz <= -3)
        flat = tri(method_scores(x, graph, gs), iu, ju)
        for name, v in flat.items():
            rows1.append(dict(embryo=emb, design="v2_contact_z", method=name, n_touch=int(touch.sum()),
                              n_apart=int(apart.sum()), n_same=int(same.sum()),
                              **three_aucs(v, touch, apart, same)))
        # ---------------- C2: contact-rate design of the parallel analysis
        X = expr[m].toarray().astype(float)
        A8 = gaussian_knn8(xy)
        Z = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
        cts = lab
        uniq = [ct for ct, cnt in pd.Series(cts).value_counts().items() if cnt >= 100]
        ct_markers = {}
        for ct in uniq:
            diff = Z[cts == ct].mean(axis=0) - Z[cts != ct].mean(axis=0)
            top = np.argsort(diff)[::-1][:3]
            ct_markers[ct] = [int(i) for i in top if diff[i] > 1.0]
        coo = A8.tocoo()
        cid = {ct: i for i, ct in enumerate(uniq)}
        ids = np.array([cid.get(c, -1) for c in cts])
        ok = (ids[coo.row] >= 0) & (ids[coo.col] >= 0)
        contact = np.zeros((len(uniq), len(uniq)))
        np.add.at(contact, (ids[coo.row[ok]], ids[coo.col[ok]]), 1.0)
        nct = np.array([(cts == ct).sum() for ct in uniq], float)
        rate = contact / np.sqrt(np.outer(nct, nct))
        adj_set, sep_set, same_set = set(), set(), set()
        for i in range(len(uniq)):
            gi = ct_markers[uniq[i]]
            for a_i, g1 in enumerate(gi):
                for g2 in gi[a_i + 1:]:
                    same_set.add((min(g1, g2), max(g1, g2)))
            for j in range(i + 1, len(uniq)):
                gj = ct_markers[uniq[j]]
                target = adj_set if rate[i, j] >= 0.15 else sep_set if rate[i, j] <= 0.01 else None
                if target is None:
                    continue
                for g1 in gi:
                    for g2 in gj:
                        if g1 != g2:
                            target.add((min(g1, g2), max(g1, g2)))
        sep_set -= adj_set
        same_set -= adj_set
        used = sorted({g for pr in adj_set | sep_set | same_set for g in pr} |
                      {g for ct in uniq for g in ct_markers[ct]})
        pos = {g: k for k, g in enumerate(used)}
        xs = X[:, used]
        sc = method_scores(xs, graph, gs)
        sc["J0_knn8_gaussian"] = jx.juxtaposition_matrix(xs, A8, scale=0)  # the parallel analysis's J
        for task, neg in (("touch_vs_apart", sep_set), ("touch_vs_apart_or_same", sep_set | same_set)):
            prs = list(adj_set) + list(neg)
            yt = np.r_[np.ones(len(adj_set)), np.zeros(len(neg))]
            i1 = np.array([pos[p[0]] for p in prs])
            i2 = np.array([pos[p[1]] for p in prs])
            for name, mat in sc.items():
                v = np.asarray(mat)[i1, i2]
                okv = np.isfinite(v)
                rows2.append(dict(embryo=emb, design="contact_rate", task=task, method=name,
                                  n_pos=len(adj_set), n_neg=len(neg),
                                  auroc=roc_auc_score(yt[okv], v[okv])))
        rates, means = [], {k: [] for k in sc}
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                gi, gj = ct_markers[uniq[i]], ct_markers[uniq[j]]
                if not gi or not gj:
                    continue
                rates.append(rate[i, j])
                ia = [pos[g] for g in gi]
                ib = [pos[g] for g in gj]
                for k_, mat in sc.items():
                    means[k_].append(float(np.nanmean(np.asarray(mat)[np.ix_(ia, ib)])))
        for k_, v in means.items():
            rho, p = stats.spearmanr(rates, v)
            rho_rows.append(dict(embryo=emb, method=k_, n_type_pairs=len(rates), spearman_rho=rho, p=p))
        print(f"C {emb} n={m.sum()} {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows1).to_csv(RESULTS / "gold_c1_seqfish_contactz.csv", index=False)
    pd.DataFrame(rows2).to_csv(RESULTS / "gold_c2_seqfish_contactrate.csv", index=False)
    pd.DataFrame(rho_rows).to_csv(RESULTS / "gold_c2_seqfish_contactrate_rho.csv", index=False)


# --------------------------------------------------------------------------------------- CRC
def crc(top=300):
    t0 = time.time()
    counts, genes, meta = load_crc()
    lib = np.asarray(counts.sum(axis=1)).ravel()
    keep_spots = lib > 0
    counts, meta = counts[keep_spots], meta[keep_spots]
    rows_, cols_ = meta.array_row.values, meta.array_col.values
    graph = jx.visium_graph(rows_, cols_)
    detect = np.asarray((counts > 0).mean(axis=0)).ravel()
    cand = np.where((detect >= 0.10) & ~pd.Series(genes).str.upper().str.startswith("MT-").values)[0]
    x = lognorm(counts)[:, cand].toarray()
    z = jx.standardize(x)
    w = jx._core.row_standardize(graph)
    moran = np.einsum("ik,ik->k", z, w @ z) / w.sum()
    sel = np.argsort(-moran)[:top]
    X = x[:, sel]
    n = X.shape[0]
    from bench_gold_calibration import Setup
    st = Setup(graph, hex_xy(rows_, cols_), spectral=True, extra_spectral=True)
    Z = zcols(X)
    iu, ju = np.triu_indices(top, k=1)
    out = {}
    r = Z.T @ Z / n
    out["pearson_t"] = t_two_sided(r, n)[iu, ju]
    R = zcols(rank_cols(X))
    out["spearman_t"] = t_two_sided(R.T @ R / n, n)[iu, ju]
    coefs = st.coef(Z)
    obs = {k: v.T @ Z for k, v in coefs.items()}
    obs_sym = {k: 0.5 * (o + o.T) for k, o in obs.items()}
    out["spatialdm_z"] = stats.norm.sf(obs_sym["spatialdm"] / st.gs.sdm_sd)[iu, ju]
    sz = st.gs.S_hot @ Z
    hmat = Z.T @ sz
    eg2 = np.einsum("ik,ik->k", sz, sz)
    za, zb = hmat / np.sqrt(eg2)[:, None], hmat / np.sqrt(eg2)[None, :]
    out["hotspot_z"] = stats.norm.sf(np.where(np.abs(za) < np.abs(zb), za, zb))[iu, ju]
    rng = np.random.default_rng(20261122)
    ge = {k: np.zeros((top, top)) for k in coefs}
    ab = {k: np.zeros((top, top)) for k in coefs}
    lim = {"spatialdm": B, "meringue": B, "lee": B_LEE, "liana": B_LIANA}
    for b in range(B):
        p = rng.permutation(n)
        zp = Z[p]
        for k, v in coefs.items():
            if b >= lim[k]:
                continue
            nv = v.T @ zp  # x fixed (rows), y permuted (columns)
            ge[k] += nv >= obs[k] - 1e-12
            ab[k] += np.abs(nv) >= np.abs(obs[k]) - 1e-12
    # average the two orientations (x permuted or y permuted), as the packages test one direction
    sym = lambda a: 0.5 * (a + a.T)  # noqa: E731
    out["spatialdm_perm"] = sym(ge["spatialdm"] / B)[iu, ju]
    out["liana_perm"] = sym((1 + ab["liana"]) / (B_LIANA + 1))[iu, ju]
    out["meringue_perm"] = sym((1 + ab["meringue"]) / (B + 1))[iu, ju]
    out["lee_perm"] = sym((1 + ge["lee"]) / (B_LEE + 1))[iu, ju]
    tj = st.ps.null.test_j(X, scale=1)
    out["J1_spectral"] = tj["p"][iu, ju]
    u = st.ps.null.U
    xh = u.T @ Z
    sq = st.ps.null.squared_coefficients(xh)
    for key, mh in st._mh.items():
        s_ = xh.T @ (mh @ xh)
        v_ = sq.T @ ((mh * mh) @ sq)
        out[f"{key}_spectral"] = (2 * stats.norm.sf(np.abs(s_ / np.sqrt(v_))))[iu, ju]
    s_ = xh.T @ xh
    v_ = sq.T @ sq
    out["pearson_spectral"] = (2 * stats.norm.sf(np.abs(s_ / np.sqrt(v_))))[iu, ju]
    rows = []
    for k, p in out.items():
        q = jx.bh_qvalues(np.asarray(p, float))
        rows.append(dict(test=k, n_pairs=len(q), frac_p05=float(np.mean(np.asarray(p) < 0.05)),
                         frac_q05=float(np.mean(q < 0.05))))
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "gold_crc_significance.csv", index=False)
    print(res.round(4).to_string(index=False), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    which = sys.argv[1]
    {"b1": b1, "b2": b2, "c": c_seqfish, "crc": crc}[which]()
