"""Axis B of PROTOCOL_v2: accuracy against the expert DLPFC layer annotation.

B1: real data, leave-one-donor-out layer markers, pair classes from layer adjacency.
B2: semi-synthetic counts on the real layer maps (known truth).
"""
import sys
import time

import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score

from common import RESULTS, hex_xy, jx
from datasets import DONOR, LAYER_ORDER, SECTIONS, load_dlpfc, lognorm

SEED = 20261122
KS = (5, 10, 20)
MIN_SPOTS = 30
RANK = {l: k for k, l in enumerate(LAYER_ORDER)}


# ----------------------------------------------------------------------------- markers
def layer_means():
    out, genes0 = {}, None
    for s in SECTIONS:
        counts, genes, meta = load_dlpfc(s)
        if genes0 is None:
            genes0 = genes
        assert np.array_equal(genes, genes0), "gene order differs between sections"
        x = lognorm(counts)
        lay = meta.layer.values
        cols = {}
        for l in LAYER_ORDER:
            m = lay == l
            if m.sum() >= MIN_SPOTS:
                cols[l] = np.asarray(x[m].mean(axis=0)).ravel()
        out[s] = pd.DataFrame(cols)
    return out, genes0


def select_markers(means, genes, test_donor, k):
    train = [s for s in SECTIONS if DONOR[s] != test_donor]
    margin_sum = {l: np.zeros(len(genes)) for l in LAYER_ORDER}
    mean_sum = {l: np.zeros(len(genes)) for l in LAYER_ORDER}
    count = {l: 0 for l in LAYER_ORDER}
    for s in train:
        df = means[s]
        for l in df.columns:
            others = df.drop(columns=l).to_numpy()
            margin_sum[l] += df[l].to_numpy() - others.max(axis=1)
            mean_sum[l] += df[l].to_numpy()
            count[l] += 1
    present = [l for l in LAYER_ORDER if count[l] > 0]
    margin = np.column_stack([margin_sum[l] / count[l] for l in present])
    mean_in = np.column_stack([mean_sum[l] / count[l] for l in present])
    best = margin.argmax(axis=1)
    best_margin = margin[np.arange(len(genes)), best]
    best_mean = mean_in[np.arange(len(genes)), best]
    ok = (best_margin > 0) & (best_mean >= 0.1) & ~pd.Series(genes).str.upper().str.startswith("MT-").values
    markers = {}
    for c, l in enumerate(present):
        idx = np.where(ok & (best == c))[0]
        idx = idx[np.argsort(-best_margin[idx])][:k]
        markers[l] = idx
    return markers


# ----------------------------------------------------------------------------- scoring
def all_scores(x, graph, null=None):
    """Dictionary of p x p score matrices oriented so that larger = 'more complementary'."""
    sc = {f"J{s}": jx.juxtaposition_matrix(x, graph, scale=s) for s in (0, 1, 2, 4)}
    sc["neg_codispersion1"] = jx.codispersion_matrix(x, graph, scale=1)
    sc["gradient_coenergy1"] = jx.gradient_coenergy_matrix(x, graph, scale=1)
    sc["neg_lee"] = -jx.lee_l_matrix(x, graph)
    sc["neg_moran"] = -jx.moran_bv_matrix(x, graph)
    sc["neg_pearson"] = -jx.pearson_matrix(x)
    if null is not None:
        sc["J1_z"] = null.test_j(x, scale=1)["z"]
    return sc


def auc(score, pos, neg):
    s = np.r_[score[pos], score[neg]]
    y = np.r_[np.ones(pos.sum()), np.zeros(neg.sum())]
    ok = np.isfinite(s)
    if len(np.unique(y[ok])) < 2:
        return np.nan
    return roc_auc_score(y[ok], s[ok])


def endpoints(sc, cls, dist):
    adj, far, same = cls == "adjacent", cls == "distant", cls == "same"
    rows = {}
    for name, v in sc.items():
        a_ad = auc(v, adj, far)
        a_as = auc(v, adj, same)
        a_ds = auc(v, far, same)
        n_ad, n_as, n_ds = adj.sum() * far.sum(), adj.sum() * same.sum(), far.sum() * same.sum()
        cidx = np.nansum([a_ad * n_ad, a_as * n_as, a_ds * n_ds]) / (n_ad + n_as + n_ds)
        rows[name] = dict(auc_adjacent_vs_distant=a_ad, auc_adjacent_vs_same=a_as,
                          auc_distant_vs_same=a_ds, ordinal_c=cidx,
                          auc_adjacent_vs_dist2=auc(v, adj, dist == 2),
                          auc_adjacent_vs_dist3plus=auc(v, adj, dist >= 3))
    return rows


def pair_frame(markers_present, gene_names):
    idx = np.concatenate([markers_present[l] for l in markers_present])
    lay = np.concatenate([[l] * len(markers_present[l]) for l in markers_present])
    return idx, lay


# ----------------------------------------------------------------------------- B1
def b1():
    t0 = time.time()
    means, genes = layer_means()
    markers = {(d, k): select_markers(means, genes, d, k) for d in sorted(set(DONOR.values())) for k in KS}
    rows, pairs_out, marker_rows = [], [], []
    for d_k, mk in markers.items():
        for l, idx in mk.items():
            for rank, g in enumerate(idx):
                marker_rows.append(dict(test_donor=d_k[0], K=d_k[1], layer=l, rank=rank + 1, gene=genes[g]))
    for s in SECTIONS:
        counts, _, meta = load_dlpfc(s)
        graph = jx.visium_graph(meta.array_row.values, meta.array_col.values)
        null = jx.SpectralNull(graph)
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
            sc = all_scores(x, graph, null if k == 10 else None)
            iu, ju = np.triu_indices(len(idx), k=1)
            dist = np.abs(np.array([RANK[l] for l in lay])[iu] - np.array([RANK[l] for l in lay])[ju])
            cls = np.where(dist == 0, "same", np.where(dist == 1, "adjacent", "distant"))
            flat = {name: v[iu, ju] for name, v in sc.items()}
            for name, ep in endpoints(flat, cls, dist).items():
                rows.append(dict(section=s, donor=DONOR[s], K=k, method=name, n_genes=len(idx),
                                 n_adjacent=int((cls == "adjacent").sum()),
                                 n_distant=int((cls == "distant").sum()),
                                 n_same=int((cls == "same").sum()), dropped_zero_var=int((~keep).sum()),
                                 **ep))
            if k == 10:
                pf = pd.DataFrame(dict(section=s, donor=DONOR[s], gene_x=genes[idx[iu]], gene_y=genes[idx[ju]],
                                       layer_x=lay[iu], layer_y=lay[ju], layer_distance=dist, pair_class=cls))
                for name, v in flat.items():
                    pf[name] = v
                pf["lee_L"] = -pf["neg_lee"]
                pf["pearson_r"] = -pf["neg_pearson"]
                pf["moran_I"] = -pf["neg_moran"]
                pf["J1_p"] = null.test_j(x, scale=1)["p"][iu, ju]
                pairs_out.append(pf)
        print(f"B1 {s} done {time.time() - t0:.0f}s", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "dlpfc_b1_auroc.csv", index=False)
    pd.concat(pairs_out).to_csv(RESULTS / "dlpfc_b1_pairs_K10.csv.gz", index=False)
    pd.DataFrame(marker_rows).to_csv(RESULTS / "dlpfc_b1_markers.csv", index=False)
    return res


# ----------------------------------------------------------------------------- B2
def b2(n_rep=5, betas=(0.5, 1.0, 2.0), per_layer=10, mean_count=2.0):
    t0 = time.time()
    rows = []
    for s in SECTIONS:
        counts, _, meta = load_dlpfc(s)
        graph = jx.visium_graph(meta.array_row.values, meta.array_col.values)
        sm = jx.smoother(graph)
        xy = hex_xy(meta.array_row.values, meta.array_col.values)
        a = np.sqrt(3) * cdist(xy, xy) / 3.0
        lc = cholesky((1 + a) * np.exp(-a) + 1e-6 * np.eye(len(xy)), lower=True, check_finite=False)
        lib = np.asarray(counts.sum(axis=1)).ravel().astype(float)
        rel = lib / lib.mean()
        lay_counts = meta.layer.value_counts()
        present = [l for l in LAYER_ORDER if lay_counts.get(l, 0) >= MIN_SPOTS]
        lay_of_gene = np.repeat(present, per_layer)
        p = len(lay_of_gene)
        ind = np.column_stack([(meta.layer.values == l).astype(float) for l in lay_of_gene])
        ind = sm @ ind  # soften boundaries by one smoothing step (spot mixing)
        rng = np.random.default_rng([SEED, int(s)])
        iu, ju = np.triu_indices(p, k=1)
        rk = np.array([RANK[l] for l in lay_of_gene])
        dist = np.abs(rk[iu] - rk[ju])
        cls = np.where(dist == 0, "same", np.where(dist == 1, "adjacent", "distant"))
        for beta in betas:
            for rep in range(n_rep):
                bg = 0.3 * (lc @ rng.normal(size=(len(xy), p)))
                logmu = beta * ind + bg
                lam = rel[:, None] * np.exp(logmu)
                lam *= mean_count / lam.mean(axis=0, keepdims=True)
                y = rng.poisson(lam)
                x = np.log1p(1e4 * y / lib[:, None])
                keep = x.std(axis=0) > 0
                sc = all_scores(x, graph)
                flat = {name: v[iu, ju] for name, v in sc.items()}
                for name, ep in endpoints(flat, cls, dist).items():
                    rows.append(dict(section=s, donor=DONOR[s], beta=beta, rep=rep, method=name,
                                     zero_var=int((~keep).sum()), **ep))
        print(f"B2 {s} done {time.time() - t0:.0f}s", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "dlpfc_b2_semisynthetic.csv", index=False)
    return res


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "b1"
    if which == "b1":
        b1()
    else:
        b2()
