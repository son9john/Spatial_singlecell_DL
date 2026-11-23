"""Amendment v3b: Jaccard-index endpoints, in the lineage of the lab's earlier evaluation
(binarize maps over a 0-1 threshold sweep, Jaccard similarity, area under the sweep).

M1  Interface localization (segmentation IoU). For each marker pair of two touching domains,
    every method's local map is compared with the expert-annotated interface: Visium edges
    joining the two adjacent cortical layers (DLPFC), or kNN edges joining the two touching
    cell types (seqFISH, contact z >= 3). Metrics: IoU at the true interface size (top-|T|
    edges), the lab's threshold-swept Jaccard AUC on min-max rescaled scores, and average
    precision.
M3  Reproducibility. Jaccard of the top-1% gene pairs between serial sections of one donor
    (adjacent sections 10 um apart; and sections 300 um apart), common 300-gene panel per donor.
M4  Redundancy. Jaccard of the top-1% gene pairs between methods within a section.
Also the lab's threshold-swept Jaccard similarity of two expression maps (co-occupancy) is
added as a comparator statistic in M3/M4 and in the DLPFC adjacency AUROC.
"""
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score

from bench_dlpfc import KS, MIN_SPOTS, RANK, layer_means, pair_frame, select_markers
from bench_gold_accuracy import method_scores, seq_gold
from bench_seqfish import MIN_CELLS, SEED, contact_z, markers_loeo
from common import RESULTS, hex_xy, jx, zcols
from datasets import DONOR, LAYER_ORDER, SECTIONS, load_dlpfc, load_seqfish, lognorm
from gold import GoldStandards, row_normalize

THRESHOLDS = np.linspace(0.05, 0.95, 19)
TOP_FRAC = 0.01
SERIAL = [("151507", "151508"), ("151509", "151510"), ("151669", "151670"),
          ("151671", "151672"), ("151673", "151674"), ("151675", "151676")]
DISTANT = [("151507", "151509"), ("151508", "151510"), ("151669", "151671"),
           ("151670", "151672"), ("151673", "151675"), ("151674", "151676")]


# ------------------------------------------------------------------ the lab's Jaccard similarity
def minmax(x):
    lo, hi = x.min(axis=0), x.max(axis=0)
    return np.divide(x - lo, hi - lo, out=np.zeros_like(x, dtype=float), where=hi > lo)


def jaccard_auc_matrix(x):
    """Mean over thresholds of Jaccard(x >= t, y >= t) on min-max rescaled maps (all pairs)."""
    s = minmax(np.asarray(x, float))
    acc = np.zeros((s.shape[1], s.shape[1]))
    for t in THRESHOLDS:
        b = (s >= t).astype(np.float32)
        inter = b.T @ b
        size = b.sum(axis=0)
        union = size[:, None] + size[None, :] - inter
        acc += np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return acc / len(THRESHOLDS)


# ------------------------------------------------------------------ local maps
class LocalMaps:
    """Edge-level maps for many gene pairs on one graph (vectorized over pairs)."""

    def __init__(self, graph, w_liana, w_spatialdm):
        up = sparse.triu(graph, k=1).tocoo()
        self.i, self.j, self.w = up.row, up.col, up.data
        n, m = graph.shape[0], len(self.i)
        inc = sparse.csr_matrix((np.ones(2 * m), (np.r_[np.arange(m), np.arange(m)], np.r_[self.i, self.j])),
                                shape=(m, n))
        self.pool = (inc @ inc.T).tocsr()  # sum over edges incident to either endpoint (focal edge twice)
        self.heat = jx.smoother(graph)
        self.lazy = jx.lazy_walk(graph)
        self.wn = row_normalize(w_liana)  # LIANA+ kernel, row-normalized for local weighted metrics
        self.w_sdm = w_spatialdm  # SpatialDM kernel for its local Moran terms

    def edges_from_nodes(self, v):
        return 0.5 * (v[self.i] + v[self.j])

    def maps(self, x_raw, pa, pb):
        """Return dict method -> (m_edges x n_pairs) edge scores, higher = predicted interface."""
        z = zcols(x_raw)
        out = {}
        for tag, op, steps, ridge in (("J_local_s1", self.heat, 1, 0.0), ("J_local_s1_ridge1", self.heat, 1, 1.0),
                                      ("J_local_lazy4_ridge1", self.lazy, 4, 1.0)):
            zs = z
            for _ in range(steps):
                zs = op @ zs
            d = (zs[self.j] - zs[self.i]) * np.sqrt(self.w)[:, None]
            dx, dy = d[:, pa], d[:, pb]
            cross = -2 * (self.pool @ (dx * dy))
            ea, eb = self.pool @ (dx * dx), self.pool @ (dy * dy)
            mass = np.asarray(self.pool @ self.w).ravel()[:, None]
            ref = (np.sum(dx * dx, axis=0) + np.sum(dy * dy, axis=0)) / self.w.sum()
            den = ea + eb + ridge * mass * ref[None, :]
            out[tag] = np.divide(cross, den, out=np.zeros_like(cross), where=den > 0)
            if tag == "J_local_s1":
                out["Q_signed_cross_energy_s1"] = cross / mass
                out["unsigned_gradient_s1"] = (self.pool @ np.abs(dx - dy)) / mass
                out["joint_gradient_energy_s1"] = (ea + eb) / mass
        # LIANA+ / SpatialDM local scores at spots (re-implemented), averaged over edge endpoints
        za, zb = z[:, pa], z[:, pb]
        out["spatialdm_local_moran"] = self.edges_from_nodes(za * (self.w_sdm @ zb) + zb * (self.w_sdm @ za))
        lag_a, lag_b = self.wn @ za, self.wn @ zb
        mab = self.wn @ (za * zb)
        maa, mbb = self.wn @ (za * za), self.wn @ (zb * zb)
        den = np.sqrt(np.clip(maa - lag_a ** 2, 0, None) * np.clip(mbb - lag_b ** 2, 0, None))
        out["liana_local_pearson"] = self.edges_from_nodes(
            np.divide(mab - lag_a * lag_b, den, out=np.zeros_like(den), where=den > 0))
        xp = np.clip(np.asarray(x_raw, float), 0, None)
        xa, xb = xp[:, pa], xp[:, pb]
        num = self.wn @ (xa * xb)
        den = np.sqrt((self.wn @ (xa * xa)) * (self.wn @ (xb * xb)))
        out["liana_local_cosine"] = self.edges_from_nodes(np.divide(num, den, out=np.zeros_like(den), where=den > 0))
        return out


def interface_metrics(scores, truth):
    """scores: (m,) edge scores; truth: (m,) bool. AP, IoU at |T|, lab Jaccard AUC, best IoU."""
    t = int(truth.sum())
    if t == 0 or t == len(truth):
        return None
    ap = average_precision_score(truth, scores)
    top = np.argpartition(-scores, t - 1)[:t]
    tp = truth[top].sum()
    iou_t = tp / (2 * t - tp)
    s = scores - scores.min()
    s = s / s.max() if s.max() > 0 else s
    j = []
    for th in THRESHOLDS:
        p = s >= th
        inter = np.sum(p & truth)
        union = np.sum(p | truth)
        j.append(inter / union if union else 0.0)
    order = np.argsort(-scores)
    cum = np.cumsum(truth[order])
    k = np.arange(1, len(truth) + 1)
    best = np.max(cum / (k + t - cum))
    return dict(ap=ap, iou_at_T=iou_t, jaccard_auc=float(np.mean(j)), best_iou=best,
                prevalence=t / len(truth))


def evaluate_maps(lm, x_raw, pa, pb, truth_list, meta):
    rows = []
    maps = lm.maps(x_raw, pa, pb)
    for name, mat in maps.items():
        for sign, tag in ((1.0, ""), (-1.0, "_negated")):
            if name.startswith(("J_local", "Q_", "unsigned", "joint")) and sign < 0:
                continue
            for k in range(mat.shape[1]):
                res = interface_metrics(sign * mat[:, k], truth_list[k])
                if res is not None:
                    rows.append(dict(**meta[k], method=name + tag, **res))
    return rows


# ------------------------------------------------------------------ M1: DLPFC
def m1_dlpfc(k_markers=10):
    t0 = time.time()
    means, genes = layer_means()
    rows = []
    for s in SECTIONS:
        mk = select_markers(means, genes, DONOR[s], k_markers)
        counts, _, meta = load_dlpfc(s)
        graph = jx.visium_graph(meta.array_row.values, meta.array_col.values)
        gs = GoldStandards(hex_xy(meta.array_row.values, meta.array_col.values), graph)
        lm = LocalMaps(graph, gs.W_liana, gs.W_sdm)
        lay = meta.layer.values
        present = [l for l in LAYER_ORDER if (lay == l).sum() >= MIN_SPOTS]
        xall = lognorm(counts)
        for a_l, b_l in zip(LAYER_ORDER[:-1], LAYER_ORDER[1:]):
            if a_l not in present or b_l not in present or not len(mk[a_l]) or not len(mk[b_l]):
                continue
            ga, gb = mk[a_l], mk[b_l]
            idx = np.r_[ga, gb]
            x = xall[:, idx].toarray()
            ok = x.std(axis=0) > 0
            pa = [k for k in range(len(ga)) if ok[k]]
            pb = [len(ga) + k for k in range(len(gb)) if ok[len(ga) + k]]
            pa_, pb_ = np.repeat(pa, len(pb)), np.tile(pb, len(pa))
            truth = ((lay[lm.i] == a_l) & (lay[lm.j] == b_l)) | ((lay[lm.i] == b_l) & (lay[lm.j] == a_l))
            meta_k = [dict(dataset="DLPFC", unit=s, interface=f"{a_l}|{b_l}", gene_x=genes[idx[p]],
                           gene_y=genes[idx[q]]) for p, q in zip(pa_, pb_)]
            rows += evaluate_maps(lm, x, pa_, pb_, [truth] * len(pa_), meta_k)
        print(f"M1 DLPFC {s} {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ M1: seqFISH
def m1_seqfish():
    t0 = time.time()
    expr, genes, meta = load_seqfish()
    expr = expr.tocsr()
    types = sorted(meta.celltype_mapped_refined.unique())
    rng = np.random.default_rng(SEED)
    rows = []
    for emb in sorted(meta.embryo.unique()):
        m = (meta.embryo == emb).values
        sub = meta[m]
        xy = sub[["x_global_affine", "y_global_affine"]].to_numpy()
        lab = sub.celltype_mapped_refined.values
        graph = jx.spatial_graph(xy, neighbors=6)
        gs, _ = seq_gold(xy, graph)
        lm = LocalMaps(graph, gs.W_liana, gs.W_sdm)
        present = [t for t in types if (lab == t).sum() >= MIN_CELLS]
        z, _ = contact_z(graph, lab, present, rng)
        mk = markers_loeo(expr, genes, meta, emb, types)
        xe = expr[m]
        for a in range(len(present)):
            for b in range(a + 1, len(present)):
                if z[a, b] < 3:
                    continue
                ta, tb = present[a], present[b]
                ga, gb = list(mk[ta]), list(mk[tb])
                if not ga or not gb:
                    continue
                idx = np.r_[ga, gb]
                x = xe[:, idx].toarray()
                ok = x.std(axis=0) > 0
                pa = [k for k in range(len(ga)) if ok[k]]
                pb = [len(ga) + k for k in range(len(gb)) if ok[len(ga) + k] and idx[len(ga) + k] not in ga]
                if not pa or not pb:
                    continue
                pa_, pb_ = np.repeat(pa, len(pb)), np.tile(pb, len(pa))
                truth = ((lab[lm.i] == ta) & (lab[lm.j] == tb)) | ((lab[lm.i] == tb) & (lab[lm.j] == ta))
                meta_k = [dict(dataset="seqFISH", unit=emb, interface=f"{ta}|{tb}", gene_x=genes[idx[p]],
                               gene_y=genes[idx[q]], contact_z=z[a, b]) for p, q in zip(pa_, pb_)]
                rows += evaluate_maps(lm, x, pa_, pb_, [truth] * len(pa_), meta_k)
        print(f"M1 seqFISH {emb} {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ M2: semi-synthetic exclusive markers
def m2_semisynthetic(betas=(1.0, 2.0, 3.0), reps=2, per_layer=3, mean_count=2.0):
    """Known-truth interfaces on the real layer maps: exclusive markers (B2 generator without the
    smooth background), Poisson counts with real library sizes; adjacent-layer marker pairs."""
    t0 = time.time()
    rows = []
    for s in SECTIONS:
        counts, _, meta = load_dlpfc(s)
        graph = jx.visium_graph(meta.array_row.values, meta.array_col.values)
        gs = GoldStandards(hex_xy(meta.array_row.values, meta.array_col.values), graph)
        lm = LocalMaps(graph, gs.W_liana, gs.W_sdm)
        sm = jx.smoother(graph)
        lay = meta.layer.values
        lib = np.asarray(counts.sum(axis=1)).ravel().astype(float)
        rel = lib / lib.mean()
        present = [l for l in LAYER_ORDER if (lay == l).sum() >= MIN_SPOTS]
        rng = np.random.default_rng([SEED, int(s), 7])
        for beta in betas:
            for rep in range(reps):
                for a_l, b_l in zip(LAYER_ORDER[:-1], LAYER_ORDER[1:]):
                    if a_l not in present or b_l not in present:
                        continue
                    ind = sm @ np.column_stack([(lay == a_l).astype(float)] * per_layer +
                                               [(lay == b_l).astype(float)] * per_layer)
                    lam = rel[:, None] * np.exp(beta * ind)
                    lam *= mean_count / lam.mean(axis=0, keepdims=True)
                    x = np.log1p(1e4 * rng.poisson(lam) / lib[:, None])
                    pa_ = np.repeat(np.arange(per_layer), per_layer)
                    pb_ = np.tile(np.arange(per_layer, 2 * per_layer), per_layer)
                    truth = ((lay[lm.i] == a_l) & (lay[lm.j] == b_l)) | ((lay[lm.i] == b_l) & (lay[lm.j] == a_l))
                    meta_k = [dict(dataset="DLPFC_semisynthetic", unit=s, interface=f"{a_l}|{b_l}",
                                   beta=beta, rep=rep, gene_x=int(p), gene_y=int(q)) for p, q in zip(pa_, pb_)]
                    rows += evaluate_maps(lm, x, pa_, pb_, [truth] * len(pa_), meta_k)
        print(f"M2 {s} {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "jaccard_m2_semisynthetic.csv.gz", index=False)


# ------------------------------------------------------------------ M3/M4: reproducibility and redundancy
def donor_panels(top=300, min_detect=0.10):
    panels = {}
    for donor in sorted(set(DONOR.values())):
        secs = [s for s in SECTIONS if DONOR[s] == donor]
        mi, det, genes0 = [], [], None
        for s in secs:
            counts, genes, meta = load_dlpfc(s)
            genes0 = genes if genes0 is None else genes0
            x = lognorm(counts)
            g = jx.visium_graph(meta.array_row.values, meta.array_col.values)
            w = jx._core.row_standardize(g)
            d = np.asarray((counts > 0).mean(axis=0)).ravel()
            cand = np.where(d >= min_detect)[0]
            xc = x[:, cand].toarray()
            zc = zcols(xc)
            m_ = np.full(len(genes), -np.inf)
            m_[cand] = np.einsum("ik,ik->k", np.nan_to_num(zc), w @ np.nan_to_num(zc)) / w.sum()
            mi.append(m_)
            det.append(d)
        mi = np.vstack(mi)
        ok = (np.vstack(det) >= min_detect).all(axis=0) & ~pd.Series(genes0).str.upper().str.startswith("MT-").values
        score = np.where(ok, mi.mean(axis=0), -np.inf)
        panels[donor] = np.argsort(-score)[:top]
    return panels, genes0


def top_set(mat, frac):
    iu, ju = np.triu_indices(mat.shape[0], k=1)
    v = np.asarray(mat)[iu, ju]
    v = np.where(np.isfinite(v), v, -np.inf)
    k = max(1, int(round(frac * len(v))))
    return set(np.argpartition(-v, k - 1)[:k].tolist())


def jac(a, b):
    return len(a & b) / len(a | b) if (a or b) else np.nan


def m3_m4():
    t0 = time.time()
    panels, genes = donor_panels()
    sets, rows4 = {}, []
    for s in SECTIONS:
        counts, _, meta = load_dlpfc(s)
        graph = jx.visium_graph(meta.array_row.values, meta.array_col.values)
        gs = GoldStandards(hex_xy(meta.array_row.values, meta.array_col.values), graph)
        x = lognorm(counts)[:, panels[DONOR[s]]].toarray()
        sc = method_scores(x, graph, gs)
        sc["jaccard_auc_similarity"] = jaccard_auc_matrix(x)
        sc["J1_concordant"] = -sc["J1"]
        sc["J2"] = jx.juxtaposition_matrix(x, graph, scale=2)
        sc["J4"] = jx.juxtaposition_matrix(x, graph, scale=4)
        sets[s] = {k: top_set(v, TOP_FRAC) for k, v in sc.items()}
        names = list(sc)
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                rows4.append(dict(section=s, method_a=names[a], method_b=names[b],
                                  jaccard=jac(sets[s][names[a]], sets[s][names[b]])))
        print(f"M3/M4 {s} {time.time() - t0:.0f}s", flush=True)
    rows3 = []
    npairs = 300 * 299 // 2
    k = int(round(TOP_FRAC * npairs))
    for kind, prs in (("serial_10um", SERIAL), ("distant_300um", DISTANT)):
        for a, b in prs:
            for mth in sets[a]:
                rows3.append(dict(kind=kind, section_a=a, section_b=b, method=mth,
                                  jaccard=jac(sets[a][mth], sets[b][mth]),
                                  chance=k / (2 * npairs - k)))
    pd.DataFrame(rows3).to_csv(RESULTS / "jaccard_m3_reproducibility.csv", index=False)
    pd.DataFrame(rows4).to_csv(RESULTS / "jaccard_m4_redundancy.csv", index=False)


def b1_jaccard_similarity():
    """DLPFC adjacency AUROC (K = 10) for the lab's Jaccard similarity, as for the other comparators."""
    means, genes = layer_means()
    rows = []
    for s in SECTIONS:
        mk = select_markers(means, genes, DONOR[s], 10)
        counts, _, meta = load_dlpfc(s)
        lay_counts = meta.layer.value_counts()
        present = [l for l in LAYER_ORDER if lay_counts.get(l, 0) >= MIN_SPOTS]
        idx, lay = pair_frame({l: mk[l] for l in present if len(mk[l])}, genes)
        x = lognorm(counts)[:, idx].toarray()
        keep = x.std(axis=0) > 0
        x, lay = x[:, keep], lay[keep]
        iu, ju = np.triu_indices(x.shape[1], k=1)
        rk = np.array([RANK[l] for l in lay])
        dist = np.abs(rk[iu] - rk[ju])
        v = jaccard_auc_matrix(x)[iu, ju]
        def auc(pos, neg):
            y = np.r_[np.ones(pos.sum()), np.zeros(neg.sum())]
            return roc_auc_score(y, np.r_[v[pos], v[neg]])
        rows.append(dict(section=s, method="jaccard_auc_similarity",
                         auc_touch_vs_apart=auc(dist == 1, dist >= 2),
                         auc_touch_vs_same=auc(dist == 1, dist == 0),
                         auc_touch_vs_apart_or_same=auc(dist == 1, dist != 1)))
    pd.DataFrame(rows).to_csv(RESULTS / "jaccard_b1_similarity_auroc.csv", index=False)


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "m1":
        pd.concat([m1_dlpfc(), m1_seqfish()], ignore_index=True).to_csv(
            RESULTS / "jaccard_m1_interfaces.csv.gz", index=False)
    elif which == "m1_dlpfc":
        m1_dlpfc().to_csv(RESULTS / "jaccard_m1_dlpfc.csv.gz", index=False)
    elif which == "m1_seqfish":
        m1_seqfish().to_csv(RESULTS / "jaccard_m1_seqfish.csv.gz", index=False)
    elif which == "m2":
        m2_semisynthetic()
    elif which == "m34":
        m3_m4()
        b1_jaccard_similarity()
