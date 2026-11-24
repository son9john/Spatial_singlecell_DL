"""Follow-up experiment: Testing two remedies for Seed-Weighted J.

Remedy 1 (Spatial Contiguity / Clustered Seed Filtering):
  - Filter out isolated/scattered noise seeds from First-Elbow using graph
    connected components (keeping clusters with size >= min_cluster_size).
  - Test both hard-filtered with graceful fallback to uniform J (SeedJ_clust)
    and coherence-gated soft blending (SeedJ_soft_clust).

Remedy 2 (Pair-Independent Map-Level Fixed Seeds + Spectral Null Calibration):
  - Extract a single anatomical seed set S_map via First-Elbow from a map-level
    average local J (independent of the test pair), construct a fixed
    distance-weighted Laplacian L_W, and test:
      (a) Pair ranking AUROC on DLPFC B2 (semi-synthetic) and DLPFC B1 (real)
      (b) Null distribution mean/std and exact empirical False-Positive Rate
          at alpha = 0.05 under the analytical Spectral Sign-Flip Null!
"""
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.linalg import cholesky, eigh
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from juxtaposition import standardize, edge_differences, spatial_weights, lee_l, juxtaposed_value
from seed_j_pilot import (
    CACHE, SECTIONS, DONOR, LAYER_ORDER, load_dlpfc, lognorm, scenario,
    scaled_scenario, build_grid_graph, build_visium_graph, smooth_signal,
    spot_local_j, find_first_elbow_seeds,
)

OUTDIR = Path(__file__).resolve().parent / "seed_j_remedies_results"
OUTDIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Remedy 1: Clustered Seed Filtering (Hard + Soft Coherence-Gated)
# =============================================================================
def filter_clustered_seeds(seeds, graph, min_cluster_size=4, two_hop=True):
    """Keep only spatially contiguous seed clusters of size >= min_cluster_size.

    If two_hop=True, connects seeds within 2 graph hops so 1-spot gaps along a
    noisy boundary do not fragment a genuine interface.
    Returns (kept_seeds, coherence_fraction, n_clusters, max_cluster_size).
    """
    if len(seeds) == 0:
        return np.array([], dtype=int), 0.0, 0, 0
    A = graph["A"]
    if two_hop:
        # 1-hop or 2-hop adjacency restricted to seeds
        sub_A1 = A[seeds][:, seeds]
        # Efficient 2-hop among seeds: (A[seeds, :] @ A[:, seeds]) > 0
        sub_A2 = (A[seeds, :] @ A[:, seeds])
        sub_adj = ((sub_A1 + sub_A2) > 0).astype(int)
    else:
        sub_adj = A[seeds][:, seeds]

    n_comp, labels = connected_components(sub_adj, directed=False)
    counts = np.bincount(labels, minlength=n_comp)
    keep_comp = np.where(counts >= min_cluster_size)[0]
    if len(keep_comp) == 0:
        return np.array([], dtype=int), 0.0, 0, int(counts.max()) if len(counts) else 0

    mask = np.isin(labels, keep_comp)
    kept = seeds[mask]
    coherence = float(len(kept)) / float(len(seeds))
    return kept, coherence, len(keep_comp), int(counts.max())


def score_remedy1(x, y, graph, sigma=4.0, ridge=1.0, min_cluster_size=4, return_diag=False):
    """Compare Standard J, Raw Seed-J, Hard Clustered Seed-J, and Soft-Gated Clustered Seed-J."""
    j_spot, dx, dy = spot_local_j(x, y, graph, ridge=ridge)
    raw_seeds, diag = find_first_elbow_seeds(j_spot)
    kept_seeds, coherence, n_clust, max_clust = filter_clustered_seeds(
        raw_seeds, graph, min_cluster_size=min_cluster_size, two_hop=True
    )

    j_std = float(juxtaposed_value(dx, dy))
    sig_phys = sigma * graph["spacing"]

    # Raw Seed-J (for direct comparison)
    d_raw = cKDTree(graph["coords"][raw_seeds]).query(graph["midpoints"])[0]
    w_raw = np.exp(-0.5 * (d_raw / sig_phys) ** 2)
    j_raw_grad = float(juxtaposed_value(dx, dy, weights=w_raw))

    # Remedy 1A: Hard Clustered Seed-J (graceful fallback to standard J if no cluster)
    # Remedy 1B: Soft Coherence-Gated Seed-J: w = (1 - alpha)*1 + alpha*w_clust
    if len(kept_seeds) > 0:
        d_cl = cKDTree(graph["coords"][kept_seeds]).query(graph["midpoints"])[0]
        w_cl = np.exp(-0.5 * (d_cl / sig_phys) ** 2)
        j_clust_grad = float(juxtaposed_value(dx, dy, weights=w_cl))
        global_e = float(np.sum(dx * dx + dy * dy))
        j_clust_gnorm = float(np.sum(w_cl * (-2.0 * dx * dy)) / global_e) if global_e > 0 else 0.0

        # Coherence-gated blend: when coherence is low (scattered noise), stays close to J_std
        alpha = coherence ** 2
        w_soft = (1.0 - alpha) * 1.0 + alpha * w_cl
        j_soft_clust = float(juxtaposed_value(dx, dy, weights=w_soft))
    else:
        w_cl = np.ones(len(dx))
        j_clust_grad = j_std
        j_clust_gnorm = j_std
        j_soft_clust = j_std

    res = dict(
        J_std=j_std,
        SeedJ_raw=j_raw_grad,
        SeedJ_clust=j_clust_grad,
        SeedJ_clust_gnorm=j_clust_gnorm,
        SeedJ_soft_clust=j_soft_clust,
        k_raw=len(raw_seeds),
        k_kept=len(kept_seeds),
        coherence=coherence,
        max_clust=max_clust,
    )
    if return_diag:
        diag["raw_seeds"] = raw_seeds
        diag["kept_seeds"] = kept_seeds
        diag["j_spot"] = j_spot
        diag["coherence"] = coherence
        return res, diag
    return res


# =============================================================================
# Remedy 2: Pair-Independent Map-Level Fixed Seeds & Weighted Spectral Null
# =============================================================================
def build_map_fixed_weights(Z_ref, graph, sigma=4.0, floor=0.1, min_cluster_size=4, max_pairs=400):
    """Compute a pair-independent map-level mean local J from a reference gene matrix,
    select fixed anatomical seeds via First-Elbow (+ cluster filter), and return
    fixed edge weights w_e in [floor, 1.0].
    """
    n, p = Z_ref.shape
    iu, ju = np.triu_indices(p, k=1)
    rng = np.random.default_rng(20261122)
    if len(iu) > max_pairs:
        pick = rng.choice(len(iu), size=max_pairs, replace=False)
        iu, ju = iu[pick], ju[pick]

    # Accumulate positive local J across reference gene pairs
    u, v = graph["edges"].T
    deg = graph["deg"]
    acc_spot = np.zeros(n, dtype=float)
    for i, j in zip(iu, ju):
        j_sp, _, _ = spot_local_j(Z_ref[:, i], Z_ref[:, j], graph, ridge=1.0)
        acc_spot += np.maximum(j_sp, 0.0)
    mean_spot_j = acc_spot / max(len(iu), 1)

    raw_seeds, diag = find_first_elbow_seeds(mean_spot_j)
    kept_seeds, coherence, _, _ = filter_clustered_seeds(
        raw_seeds, graph, min_cluster_size=min_cluster_size, two_hop=True
    )
    use_seeds = kept_seeds if len(kept_seeds) > 0 else raw_seeds
    d_edge = cKDTree(graph["coords"][use_seeds]).query(graph["midpoints"])[0]
    sig_phys = sigma * graph["spacing"]
    w_edge = floor + (1.0 - floor) * np.exp(-0.5 * (d_edge / sig_phys) ** 2)
    return w_edge, use_seeds, mean_spot_j


def build_weighted_laplacian(n, edges, w_edge):
    u, v = edges.T
    row = np.r_[u, v]
    col = np.r_[v, u]
    data = np.r_[w_edge, w_edge]
    A_w = sparse.csr_matrix((data, (row, col)), shape=(n, n))
    deg_w = np.asarray(A_w.sum(axis=1)).ravel()
    L_w = sparse.diags(deg_w) - A_w
    return L_w


# =============================================================================
# Experiment A: Amplitude Imbalance (Remedy 1)
# =============================================================================
def run_exp_a(n_rep=30, sigma=4.0, min_cluster_size=5):
    print("\n=== Exp A: Amplitude Imbalance (Remedy 1: Clustered Seed Filtering) ===", flush=True)
    shape = (42, 46)
    r, c = np.indices(shape)
    graph = build_grid_graph(shape)
    modes = ("focal", "shared", "unilateral", "displaced", "broad", "independent")
    rows = []
    diag_examples = {}

    for sd in (0.15, 0.35, 0.60):
        for amp in (1.0, 0.5, 0.25, 0.1):
            aucs = {m: [] for m in ("J_std", "SeedJ_raw", "SeedJ_clust", "SeedJ_clust_gnorm", "SeedJ_soft_clust")}
            coh_f, coh_i, kept_f, kept_i = [], [], [], []
            for rep in range(n_rep):
                sc = {m: [] for m in aucs}
                for mode in modes:
                    rng = np.random.default_rng(20261122 + rep)
                    cx, cy, side, window = scaled_scenario(mode, r, c, rng, amp)
                    x_obs = gaussian_filter(cx, 0.55) + rng.normal(0, sd, shape)
                    y_obs = gaussian_filter(cy, 0.55) + rng.normal(0, sd, shape)
                    x, y = standardize(x_obs.ravel()), standardize(y_obs.ravel())
                    want_diag = (rep == 0 and sd == 0.35 and amp == 0.5 and mode in ("focal", "independent"))
                    out = score_remedy1(x, y, graph, sigma=sigma, min_cluster_size=min_cluster_size, return_diag=want_diag)
                    if want_diag:
                        res, diag = out
                        diag_examples[mode] = dict(diag=diag, res=res, shape=shape)
                    else:
                        res = out
                    for m in sc:
                        sc[m].append(res[m])
                    if mode == "focal":
                        coh_f.append(res["coherence"])
                        kept_f.append(res["k_kept"])
                    elif mode == "independent":
                        coh_i.append(res["coherence"])
                        kept_i.append(res["k_kept"])

                labels = np.array([m == "focal" for m in modes], dtype=int)
                for m in aucs:
                    aucs[m].append(roc_auc_score(labels, sc[m]))

            row = dict(
                noise_sd=sd, amplitude=amp,
                coh_focal=np.mean(coh_f), coh_indep=np.mean(coh_i),
                kept_focal=np.mean(kept_f), kept_indep=np.mean(kept_i),
            )
            for m in aucs:
                row[m] = float(np.mean(aucs[m]))
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "expA_amplitude_remedy1.csv", index=False)
    print(df[df["noise_sd"] == 0.35][
        ["amplitude", "J_std", "SeedJ_raw", "SeedJ_clust", "SeedJ_soft_clust", "coh_focal", "coh_indep", "kept_focal", "kept_indep"]
    ].round(3).to_string(index=False))
    return df, diag_examples


# =============================================================================
# Experiment B: DLPFC B2 Semi-Synthetic (Remedy 1 + Remedy 2 Map-Level Fixed)
# =============================================================================
def run_exp_b(sections=("151507", "151669", "151673"), n_rep=2, betas=(0.5, 1.0, 2.0), per_layer=4):
    print("\n=== Exp B: DLPFC B2 Semi-Synthetic (Remedy 1 + Remedy 2 Map-Level Fixed) ===", flush=True)
    RANK = {l: k for k, l in enumerate(LAYER_ORDER)}
    rows = []

    for s in sections:
        t0 = time.time()
        counts, _, meta = load_dlpfc(s)
        graph = build_visium_graph(meta.array_row.values, meta.array_col.values)
        xy = graph["coords"]
        a = np.sqrt(3.0) * cdist(xy, xy) / 3.0
        lc = cholesky((1.0 + a) * np.exp(-a) + 1e-6 * np.eye(len(xy)), lower=True, check_finite=False)
        lib = np.asarray(counts.sum(axis=1)).ravel().astype(float)
        rel = lib / lib.mean()
        lay_counts = meta.layer.value_counts()
        present = [l for l in LAYER_ORDER if lay_counts.get(l, 0) >= 30]
        lay_of_gene = np.repeat(present, per_layer)
        p = len(lay_of_gene)
        ind = np.column_stack([(meta.layer.values == l).astype(float) for l in lay_of_gene])
        ind = graph["S"] @ ind
        rng = np.random.default_rng([20261122, int(s)])

        iu, ju = np.triu_indices(p, k=1)
        rk = np.array([RANK[l] for l in lay_of_gene])
        dist = np.abs(rk[iu] - rk[ju])
        adj, far, same = dist == 1, dist >= 2, dist == 0

        for beta in betas:
            for rep in range(n_rep):
                bg = 0.3 * (lc @ rng.normal(size=(len(xy), p)))
                logmu = beta * ind + bg
                lam = rel[:, None] * np.exp(logmu)
                lam *= 2.0 / lam.mean(axis=0, keepdims=True)
                y_cnt = rng.poisson(lam)
                X = np.log1p(1e4 * y_cnt / lib[:, None])
                Z = standardize(X)
                Z4 = smooth_signal(Z, graph["S"], 4)

                # Remedy 2: Build a separate independent reference panel on the same tissue
                # (simulating an independent set of SVGs on the same section for map-level seeds)
                bg_ref = 0.3 * (lc @ rng.normal(size=(len(xy), p)))
                lam_ref = rel[:, None] * np.exp(beta * ind + bg_ref)
                lam_ref *= 2.0 / lam_ref.mean(axis=0, keepdims=True)
                Z4_ref = smooth_signal(standardize(np.log1p(1e4 * rng.poisson(lam_ref) / lib[:, None])), graph["S"], 4)
                w_map_fixed, map_seeds, _ = build_map_fixed_weights(Z4_ref, graph, sigma=4.0, floor=0.1, min_cluster_size=5)

                methods = ("J4_std", "SeedJ4_raw", "SeedJ4_clust", "SeedJ4_soft_clust", "SeedJ4_mapfixed")
                sc = {m: np.zeros(len(iu)) for m in methods}

                for idx_pair, (i, j) in enumerate(zip(iu, ju)):
                    r4 = score_remedy1(Z4[:, i], Z4[:, j], graph, sigma=4.0, min_cluster_size=5)
                    dx, dy = edge_differences(Z4[:, i], Z4[:, j], graph["edges"])
                    sc["J4_std"][idx_pair] = r4["J_std"]
                    sc["SeedJ4_raw"][idx_pair] = r4["SeedJ_raw"]
                    sc["SeedJ4_clust"][idx_pair] = r4["SeedJ_clust"]
                    sc["SeedJ4_soft_clust"][idx_pair] = r4["SeedJ_soft_clust"]
                    sc["SeedJ4_mapfixed"][idx_pair] = float(juxtaposed_value(dx, dy, weights=w_map_fixed))

                for m in methods:
                    a_ad = roc_auc_score(np.r_[np.ones(adj.sum()), np.zeros(far.sum())],
                                         np.r_[sc[m][adj], sc[m][far]])
                    a_as = roc_auc_score(np.r_[np.ones(adj.sum()), np.zeros(same.sum())],
                                         np.r_[sc[m][adj], sc[m][same]])
                    rows.append(dict(section=s, beta=beta, rep=rep, method=m,
                                     auc_adj_vs_far=a_ad, auc_adj_vs_same=a_as, n_map_seeds=len(map_seeds)))
        print(f"  Section {s} done in {time.time() - t0:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "expB_dlpfc_b2_remedies.csv", index=False)
    print(df.groupby(["beta", "method"])[["auc_adj_vs_far", "auc_adj_vs_same"]].mean().unstack("beta").round(3))
    return df


# =============================================================================
# Experiment C: Real Observed DLPFC B1 (Remedy 1 + Remedy 2 Map-Level Fixed)
# =============================================================================
def run_exp_c(sections=("151507", "151669", "151673"), k_markers=5):
    print("\n=== Exp C: Real Observed DLPFC B1 (Remedy 1 + Remedy 2 Map-Level Fixed) ===", flush=True)
    RANK = {l: k for k, l in enumerate(LAYER_ORDER)}
    means, genes0 = {}, None
    for s in SECTIONS:
        counts, genes, meta = load_dlpfc(s)
        if genes0 is None:
            genes0 = genes
        x = lognorm(counts)
        lay = meta.layer.values
        cols = {}
        for l in LAYER_ORDER:
            m = lay == l
            if m.sum() >= 30:
                cols[l] = np.asarray(x[m].mean(axis=0)).ravel()
        means[s] = pd.DataFrame(cols)

    rows = []
    for s in sections:
        t0 = time.time()
        test_donor = DONOR[s]
        train = [ts for ts in SECTIONS if DONOR[ts] != test_donor]
        margin_sum = {l: np.zeros(len(genes0)) for l in LAYER_ORDER}
        mean_sum = {l: np.zeros(len(genes0)) for l in LAYER_ORDER}
        cnt = {l: 0 for l in LAYER_ORDER}
        for ts in train:
            df_m = means[ts]
            for l in df_m.columns:
                others = df_m.drop(columns=l).to_numpy()
                margin_sum[l] += df_m[l].to_numpy() - others.max(axis=1)
                mean_sum[l] += df_m[l].to_numpy()
                cnt[l] += 1
        present_all = [l for l in LAYER_ORDER if cnt[l] > 0]
        margin = np.column_stack([margin_sum[l] / cnt[l] for l in present_all])
        mean_in = np.column_stack([mean_sum[l] / cnt[l] for l in present_all])
        best = margin.argmax(axis=1)
        best_margin = margin[np.arange(len(genes0)), best]
        best_mean = mean_in[np.arange(len(genes0)), best]
        ok = (best_margin > 0) & (best_mean >= 0.1) & ~pd.Series(genes0).str.upper().str.startswith("MT-").values
        markers = {}
        for c_idx, l in enumerate(present_all):
            idx = np.where(ok & (best == c_idx))[0]
            idx = idx[np.argsort(-best_margin[idx])][:k_markers]
            markers[l] = idx

        counts, _, meta = load_dlpfc(s)
        graph = build_visium_graph(meta.array_row.values, meta.array_col.values)
        lay_counts = meta.layer.value_counts()
        present = [l for l in LAYER_ORDER if lay_counts.get(l, 0) >= 30 and l in markers]
        idx_list = np.concatenate([markers[l] for l in present])
        lay_list = np.concatenate([[l] * len(markers[l]) for l in present])

        X_all = lognorm(counts)
        X = X_all[:, idx_list].toarray()
        keep = X.std(axis=0) > 0
        X, lay_list = X[:, keep], lay_list[keep]
        Z4 = smooth_signal(standardize(X), graph["S"], 4)

        # Remedy 2: Map-level fixed seeds from top-40 non-marker high-variance genes on the section
        marker_set = set(idx_list.tolist())
        # Compute detection rate & variance quickly on top expressed genes
        nnz_per_gene = np.asarray((X_all > 0).sum(axis=0)).ravel()
        cand_genes = np.where((nnz_per_gene >= 0.15 * X_all.shape[0]) &
                              ~pd.Series(genes0).str.upper().str.startswith("MT-").values)[0]
        cand_genes = np.array([g for g in cand_genes if g not in marker_set])
        # Pick top 40 by variance in lognorm expression
        sub_ref = X_all[:, cand_genes[:1000]].toarray()
        top_var_idx = np.argsort(-sub_ref.var(axis=0))[:40]
        Z4_ref = smooth_signal(standardize(sub_ref[:, top_var_idx]), graph["S"], 4)
        w_map_fixed, map_seeds, _ = build_map_fixed_weights(Z4_ref, graph, sigma=4.0, floor=0.1, min_cluster_size=5)

        p = Z4.shape[1]
        iu, ju = np.triu_indices(p, k=1)
        rk = np.array([RANK[l] for l in lay_list])
        dist = np.abs(rk[iu] - rk[ju])
        adj, far, same = dist == 1, dist >= 2, dist == 0

        methods = ("J4_std", "SeedJ4_raw", "SeedJ4_clust", "SeedJ4_soft_clust", "SeedJ4_mapfixed")
        sc = {m: np.zeros(len(iu)) for m in methods}
        for idx_pair, (i, j) in enumerate(zip(iu, ju)):
            r4 = score_remedy1(Z4[:, i], Z4[:, j], graph, sigma=4.0, min_cluster_size=5)
            dx, dy = edge_differences(Z4[:, i], Z4[:, j], graph["edges"])
            sc["J4_std"][idx_pair] = r4["J_std"]
            sc["SeedJ4_raw"][idx_pair] = r4["SeedJ_raw"]
            sc["SeedJ4_clust"][idx_pair] = r4["SeedJ_clust"]
            sc["SeedJ4_soft_clust"][idx_pair] = r4["SeedJ_soft_clust"]
            sc["SeedJ4_mapfixed"][idx_pair] = float(juxtaposed_value(dx, dy, weights=w_map_fixed))

        for m in methods:
            a_ad = roc_auc_score(np.r_[np.ones(adj.sum()), np.zeros(far.sum())],
                                 np.r_[sc[m][adj], sc[m][far]])
            a_as = roc_auc_score(np.r_[np.ones(adj.sum()), np.zeros(same.sum())],
                                 np.r_[sc[m][adj], sc[m][same]])
            rows.append(dict(section=s, method=m, auc_adj_vs_far=a_ad, auc_adj_vs_same=a_as))
        print(f"  Real DLPFC {s} done in {time.time() - t0:.1f}s (map_seeds={len(map_seeds)})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "expC_dlpfc_b1_remedies.csv", index=False)
    print(df.groupby("method")[["auc_adj_vs_far", "auc_adj_vs_same"]].mean().round(3))
    return df


# =============================================================================
# Experiment D: Null Bias & Exact Analytical Spectral Null Calibration (FPR @ 0.05)
# =============================================================================
def run_exp_d(n_pairs=400, sigma=4.0, min_cluster_size=5):
    print("\n=== Exp D: Null Bias & Spectral Sign-Flip FPR Calibration (alpha=0.05) ===", flush=True)
    # Use a 32x34 grid (1,088 spots) so dense eigendecomposition is instant (<0.4s)
    shape = (32, 34)
    r, c = np.indices(shape)
    graph = build_grid_graph(shape)
    xy = graph["coords"]
    n = len(xy)

    # Build a fixed anatomical map-level seed weight (from an independent focal tissue boundary)
    rng_ref = np.random.default_rng(999)
    cx, cy, _, _ = scenario("focal", r, c, rng_ref)
    x_ref = standardize((gaussian_filter(cx, 0.55) + rng_ref.normal(0, 0.35, shape)).ravel())
    y_ref = standardize((gaussian_filter(cy, 0.55) + rng_ref.normal(0, 0.35, shape)).ravel())
    j_sp_ref, _, _ = spot_local_j(x_ref, y_ref, graph, ridge=1.0)
    raw_ref, _ = find_first_elbow_seeds(j_sp_ref)
    kept_ref, _, _, _ = filter_clustered_seeds(raw_ref, graph, min_cluster_size=min_cluster_size, two_hop=True)
    d_edge_fixed = cKDTree(graph["coords"][kept_ref]).query(graph["midpoints"])[0]
    w_fixed = 0.1 + 0.9 * np.exp(-0.5 * (d_edge_fixed / (sigma * graph["spacing"])) ** 2)

    # Eigendecompositions for analytical spectral sign-flip null:
    # 1) Unweighted Laplacian L (for standard J)
    mu_std, U_std = eigh(graph["L"].toarray())
    # 2) Fixed weighted Laplacian L_W (for Remedy 2 Map-Fixed Seed-J)
    L_w = build_weighted_laplacian(n, graph["edges"], w_fixed).toarray()
    mu_w, U_w = eigh(L_w)

    # Matern nu=5/2, range=3 Cholesky for spatially autocorrelated independent null fields
    a = np.sqrt(5.0) * cdist(xy, xy) / 3.0
    K_mat = (1.0 + a + (a * a) / 3.0) * np.exp(-a) + 1e-6 * np.eye(n)
    L_chol = cholesky(K_mat, lower=True, check_finite=False)

    rng = np.random.default_rng(20261122)
    records = []
    for regime in ("white_noise", "matern_52_r3"):
        for _ in range(n_pairs):
            if regime == "white_noise":
                x = standardize(rng.normal(size=n))
                y = standardize(rng.normal(size=n))
            else:
                x = standardize(L_chol @ rng.normal(size=n))
                y = standardize(L_chol @ rng.normal(size=n))

            r1 = score_remedy1(x, y, graph, sigma=sigma, min_cluster_size=min_cluster_size)
            dx, dy = edge_differences(x, y, graph["edges"])
            j_mapfixed = float(juxtaposed_value(dx, dy, weights=w_fixed))

            # Analytical spectral sign-flip z-score for Standard J
            xh = U_std.T @ x
            yh = U_std.T @ y
            C_std = float(np.sum(mu_std * xh * yh))
            var_std = float(np.sum((mu_std ** 2) * (xh ** 2) * (yh ** 2)))
            z_std = -C_std / np.sqrt(max(var_std, 1e-18))
            p_std = float(2.0 * norm.sf(abs(z_std)))

            # Analytical spectral sign-flip z-score for Map-Fixed Weighted Laplacian L_W
            xh_w = U_w.T @ x
            yh_w = U_w.T @ y
            C_w = float(np.sum(mu_w * xh_w * yh_w))
            var_w = float(np.sum((mu_w ** 2) * (xh_w ** 2) * (yh_w ** 2)))
            z_mapfixed = -C_w / np.sqrt(max(var_w, 1e-18))
            p_mapfixed = float(2.0 * norm.sf(abs(z_mapfixed)))

            records.append(dict(
                regime=regime,
                J_std=r1["J_std"],
                SeedJ_raw=r1["SeedJ_raw"],
                SeedJ_clust=r1["SeedJ_clust"],
                SeedJ_soft_clust=r1["SeedJ_soft_clust"],
                SeedJ_mapfixed=j_mapfixed,
                coherence=r1["coherence"],
                k_kept=r1["k_kept"],
                reject_std=int(p_std < 0.05),
                reject_mapfixed=int(p_mapfixed < 0.05),
            ))

    df = pd.DataFrame(records)
    df.to_csv(OUTDIR / "expD_null_calibration_remedies.csv", index=False)
    for reg, g in df.groupby("regime"):
        print(f"\nRegime: {reg} (n={len(g)})")
        print(f"  Mean scores -> J_std: {g['J_std'].mean():+.4f}, SeedJ_raw: {g['SeedJ_raw'].mean():+.4f}, "
              f"SeedJ_clust: {g['SeedJ_clust'].mean():+.4f}, SeedJ_soft_clust: {g['SeedJ_soft_clust'].mean():+.4f}, "
              f"SeedJ_mapfixed: {g['SeedJ_mapfixed'].mean():+.4f}")
        print(f"  Mean cluster coherence: {g['coherence'].mean():.3f}, fallback-to-std rate (k_kept==0): {(g['k_kept']==0).mean()*100:.1f}%")
        print(f"  Spectral Null FPR @ alpha=0.05 -> J_std: {g['reject_std'].mean()*100:.2f}%, "
              f"SeedJ_mapfixed (L_W): {g['reject_mapfixed'].mean()*100:.2f}%")
    return df


# =============================================================================
# Plotting
# =============================================================================
def make_figure(dfA, diag_ex, dfB, dfC, dfD):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.2), dpi=160)

    # Panel A: Amplitude Imbalance (SD=0.35)
    ax = axes[0, 0]
    sub = dfA[dfA["noise_sd"] == 0.35].sort_values("amplitude")
    amps = sub["amplitude"].values
    ax.plot(amps, sub["J_std"], "o--", lw=2.0, color="#7f8c8d", label="Standard J")
    ax.plot(amps, sub["SeedJ_raw"], "s:", lw=1.8, color="#e74c3c", label="Seed-J (Raw First-Elbow)")
    ax.plot(amps, sub["SeedJ_clust"], "^-", lw=2.3, color="#27ae60", label="Remedy 1A: Clustered Seed-J")
    ax.plot(amps, sub["SeedJ_soft_clust"], "d-", lw=2.3, color="#2980b9", label="Remedy 1B: Coherence-Gated Seed-J")
    ax.set_title("A. Amplitude Imbalance (Noise SD=0.35)", fontweight="bold")
    ax.set_xlabel("Second-marker contrast amplitude ratio")
    ax.set_ylabel("Pair AUROC (Focal vs 5 Distractors)")
    ax.set_ylim(0.45, 1.03)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.25)

    # Panel B: DLPFC B2 Semi-Synthetic (Adj vs Distant)
    ax = axes[0, 1]
    b2 = dfB.groupby(["beta", "method"])["auc_adj_vs_far"].mean().unstack("method")
    betas = b2.index.values
    for col, lbl, clr, ls in [
        ("J4_std", "J4 (Standard)", "#7f8c8d", "--"),
        ("SeedJ4_raw", "Seed-J4 (Raw)", "#e74c3c", ":"),
        ("SeedJ4_clust", "Remedy 1A: Clustered", "#27ae60", "-"),
        ("SeedJ4_soft_clust", "Remedy 1B: Soft-Gated", "#2980b9", "-"),
        ("SeedJ4_mapfixed", "Remedy 2: Map-Fixed", "#8e44ad", "-"),
    ]:
        if col in b2.columns:
            ax.plot(betas, b2[col], marker="o", lw=2.2, ls=ls, color=clr, label=lbl)
    ax.set_title("B. DLPFC B2 Semi-Synthetic (Adj vs Distant)", fontweight="bold")
    ax.set_xlabel("Layer effect size β (log-fold)")
    ax.set_ylabel("Mean AUROC (3 donors)")
    ax.set_ylim(0.55, 1.02)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.25)

    # Panel C: Null Distribution Shift (Matern nu=5/2, r=3)
    ax = axes[1, 0]
    subD = dfD[dfD["regime"] == "matern_52_r3"]
    for col, lbl, clr in [
        ("J_std", "Standard J", "#7f8c8d"),
        ("SeedJ_raw", "Seed-J (Raw)", "#e74c3c"),
        ("SeedJ_soft_clust", "Remedy 1B: Soft-Gated", "#2980b9"),
        ("SeedJ_mapfixed", "Remedy 2: Map-Fixed", "#8e44ad"),
    ]:
        ax.hist(subD[col], bins=26, alpha=0.45, color=clr,
                label=f"{lbl} (μ={subD[col].mean():+.3f})", density=True)
    ax.axvline(0.0, color="black", ls="--", lw=1.2)
    ax.set_title("C. Null Distribution Under Independent Matérn Fields", fontweight="bold")
    ax.set_xlabel("Score value under true null")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8.2)
    ax.grid(alpha=0.25)

    # Panel D: Spectral Null FPR @ alpha=0.05 + Cluster Filter Effect
    ax = axes[1, 1]
    regimes = ["white_noise", "matern_52_r3"]
    labels_x = ["White Noise\n(400 pairs)", "Matérn ν=5/2, r=3\n(400 pairs)"]
    fpr_std = [dfD[dfD["regime"] == rg]["reject_std"].mean() * 100 for rg in regimes]
    fpr_map = [dfD[dfD["regime"] == rg]["reject_mapfixed"].mean() * 100 for rg in regimes]
    x_pos = np.arange(len(regimes))
    w = 0.32
    b1 = ax.bar(x_pos - w/2, fpr_std, w, color="#7f8c8d", label="Standard J (Spectral Null)")
    b2 = ax.bar(x_pos + w/2, fpr_map, w, color="#8e44ad", label="Remedy 2: Map-Fixed J (L_W Spectral Null)")
    ax.axhline(5.0, color="#e74c3c", ls="--", lw=1.5, label="Nominal α = 5.0%")
    ax.axhspan(2.5, 7.5, color="#e74c3c", alpha=0.10, label="Bradley interval [2.5%, 7.5%]")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_x)
    ax.set_ylim(0, 12.5)
    ax.set_ylabel("False Positive Rate at α = 0.05 (%)")
    ax.set_title("D. Analytical Spectral Null Calibration (Remedy 2)", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    fig.savefig(OUTDIR / "fig_remedies_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure to {OUTDIR / 'fig_remedies_comparison.png'}", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    dfA, diag_ex = run_exp_a(n_rep=30, sigma=4.0, min_cluster_size=5)
    dfB = run_exp_b(sections=("151507", "151669", "151673"), n_rep=2, betas=(0.5, 1.0, 2.0), per_layer=4)
    dfC = run_exp_c(sections=("151507", "151669", "151673"), k_markers=5)
    dfD = run_exp_d(n_pairs=400, sigma=4.0, min_cluster_size=5)
    make_figure(dfA, diag_ex, dfB, dfC, dfD)
    print(f"\nALL REMEDY EXPERIMENTS COMPLETED in {time.time() - t0:.1f}s")
