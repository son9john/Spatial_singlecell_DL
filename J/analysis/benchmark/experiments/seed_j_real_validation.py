"""Real-tissue follow-up validations for Remedy 1B and Remedy 2.

Validation 1 — Serial-Section Reproducibility (DLPFC 151507 vs 151508, 10 um apart):
  Tests whether Remedy 1B (SeedJ_soft_clust) and Remedy 1A (SeedJ_clust) improve
  top-1% / top-2% / top-5% Jaccard reproducibility and rank correlation over
  standard Boundary J (scales 1 and 4) on the top-80 spatially variable genes
  (3,160 pairs) shared between serial sections of donor Br5292.

Validation 2 — Single-Boundary Targeted Screening + Exact U_std Spectral Null
  (Real DLPFC 151507 L6-WM boundary; note: CRC .h5 requires h5py not installed locally):
  Uses an independent reference contrast (or held-out L6-WM genes) to fix a
  single anatomical boundary seed set S_map -> fixed weighted Laplacian L_W.
  Evaluates:
    (a) Power (|z|-score & BH q < 0.05) and AUROC on held-out L6-WM cross-boundary
        pairs vs within-compartment / housekeeping controls.
    (b) Real-data null calibration (FPR at alpha = 0.05) on 2,500 cross-donor
        rotated pairs (151507 x 151673, 50 x 50 top SVGs).
"""
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import eigh
from scipy.spatial import cKDTree
from scipy.stats import norm, spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from juxtaposition import standardize, spatial_weights, lee_l, juxtaposed_value
from seed_j_pilot import (
    load_dlpfc, lognorm, build_visium_graph, smooth_signal,
    spot_local_j, find_first_elbow_seeds,
)
from seed_j_remedies import filter_clustered_seeds, score_remedy1, build_weighted_laplacian

OUTDIR = Path(__file__).resolve().parent / "seed_j_real_validation_results"
OUTDIR.mkdir(parents=True, exist_ok=True)


def moran_i_vec(Z, W):
    """Univariate Moran's I for each column of standardized matrix Z."""
    WZ = W @ Z
    s0 = float(W.sum())
    return np.sum(Z * WZ, axis=0) / s0


def jaccard_topk(scores_a, scores_b, frac):
    k = max(1, int(round(len(scores_a) * frac)))
    top_a = set(np.argsort(-scores_a)[:k].tolist())
    top_b = set(np.argsort(-scores_b)[:k].tolist())
    return len(top_a & top_b) / len(top_a | top_b)


def bh_qvalues(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


# =============================================================================
# Validation 1: Serial-Section Reproducibility (DLPFC 151507 vs 151508)
# =============================================================================
def run_validation1(n_genes=80):
    print("\n=== Validation 1: Serial-Section Reproducibility (151507 vs 151508) ===", flush=True)
    t0 = time.time()
    c1, g1, m1 = load_dlpfc("151507")
    c2, g2, m2 = load_dlpfc("151508")
    assert np.array_equal(g1, g2)

    x1_sp, x2_sp = lognorm(c1), lognorm(c2)
    det1 = np.asarray((c1 > 0).mean(axis=0)).ravel()
    det2 = np.asarray((c2 > 0).mean(axis=0)).ravel()
    is_mt = pd.Series(g1).str.upper().str.startswith("MT-").values
    elig = np.where((det1 >= 0.10) & (det2 >= 0.10) & ~is_mt)[0]

    gr1 = build_visium_graph(m1.array_row.values, m1.array_col.values)
    gr2 = build_visium_graph(m2.array_row.values, m2.array_col.values)
    w1 = spatial_weights(len(m1), gr1["edges"])
    w2 = spatial_weights(len(m2), gr2["edges"])

    # Compute Moran's I on eligible genes to pick top n_genes SVGs (matching Fig 6A protocol)
    Z1_el = standardize(x1_sp[:, elig].toarray())
    Z2_el = standardize(x2_sp[:, elig].toarray())
    mi = 0.5 * (moran_i_vec(Z1_el, w1) + moran_i_vec(Z2_el, w2))
    top_local = np.argsort(-mi)[:n_genes]
    Z1 = Z1_el[:, top_local]
    Z2 = Z2_el[:, top_local]
    gene_names = g1[elig[top_local]]

    Z1_s1, Z2_s1 = smooth_signal(Z1, gr1["S"], 1), smooth_signal(Z2, gr2["S"], 1)
    Z1_s4, Z2_s4 = smooth_signal(Z1, gr1["S"], 4), smooth_signal(Z2, gr2["S"], 4)

    iu, ju = np.triu_indices(n_genes, k=1)
    n_pairs = len(iu)
    print(f"  Scoring {n_pairs} pairs ({n_genes} SVGs) on 151507 and 151508...", flush=True)

    methods = ("J1_std", "SeedJ1_raw", "SeedJ1_clust", "SeedJ1_soft_clust",
               "J4_std", "SeedJ4_raw", "SeedJ4_clust", "SeedJ4_soft_clust", "Lee_L")
    s1_scores = {m: np.zeros(n_pairs) for m in methods}
    s2_scores = {m: np.zeros(n_pairs) for m in methods}

    # Vectorized Lee's L
    def lee_pairs(Z, W):
        WZ = W @ Z
        WZ = WZ - WZ.mean(axis=0, keepdims=True)
        mass = float(np.sum(np.asarray(W.sum(axis=1)).ravel() ** 2))
        norms = np.linalg.norm(Z, axis=0)
        Lmat = (len(Z) / mass) * (WZ.T @ WZ) / np.outer(norms, norms)
        return Lmat[iu, ju]

    s1_scores["Lee_L"] = lee_pairs(Z1, w1)
    s2_scores["Lee_L"] = lee_pairs(Z2, w2)

    for idx, (i, j) in enumerate(zip(iu, ju)):
        r1_1 = score_remedy1(Z1_s1[:, i], Z1_s1[:, j], gr1, sigma=4.0, min_cluster_size=5)
        r2_1 = score_remedy1(Z2_s1[:, i], Z2_s1[:, j], gr2, sigma=4.0, min_cluster_size=5)
        r1_4 = score_remedy1(Z1_s4[:, i], Z1_s4[:, j], gr1, sigma=4.0, min_cluster_size=5)
        r2_4 = score_remedy1(Z2_s4[:, i], Z2_s4[:, j], gr2, sigma=4.0, min_cluster_size=5)

        s1_scores["J1_std"][idx] = r1_1["J_std"]
        s2_scores["J1_std"][idx] = r2_1["J_std"]
        s1_scores["SeedJ1_raw"][idx] = r1_1["SeedJ_raw"]
        s2_scores["SeedJ1_raw"][idx] = r2_1["SeedJ_raw"]
        s1_scores["SeedJ1_clust"][idx] = r1_1["SeedJ_clust"]
        s2_scores["SeedJ1_clust"][idx] = r2_1["SeedJ_clust"]
        s1_scores["SeedJ1_soft_clust"][idx] = r1_1["SeedJ_soft_clust"]
        s2_scores["SeedJ1_soft_clust"][idx] = r2_1["SeedJ_soft_clust"]

        s1_scores["J4_std"][idx] = r1_4["J_std"]
        s2_scores["J4_std"][idx] = r2_4["J_std"]
        s1_scores["SeedJ4_raw"][idx] = r1_4["SeedJ_raw"]
        s2_scores["SeedJ4_raw"][idx] = r2_4["SeedJ_raw"]
        s1_scores["SeedJ4_clust"][idx] = r1_4["SeedJ_clust"]
        s2_scores["SeedJ4_clust"][idx] = r2_4["SeedJ_clust"]
        s1_scores["SeedJ4_soft_clust"][idx] = r1_4["SeedJ_soft_clust"]
        s2_scores["SeedJ4_soft_clust"][idx] = r2_4["SeedJ_soft_clust"]

    rows = []
    for m in methods:
        a, b = s1_scores[m], s2_scores[m]
        rows.append(dict(
            method=m,
            jaccard_top1=jaccard_topk(a, b, 0.01),
            jaccard_top2=jaccard_topk(a, b, 0.02),
            jaccard_top5=jaccard_topk(a, b, 0.05),
            spearman_r=float(spearmanr(a, b).statistic),
            pearson_r=float(np.corrcoef(a, b)[0, 1]),
        ))
    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "val1_serial_reproducibility.csv", index=False)
    print(df.round(3).to_string(index=False))
    print(f"  Validation 1 done in {time.time() - t0:.1f}s", flush=True)
    return df


# =============================================================================
# Validation 2: Real DLPFC 151507 Single-Boundary Targeted Screening (L6-WM)
#               + Exact U_std Spectral Null Power & Cross-Donor Calibration
# =============================================================================
def run_validation2():
    print("\n=== Validation 2: Real DLPFC 151507 L6-WM Targeted Screening (Remedy 2) ===", flush=True)
    t0 = time.time()
    c1, genes, m1 = load_dlpfc("151507")
    gr1 = build_visium_graph(m1.array_row.values, m1.array_col.values)
    x1_sp = lognorm(c1)
    n = len(m1)

    # Identify top L6 and WM markers using OTHER donors (151669-151676) to avoid leakage
    from seed_j_pilot import SECTIONS, DONOR, LAYER_ORDER
    train_secs = [s for s in SECTIONS if DONOR[s] != "Br5292"]
    l6_margin = np.zeros(len(genes))
    wm_margin = np.zeros(len(genes))
    l6_mean = np.zeros(len(genes))
    wm_mean = np.zeros(len(genes))
    for ts in train_secs:
        ct, _, mt = load_dlpfc(ts)
        xt = lognorm(ct)
        lay = mt.layer.values
        means_l = {}
        for l in LAYER_ORDER:
            mask = lay == l
            if mask.sum() >= 30:
                means_l[l] = np.asarray(xt[mask].mean(axis=0)).ravel()
        others_l6 = np.column_stack([v for k, v in means_l.items() if k != "L6"]).max(axis=1)
        others_wm = np.column_stack([v for k, v in means_l.items() if k != "WM"]).max(axis=1)
        l6_margin += means_l["L6"] - others_l6
        wm_margin += means_l["WM"] - others_wm
        l6_mean += means_l["L6"]
        wm_mean += means_l["WM"]
    l6_margin /= len(train_secs)
    wm_margin /= len(train_secs)
    l6_mean /= len(train_secs)
    wm_mean /= len(train_secs)
    is_mt = pd.Series(genes).str.upper().str.startswith("MT-").values

    l6_idx = np.where((l6_margin > 0) & (l6_mean >= 0.1) & ~is_mt)[0]
    l6_idx = l6_idx[np.argsort(-l6_margin[l6_idx])][:15]
    wm_idx = np.where((wm_margin > 0) & (wm_mean >= 0.1) & ~is_mt)[0]
    wm_idx = wm_idx[np.argsort(-wm_margin[wm_idx])][:15]

    # Split into: (1) Reference seed-building pair set (top 3 L6 x top 3 WM = 9 pairs)
    #             (2) Held-out evaluation set (next 12 L6 x next 12 WM = 144 true L6-WM pairs
    #                 + within-L6 + within-WM + 10 non-boundary control genes)
    ref_l6, test_l6 = l6_idx[:3], l6_idx[3:]
    ref_wm, test_wm = wm_idx[:3], wm_idx[3:]

    # Build fixed L6-WM anatomical boundary seed map ONLY from the 3x3 reference genes (scale=4)
    Z_ref_l6 = smooth_signal(standardize(x1_sp[:, ref_l6].toarray()), gr1["S"], 4)
    Z_ref_wm = smooth_signal(standardize(x1_sp[:, ref_wm].toarray()), gr1["S"], 4)
    acc_spot = np.zeros(n, dtype=float)
    for a in range(Z_ref_l6.shape[1]):
        for b in range(Z_ref_wm.shape[1]):
            jsp, _, _ = spot_local_j(Z_ref_l6[:, a], Z_ref_wm[:, b], gr1, ridge=1.0)
            acc_spot += np.maximum(jsp, 0.0)
    mean_l6wm_j = acc_spot / (Z_ref_l6.shape[1] * Z_ref_wm.shape[1])

    raw_seeds, _ = find_first_elbow_seeds(mean_l6wm_j)
    kept_seeds, coh, _, _ = filter_clustered_seeds(raw_seeds, gr1, min_cluster_size=5, two_hop=True)
    use_seeds = kept_seeds if len(kept_seeds) > 0 else raw_seeds
    d_edge = cKDTree(gr1["coords"][use_seeds]).query(gr1["midpoints"])[0]
    w_fixed = 0.05 + 0.95 * np.exp(-0.5 * (d_edge / 4.0) ** 2)

    # Check how well the automatically extracted seeds overlap with the true L6-WM boundary
    u, v = gr1["edges"].T
    lay_arr = m1.layer.values
    true_l6wm_edge = ((lay_arr[u] == "L6") & (lay_arr[v] == "WM")) | ((lay_arr[u] == "WM") & (lay_arr[v] == "L6"))
    seed_set = set(use_seeds.tolist())
    seed_near_l6wm = np.mean([((u[e] in seed_set or v[e] in seed_set)) for e in np.where(true_l6wm_edge)[0]])
    print(f"  L6-WM Reference Seeds: k={len(use_seeds)} (coherence={coh:.2f}), "
          f"true L6-WM edges touched by seeds: {seed_near_l6wm*100:.1f}%", flush=True)

    # Eigendecomposition of unweighted tissue Laplacian L (one-time per section)
    print(f"  Computing eigendecomposition of {n}x{n} unweighted Laplacian...", flush=True)
    mu_std, U_std = eigh(gr1["L"].toarray())
    d_max = float(gr1["deg"].max())
    # Scale-4 smoothing eigenvalues: s_k = (1 - mu_k / (2*d_max))^4
    s_eig = (1.0 - mu_std / (2.0 * d_max)) ** 4
    h_std = mu_std * (s_eig ** 2)

    # Fixed weighted Laplacian L_W in unweighted eigenbasis U_std (with scale-4 smoother S^4)
    L_w = build_weighted_laplacian(n, gr1["edges"], w_fixed).toarray()
    M_hat = (U_std.T @ L_w @ U_std) * np.outer(s_eig, s_eig)
    M_hat_sq = M_hat ** 2

    # Evaluate held-out test panel: 12 L6 + 12 WM + 10 non-differential expressed control genes
    used_set = set(l6_idx.tolist() + wm_idx.tolist())
    det1 = np.asarray((c1 > 0).mean(axis=0)).ravel()
    ctrl_pool = np.where((det1 >= 0.30) & ~is_mt & (np.abs(l6_mean - wm_mean) < 0.05))[0]
    ctrl_pool = np.array([g for g in ctrl_pool if g not in used_set])
    rng = np.random.default_rng(20261122)
    ctrl_idx = rng.choice(ctrl_pool, size=min(10, len(ctrl_pool)), replace=False)

    panel_idx = np.concatenate([test_l6, test_wm, ctrl_idx])
    panel_grp = np.array(["L6"] * len(test_l6) + ["WM"] * len(test_wm) + ["CTRL"] * len(ctrl_idx))
    panel_names = genes[panel_idx]
    Z_panel = standardize(x1_sp[:, panel_idx].toarray())
    Z_panel_s1 = smooth_signal(Z_panel, gr1["S"], 4)

    # Standard J1 + exact spectral z/p/q
    XH = U_std.T @ Z_panel
    XH2 = XH ** 2
    C_std = Z_panel_s1.T @ (gr1["L"] @ Z_panel_s1)
    E_std = np.diag(C_std)
    J_std_mat = np.clip(-2.0 * C_std / np.maximum(E_std[:, None] + E_std[None, :], 1e-18), -1.0, 1.0)
    Var_std = XH2.T @ ((h_std ** 2)[:, None] * XH2)
    Z_std_mat = -C_std / np.sqrt(np.maximum(Var_std, 1e-18))

    # Remedy 2 Map-Fixed Seed-J1 + exact U_std quadratic-form spectral z/p/q
    C_fix = Z_panel_s1.T @ (L_w @ Z_panel_s1)
    E_fix = np.diag(C_fix)
    J_fix_mat = np.clip(-2.0 * C_fix / np.maximum(E_fix[:, None] + E_fix[None, :], 1e-18), -1.0, 1.0)
    Var_fix = XH2.T @ (M_hat_sq @ XH2)
    Z_fix_mat = -C_fix / np.sqrt(np.maximum(Var_fix, 1e-18))

    iu, ju = np.triu_indices(len(panel_idx), k=1)
    p_std = 2.0 * norm.sf(np.abs(Z_std_mat[iu, ju]))
    p_fix = 2.0 * norm.sf(np.abs(Z_fix_mat[iu, ju]))
    q_std = bh_qvalues(p_std)
    q_fix = bh_qvalues(p_fix)

    g_u, g_v = panel_grp[iu], panel_grp[ju]
    is_target = ((g_u == "L6") & (g_v == "WM")) | ((g_u == "WM") & (g_v == "L6"))
    is_within = ((g_u == "L6") & (g_v == "L6")) | ((g_u == "WM") & (g_v == "WM"))
    is_ctrl = (g_u == "CTRL") | (g_v == "CTRL")

    pair_df = pd.DataFrame(dict(
        gene_x=panel_names[iu], gene_y=panel_names[ju],
        grp_x=g_u, grp_y=g_v,
        pair_type=np.where(is_target, "L6_vs_WM", np.where(is_within, "Within_L6_WM", "Control")),
        J1_std=J_std_mat[iu, ju], z_std=Z_std_mat[iu, ju], q_std=q_std,
        J1_mapfixed=J_fix_mat[iu, ju], z_mapfixed=Z_fix_mat[iu, ju], q_mapfixed=q_fix,
    ))
    pair_df.to_csv(OUTDIR / "val2_dlpfc151507_l6wm_pairs.csv", index=False)

    # Metrics on target L6-WM pairs
    tgt = pair_df[pair_df["pair_type"] == "L6_vs_WM"]
    auc_std_j = roc_auc_score(is_target, pair_df["J1_std"])
    auc_fix_j = roc_auc_score(is_target, pair_df["J1_mapfixed"])
    auc_std_z = roc_auc_score(is_target, pair_df["z_std"])
    auc_fix_z = roc_auc_score(is_target, pair_df["z_mapfixed"])

    # Power: fraction of L6-WM pairs called significant (positive opposition, q < 0.05)
    pow_std = float(((tgt["q_std"] < 0.05) & (tgt["z_std"] > 0)).mean())
    pow_fix = float(((tgt["q_mapfixed"] < 0.05) & (tgt["z_mapfixed"] > 0)).mean())
    med_z_std = float(tgt["z_std"].median())
    med_z_fix = float(tgt["z_mapfixed"].median())
    med_j_std = float(tgt["J1_std"].median())
    med_j_fix = float(tgt["J1_mapfixed"].median())

    print(f"  Held-out L6-WM pairs (n={len(tgt)}):")
    print(f"    Standard J1   -> median J={med_j_std:+.3f}, median z={med_z_std:+.2f}, power (q<0.05, z>0)={pow_std*100:.1f}%, AUROC(J)={auc_std_j:.3f}, AUROC(z)={auc_std_z:.3f}")
    print(f"    Map-Fixed J1  -> median J={med_j_fix:+.3f}, median z={med_z_fix:+.2f}, power (q<0.05, z>0)={pow_fix*100:.1f}%, AUROC(J)={auc_fix_j:.3f}, AUROC(z)={auc_fix_z:.3f}")

    # Real cross-donor rotated null check (151507 x 151673 rotated 180 deg)
    c_other, g_other, m_other = load_dlpfc("151673")
    x_other_sp = lognorm(c_other)
    # Match spots after 180-deg rotation: r -> 77 - r, c -> 127 - c
    rc1 = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(m1.array_row.values, m1.array_col.values))}
    rc2_rot = {(77 - int(r), 127 - int(c)): i for i, (r, c) in enumerate(zip(m_other.array_row.values, m_other.array_col.values))}
    shared_keys = sorted(set(rc1.keys()) & set(rc2_rot.keys()))
    idx1 = np.array([rc1[k] for k in shared_keys])
    idx2 = np.array([rc2_rot[k] for k in shared_keys])

    # Build graph on shared spots
    gr_sh = build_visium_graph(m1.array_row.values[idx1], m1.array_col.values[idx1])
    n_sh = len(idx1)
    # Map L6-WM seeds onto shared spots
    orig_to_sh = {orig: new for new, orig in enumerate(idx1)}
    sh_seeds = np.array([orig_to_sh[s] for s in use_seeds if s in orig_to_sh], dtype=int)
    d_sh = cKDTree(gr_sh["coords"][sh_seeds]).query(gr_sh["midpoints"])[0]
    w_sh = 0.05 + 0.95 * np.exp(-0.5 * (d_sh / 3.0) ** 2)

    mu_sh, U_sh = eigh(gr_sh["L"].toarray())
    s_sh = 1.0 - mu_sh / (2.0 * float(gr_sh["deg"].max()))
    h_sh = mu_sh * (s_sh ** 2)
    L_w_sh = build_weighted_laplacian(n_sh, gr_sh["edges"], w_sh).toarray()
    M_sh_sq = ((U_sh.T @ L_w_sh @ U_sh) * np.outer(s_sh, s_sh)) ** 2

    # Top 50 SVGs from each donor
    w1_sh = spatial_weights(n_sh, gr_sh["edges"])
    det_a = np.asarray((c1[idx1] > 0).mean(axis=0)).ravel()
    det_b = np.asarray((c_other[idx2] > 0).mean(axis=0)).ravel()
    el_a = np.where((det_a >= 0.10) & ~is_mt)[0]
    el_b = np.where((det_b >= 0.10) & ~is_mt)[0]
    Za_all = standardize(x1_sp[idx1][:, el_a].toarray())
    Zb_all = standardize(x_other_sp[idx2][:, el_b].toarray())
    Za = Za_all[:, np.argsort(-moran_i_vec(Za_all, w1_sh))[:50]]
    Zb = Zb_all[:, np.argsort(-moran_i_vec(Zb_all, w1_sh))[:50]]
    Za_s1 = smooth_signal(Za, gr_sh["S"], 1)
    Zb_s1 = smooth_signal(Zb, gr_sh["S"], 1)

    Ha2 = (U_sh.T @ Za) ** 2
    Hb2 = (U_sh.T @ Zb) ** 2

    C_null_std = Za_s1.T @ (gr_sh["L"] @ Zb_s1)
    Var_null_std = Ha2.T @ ((h_sh ** 2)[:, None] * Hb2)
    Z_null_std = -C_null_std / np.sqrt(np.maximum(Var_null_std, 1e-18))
    fpr_null_std = float((2.0 * norm.sf(np.abs(Z_null_std)) < 0.05).mean())

    C_null_fix = Za_s1.T @ (L_w_sh @ Zb_s1)
    Var_null_fix = Ha2.T @ (M_sh_sq @ Hb2)
    Z_null_fix = -C_null_fix / np.sqrt(np.maximum(Var_null_fix, 1e-18))
    fpr_null_fix = float((2.0 * norm.sf(np.abs(Z_null_fix)) < 0.05).mean())

    print(f"  Real Cross-Donor Rotated Null (151507 x 151673, 2,500 pairs, alpha=0.05):")
    print(f"    Standard J1 Spectral FPR   = {fpr_null_std*100:.2f}%")
    print(f"    Map-Fixed J1 Spectral FPR  = {fpr_null_fix*100:.2f}%")

    summary = dict(
        med_j_std=med_j_std, med_j_fix=med_j_fix,
        med_z_std=med_z_std, med_z_fix=med_z_fix,
        pow_std=pow_std, pow_fix=pow_fix,
        auc_std_j=auc_std_j, auc_fix_j=auc_fix_j,
        auc_std_z=auc_std_z, auc_fix_z=auc_fix_z,
        fpr_null_std=fpr_null_std, fpr_null_fix=fpr_null_fix,
        k_seeds=len(use_seeds),
    )
    pd.DataFrame([summary]).to_csv(OUTDIR / "val2_summary.csv", index=False)
    print(f"  Validation 2 done in {time.time() - t0:.1f}s", flush=True)
    return pair_df, summary, dict(coords=gr1["coords"], lay=lay_arr, seeds=use_seeds, mean_j=mean_l6wm_j)


def make_figure(df1, pair_df, val2_sum, map_info):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.2), dpi=160)

    # Panel A: Serial-Section Reproducibility (151507 vs 151508)
    ax = axes[0, 0]
    order = ["J1_std", "SeedJ1_raw", "SeedJ1_clust", "SeedJ1_soft_clust",
             "J4_std", "SeedJ4_raw", "SeedJ4_clust", "SeedJ4_soft_clust", "Lee_L"]
    sub = df1.set_index("method").loc[order]
    x = np.arange(len(sub))
    w = 0.27
    ax.bar(x - w, sub["jaccard_top1"], w, color="#e74c3c", label="Top-1% Jaccard")
    ax.bar(x, sub["jaccard_top5"], w, color="#3498db", label="Top-5% Jaccard")
    ax.bar(x + w, sub["spearman_r"], w, color="#2ecc71", label="All-pairs Spearman r")
    ax.set_xticks(x)
    ax.set_xticklabels(sub.index, rotation=28, ha="right", fontsize=8.2)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Reproducibility (151507 vs 151508)")
    ax.set_title("A. Serial-Section Reproducibility (80 SVGs, 3,160 pairs)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    # Panel B: L6-WM Reference Seed Map on DLPFC 151507
    ax = axes[0, 1]
    coords = map_info["coords"]
    lay = map_info["lay"]
    seeds = map_info["seeds"]
    is_l6 = lay == "L6"
    is_wm = lay == "WM"
    other = ~(is_l6 | is_wm)
    ax.scatter(coords[other, 0], -coords[other, 1], c="#dfe6e9", s=7, label="L1–L5")
    ax.scatter(coords[is_l6, 0], -coords[is_l6, 1], c="#74b9ff", s=9, label="L6")
    ax.scatter(coords[is_wm, 0], -coords[is_wm, 1], c="#ffeaa7", s=9, label="WM")
    ax.scatter(coords[seeds, 0], -coords[seeds, 1], facecolors="none", edgecolors="#d63031",
               s=28, lw=1.2, label=f"Map-Fixed Seeds (k={len(seeds)})")
    ax.set_title("B. DLPFC 151507: Auto First-Elbow L6–WM Seeds", fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

    # Panel C: z-score scatter (Standard J1 vs Map-Fixed J1) on held-out pairs
    ax = axes[1, 0]
    for ptype, clr, lbl, alp in [
        ("Control", "#95a5a6", "Control pairs", 0.35),
        ("Within_L6_WM", "#3498db", "Within L6 / WM", 0.50),
        ("L6_vs_WM", "#e74c3c", "Held-out L6 vs WM", 0.85),
    ]:
        s = pair_df[pair_df["pair_type"] == ptype]
        ax.scatter(s["z_std"], s["z_mapfixed"], c=clr, s=18, alpha=alp, label=lbl)
    lim = max(abs(pair_df["z_std"]).max(), abs(pair_df["z_mapfixed"]).max()) + 0.5
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.1, alpha=0.6, label="y = x")
    ax.axhline(1.96, color="#e74c3c", ls=":", lw=1, alpha=0.6)
    ax.axvline(1.96, color="#e74c3c", ls=":", lw=1, alpha=0.6)
    ax.set_xlabel("Standard J1 Spectral z-score")
    ax.set_ylabel("Remedy 2 Map-Fixed J1 Spectral z-score")
    ax.set_title("C. Targeted L6–WM Power Gain (Exact U_std Spectral z)", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)

    # Panel D: Summary bar (Power, AUROC, Real Cross-Donor Rotated Null FPR)
    ax = axes[1, 1]
    metrics_lbl = ["L6–WM Power\n(q<0.05, %)", "L6–WM AUROC\n(z-score, %)", "Cross-Donor Null\nFPR (α=5%, %)"]
    std_vals = [val2_sum["pow_std"] * 100, val2_sum["auc_std_z"] * 100, val2_sum["fpr_null_std"] * 100]
    fix_vals = [val2_sum["pow_fix"] * 100, val2_sum["auc_fix_z"] * 100, val2_sum["fpr_null_fix"] * 100]
    x_pos = np.arange(3)
    w = 0.34
    b1 = ax.bar(x_pos - w/2, std_vals, w, color="#7f8c8d", label="Standard J1")
    b2 = ax.bar(x_pos + w/2, fix_vals, w, color="#8e44ad", label="Remedy 2: Map-Fixed J1")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.8, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_lbl)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("D. Real Visium Power vs Cross-Donor Null Calibration", fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    fig.savefig(OUTDIR / "fig_real_validation.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure to {OUTDIR / 'fig_real_validation.png'}", flush=True)


if __name__ == "__main__":
    t_all = time.time()
    val1_csv = OUTDIR / "val1_serial_reproducibility.csv"
    if val1_csv.exists():
        df1 = pd.read_csv(val1_csv)
        print("Loaded cached Validation 1 results.")
    else:
        df1 = run_validation1(n_genes=80)
    pair_df, val2_sum, map_info = run_validation2()
    make_figure(df1, pair_df, val2_sum, map_info)
    print(f"\nALL REAL VALIDATIONS COMPLETED in {time.time() - t_all:.1f}s")
