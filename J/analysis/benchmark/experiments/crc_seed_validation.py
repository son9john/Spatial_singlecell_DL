"""CRC GSM8594561 Tumor-Stroma Boundary Targeted Power & Null Calibration Validation.

Uses the new `juxtaposed` extension module (`_seed_extensions.py`):
  - Loads all 4,269 in-tissue Visium CytAssist spots of CRC GSM8594561.
  - Selects the exact top 300 spatially variable genes (detected >= 10%, non-MT,
    highest Moran's I) matching the manuscript's 44,850-pair screen.
  - Anchors a fixed tumor-stroma boundary seed set S_CRC via First-Elbow + 2-hop
    cluster filter using ONLY 2 canonical anchor genes per compartment
    (Epithelial: EPCAM, ELF3 vs Stromal: MGP, VIM) — strictly excluded from the
    held-out evaluation pairs!
  - Evaluates Standard J1 (Spectral Null) vs Remedy 2 Map-Fixed J1 (Exact U_std
    Quadratic-Form Spectral Null) on:
      1. Held-out epithelial-stromal marker pairs (strictly disjoint from anchors)
      2. Full 300-gene screen (44,850 pairs, BH q < 0.05) & AUROC
      3. Real-data 180-deg rotated CRC null (50 x 50 top SVGs = 2,500 pairs)
"""
from pathlib import Path
import gzip
import sys
import time
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import eigh
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import juxtaposed as jx
from juxtaposition import standardize, spatial_weights, visium_edges

OUTDIR = Path(__file__).resolve().parent / "crc_seed_validation_results"
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_crc():
    data_dir = ROOT / "data" / "crc"
    with h5py.File(data_dir / "filtered_feature_bc_matrix.h5", "r") as f:
        m = f["matrix"]
        shape = tuple(int(v) for v in m["shape"][:])
        mat = sparse.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=shape)
        ftype = m["features/feature_type"].asstr()[:]
        genes = m["features/name"].asstr()[:]
        barcodes = m["barcodes"].asstr()[:]
    keep_g = ftype == "Gene Expression"
    counts = mat[keep_g].T.tocsr()
    genes = genes[keep_g]

    with gzip.open(data_dir / "tissue_positions.csv.gz", "rt") as fh:
        pos = pd.read_csv(fh)
    pos = pos.set_index(pos.columns[0])
    meta = pos.reindex(barcodes)
    keep_s = (meta["in_tissue"].values == 1) & (np.asarray(counts.sum(axis=1)).ravel() > 0)
    return counts[keep_s].tocsr(), np.asarray(genes), meta[keep_s].copy()


def lognorm(counts, scale=1e4):
    lib = np.asarray(counts.sum(axis=1)).ravel().astype(float)
    norm_m = sparse.diags(scale / lib) @ counts.astype(float)
    norm_m.data = np.log1p(norm_m.data)
    return norm_m.tocsr()


def build_visium_hex(meta):
    ar = meta["array_row"].values.astype(int)
    ac = meta["array_col"].values.astype(int)
    edges = visium_edges(np.column_stack([ar, ac]))
    xy = np.column_stack([ac * 0.5, ar * (np.sqrt(3.0) / 2.0)]).astype(float)
    n = len(xy)
    u, v = edges.T
    A = sparse.csr_matrix((np.ones(2 * len(u)), (np.r_[u, v], np.r_[v, u])), shape=(n, n))
    deg = np.asarray(A.sum(axis=1)).ravel()
    return xy, edges, A, deg


def moran_i_vec(Z, W):
    return np.sum(Z * (W @ Z), axis=0) / float(W.sum())


def run():
    t0 = time.time()
    print("=== Loading CRC GSM8594561 (Visium CytAssist) ===", flush=True)
    counts, genes, meta = load_crc()
    n_spots = counts.shape[0]
    xy, edges, A, deg = build_visium_hex(meta)
    L_std, S, _, d_max = jx.build_heat_smoother(A)
    W_sp = spatial_weights(n_spots, edges)
    x_sp = lognorm(counts)
    print(f"  Loaded {n_spots} in-tissue spots, {len(edges)} edges, {len(genes)} genes.", flush=True)

    # Top 300 SVGs (detected >= 10%, non-MT, highest Moran's I)
    det = np.asarray((counts > 0).mean(axis=0)).ravel()
    is_mt = pd.Series(genes).str.upper().str.startswith("MT-").values
    elig = np.where((det >= 0.10) & ~is_mt)[0]
    Z_elig = standardize(x_sp[:, elig].toarray())
    mi = moran_i_vec(Z_elig, W_sp)
    top300_local = np.argsort(-mi)[:300]
    top300_idx = elig[top300_local]
    top300_genes = genes[top300_idx]
    Z300 = Z_elig[:, top300_local]
    gene_to_pos = {g: i for i, g in enumerate(top300_genes)}

    # Reference anchor genes (STRICTLY excluded from held-out epithelial-stromal evaluation)
    anchor_epi = [g for g in ("EPCAM", "ELF3") if g in gene_to_pos]
    anchor_str = [g for g in ("MGP", "VIM") if g in gene_to_pos]

    # Held-out epithelial & stromal markers from manuscript Table S4 (disjoint from anchors)
    held_epi = [g for g in ("CEACAM5", "KLF5", "CLDN7", "LGALS4", "MUC12", "ST14", "PHGR1", "PRR15L") if g in gene_to_pos]
    held_str = [g for g in ("C1S", "C1R", "COL1A1", "COL3A1", "COL1A2", "COL6A3", "MMP2", "DCN", "GREM1") if g in gene_to_pos]
    print(f"  Anchor genes -> Epi: {anchor_epi}, Str: {anchor_str}")
    print(f"  Held-out genes -> Epi ({len(held_epi)}): {held_epi}, Str ({len(held_str)}): {held_str}", flush=True)

    # Build fixed tumor-stroma boundary seed weights from ONLY the 2x2 anchor pairs (scale=1)
    Z_ref_a = jx.diffuse_heat(Z300[:, [gene_to_pos[g] for g in anchor_epi]], S, scale=1)
    Z_ref_b = jx.diffuse_heat(Z300[:, [gene_to_pos[g] for g in anchor_str]], S, scale=1)
    w_fixed, seeds, mean_spot_j, seed_diag = jx.build_map_fixed_weights(
        Z_ref_a, Z_ref_b, xy, edges, A, deg,
        sigma=4.0, floor=0.05, ridge=1.0, min_cluster_size=5, spacing=1.0
    )
    print(f"  CRC Map-Fixed Seeds: k_used={seed_diag['k_used']} (k_raw={seed_diag['k_raw']}, "
          f"coherence={seed_diag['coherence']:.2f}, clusters={seed_diag['n_clusters']})", flush=True)

    # Eigendecomposition of unweighted tissue Laplacian L (one-time)
    t_eig = time.time()
    print(f"  Computing eigendecomposition of {n_spots}x{n_spots} unweighted Laplacian...", flush=True)
    eig = eigh(L_std.toarray())
    print(f"  Eigendecomposition done in {time.time() - t_eig:.1f}s.", flush=True)

    # Standard J1 Spectral Null (w_edge = 1) vs Remedy 2 Map-Fixed J1 Spectral Null
    null_std = jx.MapFixedSpectralNull(A, edges, np.ones(len(edges)), eigendecomp=eig)
    null_fix = jx.MapFixedSpectralNull(A, edges, w_fixed, eigendecomp=eig)

    res_std = null_std.test_all_pairs(Z300, scale=1)
    res_fix = null_fix.test_all_pairs(Z300, scale=1)

    iu, ju = np.triu_indices(300, k=1)
    gx, gy = top300_genes[iu], top300_genes[ju]
    anchor_set = set(anchor_epi + anchor_str)
    epi_set, str_set = set(held_epi), set(held_str)

    no_anchor = np.array([(a not in anchor_set) and (b not in anchor_set) for a, b in zip(gx, gy)])
    is_held_es = np.array([((a in epi_set and b in str_set) or (a in str_set and b in epi_set))
                           for a, b in zip(gx, gy)])
    is_held_same = np.array([((a in epi_set and b in epi_set) or (a in str_set and b in str_set))
                             for a, b in zip(gx, gy)])

    df_pairs = pd.DataFrame(dict(
        gene_x=gx, gene_y=gy,
        is_held_es=is_held_es, is_held_same=is_held_same, no_anchor=no_anchor,
        J1_std=res_std["J"][iu, ju], z_std=res_std["z"][iu, ju], q_std=res_std["q"][iu, ju],
        J1_fix=res_fix["J"][iu, ju], z_fix=res_fix["z"][iu, ju], q_fix=res_fix["q"][iu, ju],
    ))

    # Metrics on held-out epithelial-stromal pairs (strictly disjoint from anchor genes)
    es = df_pairs[df_pairs["is_held_es"]]
    eval_pool = df_pairs[df_pairs["no_anchor"]]
    auc_std_j = roc_auc_score(eval_pool["is_held_es"], eval_pool["J1_std"])
    auc_fix_j = roc_auc_score(eval_pool["is_held_es"], eval_pool["J1_fix"])
    auc_std_z = roc_auc_score(eval_pool["is_held_es"], eval_pool["z_std"])
    auc_fix_z = roc_auc_score(eval_pool["is_held_es"], eval_pool["z_fix"])

    # Also test Mode 1 (SeedJ_soft_clust) on the held-out epithelial-stromal pairs vs held-out same-compartment pairs
    Z300_s1 = jx.diffuse_heat(Z300, S, scale=1)
    es_soft, same_soft = [], []
    for a, b in zip(es["gene_x"], es["gene_y"]):
        sc_v, _ = jx.seed_soft_j(Z300_s1[:, gene_to_pos[a]], Z300_s1[:, gene_to_pos[b]], xy, edges, A, deg, sigma=4.0)
        es_soft.append(sc_v)
    same_df = df_pairs[df_pairs["is_held_same"]]
    for a, b in zip(same_df["gene_x"], same_df["gene_y"]):
        sc_v, _ = jx.seed_soft_j(Z300_s1[:, gene_to_pos[a]], Z300_s1[:, gene_to_pos[b]], xy, edges, A, deg, sigma=4.0)
        same_soft.append(sc_v)
    auc_es_vs_same_std = roc_auc_score(np.r_[np.ones(len(es)), np.zeros(len(same_df))],
                                       np.r_[es["J1_std"].values, same_df["J1_std"].values])
    auc_es_vs_same_soft = roc_auc_score(np.r_[np.ones(len(es)), np.zeros(len(same_df))],
                                        np.r_[es_soft, same_soft])

    print(f"\n  Held-out Epithelial-Stromal pairs (n={len(es)}):")
    print(f"    Standard J1   -> median J={es['J1_std'].median():+.3f}, median z={es['z_std'].median():+.2f}, "
          f"q<0.05 (z>0)={((es['q_std']<0.05)&(es['z_std']>0)).mean()*100:.1f}%, all-pair AUROC(J)={auc_std_j:.4f}, AUROC(z)={auc_std_z:.4f}")
    print(f"    Map-Fixed J1  -> median J={es['J1_fix'].median():+.3f}, median z={es['z_fix'].median():+.2f}, "
          f"q<0.05 (z>0)={((es['q_fix']<0.05)&(es['z_fix']>0)).mean()*100:.1f}%, all-pair AUROC(J)={auc_fix_j:.4f}, AUROC(z)={auc_fix_z:.4f}")
    print(f"    Mode 1 SeedJ_soft_clust -> median J(Epi-Str)={np.median(es_soft):+.3f}, median J(Same)={np.median(same_soft):+.3f}, "
          f"AUROC(Epi-Str vs Same)={auc_es_vs_same_soft:.4f} (std={auc_es_vs_same_std:.4f})")
    print(f"  Whole 300-gene screen (44,850 pairs, q<0.05):")
    print(f"    Standard J1 called: {(df_pairs['q_std']<0.05).mean()*100:.1f}%  |  Map-Fixed J1 called: {(df_pairs['q_fix']<0.05).mean()*100:.1f}%")

    # Real-data 180-deg rotated CRC null calibration check (top 50 SVGs x top 50 SVGs rotated = 2,500 pairs)
    ar = meta["array_row"].values.astype(int)
    ac = meta["array_col"].values.astype(int)
    r_max, c_max = int(ar.max()), int(ac.max())
    rc_orig = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(ar, ac))}
    rc_rot = {(r_max - int(r), c_max - int(c)): i for i, (r, c) in enumerate(zip(ar, ac))}
    shared_keys = sorted(set(rc_orig.keys()) & set(rc_rot.keys()))
    idx_a = np.array([rc_orig[k] for k in shared_keys])
    idx_b = np.array([rc_rot[k] for k in shared_keys])

    xy_sh, edges_sh, A_sh, deg_sh = build_visium_hex(meta.iloc[idx_a])
    orig_to_sh = {orig: new for new, orig in enumerate(idx_a)}
    sh_seeds = np.array([orig_to_sh[s] for s in seeds if s in orig_to_sh], dtype=int)
    d_sh = cKDTree(xy_sh[sh_seeds]).query(xy_sh[edges_sh].mean(axis=1))[0]
    w_sh = 0.05 + 0.95 * np.exp(-0.5 * (d_sh / 4.0) ** 2)

    L_sh, _, _, _ = jx.build_heat_smoother(A_sh)
    eig_sh = eigh(L_sh.toarray())
    null_sh_std = jx.MapFixedSpectralNull(A_sh, edges_sh, np.ones(len(edges_sh)), eigendecomp=eig_sh)
    null_sh_fix = jx.MapFixedSpectralNull(A_sh, edges_sh, w_sh, eigendecomp=eig_sh)

    Za = standardize(x_sp[idx_a][:, top300_idx[:50]].toarray())
    Zb = standardize(x_sp[idx_b][:, top300_idx[:50]].toarray())
    rot_std = null_sh_std.test_cross_panels(Za, Zb, scale=1)
    rot_fix = null_sh_fix.test_cross_panels(Za, Zb, scale=1)
    fpr_rot_std = float((rot_std["p"] < 0.05).mean())
    fpr_rot_fix = float((rot_fix["p"] < 0.05).mean())
    print(f"  CRC 180-deg Rotated Self-Null ({len(shared_keys)} shared spots, 2,500 pairs, alpha=0.05):")
    print(f"    Standard J1 Spectral FPR  = {fpr_rot_std*100:.2f}%")
    print(f"    Map-Fixed J1 Spectral FPR = {fpr_rot_fix*100:.2f}%", flush=True)

    # Save tables
    es.sort_values("z_fix", ascending=False).to_csv(OUTDIR / "crc_heldout_epi_stroma_pairs.csv", index=False)
    summary = dict(
        n_spots=n_spots, k_seeds=seed_diag["k_used"], coherence=seed_diag["coherence"],
        n_held_es=len(es),
        med_j_std=float(es["J1_std"].median()), med_j_fix=float(es["J1_fix"].median()),
        med_z_std=float(es["z_std"].median()), med_z_fix=float(es["z_fix"].median()),
        auc_std_j=auc_std_j, auc_fix_j=auc_fix_j,
        auc_std_z=auc_std_z, auc_fix_z=auc_fix_z,
        pct_screen_std=float((df_pairs["q_std"] < 0.05).mean()),
        pct_screen_fix=float((df_pairs["q_fix"] < 0.05).mean()),
        fpr_rot_std=fpr_rot_std, fpr_rot_fix=fpr_rot_fix,
    )
    pd.DataFrame([summary]).to_csv(OUTDIR / "crc_validation_summary.csv", index=False)

    # 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.2), dpi=160)

    # Panel A: Anchor EPCAM - MGP difference + automatically extracted seeds
    ax = axes[0, 0]
    epi_m_str = Z300[:, gene_to_pos["EPCAM"]] - Z300[:, gene_to_pos["MGP"]]
    lim_v = np.percentile(np.abs(epi_m_str), 97)
    sc1 = ax.scatter(xy[:, 0], -xy[:, 1], c=epi_m_str, cmap="RdBu_r", vmin=-lim_v, vmax=lim_v, s=6, alpha=0.85)
    ax.scatter(xy[seeds, 0], -xy[seeds, 1], facecolors="none", edgecolors="lime", s=22, lw=0.9,
               label=f"First-Elbow Clustered Seeds (k={len(seeds)})")
    ax.set_title("A. CRC GSM8594561: EPCAM−MGP & Auto Tumor–Stroma Seeds", fontweight="bold", fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    plt.colorbar(sc1, ax=ax, fraction=0.04, pad=0.02, label="EPCAM − MGP (z)")

    # Panel B: Fixed distance-decay spot weight field W(S_CRC)
    ax = axes[0, 1]
    d_spot = cKDTree(xy[seeds]).query(xy)[0]
    w_spot = 0.05 + 0.95 * np.exp(-0.5 * (d_spot / 4.0) ** 2)
    sc2 = ax.scatter(xy[:, 0], -xy[:, 1], c=w_spot, cmap="magma", vmin=0, vmax=1, s=6)
    ax.set_title("B. Fixed Tumor–Stroma Distance Weight Field W(S_CRC)", fontweight="bold", fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.colorbar(sc2, ax=ax, fraction=0.04, pad=0.02, label="Weight w")

    # Panel C: Held-out Epithelial-Stromal z-score scatter (Standard J1 vs Map-Fixed J1)
    ax = axes[1, 0]
    bg_sub = eval_pool[~eval_pool["is_held_es"]].sample(n=min(3000, len(eval_pool)), random_state=42)
    ax.scatter(bg_sub["z_std"], bg_sub["z_fix"], c="#bdc3c7", s=8, alpha=0.30, label="Other pairs (3k sample)")
    ax.scatter(es["z_std"], es["z_fix"], c="#e74c3c", s=28, alpha=0.9, edgecolors="white", lw=0.3,
               label=f"Held-out Epi–Stroma (n={len(es)})")
    lim_z = max(abs(es["z_std"]).max(), abs(es["z_fix"]).max()) + 1.0
    ax.plot([-lim_z, lim_z], [-lim_z, lim_z], "k--", lw=1.2, alpha=0.65, label="y = x")
    ax.set_xlabel("Standard J1 Spectral z-score")
    ax.set_ylabel("Remedy 2 Map-Fixed J1 Spectral z-score")
    ax.set_title("C. Held-out Epithelial–Stromal Opposition Boost (Exact U_std z)", fontweight="bold", fontsize=10)
    ax.legend(fontsize=8.2, loc="upper left")
    ax.grid(alpha=0.25)

    # Panel D: Summary comparison (Median J, All-pair AUROC, Rotated CRC Null FPR)
    ax = axes[1, 1]
    lbls = ["Held-out Epi–Str\nMedian J (×100)", "All-pair AUROC\n(Epi–Str vs Rest, %)", "180° Rotated CRC\nNull FPR (α=5%, %)"]
    v_std = [summary["med_j_std"] * 100, summary["auc_std_j"] * 100, summary["fpr_rot_std"] * 100]
    v_fix = [summary["med_j_fix"] * 100, summary["auc_fix_j"] * 100, summary["fpr_rot_fix"] * 100]
    x_pos = np.arange(3)
    w = 0.34
    b1 = ax.bar(x_pos - w/2, v_std, w, color="#7f8c8d", label="Standard J1")
    b2 = ax.bar(x_pos + w/2, v_fix, w, color="#8e44ad", label="Remedy 2: Map-Fixed J1")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(lbls)
    ax.set_ylim(0, 114)
    ax.set_ylabel("Value / Percentage")
    ax.set_title("D. CRC Tumor–Stroma Effect, Ranking & Null Calibration", fontweight="bold", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    fig.savefig(OUTDIR / "fig_crc_targeted_validation.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure to {OUTDIR / 'fig_crc_targeted_validation.png'} (total {time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    run()
