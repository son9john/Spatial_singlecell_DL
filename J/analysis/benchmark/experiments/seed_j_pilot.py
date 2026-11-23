"""Pilot simulation experiment: Seed-Anchored Distance-Weighted J with Automatic First-Elbow Seed Selection.

Compares standard Boundary J against proposed Seed-Weighted J variants on
benchmarks where current J struggles:
  Exp 1: Controlled contrast & amplitude imbalance (SD=0.15, 0.35, 0.60; amp=1.0, 0.5, 0.25, 0.1)
  Exp 2: Weak-signal semi-synthetic DLPFC B2 (sections 151507, 151669, 151673; beta=0.5, 1.0, 2.0)
  Exp 3: Real observed DLPFC B1 (adjacent vs distant, adjacent vs same; sections 151507, 151669, 151673)
  Exp 4: Null calibration / post-selection inflation check (independent white-noise & Matern fields)
"""
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import cholesky
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks" / "v2"))

from juxtaposition import (
    standardize, edge_differences, spatial_weights, lee_l, juxtaposed_value,
)

CACHE = ROOT / "cache"
SECTIONS = tuple(map(str, (*range(151507, 151511), *range(151669, 151677))))
DONOR = {s: ("Br5292" if s.startswith("15150") or s == "151510" else
             "Br5595" if s in ("151669", "151670", "151671", "151672") else "Br8100")
         for s in SECTIONS}
LAYER_ORDER = ("L1", "L2", "L3", "L4", "L5", "L6", "WM")


def load_dlpfc(section):
    counts = sparse.load_npz(CACHE / f"dlpfc_{section}.npz").tocsr()
    meta = pd.read_csv(CACHE / f"dlpfc_{section}_meta.csv", index_col=0)
    genes = np.load(CACHE / f"dlpfc_{section}_genes.npy", allow_pickle=True)
    return counts, genes, meta


def lognorm(counts, scale=1e4):
    lib = np.asarray(counts.sum(axis=1)).ravel().astype(float)
    norm = sparse.diags(scale / lib) @ counts.astype(float)
    norm.data = np.log1p(norm.data)
    return norm.tocsr()


def scenario(mode, r, c, rng):
    """Published 42x46 synthetic tissue scenarios from analysis.py."""
    phase = rng.uniform(0, 2 * np.pi)
    boundary = 22 + 3 * np.sin((r - 3) / 9 + phase)
    side = np.where(c >= boundary, 1.0, -1.0)
    vertical = 10 + rng.integers(-3, 4)
    window = 0.5 * (np.tanh((r - vertical) / 4) - np.tanh((r - vertical - 22) / 4))
    background_x = 0.30 * np.sin(r / 9 + phase) + 0.25 * np.cos(c / 11 + phase)
    background_y = 0.30 * np.sin(r / 9 + phase) - 0.25 * np.sin(c / 11 + phase)
    interface = 1.8 * side * window
    if mode == "focal":
        return background_x + interface, background_y - interface, side, window
    if mode == "shared":
        return background_x + interface, background_y + interface, side, window
    if mode == "unilateral":
        return background_x + interface, background_y, side, window
    if mode == "displaced":
        other_side = np.where(c >= boundary + 6, 1.0, -1.0)
        return background_x + interface, background_y - 1.8 * other_side * window, side, window
    if mode == "broad":
        wave = 1.5 * np.sin(c / 6 + phase) + 0.6 * np.cos(r / 6 + phase)
        return background_x + wave, background_y - wave, side, window
    return background_x + 0.7 * np.sin(r / 5 + phase), background_y + 0.7 * np.cos(c / 5 - phase), side, window

OUTDIR = Path(__file__).resolve().parent / "seed_j_pilot_results"
OUTDIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. Core Graph & Local J Helpers
# =============================================================================
def build_grid_graph(shape=(42, 46)):
    r, c = np.indices(shape)
    coords = np.column_stack([r.ravel(), c.ravel()]).astype(float)
    keys = np.arange(r.size).reshape(shape)
    edges = np.vstack((
        np.column_stack([keys[:, :-1].ravel(), keys[:, 1:].ravel()]),
        np.column_stack([keys[:-1].ravel(), keys[1:].ravel()]),
    ))
    n = len(coords)
    u, v = edges.T
    row = np.r_[u, v]
    col = np.r_[v, u]
    A = sparse.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
    deg = np.asarray(A.sum(axis=1)).ravel()
    L = sparse.diags(deg) - A
    d_max = float(deg.max())
    S = sparse.eye(n, format="csr") - L / (2.0 * d_max)
    return dict(coords=coords, edges=edges, A=A, deg=deg, L=L, S=S,
                midpoints=coords[edges].mean(axis=1), spacing=1.0)


def build_visium_graph(array_row, array_col):
    coords_int = np.column_stack([array_row, array_col]).astype(int)
    index = {tuple(rc): i for i, rc in enumerate(coords_int)}
    edges = [
        (i, j) for (r, c), i in index.items()
        for dr, dc in ((0, 2), (1, -1), (1, 1))
        if (j := index.get((r + dr, c + dc))) is not None
    ]
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    # Physical hex coordinates (unit spacing = 1.0 between nearest neighbors)
    xy = np.column_stack([array_col * 0.5, array_row * (np.sqrt(3) / 2.0)]).astype(float)
    n = len(xy)
    u, v = edges.T
    row = np.r_[u, v]
    col = np.r_[v, u]
    A = sparse.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
    deg = np.asarray(A.sum(axis=1)).ravel()
    L = sparse.diags(deg) - A
    d_max = float(deg.max())
    S = sparse.eye(n, format="csr") - L / (2.0 * d_max)
    return dict(coords=xy, edges=edges, A=A, deg=deg, L=L, S=S,
                midpoints=xy[edges].mean(axis=1), spacing=1.0)


def smooth_signal(x, S, steps):
    out = x.copy()
    for _ in range(steps):
        out = S @ out
    return out


def spot_local_j(x, y, graph, ridge=1.0):
    """Regularized spot-level average local J over incident edges."""
    u, v = graph["edges"].T
    n = len(x)
    dx = x[v] - x[u]
    dy = y[v] - y[u]
    num_e = -2.0 * dx * dy
    den_e = dx * dx + dy * dy
    mean_energy = float(np.mean(den_e)) if len(den_e) else 0.0

    # Accumulate incident edges for both endpoints
    num_spot = np.bincount(u, weights=num_e, minlength=n) + np.bincount(v, weights=num_e, minlength=n)
    den_spot = np.bincount(u, weights=den_e, minlength=n) + np.bincount(v, weights=den_e, minlength=n)
    deg = graph["deg"]
    denom = den_spot + ridge * deg * mean_energy
    j_spot = np.divide(num_spot, denom, out=np.zeros(n), where=denom > 0)
    return np.clip(j_spot, -1.0, 1.0), dx, dy


# =============================================================================
# 2. Automatic First-Elbow Seed Selection Algorithm
# =============================================================================
def find_first_elbow_seeds(spot_scores, min_seeds=5, max_frac=0.15):
    """Automatically find the FIRST bending point (first elbow) on the descending
    sorted positive spot_scores curve, avoiding the global 'all-or-none' tail split.

    Returns (seed_indices, diagnostics_dict).
    """
    n = len(spot_scores)
    order = np.argsort(-spot_scores)
    sorted_vals = spot_scores[order]

    # Restrict to strictly positive scores for elbow detection
    pos_count = int(np.sum(sorted_vals > 0))
    max_seeds = max(min_seeds + 2, int(round(n * max_frac)))

    if pos_count <= min_seeds:
        k = min(min_seeds, n)
        return order[:k], dict(k_first=k, k_global=k, pos_count=pos_count, sorted_vals=sorted_vals)

    # Search window: positive prefix capped at 50% of tissue to avoid all-or-none tail
    M = min(pos_count, max(max_seeds * 2, int(0.5 * n)))
    y_raw = sorted_vals[:M]
    y_span = y_raw[0] - y_raw[-1]
    if y_span <= 1e-12:
        k = min_seeds
        return order[:k], dict(k_first=k, k_global=k, pos_count=pos_count, sorted_vals=sorted_vals)

    # Normalize to [0, 1] x [0, 1]
    u = np.linspace(0.0, 1.0, M)
    y = (y_raw - y_raw[-1]) / y_span

    # Light smoothing to remove single-spot jitter while preserving sharp initial drop
    sigma = max(1.5, 0.01 * M)
    y_smooth = gaussian_filter1d(y, sigma=sigma, mode="nearest")

    # 1) Global Kneedle distance-to-chord (for reference / comparison)
    kneedle_d = (1.0 - u) - y_smooth
    k_global = int(np.argmax(kneedle_d)) + 1

    # 2) Signed discrete curvature kappa(u) = y'' / (1 + y'^2)^(3/2)
    # On a steeply falling curve that bends flat, y' < 0 and y'' > 0 -> positive curvature peak!
    du = 1.0 / max(M - 1, 1)
    dy = np.gradient(y_smooth, du)
    d2y = np.gradient(dy, du)
    kappa = d2y / np.power(1.0 + dy * dy, 1.5)

    # Search for the FIRST prominent local maximum of curvature in [min_seeds-1, max_seeds-1]
    hi = min(max_seeds, M - 2)
    lo = max(min_seeds - 1, 2)
    search_k = kappa[lo:hi]

    k_first = None
    if len(search_k) >= 3:
        # Local maxima of curvature
        is_peak = (search_k[1:-1] > search_k[:-2]) & (search_k[1:-1] >= search_k[2:])
        peak_pos = np.where(is_peak)[0] + 1 + lo
        if len(peak_pos) > 0:
            max_kappa = float(np.max(search_k))
            # First peak reaching at least 25% of max curvature in the search window
            prominent = peak_pos[kappa[peak_pos] >= 0.25 * max(max_kappa, 1e-9)]
            if len(prominent) > 0:
                k_first = int(prominent[0]) + 1

    if k_first is None:
        # Fallback: first point where Kneedle distance reaches 75% of its local max in [lo, hi]
        sub_d = kneedle_d[lo:hi]
        if len(sub_d) > 0 and np.max(sub_d) > 0:
            thresh = 0.75 * np.max(sub_d)
            k_first = int(np.argmax(sub_d >= thresh)) + lo + 1
        else:
            k_first = min(k_global, max_seeds)

    k_first = int(np.clip(k_first, min_seeds, max_seeds))
    return order[:k_first], dict(
        k_first=k_first,
        k_global=k_global,
        pos_count=pos_count,
        sorted_vals=sorted_vals,
        kappa=kappa,
        kneedle_d=kneedle_d,
        M=M,
    )


# =============================================================================
# 3. Seed-Anchored Distance-Weighted J Variants
# =============================================================================
def score_seed_variants(x, y, graph, scale=0, sigma=3.0, ridge=1.0, return_diag=False):
    """Compute standard J and the proposed Seed-Anchored Distance-Weighted J variants."""
    xs = smooth_signal(x, graph["S"], scale) if scale > 0 else x
    ys = smooth_signal(y, graph["S"], scale) if scale > 0 else y

    j_spot, dx, dy = spot_local_j(xs, ys, graph, ridge=ridge)
    seeds, diag = find_first_elbow_seeds(j_spot)

    seed_coords = graph["coords"][seeds]
    tree = cKDTree(seed_coords)

    # Distance from every spot and every edge midpoint to nearest seed
    d_spot = tree.query(graph["coords"])[0]
    d_edge = tree.query(graph["midpoints"])[0]

    sig_phys = sigma * graph["spacing"]
    w_spot = np.exp(-0.5 * (d_spot / sig_phys) ** 2)
    w_edge = np.exp(-0.5 * (d_edge / sig_phys) ** 2)

    # Standard whole-graph J
    j_std = float(juxtaposed_value(dx, dy))

    # Variant A1: Seed-Weighted Gradient J (local-weighted denominator)
    j_seed_grad = float(juxtaposed_value(dx, dy, weights=w_edge))

    # Variant A2: Seed-Weighted Gradient J with Global Energy Denominator
    # Prevents tiny-energy random null seeds from inflating to +1
    global_energy = float(np.sum(dx * dx + dy * dy))
    j_seed_gnorm = float(np.sum(w_edge * (-2.0 * dx * dy)) / global_energy) if global_energy > 0 else 0.0

    # Variant B1: Seed-Weighted Whole-Gene Expression Product (opposing sign: -2*sum w*x*y / sum w*(x^2+y^2))
    expr_den = float(np.sum(w_spot * (xs * xs + ys * ys)))
    j_seed_expr_opp = float(-2.0 * np.sum(w_spot * xs * ys) / expr_den) if expr_den > 0 else 0.0

    # Variant B2: Seed-Weighted Whole-Gene Expression Product (positive sign, for co-localization/proximity)
    j_seed_expr_pos = -j_seed_expr_opp

    # Variant B3: Seed-Anchored Cross-Spot Product (each seed s vs all spots j weighted by K(d_sj))
    # Compute K(d_sj) for s in seeds, j in all spots (shape: k_seeds x n)
    d_mat = cdist(seed_coords, graph["coords"])
    K = np.exp(-0.5 * (d_mat / sig_phys) ** 2)
    K_row_sum = np.maximum(K.sum(axis=1), 1e-12)
    x_bar = (K @ xs) / K_row_sum
    y_bar = (K @ ys) / K_row_sum
    x_s = xs[seeds]
    y_s = ys[seeds]
    anchor_num = -float(np.sum(x_s * y_bar + y_s * x_bar))
    anchor_den = 0.5 * float(np.sum(x_s**2 + y_s**2 + x_bar**2 + y_bar**2))
    j_seed_anchor = float(np.clip(anchor_num / anchor_den, -1.0, 1.0)) if anchor_den > 0 else 0.0

    res = {
        "J_std": j_std,
        "SeedJ_grad": j_seed_grad,
        "SeedJ_gnorm": j_seed_gnorm,
        "SeedJ_expr_opp": j_seed_expr_opp,
        "SeedJ_expr_pos": j_seed_expr_pos,
        "SeedJ_anchor": j_seed_anchor,
        "k_seeds": len(seeds),
    }
    if return_diag:
        diag["seeds"] = seeds
        diag["j_spot"] = j_spot
        diag["w_spot"] = w_spot
        return res, diag
    return res


# =============================================================================
# Experiment 1: Controlled Contrast & Amplitude Imbalance
# =============================================================================
def scaled_scenario(mode, r, c, rng, amplitude):
    state = rng.bit_generator.state
    clean_x, clean_y, side, window = scenario(mode, r, c, rng)
    if mode == "focal" and amplitude != 1.0:
        baseline = np.random.default_rng()
        baseline.bit_generator.state = state
        _, background_y, _, _ = scenario("unilateral", r, c, baseline)
        clean_y = background_y + amplitude * (clean_y - background_y)
    return clean_x, clean_y, side, window


def run_exp1(n_rep=30, sigmas=(2.0, 4.0)):
    print("\n=== Experiment 1: Controlled Contrast & Amplitude Imbalance ===", flush=True)
    shape = (42, 46)
    r, c = np.indices(shape)
    graph = build_grid_graph(shape)
    modes = ("focal", "shared", "unilateral", "displaced", "broad", "independent")
    rows = []
    diag_examples = {}

    for sd in (0.15, 0.35, 0.60):
        for amp in (1.0, 0.5, 0.25, 0.1):
            # Track per-replicate AUROC for each method
            method_aucs = {}
            k_focal, k_indep = [], []
            for rep in range(n_rep):
                scores_by_method = {}
                for mode in modes:
                    rng = np.random.default_rng(20261122 + rep)
                    cx, cy, side, window = scaled_scenario(mode, r, c, rng, amp)
                    x_obs = gaussian_filter(cx, 0.55) + rng.normal(0, sd, shape)
                    y_obs = gaussian_filter(cy, 0.55) + rng.normal(0, sd, shape)
                    x, y = standardize(x_obs.ravel()), standardize(y_obs.ravel())
                    dx, dy = edge_differences(x, y, graph["edges"])

                    # Standard J & negative codispersion
                    scores_by_method.setdefault("J_std", []).append(float(juxtaposed_value(dx, dy, power=1.0)))
                    scores_by_method.setdefault("NegCodisp", []).append(float(juxtaposed_value(dx, dy, power=0.0)))

                    for sig in sigmas:
                        want_diag = (rep == 0 and sd == 0.35 and amp == 0.5 and sig == 4.0
                                     and mode in ("focal", "displaced", "independent"))
                        out = score_seed_variants(x, y, graph, scale=0, sigma=sig, return_diag=want_diag)
                        if want_diag:
                            res, diag = out
                            diag_examples[mode] = dict(x=x_obs, y=y_obs, diag=diag, res=res, shape=shape)
                        else:
                            res = out
                        for m in ("SeedJ_grad", "SeedJ_gnorm", "SeedJ_expr_opp", "SeedJ_anchor"):
                            scores_by_method.setdefault(f"{m}_s{int(sig)}", []).append(res[m])
                        if sig == sigmas[0]:
                            if mode == "focal":
                                k_focal.append(res["k_seeds"])
                            elif mode == "independent":
                                k_indep.append(res["k_seeds"])

                labels = np.array([m == "focal" for m in modes], dtype=int)
                for m, vals in scores_by_method.items():
                    method_aucs.setdefault(m, []).append(roc_auc_score(labels, vals))

            row = dict(noise_sd=sd, amplitude=amp, mean_k_focal=np.mean(k_focal), mean_k_indep=np.mean(k_indep))
            for m, vals in method_aucs.items():
                row[m] = float(np.mean(vals))
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "exp1_contrast_imbalance.csv", index=False)
    print(df[df["noise_sd"] == 0.35][
        ["amplitude", "J_std", "NegCodisp", "SeedJ_grad_s4", "SeedJ_gnorm_s4", "SeedJ_expr_opp_s4", "SeedJ_anchor_s4", "mean_k_focal"]
    ].round(3).to_string(index=False))
    print("\nHigh-noise SD=0.60, amp=1.0:")
    print(df[(df["noise_sd"] == 0.60) & (df["amplitude"] == 1.0)][
        ["J_std", "NegCodisp", "SeedJ_grad_s2", "SeedJ_grad_s4", "SeedJ_gnorm_s4", "SeedJ_expr_opp_s4", "SeedJ_anchor_s4"]
    ].round(3).to_string(index=False))
    return df, diag_examples


# =============================================================================
# Experiment 2: Weak-Signal Semi-Synthetic DLPFC B2
# =============================================================================
def run_exp2(sections=("151507", "151669", "151673"), n_rep=2, betas=(0.5, 1.0, 2.0), per_layer=4):
    print("\n=== Experiment 2: Weak-Signal Semi-Synthetic DLPFC B2 ===", flush=True)
    RANK = {l: k for k, l in enumerate(LAYER_ORDER)}
    rows = []

    for s in sections:
        t0 = time.time()
        counts, _, meta = load_dlpfc(s)
        graph = build_visium_graph(meta.array_row.values, meta.array_col.values)
        xy = graph["coords"]
        # Matern(nu=3/2, range=3) background
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
        adj = dist == 1
        far = dist >= 2
        same = dist == 0

        for beta in betas:
            for rep in range(n_rep):
                bg = 0.3 * (lc @ rng.normal(size=(len(xy), p)))
                logmu = beta * ind + bg
                lam = rel[:, None] * np.exp(logmu)
                lam *= 2.0 / lam.mean(axis=0, keepdims=True)
                y_cnt = rng.poisson(lam)
                X = np.log1p(1e4 * y_cnt / lib[:, None])
                Z = standardize(X)
                # Pre-smooth at scale=1 and scale=4
                Z1 = smooth_signal(Z, graph["S"], 1)
                Z4 = smooth_signal(Z, graph["S"], 4)

                # Evaluate pairs
                methods = ("J1_std", "J4_std", "SeedJ1_grad", "SeedJ1_gnorm", "SeedJ1_expr_opp",
                           "SeedJ4_grad", "SeedJ4_gnorm", "SeedJ4_expr_opp", "SeedJ4_anchor", "Lee_L_pos")
                sc = {m: np.zeros(len(iu)) for m in methods}
                w_sp = spatial_weights(len(xy), graph["edges"])

                for idx_pair, (i, j) in enumerate(zip(iu, ju)):
                    r1 = score_seed_variants(Z1[:, i], Z1[:, j], graph, scale=0, sigma=4.0)
                    r4 = score_seed_variants(Z4[:, i], Z4[:, j], graph, scale=0, sigma=4.0)
                    sc["J1_std"][idx_pair] = r1["J_std"]
                    sc["SeedJ1_grad"][idx_pair] = r1["SeedJ_grad"]
                    sc["SeedJ1_gnorm"][idx_pair] = r1["SeedJ_gnorm"]
                    sc["SeedJ1_expr_opp"][idx_pair] = r1["SeedJ_expr_opp"]
                    sc["J4_std"][idx_pair] = r4["J_std"]
                    sc["SeedJ4_grad"][idx_pair] = r4["SeedJ_grad"]
                    sc["SeedJ4_gnorm"][idx_pair] = r4["SeedJ_gnorm"]
                    sc["SeedJ4_expr_opp"][idx_pair] = r4["SeedJ_expr_opp"]
                    sc["SeedJ4_anchor"][idx_pair] = r4["SeedJ_anchor"]
                    sc["Lee_L_pos"][idx_pair] = lee_l(Z[:, i], Z[:, j], w_sp)

                for m in methods:
                    a_ad = roc_auc_score(np.r_[np.ones(adj.sum()), np.zeros(far.sum())],
                                         np.r_[sc[m][adj], sc[m][far]])
                    a_as = roc_auc_score(np.r_[np.ones(adj.sum()), np.zeros(same.sum())],
                                         np.r_[sc[m][adj], sc[m][same]])
                    rows.append(dict(section=s, beta=beta, rep=rep, method=m,
                                     auc_adj_vs_far=a_ad, auc_adj_vs_same=a_as))
        print(f"  Section {s} done in {time.time() - t0:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "exp2_dlpfc_b2_semisynthetic.csv", index=False)
    piv = df.groupby(["beta", "method"])[["auc_adj_vs_far", "auc_adj_vs_same"]].mean().unstack("beta")
    print(piv.round(3))
    return df


# =============================================================================
# Experiment 3: Real Observed DLPFC B1 (Adjacent vs Distant & Adjacent vs Same)
# =============================================================================
def run_exp3(sections=("151507", "151669", "151673"), k_markers=5):
    print("\n=== Experiment 3: Real Observed DLPFC B1 ===", flush=True)
    RANK = {l: k for k, l in enumerate(LAYER_ORDER)}
    # Compute leave-one-donor-out markers using all 12 sections
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
        X = lognorm(counts)[:, idx_list].toarray()
        keep = X.std(axis=0) > 0
        X, lay_list = X[:, keep], lay_list[keep]
        Z = standardize(X)
        Z1 = smooth_signal(Z, graph["S"], 1)
        Z4 = smooth_signal(Z, graph["S"], 4)

        p = Z.shape[1]
        iu, ju = np.triu_indices(p, k=1)
        rk = np.array([RANK[l] for l in lay_list])
        dist = np.abs(rk[iu] - rk[ju])
        adj = dist == 1
        far = dist >= 2
        same = dist == 0

        methods = ("J1_std", "J4_std", "SeedJ1_grad", "SeedJ1_gnorm", "SeedJ1_expr_opp", "SeedJ1_expr_pos",
                   "SeedJ4_grad", "SeedJ4_gnorm", "SeedJ4_expr_opp", "SeedJ4_expr_pos", "Lee_L_pos")
        sc = {m: np.zeros(len(iu)) for m in methods}
        w_sp = spatial_weights(len(meta), graph["edges"])

        for idx_pair, (i, j) in enumerate(zip(iu, ju)):
            r1 = score_seed_variants(Z1[:, i], Z1[:, j], graph, scale=0, sigma=4.0)
            r4 = score_seed_variants(Z4[:, i], Z4[:, j], graph, scale=0, sigma=4.0)
            sc["J1_std"][idx_pair] = r1["J_std"]
            sc["SeedJ1_grad"][idx_pair] = r1["SeedJ_grad"]
            sc["SeedJ1_gnorm"][idx_pair] = r1["SeedJ_gnorm"]
            sc["SeedJ1_expr_opp"][idx_pair] = r1["SeedJ_expr_opp"]
            sc["SeedJ1_expr_pos"][idx_pair] = r1["SeedJ_expr_pos"]
            sc["J4_std"][idx_pair] = r4["J_std"]
            sc["SeedJ4_grad"][idx_pair] = r4["SeedJ_grad"]
            sc["SeedJ4_gnorm"][idx_pair] = r4["SeedJ_gnorm"]
            sc["SeedJ4_expr_opp"][idx_pair] = r4["SeedJ_expr_opp"]
            sc["SeedJ4_expr_pos"][idx_pair] = r4["SeedJ_expr_pos"]
            sc["Lee_L_pos"][idx_pair] = lee_l(Z[:, i], Z[:, j], w_sp)

        for m in methods:
            a_ad = roc_auc_score(np.r_[np.ones(adj.sum()), np.zeros(far.sum())],
                                 np.r_[sc[m][adj], sc[m][far]])
            a_as = roc_auc_score(np.r_[np.ones(adj.sum()), np.zeros(same.sum())],
                                 np.r_[sc[m][adj], sc[m][same]])
            rows.append(dict(section=s, method=m, auc_adj_vs_far=a_ad, auc_adj_vs_same=a_as))
        print(f"  Real DLPFC {s} done in {time.time() - t0:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "exp3_dlpfc_b1_real.csv", index=False)
    print(df.groupby("method")[["auc_adj_vs_far", "auc_adj_vs_same"]].mean().round(3))
    return df


# =============================================================================
# Experiment 4: Null Calibration / Post-Selection Bias Check
# =============================================================================
def run_exp4(n_pairs=200):
    print("\n=== Experiment 4: Null Calibration / Selection Bias Check ===", flush=True)
    shape = (42, 46)
    graph = build_grid_graph(shape)
    xy = graph["coords"]
    a = np.sqrt(5.0) * cdist(xy, xy) / 3.0
    K_mat = (1.0 + a + (a * a) / 3.0) * np.exp(-a) + 1e-6 * np.eye(len(xy))
    L_mat = cholesky(K_mat, lower=True, check_finite=False)

    rng = np.random.default_rng(20261122)
    records = []
    for regime in ("white_noise", "matern_52_r3"):
        for _ in range(n_pairs):
            if regime == "white_noise":
                x = standardize(rng.normal(size=len(xy)))
                y = standardize(rng.normal(size=len(xy)))
            else:
                x = standardize(L_mat @ rng.normal(size=len(xy)))
                y = standardize(L_mat @ rng.normal(size=len(xy)))
            res = score_seed_variants(x, y, graph, scale=0, sigma=4.0)
            res["regime"] = regime
            records.append(res)

    df = pd.DataFrame(records)
    df.to_csv(OUTDIR / "exp4_null_bias.csv", index=False)
    summary = df.groupby("regime")[["J_std", "SeedJ_grad", "SeedJ_gnorm", "SeedJ_expr_opp", "SeedJ_anchor", "k_seeds"]].agg(["mean", "std"])
    print(summary.round(4))
    return df


# =============================================================================
# Plotting
# =============================================================================
def make_figures(df1, diag_examples, df2, df3, df4):
    # Figure 1: First-Elbow Algorithm & Seed Weight Maps
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 11.5), dpi=160)
    modes = ["focal", "displaced", "independent"]
    titles = ["Focal (Touching Opposing)", "Displaced (Separated 6 spots)", "Independent (Null)"]

    for row_i, (mode, title) in enumerate(zip(modes, titles)):
        ex = diag_examples[mode]
        d = ex["diag"]
        shape = ex["shape"]
        j_map = d["j_spot"].reshape(shape)
        w_map = d["w_spot"].reshape(shape)
        seeds = d["seeds"]
        sr, sc = np.unravel_index(seeds, shape)

        # Col 0: Sorted local J curve + first-elbow vs global Kneedle
        ax = axes[row_i, 0]
        M = min(d["M"], 400)
        ranks = np.arange(1, M + 1)
        ax.plot(ranks, d["sorted_vals"][:M], color="#2c3e50", lw=2, label="Sorted spot local J")
        kf = d["k_first"]
        kg = d["k_global"]
        ax.axvline(kf, color="#e74c3c", ls="--", lw=1.8, label=f"First-Elbow (k={kf})")
        if kg <= M:
            ax.axvline(kg, color="#3498db", ls=":", lw=1.6, label=f"Global Kneedle (k={kg})")
        ax.scatter([kf], [d["sorted_vals"][kf - 1]], color="#e74c3c", zorder=5, s=45)
        ax.set_title(f"{title}\nFirst-Elbow vs Global Kneedle", fontsize=10, fontweight="bold")
        ax.set_xlabel("Spot rank (descending local J)")
        ax.set_ylabel("Regularized spot J")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.25)

        # Col 1: Spot local J map + selected seeds
        ax = axes[row_i, 1]
        im1 = ax.imshow(j_map, cmap="RdBu_r", vmin=-0.5, vmax=0.5, origin="lower")
        ax.scatter(sc, sr, s=16, facecolors="none", edgecolors="lime", lw=1.1, label=f"Seeds (k={kf})")
        ax.set_title(f"Local J Map + Selected Seeds\nJ_std={ex['res']['J_std']:.3f} -> SeedJ_grad={ex['res']['SeedJ_grad']:.3f}", fontsize=9.5)
        ax.legend(fontsize=8, loc="upper right")
        plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

        # Col 2: Distance-decay weight map W(S)
        ax = axes[row_i, 2]
        im2 = ax.imshow(w_map, cmap="magma", vmin=0, vmax=1, origin="lower")
        ax.set_title(f"Seed Distance Weight W(S) (sigma=4)\nSeedJ_gnorm={ex['res']['SeedJ_gnorm']:.3f}, expr={ex['res']['SeedJ_expr_opp']:.3f}", fontsize=9.5)
        plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(OUTDIR / "fig1_first_elbow_and_seed_weights.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: 4-Panel Benchmark Summary
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=160)

    # Panel A: Amplitude imbalance at SD=0.35
    ax = axes[0, 0]
    sub1 = df1[df1["noise_sd"] == 0.35].sort_values("amplitude")
    amps = sub1["amplitude"].values
    ax.plot(amps, sub1["J_std"], "o-", lw=2.2, color="#7f8c8d", label="Standard J")
    ax.plot(amps, sub1["NegCodisp"], "s--", lw=1.6, color="#95a5a6", label="Neg. Codispersion")
    ax.plot(amps, sub1["SeedJ_grad_s4"], "o-", lw=2.2, color="#e74c3c", label="Seed-J (grad, σ=4)")
    ax.plot(amps, sub1["SeedJ_gnorm_s4"], "^-", lw=2.2, color="#e67e22", label="Seed-J (global-norm, σ=4)")
    ax.plot(amps, sub1["SeedJ_expr_opp_s4"], "d-", lw=2.0, color="#2980b9", label="Seed-J (expr-opp, σ=4)")
    ax.plot(amps, sub1["SeedJ_anchor_s4"], "v:", lw=1.8, color="#8e44ad", label="Seed-J (anchor-cross, σ=4)")
    ax.set_title("A. Amplitude Imbalance (Noise SD=0.35)", fontweight="bold")
    ax.set_xlabel("Second-marker contrast amplitude ratio")
    ax.set_ylabel("Pair AUROC (Focal vs 5 Distractors)")
    ax.set_ylim(0.35, 1.03)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)

    # Panel B: DLPFC B2 Semi-Synthetic (Adj vs Far)
    ax = axes[0, 1]
    b2_mean = df2.groupby(["beta", "method"])["auc_adj_vs_far"].mean().unstack("method")
    betas = b2_mean.index.values
    for col, lbl, clr, ls in [
        ("J1_std", "J1 (std)", "#7f8c8d", "--"),
        ("J4_std", "J4 (std)", "#2c3e50", "-"),
        ("SeedJ4_grad", "Seed-J4 (grad)", "#e74c3c", "-"),
        ("SeedJ4_gnorm", "Seed-J4 (global-norm)", "#e67e22", "-"),
        ("SeedJ4_expr_opp", "Seed-J4 (expr-opp)", "#2980b9", "-"),
        ("Lee_L_pos", "Lee's L (+)", "#27ae60", ":"),
    ]:
        if col in b2_mean.columns:
            ax.plot(betas, b2_mean[col], marker="o", lw=2.1, ls=ls, color=clr, label=lbl)
    ax.set_title("B. DLPFC Semi-Synthetic Exclusive Markers (Adj vs Distant)", fontweight="bold")
    ax.set_xlabel("Layer effect size β (log-fold)")
    ax.set_ylabel("Mean AUROC (3 donors)")
    ax.set_ylim(0.4, 1.03)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)

    # Panel C: Real DLPFC B1 (Adj vs Distant & Adj vs Same)
    ax = axes[1, 0]
    b1_m = df3.groupby("method")[["auc_adj_vs_far", "auc_adj_vs_same"]].mean()
    keep_m = ["J4_std", "SeedJ4_grad", "SeedJ4_gnorm", "SeedJ4_expr_opp", "SeedJ4_expr_pos", "Lee_L_pos"]
    b1_sub = b1_m.loc[[m for m in keep_m if m in b1_m.index]]
    x_pos = np.arange(len(b1_sub))
    w = 0.36
    ax.bar(x_pos - w/2, b1_sub["auc_adj_vs_far"], w, color="#3498db", label="Adjacent vs Distant")
    ax.bar(x_pos + w/2, b1_sub["auc_adj_vs_same"], w, color="#e74c3c", label="Adjacent vs Same")
    ax.axhline(0.5, color="black", ls="--", lw=1, alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(b1_sub.index, rotation=22, ha="right", fontsize=8.5)
    ax.set_title("C. Real Observed DLPFC B1 (3 donors, K=5)", fontweight="bold")
    ax.set_ylabel("Mean AUROC")
    ax.set_ylim(0.0, 0.95)
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.25)

    # Panel D: Null Calibration / Post-Selection Bias (Matern nu=5/2, r=3)
    ax = axes[1, 1]
    sub4 = df4[df4["regime"] == "matern_52_r3"]
    for col, lbl, clr in [
        ("J_std", "Standard J", "#2c3e50"),
        ("SeedJ_grad", "Seed-J (grad)", "#e74c3c"),
        ("SeedJ_gnorm", "Seed-J (global-norm)", "#e67e22"),
        ("SeedJ_expr_opp", "Seed-J (expr-opp)", "#2980b9"),
    ]:
        ax.hist(sub4[col], bins=24, alpha=0.45, color=clr, label=f"{lbl} (μ={sub4[col].mean():.3f})", density=True)
    ax.axvline(0.0, color="black", ls="--", lw=1.2)
    ax.set_title("D. Null Distribution Shift (Independent Matérn ν=5/2, r=3)", fontweight="bold")
    ax.set_xlabel("Score value under true null")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(OUTDIR / "fig2_benchmark_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figures to {OUTDIR}", flush=True)


if __name__ == "__main__":
    t_start = time.time()
    df1, diag_ex = run_exp1(n_rep=30, sigmas=(2.0, 4.0))
    df2 = run_exp2(sections=("151507", "151669", "151673"), n_rep=2, betas=(0.5, 1.0, 2.0), per_layer=4)
    df3 = run_exp3(sections=("151507", "151669", "151673"), k_markers=5)
    df4 = run_exp4(n_pairs=200)
    make_figures(df1, diag_ex, df2, df3, df4)
    print(f"\nALL EXPERIMENTS COMPLETED in {time.time() - t_start:.1f}s")
