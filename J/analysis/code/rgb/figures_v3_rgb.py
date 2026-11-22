"""Figures for the combined manuscript (protocol v2 + amendment v3 + Jaccard endpoints)."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from figures_v2 import (AQUA, BASE, BLUE, DIVERGING, GRID, GREEN, INK, INK2, MAGENTA, MUTED, ORANGE,
                        RED, SEQ, VIOLET, YELLOW, hgrid, panel)
from common import RESULTS, hex_xy, jx

FIG = RESULTS.parent.parent / "figures" / "v3"
FIG.mkdir(exist_ok=True)
EXT = RESULTS / "external_parallel"  # snapshot of the parallel analyses' result tables (see MANIFEST.json)

# colour follows the package (entity), fixed order; neutral grey for baselines
PKG = {"J": BLUE, "SpatialDM": ORANGE, "LIANA+": AQUA, "Scanpy/Seurat": YELLOW, "Hotspot": MAGENTA,
       "MERINGUE": GREEN, "Lee": VIOLET, "Giotto": RED, "baseline": "#7F7F7F"}

DEFAULT_TESTS = [  # column, label, package, one-sided
    ("pearson_t", "Pearson t-test", "Scanpy/Seurat", False),
    ("spearman_t", "Spearman t-test", "Scanpy/Seurat", False),
    ("spatialdm_z", "SpatialDM z-test", "SpatialDM", True),
    ("spatialdm_perm", "SpatialDM, 1,000 perm.", "SpatialDM", True),
    ("liana_perm", "LIANA+ Moran's R, 100 perm.", "LIANA+", False),
    ("hotspot_z", "Hotspot Z", "Hotspot", True),
    ("meringue_perm", "MERINGUE, 1,000 perm.", "MERINGUE", False),
    ("lee_perm", "Lee's L, 999 perm.", "Lee", True),
    ("J1_spectral", "J, spectral null (this work)", "J", False),
]
SPECTRAL = [("pearson_spectral", "Pearson r", "Scanpy/Seurat"), ("spatialdm_spectral", "SpatialDM R", "SpatialDM"),
            ("liana_spectral", "LIANA+ Moran's R", "LIANA+"), ("meringue_spectral", "MERINGUE I", "MERINGUE"),
            ("lee_spectral", "Lee's L", "Lee"), ("J1_spectral", "J", "J")]
SCORES = [  # pair-level score column, label, package
    ("J1", "J (scale 1)", "J"), ("pearson", "Pearson r", "Scanpy/Seurat"),
    ("spearman", "Spearman rho", "Scanpy/Seurat"), ("spatialdm_R", "SpatialDM R", "SpatialDM"),
    ("liana_morans_R", "LIANA+ Moran's R", "LIANA+"), ("liana_cosine_mean", "LIANA+ cosine", "LIANA+"),
    ("hotspot_Z", "Hotspot Z", "Hotspot"), ("meringue_moran", "MERINGUE I", "MERINGUE"),
    ("lee_l", "Lee's L", "Lee"), ("giotto_spatial_cor", "Giotto spatial cor.", "Giotto")]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def hbar(ax, labels, values, colors, err=None, hatches=None, xlabel="", xmax=None, fmt="{:.0f}"):
    y = np.arange(len(labels))[::-1]
    bars = ax.barh(y, values, color=colors, height=0.66, edgecolor="white", linewidth=0.6)
    if hatches:
        for b, h in zip(bars, hatches):
            if h:
                b.set_hatch(h)
                b.set_edgecolor("white")
    if err is not None:
        lo, hi = err
        ax.errorbar(values, y, xerr=[np.asarray(values) - lo, hi - np.asarray(values)], fmt="none",
                    ecolor=INK2, elinewidth=0.6, capsize=1.5)
    ends = np.asarray(err[1]) if err is not None else np.asarray(values)
    for yy, v, e in zip(y, values, ends):
        ax.text(max(v, e) + (xmax or max(values)) * 0.015, yy, fmt.format(v), va="center", fontsize=6.3, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    if xmax:
        ax.set_xlim(0, xmax)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    return y


# ------------------------------------------------------------------------------------- Fig 2
def figure2():
    cal = pd.read_csv(RESULTS / "gold_calibration_all.csv.gz", low_memory=False)
    crc = pd.read_csv(RESULTS / "gold_crc_significance.csv").set_index("test")
    sp = pd.read_csv(RESULTS / "gold_speed.csv")
    fig = plt.figure(figsize=(7.4, 8.6))
    gs = fig.add_gridspec(3, 2, hspace=0.62, wspace=0.95, height_ratios=[1.15, 1.0, 1.0])
    rows = []
    for setting, letter, title in (("rotated", "A", "Unrelated genes, cross-donor null (array positions decoupled)"),
                                   ("aligned", None, None)):
        d = cal[(cal.axis == "A2") & (cal.setting == setting)]
        for col, lab, pkg, one in DEFAULT_TESTS:
            k = int((d[col] < 0.05).sum())
            lo, hi = wilson(k, len(d))
            rows.append(dict(null=setting, test=lab, column=col, package=pkg, one_sided=one,
                             fpr=100 * k / len(d), lo=100 * lo, hi=100 * hi, n=len(d)))
    tab = pd.DataFrame(rows)
    ax = fig.add_subplot(gs[0, :])
    t = tab[tab.null == "rotated"]
    hbar(ax, [r.test + (" †" if r.one_sided else "") for r in t.itertuples()], t.fpr.values,
         [PKG[p] for p in t.package], err=(t.lo.values, t.hi.values), xlabel="false-positive rate at α = 0.05 (%)",
         xmax=100)
    ax.axvspan(2.5, 7.5, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.axvline(5, color=INK2, lw=0.8, ls="--", zorder=0.5)
    ax.axvline(50, color=MUTED, lw=0.6, ls=":", zorder=0.5)
    ax.text(50.8, len(DEFAULT_TESTS) - 0.35, "ceiling of a one-sided test (†)", fontsize=6, color=MUTED, va="center")
    ax.set_title(title, fontweight="normal", fontsize=8)
    panel(ax, "A")
    # B: the same statistics under the spectral null
    ax = fig.add_subplot(gs[1, 0])
    d = cal[(cal.axis == "A2") & (cal.setting == "rotated")]
    vals = [100 * (d[c] < 0.05).mean() for c, _, _ in SPECTRAL]
    ci = [wilson(int((d[c] < 0.05).sum()), len(d)) for c, _, _ in SPECTRAL]
    hbar(ax, [l for _, l, _ in SPECTRAL], vals, [PKG[p] for _, _, p in SPECTRAL],
         err=(100 * np.array([c[0] for c in ci]), 100 * np.array([c[1] for c in ci])),
         xlabel="false-positive rate (%)", xmax=10, fmt="{:.1f}")
    ax.axvspan(2.5, 7.5, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.axvline(5, color=INK2, lw=0.8, ls="--", zorder=0.5)
    ax.set_title("Same statistics, spectral null", fontweight="normal", fontsize=8)
    panel(ax, "B")
    # C: synthetic fields, by range (nu = 0.5), default tests
    ax = fig.add_subplot(gs[1, 1])
    a1 = cal[cal.axis == "A1"]
    xs = [0, 1, 3, 10]
    for col, lab, pkg, one in DEFAULT_TESTS:
        ys = [100 * (a1[a1.setting == "white"][col] < 0.05).mean()] + \
             [100 * (a1[a1.setting == f"matern0.5_range{r}"][col] < 0.05).mean() for r in (1, 3, 10)]
        ax.plot(range(4), ys, marker="o", ms=2.8, lw=1.4 if pkg == "J" else 1.0, color=PKG[pkg],
                ls="--" if one else "-", zorder=3 if pkg == "J" else 2)
    ax.axhspan(2.5, 7.5, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.text(3.05, 2.0, "dashed: one-sided test", fontsize=5.8, color=MUTED, ha="right", va="top")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["white\nnoise", "1", "3", "10"])
    ax.set_xlabel("Matérn ν = ½ range (spot spacings)")
    ax.set_ylabel("false-positive rate (%)")
    ax.set_ylim(0, 100)
    hgrid(ax)
    ax.set_title("Gaussian random fields (3,000 pairs each)", fontweight="normal", fontsize=8)
    handles = [Patch(color=PKG[p], label=p) for p in ("J", "Scanpy/Seurat", "SpatialDM", "LIANA+", "Hotspot",
                                                       "MERINGUE", "Lee")]
    ax.legend(handles=handles, frameon=False, fontsize=5.8, loc="upper left", ncol=1, handlelength=1)
    panel(ax, "C")
    # D: real section, fraction of all pairs called significant
    ax = fig.add_subplot(gs[2, 0])
    labs = [l for _, l, _, _ in DEFAULT_TESTS]
    cols = [c for c, _, _, _ in DEFAULT_TESTS]
    vals = [100 * crc.loc[c, "frac_q05"] for c in cols]
    hbar(ax, [l + (" †" if o else "") for (_, l, _, o) in DEFAULT_TESTS], vals,
         [PKG[p] for _, _, p, _ in DEFAULT_TESTS], xlabel="pairs with q < 0.05 (%)", xmax=110)
    ax.axhline(0.5, color=INK2, lw=0.6, ls=":")
    ax.text(108, 0.05, "calibrated", fontsize=5.8, color=INK2, ha="right", va="center")
    ax.set_title("Colorectal section: 44,850 pairs, default tests", fontweight="normal", fontsize=8)
    panel(ax, "D")
    # E: cost-calibration plane
    ax = fig.add_subplot(gs[2, 1])
    name_map = {"Pearson t-test (Scanpy/Seurat)": "pearson_t", "Spearman t-test (Scanpy/Seurat)": "spearman_t",
                "SpatialDM z-test": "spatialdm_z", "SpatialDM permutation (1,000)": "spatialdm_perm",
                "LIANA+ Moran's R, 100 permutations": "liana_perm", "Hotspot Z": "hotspot_z",
                "MERINGUE, 1,000 permutations": "meringue_perm", "Lee's L, 999 permutations": "lee_perm",
                "J, spectral null (this work)": "J1_spectral"}
    rot = tab[tab.null == "rotated"].set_index("column")
    for r in sp.itertuples():
        col = name_map[r.method]
        row = rot.loc[col]
        ax.scatter(r.seconds_total, row.fpr, s=34, color=PKG[row.package], edgecolor="white", lw=0.8,
                   marker="^" if row.one_sided else "o", zorder=3)
        short = {"pearson_t": "Pearson", "spearman_t": "Spearman", "spatialdm_z": "SpatialDM z",
                 "spatialdm_perm": "SpatialDM perm.", "liana_perm": "LIANA+", "hotspot_z": "Hotspot",
                 "meringue_perm": "MERINGUE", "lee_perm": "Lee", "J1_spectral": "J (spectral)"}[col]
        dy = {"spearman_t": -6, "pearson_t": 5, "hotspot_z": -6, "spatialdm_z": 4, "lee_perm": -6,
              "spatialdm_perm": 5, "meringue_perm": -6, "liana_perm": 5, "J1_spectral": 6}[col]
        ax.annotate(short, (r.seconds_total, row.fpr), (0, dy), textcoords="offset points", fontsize=5.8,
                    ha="center", color=INK)
    ax.axhspan(2.5, 7.5, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(1, 8000)
    ax.set_ylim(0, 100)
    ax.set_xlabel("seconds to p-values, all 8.0 M pairs\n(4,000 genes, 1 thread; perm. extrapolated)")
    ax.set_ylabel("false-positive rate (%)")
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_title("Cost and calibration", fontweight="normal", fontsize=8)
    panel(ax, "E")
    save(fig, "fig2_calibration_gold")
    return tab


# ------------------------------------------------------------------------------------- Fig 3
def dotplot(ax, df, value, by, order, colors, xlabel, ref=0.5):
    for k, (col, lab, pkg) in enumerate(order):
        v = df[df.method == col].groupby(by)[value].mean()
        y = len(order) - 1 - k
        ax.scatter(v.values, np.full(len(v), y) + np.random.default_rng(k).uniform(-0.18, 0.18, len(v)),
                   s=7, color=colors[pkg], alpha=0.75, linewidths=0)
        ax.plot([v.mean()] * 2, [y - 0.32, y + 0.32], color=INK, lw=1.4)
    ax.axvline(ref, color=INK2, lw=0.7, ls="--")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([lab for _, lab, _ in order][::-1])
    ax.set_xlabel(xlabel)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def figure3():
    b1 = pd.read_csv(RESULTS / "gold_b1_auroc.csv", dtype={"section": str})
    b2 = pd.read_csv(RESULTS / "gold_b2_semisynthetic.csv", dtype={"section": str})
    c1 = pd.read_csv(RESULTS / "gold_c1_seqfish_contactz.csv")
    c2 = pd.read_csv(RESULTS / "gold_c2_seqfish_contactrate.csv")
    rho = pd.read_csv(RESULTS / "gold_c2_seqfish_contactrate_rho.csv")
    rep = pd.read_csv(EXT / "seqfish_global_benchmark.csv")
    fig = plt.figure(figsize=(7.4, 7.6))
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=1.0)
    order = SCORES
    ax = fig.add_subplot(gs[0, 0])
    dotplot(ax, b1[b1.K == 10], "auc_touch_vs_apart", "section", order, PKG,
            "AUROC, adjacent vs distant layers")
    ax.set_xlim(0.3, 0.9)
    ax.set_title("DLPFC, 12 sections (points)", fontweight="normal", fontsize=8)
    panel(ax, "A")
    ax = fig.add_subplot(gs[0, 1])
    dotplot(ax, c1, "auc_touch_vs_apart", "embryo", order, PKG, "AUROC, touching vs separated types")
    ax.set_xlim(0.2, 0.9)
    ax.set_title("seqFISH, 3 embryos (contact z design)", fontweight="normal", fontsize=8)
    panel(ax, "B")
    # C: the contact-rate design as reported (sign-reversed) and in natural orientation
    ax = fig.add_subplot(gs[1, 0])
    rep = rep[rep.task == "Adjacent vs. Distant"]
    mapping = [("-Pearson r (Scanpy/Seurat)", "Pearson r", "Scanpy/Seurat"),
               ("-Hotspot Spatial Cov. (Scanpy)", "Hotspot covariance", "Hotspot"),
               ("-Bivariate Moran's I (SpatialDM/MERINGUE)", "Moran's I (SpatialDM/MERINGUE)", "MERINGUE"),
               ("-Lee's L (Lee 2001 / Prior)", "Lee's L", "Lee"),
               ("-LIANA+ Spatial Cosine (Scanpy)", "LIANA+ cosine", "LIANA+"),
               ("J (Weighted Laplacian)", "J", "J")]
    y = np.arange(len(mapping))[::-1]
    for yy, (rlab, lab, pkg) in zip(y, mapping):
        reported = rep[rep.method == rlab].auroc.mean()
        natural = 1 - reported if rlab.startswith("-") else reported
        ax.plot([reported, natural], [yy, yy], color=GRID, lw=2, zorder=1)
        ax.scatter([reported], [yy], s=26, facecolor="white", edgecolor=PKG[pkg], lw=1.2, zorder=3)
        ax.scatter([natural], [yy], s=26, color=PKG[pkg], zorder=3)
    ax.axvline(0.5, color=INK2, lw=0.7, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([m[1] for m in mapping])
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(-1.6, len(mapping) - 0.5)
    ax.set_xlabel("AUROC, touching vs separated (contact-rate design)")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.scatter([], [], s=20, facecolor="white", edgecolor=INK2, label="as reported, sign reversed")
    ax.scatter([], [], s=20, color=INK2, label="same statistic, natural sign")
    ax.legend(frameon=False, fontsize=6, loc="lower left", ncol=2, columnspacing=0.8, handletextpad=0.2)
    ax.set_title("Sign convention decides the winner", fontweight="normal", fontsize=8)
    panel(ax, "C")
    # D: semi-synthetic exclusive markers
    ax = fig.add_subplot(gs[1, 1])
    for col, lab, pkg in [("J1", "J", "J"), ("hotspot_Z", "Hotspot Z", "Hotspot"),
                          ("liana_morans_R", "LIANA+ R", "LIANA+"), ("lee_l", "Lee's L", "Lee"),
                          ("spatialdm_R", "SpatialDM R", "SpatialDM"), ("pearson", "Pearson r", "Scanpy/Seurat")]:
        m = b2[b2.method == col].groupby("beta").auc_touch_vs_apart.mean()
        ax.plot(m.index, m.values, marker="o", ms=3.2, lw=1.6 if pkg == "J" else 1.0, color=PKG[pkg], label=lab)
    ax.axhline(0.5, color=INK2, lw=0.7, ls="--")
    ax.set_xticks([0.5, 1, 2])
    ax.set_xlabel("layer effect β (log scale)")
    ax.set_ylabel("AUROC, adjacent vs distant")
    ax.set_ylim(0.45, 1.0)
    hgrid(ax)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    ax.set_title("Exclusive markers on real layer maps", fontweight="normal", fontsize=8)
    panel(ax, "D")
    save(fig, "fig3_accuracy_gold")
    return rho.groupby("method").spearman_rho.mean()


# ------------------------------------------------------------------------------------- Fig 5
LOCAL = [("J_local_lazy4_ridge1", "J, local (ridge 1)", "J"), ("Q_signed_cross_energy_s1", "Q, signed cross-energy", "J"),
         ("unsigned_gradient_s1", "|Δx − Δy| gradient", "baseline"),
         ("joint_gradient_energy_s1", "joint gradient energy", "baseline"),
         ("spatialdm_local_moran", "SpatialDM local Moran", "SpatialDM"),
         ("liana_local_pearson_negated", "LIANA+ local Pearson (−)", "LIANA+"),
         ("liana_local_cosine", "LIANA+ local cosine", "LIANA+")]


def figure5():
    rep = pd.read_csv(EXT / "local_replicates.csv")
    pc = rep.groupby(["section", "regime", "method"])[["average_precision", "prevalence"]].median().reset_index()
    m2 = pd.read_csv(RESULTS / "jaccard_m2_semisynthetic.csv.gz", dtype={"unit": str})
    m1 = pd.read_csv(RESULTS / "jaccard_m1_interfaces.csv.gz", dtype={"unit": str})
    fig = plt.figure(figsize=(7.4, 8.2))
    gs = fig.add_gridspec(2, 2, hspace=0.75, wspace=0.95)
    ax = fig.add_subplot(gs[0, 0])
    order = [("j_regularized", "J, local (ridge 1)", "J"), ("j", "J, local (no ridge)", "J"),
             ("signed_gradient", "signed gradient (Q)", "J"), ("unsigned_gradient", "|Δx − Δy| gradient", "baseline"),
             ("liana_pearson", "LIANA+ weighted Pearson", "LIANA+"), ("liana_masked_spearman", "LIANA+ masked Spearman", "LIANA+"),
             ("liana_spearman", "LIANA+ weighted Spearman", "LIANA+"), ("liana_cosine", "LIANA+ cosine", "LIANA+"),
             ("spatialdm_morans", "SpatialDM local Moran", "SpatialDM")]
    vals = [pc[pc.method == c].average_precision.median() for c, _, _ in order]
    hbar(ax, [l for _, l, _ in order], vals, [PKG[p] for _, _, p in order], xlabel="average precision (median of 4 strata)",
         xmax=0.5, fmt="{:.2f}")
    ax.axvline(pc.prevalence.median(), color=INK2, lw=0.7, ls="--")
    ax.set_title("Known truth, held-out MERFISH geometries\n(official LIANA+ 2.1.1 / SpatialDM 0.3.1)",
                 fontweight="normal", fontsize=8)
    panel(ax, "A")
    ax = fig.add_subplot(gs[0, 1])
    for col, lab, pkg in LOCAL:
        d = m2[m2.method == col].groupby(["beta", "unit"]).iou_at_T.median().groupby("beta").mean()
        ax.plot(d.index, d.values, marker="o", ms=3, lw=1.6 if pkg == "J" else 1.0, color=PKG[pkg],
                ls="--" if col.startswith(("Q_", "joint")) else ":" if col.endswith("cosine") else "-", label=lab)
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("marker effect β (log scale)")
    ax.set_ylabel("Jaccard (IoU) at true interface size")
    hgrid(ax)
    ax.legend(frameon=False, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2,
              columnspacing=1.0)
    ax.set_title("Exclusive markers on DLPFC layer maps", fontweight="normal", fontsize=8)
    panel(ax, "B")
    ax = fig.add_subplot(gs[1, 0])
    rows = []
    for ds in ("DLPFC", "seqFISH"):
        d = m1[m1.dataset == ds].groupby(["unit", "method"]).iou_at_T.median().groupby("method").mean()
        prev = m1[m1.dataset == ds].prevalence.median()
        for col, lab, pkg in LOCAL:
            rows.append(dict(ds=ds, lab=lab, pkg=pkg, v=d.get(col, np.nan), chance=prev / (2 - prev)))
    t = pd.DataFrame(rows)
    y = np.arange(len(LOCAL))[::-1]
    for off, ds, hatch in ((0.18, "DLPFC", None), (-0.18, "seqFISH", "////")):
        s = t[t.ds == ds]
        bars = ax.barh(y + off, s.v.values, height=0.34, color=[PKG[p] for p in s.pkg], edgecolor="white", lw=0.5)
        if hatch:
            for b in bars:
                b.set_hatch(hatch)
        ax.axvline(s.chance.iloc[0], color=INK2, lw=0.6, ls=":" if hatch else "--")
        ax.text(s.chance.iloc[0], len(LOCAL) - 0.35, "chance" if not hatch else "", fontsize=5.6, color=INK2,
                ha="left", va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([l for _, l, _ in LOCAL])
    ax.set_xlabel("Jaccard (IoU) at true interface size")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=MUTED, label="DLPFC layer boundaries"),
                       Patch(facecolor=MUTED, hatch="////", edgecolor="white", label="seqFISH cell-type contacts")],
              frameon=False, fontsize=6, loc="lower right")
    ax.set_title("Real marker pairs, expert-annotated interfaces", fontweight="normal", fontsize=8)
    panel(ax, "C")
    ax = fig.add_subplot(gs[1, 1])
    example_map(ax)
    panel(ax, "D")
    save(fig, "fig5_local_jaccard")
    return t


def example_map(ax, section="151673", beta=2.0):
    from bench_jaccard import LocalMaps
    from datasets import load_dlpfc
    from gold import GoldStandards
    counts, _, meta = load_dlpfc(section)
    g = jx.visium_graph(meta.array_row.values, meta.array_col.values)
    gs = GoldStandards(hex_xy(meta.array_row.values, meta.array_col.values), g)
    lm = LocalMaps(g, gs.W_liana, gs.W_sdm)
    lay = meta.layer.values
    lib = np.asarray(counts.sum(axis=1)).ravel().astype(float)
    ind = jx.smoother(g) @ np.column_stack([(lay == "L6").astype(float), (lay == "WM").astype(float)])
    lam = (lib / lib.mean())[:, None] * np.exp(beta * ind)
    lam *= 2.0 / lam.mean(axis=0, keepdims=True)
    x = np.log1p(1e4 * np.random.default_rng(3).poisson(lam) / lib[:, None])
    truth = ((lay[lm.i] == "L6") & (lay[lm.j] == "WM")) | ((lay[lm.i] == "WM") & (lay[lm.j] == "L6"))
    maps = lm.maps(x, np.array([0]), np.array([1]))
    xy = hex_xy(meta.array_row.values, meta.array_col.values)
    xy[:, 1] *= -1
    ax.scatter(xy[:, 0], xy[:, 1], s=1.2, color="#eeede8", linewidths=0, rasterized=True)
    t = int(truth.sum())
    for col, colr, lab in (("J_local_lazy4_ridge1", BLUE, "J top-|T| edges"),
                           ("spatialdm_local_moran", ORANGE, "SpatialDM local top-|T|")):
        sc = maps[col][:, 0]
        top = np.argpartition(-sc, t - 1)[:t]
        tp = truth[top].sum()
        seg = np.stack([xy[lm.i[top]], xy[lm.j[top]]], axis=1)
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(seg, colors=colr, linewidths=1.3, alpha=0.9,
                                         label=f"{lab} (IoU {tp / (2 * t - tp):.2f})"))
    seg = np.stack([xy[lm.i[truth]], xy[lm.j[truth]]], axis=1)
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(seg, colors=INK, linewidths=0.5, alpha=0.6, label="annotated L6–WM edges"))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spn in ax.spines.values():
        spn.set_visible(False)
    ax.legend(frameon=False, fontsize=5.8, loc="lower left", bbox_to_anchor=(0, -0.22))
    ax.set_title(f"Example, section {section}: exclusive L6/WM markers (β = {beta:g})", fontweight="normal", fontsize=8)


# ------------------------------------------------------------------------------------- Fig 6
REPRO = [("J1", "J (scale 1)", "J"), ("J2", "J (scale 2)", "J"), ("J4", "J (scale 4)", "J"),
         ("J1_concordant", "−J (concordant)", "J"), ("pearson", "Pearson r", "Scanpy/Seurat"),
         ("spearman", "Spearman rho", "Scanpy/Seurat"), ("spatialdm_R", "SpatialDM R", "SpatialDM"),
         ("liana_morans_R", "LIANA+ Moran's R", "LIANA+"), ("hotspot_Z", "Hotspot Z", "Hotspot"),
         ("meringue_moran", "MERINGUE I", "MERINGUE"), ("lee_l", "Lee's L", "Lee"),
         ("giotto_spatial_cor", "Giotto spatial cor.", "Giotto"),
         ("jaccard_auc_similarity", "Jaccard similarity (lab)", "baseline")]


def figure6():
    r = pd.read_csv(RESULTS / "jaccard_m3_reproducibility.csv", dtype={"section_a": str, "section_b": str})
    m = pd.read_csv(RESULTS / "jaccard_m4_redundancy.csv", dtype={"section": str})
    fig = plt.figure(figsize=(7.4, 3.9))
    gs = fig.add_gridspec(1, 2, wspace=0.95, width_ratios=[1, 1.15])
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(REPRO))[::-1]
    for off, kind, alpha in ((0.18, "serial_10um", 1.0), (-0.18, "distant_300um", 0.45)):
        v = [r[(r.method == c) & (r.kind == kind)].jaccard.mean() for c, _, _ in REPRO]
        ax.barh(y + off, v, height=0.34, color=[PKG[p] for _, _, p in REPRO], alpha=alpha, edgecolor="white", lw=0.5)
    ax.axvline(r.chance.iloc[0], color=INK2, lw=0.7, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([l for _, l, _ in REPRO])
    ax.set_xlabel("Jaccard of top-1% pairs between sections")
    ax.set_xlim(0, 0.8)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=MUTED, label="serial, 10 µm apart"),
                       Patch(facecolor=MUTED, alpha=0.45, label="300 µm apart")], frameon=False, fontsize=6,
              loc="upper right")
    ax.text(r.chance.iloc[0] + 0.01, -0.9, f"chance {r.chance.iloc[0]:.3f}", fontsize=5.6, color=INK2)
    ax.set_title("Reproducibility across sections (same donor)", fontweight="normal", fontsize=8)
    panel(ax, "A")
    ax = fig.add_subplot(gs[0, 1])
    names = [c for c, _, _ in REPRO]
    mm = m.groupby(["method_a", "method_b"]).jaccard.mean()
    M = np.eye(len(names))
    for a in range(len(names)):
        for b in range(len(names)):
            if a != b:
                M[a, b] = mm.get((names[a], names[b]), mm.get((names[b], names[a]), np.nan))
    im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels([l for _, l, _ in REPRO], rotation=60, ha="right", fontsize=6)
    ax.set_yticklabels([l for _, l, _ in REPRO], fontsize=6)
    for a in range(len(names)):
        for b in range(len(names)):
            if a != b and np.isfinite(M[a, b]):
                ax.text(b, a, f"{M[a, b]:.2f}"[1:] if M[a, b] < 1 else "1", ha="center", va="center", fontsize=4.6,
                        color="white" if M[a, b] > 0.55 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Jaccard of top-1% pairs", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    ax.set_title("Redundancy between methods (same section)", fontweight="normal", fontsize=8)
    panel(ax, "B")
    save(fig, "fig6_jaccard_repro_redundancy")


if __name__ == "__main__":
    which = sys.argv[1:] or ["2", "3", "5", "6"]
    if "2" in which:
        figure2().to_csv(RESULTS / "gold_calibration_fpr_table.csv", index=False)
    if "3" in which:
        print(figure3())
    if "5" in which:
        figure5().to_csv(RESULTS / "jaccard_m1_summary.csv", index=False)
    if "6" in which:
        figure6()


# ------------------------------------------------------------------------------------- Fig 7
def figure7():
    s = pd.read_csv(RESULTS / "speed_v3.csv")
    g = pd.read_csv(RESULTS / "gold_speed.csv").set_index("method")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw=dict(wspace=1.05))
    ax = axes[0]
    order = [("J scale 1 (v0.3)", "J (scale 1)", "J"), ("J scale 1 (v0.3), float32", "J (scale 1), float32", "J"),
             ("J scale 0 (v0.3)", "J (scale 0)", "J"), ("J scale 1, general product (v0.2)", "J, v0.2 kernel", "baseline"),
             ("Lee's L", "Lee's L", "Lee"), ("bivariate Moran's I", "bivariate Moran's I", "MERINGUE"),
             ("Pearson r", "Pearson r", "Scanpy/Seurat")]
    d1 = s[(s.threads == 1) & (s.genes == 4000)].set_index("task")
    d2 = s[(s.threads == 2) & (s.genes == 4000)].set_index("task")
    y = np.arange(len(order))[::-1]
    ax.barh(y + 0.18, [d1.loc[t, "seconds"] for t, _, _ in order], height=0.34,
            color=[PKG[p] for _, _, p in order], edgecolor="white", lw=0.5)
    ax.barh(y - 0.18, [d2.loc[t, "seconds"] for t, _, _ in order], height=0.34,
            color=[PKG[p] for _, _, p in order], alpha=0.45, edgecolor="white", lw=0.5)
    for yy, (t, _, _) in zip(y, order):
        ax.text(d1.loc[t, "seconds"] + 0.03, yy + 0.18, f"{d1.loc[t, 'seconds']:.2f}", va="center", fontsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels([l for _, l, _ in order])
    ax.set_xlabel("seconds, all 8.0 M pairs of 4,000 genes")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=MUTED, label="1 thread"), Patch(facecolor=MUTED, alpha=0.45, label="2 threads")],
              frameon=False, fontsize=6, loc="lower right")
    ax.set_title("All-pairs statistic, 3,611 spots", fontweight="normal", fontsize=8)
    panel(ax, "A")
    ax = axes[1]
    rows = [("J, spectral null (this work)", "J, calibrated p-values", "J"),
            ("Pearson t-test (Scanpy/Seurat)", "Pearson t-test", "Scanpy/Seurat"),
            ("SpatialDM z-test", "SpatialDM z-test", "SpatialDM"), ("Hotspot Z", "Hotspot Z", "Hotspot"),
            ("LIANA+ Moran's R, 100 permutations", "LIANA+, 100 perm.", "LIANA+"),
            ("Lee's L, 999 permutations", "Lee's L, 999 perm.", "Lee"),
            ("MERINGUE, 1,000 permutations", "MERINGUE, 1,000 perm.", "MERINGUE"),
            ("SpatialDM permutation (1,000)", "SpatialDM, 1,000 perm.", "SpatialDM")]
    vals = [g.loc[m, "seconds_total"] for m, _, _ in rows]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, height=0.62, color=[PKG[p] for _, _, p in rows], edgecolor="white", lw=0.5)
    for yy, v, (m, _, _) in zip(y, vals, rows):
        lab = f"{v:.0f} s" if v < 120 else f"{v / 60:.0f} min"
        ax.text(v * 1.12, yy, lab + (" (calibrated)" if m.startswith("J") else ""), va="center", fontsize=6)
    ax.set_xscale("log")
    ax.set_xlim(1, 3e4)
    ax.set_yticks(y)
    ax.set_yticklabels([l for _, l, _ in rows])
    ax.set_xlabel("seconds to p-values, all pairs (1 thread)")
    ax.xaxis.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Testing every pair", fontweight="normal", fontsize=8)
    panel(ax, "B")
    save(fig, "fig7_speed")


if __name__ == "__main__" and "7" in sys.argv[1:]:
    figure7()
