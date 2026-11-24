"""Rebuild Figure 4 and the observed-data supplement from the matched benchmarks."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "J-analysis"
FIG = ROOT / "figures" / "v4"
KEYS = ["section", "beta", "rep", "method"]
ENDPOINTS = ["adjacent_vs_distant", "adjacent_vs_same"]
LABELS = {
    "J0": "J₀", "J1": "J₁", "J2": "J₂", "J4": "J₄",
    "Q1_signed_cross_energy": "Signed gradient Q",
    "neg_codispersion1": "Negative codispersion",
    "gradient_coenergy1": "Gradient co-energy",
    "hotspot_Z": "Hotspot Z", "liana_morans_R": "LIANA+ Moran",
    "giotto_spatial_cor": "Giotto correlation", "lee_l": "Lee’s L",
    "spatialdm_R": "SpatialDM R", "meringue_moran": "MERINGUE",
    "spearman": "Spearman ρ", "pearson": "Pearson r",
    "liana_cosine_mean": "LIANA+ cosine",
}
COLOCAL = {
    "hotspot_Z", "liana_morans_R", "giotto_spatial_cor", "lee_l",
    "spatialdm_R", "meringue_moran", "spearman", "pearson", "liana_cosine_mean",
}
BLUE, DARK, TEAL, GRID = "#F8766D", "#7F7F7F", "#619CFF", "#e5eaf0"
J_VARIANTS = {"J0", "J1", "J2", "J4"}


def matched_simulations():
    """Combine distinct methods after verifying the benchmarks share the simulated draws."""
    results = ROOT / "results"
    v2 = pd.read_csv(results / "v2" / "dlpfc_b2_semisynthetic.csv", dtype={"section": str})
    v3 = pd.read_csv(results / "v3" / "gold_b2_semisynthetic.csv", dtype={"section": str})
    v2 = v2.rename(columns={f"auc_{name}": name for name in ENDPOINTS})
    v3 = v3.rename(columns={"auc_touch_vs_apart": ENDPOINTS[0], "auc_touch_vs_same": ENDPOINTS[1]})
    for method in ("J0", "J1"):
        columns = KEYS + ENDPOINTS
        left = v2.loc[v2.method.eq(method), columns].sort_values(KEYS).reset_index(drop=True)
        right = v3.loc[v3.method.eq(method), columns].sort_values(KEYS).reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    extra = v2.method.isin(["J2", "J4", "neg_codispersion1", "gradient_coenergy1"])
    data = pd.concat([v3[KEYS + ENDPOINTS], v2.loc[extra, KEYS + ENDPOINTS]], ignore_index=True)
    counts = data.groupby(["beta", "method"]).size()
    if len(counts) != 3 * len(LABELS) or not counts.eq(60).all() or data.duplicated(KEYS).any():
        raise ValueError("Expected 12 sections × 5 replicates for every method and effect")
    # Positive co-occurrence is the natural orientation for adjacent vs distant.
    # Lower co-occurrence orders adjacent ahead of same-layer pairs.
    data["adjacent_vs_same_oriented"] = data[ENDPOINTS[1]]
    reverse = data.method.isin(COLOCAL)
    data.loc[reverse, "adjacent_vs_same_oriented"] = 1 - data.loc[reverse, ENDPOINTS[1]]
    return data


def color(method):
    return BLUE if method in J_VARIANTS else TEAL if method in COLOCAL else DARK


def label_panel(ax, letter, title):
    ax.set_title(f"{letter}   {title}", loc="left", fontsize=10, fontweight="bold", pad=8)
    ax.spines[["top", "right"]].set_visible(False)


def dot_ranks(ax, table, column, ids, *, reference):
    summary = table.groupby(["method", "section"])[column].mean().unstack("section")
    for y, method in enumerate(ids):
        data = summary.loc[method].to_numpy()
        ax.scatter(data, y + np.linspace(-0.15, 0.15, len(data)), s=7,
                   color=color(method), alpha=0.5, edgecolor="none")
        ax.scatter(data.mean(), y, s=24, marker="D", color=color(method), edgecolor="white", linewidth=0.45)
    ax.axvline(reference, color="#87939d", linestyle="--", linewidth=0.7, zorder=0)
    ax.set_yticks(range(len(ids)), [LABELS[m] for m in ids], fontsize=6.9)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def simulation_figure(data, stamp):
    at_two = data.loc[data.beta.eq(2)]
    means = data.groupby(["beta", "method"]).adjacent_vs_distant.mean().unstack()
    section_means = at_two.groupby(["section", "method"]).adjacent_vs_distant.mean().unstack()
    baseline = section_means.drop(columns=sorted(J_VARIANTS)).max(axis=1)
    fig, axs = plt.subplots(2, 2, figsize=(8.0, 7.7),
                            gridspec_kw={"height_ratios": [1.18, 0.95]})
    fig.subplots_adjust(left=0.188, right=0.972, top=0.948, bottom=0.09, wspace=0.46, hspace=0.41)
    ax = axs[0, 0]
    order = means.loc[2].sort_values(ascending=False).index
    dot_ranks(ax, at_two, ENDPOINTS[0], order, reference=0.5)
    ax.set(xlim=(0.27, 1.025), xlabel="AUROC  ·  adjacent vs distant")
    label_panel(ax, "A", "All methods at β = 2")

    ax = axs[0, 1]
    ax.plot([0.83, 1.006], [0.83, 1.006], color="#87939d", linestyle="--", linewidth=0.75)
    ax.scatter(baseline, section_means.J4, color=BLUE, s=24, label="J₄", alpha=0.8)
    ax.scatter(baseline, section_means.J1, facecolors="none", edgecolors=DARK, linewidths=0.85,
               s=24, label="J₁")
    ax.set(xlim=(0.83, 1.008), ylim=(0.83, 1.008), aspect="equal",
           xlabel="Best non-J AUROC in the section", ylabel="J AUROC")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    label_panel(ax, "B", "Paired sections at β = 2")

    ax = axs[1, 0]
    subset = ["J4", "J2", "J1", "neg_codispersion1", "Q1_signed_cross_energy",
              "hotspot_Z", "lee_l", "pearson"]
    dot_ranks(ax, at_two, "adjacent_vs_same_oriented", subset, reference=1.0)
    ax.set_yticklabels([("−" if m in COLOCAL else "") + LABELS[m] for m in subset])
    ax.set(xlim=(0.994, 1.0009), xticks=[0.995, 0.9975, 1.0],
           xlabel="AUROC  ·  adjacent vs same layer")
    ax.set_xticklabels(["0.995", "0.9975", "1.000"])
    label_panel(ax, "C", "Secondary task at β = 2")

    ax = axs[1, 1]
    comparison = means.drop(columns=sorted(J_VARIANTS)).max(axis=1)
    ax.plot(means.index, comparison, color=DARK, marker="s", markersize=4, linestyle="--",
            label="Best non-J at each β")
    for method, marker, style in (("J4", "o", "-"), ("J1", "^", ":"), ("J2", "P", "-.")):
        ax.plot(means.index, means[method], color=BLUE, label=LABELS[method], marker=marker,
                markersize=4, linewidth=1.2, linestyle=style)
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.5, 1, 2], ["0.5", "1", "2"])
    ax.set(xlim=(0.455, 2.2), ylim=(0.48, 1.025), xlabel="Layer effect β  ·  log₂ axis",
           ylabel="Mean AUROC  ·  adjacent vs distant")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    label_panel(ax, "D", "Effect-size sweep")
    save(fig, "figure4_pair_accuracy", stamp)


def observed_figure(stamp):
    """Retain all three observed-data analyses from v3 in a single figure."""
    r = ROOT / "results" / "v3"
    dl = pd.read_csv(r / "gold_b1_auroc.csv").query("K == 10").rename(columns={"section": "unit"})
    sq = pd.read_csv(r / "gold_c1_seqfish_contactz.csv").rename(columns={"embryo": "unit"})
    rate = pd.read_csv(r / "external_parallel" / "seqfish_global_benchmark.csv")
    rate = rate.loc[rate.task.eq("Adjacent vs. Distant")].groupby("method").auroc.mean()
    order = ["J1", "pearson", "spearman", "spatialdm_R", "liana_morans_R", "hotspot_Z",
             "meringue_moran", "lee_l", "giotto_spatial_cor"]
    fig, axs = plt.subplots(3, 1, figsize=(7.0, 8.4))
    fig.subplots_adjust(left=0.26, right=0.959, bottom=0.075, top=0.949, hspace=0.54)
    for ax, df, letter, title in (
        (axs[0], dl, "A", "DLPFC · adjacent vs distant layers"),
        (axs[1], sq, "B", "seqFISH · contact z design"),
    ):
        dot_ranks(ax, df.rename(columns={"unit": "section"}), "auc_touch_vs_apart", order, reference=0.5)
        ax.set(xlim=(0.2, 0.91), xlabel="AUROC")
        label_panel(ax, letter, title)
    ax = axs[2]
    keys = [
        ("J (Weighted Laplacian)", "J"),
        ("-Pearson r (Scanpy/Seurat)", "Pearson r"),
        ("-Hotspot Spatial Cov. (Scanpy)", "Hotspot covariance"),
        ("-Bivariate Moran's I (SpatialDM/MERINGUE)", "Bivariate Moran"),
        ("-Lee's L (Lee 2001 / Prior)", "Lee’s L"),
        ("-LIANA+ Spatial Cosine (Scanpy)", "LIANA+ cosine"),
    ]
    for y, (method, _) in enumerate(keys):
        reported = rate[method]
        if method.startswith("-"):
            natural = 1 - reported
            ax.plot([reported, natural], [y, y], color=GRID, linewidth=1.5, zorder=0)
            ax.scatter(reported, y, s=20, facecolor="white", edgecolor=TEAL)
            ax.scatter(natural, y, s=20, color=TEAL)
        else:
            ax.scatter(reported, y, s=24, color=BLUE)
    ax.axvline(0.5, color="#87939d", linestyle="--", linewidth=0.7, zorder=0)
    ax.set_yticks(range(len(keys)), [name for _, name in keys], fontsize=8)
    ax.invert_yaxis()
    ax.set(xlim=(0.1, 0.91), xlabel="AUROC  ·  open, reversed co-localization  ·  filled, natural sign")
    label_panel(ax, "C", "seqFISH · contact-rate design")
    save(fig, "figure_s1_observed_pair_accuracy", stamp)


def save(fig, label, stamp):
    FIG.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(FIG / f"{label}_{stamp}.{extension}", dpi=330, facecolor="white")
    plt.close(fig)
