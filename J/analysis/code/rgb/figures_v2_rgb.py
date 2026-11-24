"""Publication figures for the J manuscript (v2 benchmark)."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from common import RESULTS, jx
from datasets import ROOT, load_crc, load_dlpfc, lognorm

FIG = ROOT / "figures" / "v3"
FIG.mkdir(exist_ok=True)

# Reference categorical palette, fixed order (validated: scripts/validate_palette.js)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#F8766D", "#CD9600", "#7CAE00", "#00BE67", "#00BFC4", "#00A9FF", "#C77CFF", "#FF61CC")
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
METHOD_COLOR = {"J": BLUE, "Lee's L": VIOLET, "Moran's I": AQUA, "Pearson r": YELLOW,
                "Gradient co-energy": GREEN, "J (scale 0)": RED}
DIVERGING = LinearSegmentedColormap.from_list("bgr", ["#3A3A98", "#619CFF", "#F7F7F7", "#F8766D", "#832424"])
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#F2F2F2", "#9DCDF9", "#56B1F7", "#2F6E9E", "#132B43"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": BASE,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False, "figure.dpi": 150, "savefig.dpi": 300,
    "pdf.fonttype": 42, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def panel(ax, letter):
    ax.text(-0.14, 1.06, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


def hgrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------- Figure 1
def figure1():
    rng = np.random.default_rng(0)
    xy = np.column_stack(np.unravel_index(np.arange(1600), (40, 40))).astype(float)
    col = xy[:, 1]
    a, b, c = col < 13, (col >= 13) & (col < 27), col >= 27
    noise = lambda: 0.3 * rng.normal(size=1600)  # noqa: E731
    expr = np.column_stack([a + noise(), b + noise(), c + noise(), a + noise()])
    g = jx.spatial_graph(xy, neighbors=4)
    J = jx.juxtaposition_matrix(expr, g, scale=2)
    L = jx.lee_l_matrix(expr, g)
    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(2, 6, height_ratios=[0.85, 1.05], hspace=0.5, wspace=1.4)
    for k, (name, idx) in enumerate([("gene A", 0), ("gene B", 1), ("gene C", 2)]):
        ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2])
        ax.scatter(xy[:, 1], xy[:, 0], c=expr[:, idx], cmap=SEQ, s=3.2, marker="s", linewidths=0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{name} (domain {'ABC'[idx]})", fontweight="normal", fontsize=8)
        if k == 0:
            panel(ax, "A")
    pairs = [("A–B\ntouch", 0, 1), ("A–C\napart", 0, 2), ("A–A′\nsame domain", 0, 3)]
    labels = [p[0] for p in pairs]
    ax = fig.add_subplot(gs[1, 0:2])
    vals = [L[i, j] for _, i, j in pairs]
    ax.bar(range(3), vals, color=ORANGE, width=0.6, edgecolor="white", linewidth=1)
    ax.axhline(0, color=BASE, lw=0.8)
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("Lee's L")
    ax.set_ylim(-0.6, 0.85)
    for k, v in enumerate(vals):
        ax.text(k, v + (0.04 if v >= 0 else -0.04), f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=7, color=INK)
    ax.set_title("Lee's L", fontweight="normal", fontsize=8)
    hgrid(ax)
    panel(ax, "B")
    ax = fig.add_subplot(gs[1, 2:4])
    vals = [J[i, j] for _, i, j in pairs]
    ax.bar(range(3), vals, color=BLUE, width=0.6, edgecolor="white", linewidth=1)
    ax.axhline(0, color=BASE, lw=0.8)
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("J")
    ax.set_ylim(-0.3, 0.3)
    for k, v in enumerate(vals):
        ax.text(k, v + (0.015 if v >= 0 else -0.015), f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=7, color=INK)
    ax.set_title("J", fontweight="normal", fontsize=8)
    hgrid(ax)
    panel(ax, "C")
    # spectral filters on a real Visium section (actual Laplacian spectrum)
    counts, genes, meta = load_dlpfc("151673")
    graph = jx.visium_graph(meta.array_row.values, meta.array_col.values)
    lap, deg = jx.laplacian(graph)
    mu_max = float(np.linalg.eigvalsh(lap.toarray())[-1])
    mu = np.linspace(0, mu_max, 400)
    dmax = deg.max()
    dbar = 6.0
    ax = fig.add_subplot(gs[1, 4:6])
    curves = [("Pearson r", np.ones_like(mu), YELLOW),
              ("Lee's L", (1 - mu / dbar) ** 2, ORANGE),
              ("J (scale 0)", mu, VIOLET),
              ("J (scale 1)", mu * (1 - mu / (2 * dmax)) ** 2, BLUE)]
    for name, h, c in curves:
        ax.plot(mu / mu_max, h / np.max(np.abs(h)), color=c, lw=1.6, label=name)
    ax.axhline(0, color=BASE, lw=0.8)
    ax.set_xlabel("graph frequency (smooth → sharp)")
    ax.set_ylabel("relative weight")
    ax.set_title("Frequency weighting", fontweight="normal", fontsize=8)
    ax.set_ylim(-0.05, 1.42)
    ax.set_yticks([0, 0.5, 1])
    ax.legend(frameon=False, loc="upper center", fontsize=6.2, handlelength=1.2, ncol=2,
              columnspacing=0.8)
    ax.set_xticks([0, 0.5, 1])
    hgrid(ax)
    panel(ax, "D")
    save(fig, "fig1_concept")
    return dict(L_touch=L[0, 1], L_apart=L[0, 2], L_same=L[0, 3], J_touch=J[0, 1], J_apart=J[0, 2],
                J_same=J[0, 3])


# ----------------------------------------------------------------------------- Figure 2
CAL_METHODS = [("pearson", "Pearson r"), ("lee", "Lee's L"), ("moran", "Moran's I"),
               ("J0", "J (scale 0)"), ("J1", "J")]


def fpr_table(df, alpha=0.05):
    rows = []
    for (axis, setting), d in df.groupby(["axis", "setting"]):
        for key, label in CAL_METHODS:
            naive = d["pearson_t"] if key == "pearson" else d[f"{key}_perm"]
            rows.append(dict(axis=axis, setting=setting, method=label, inference="naive",
                             fpr=float((naive < alpha).mean()), n=len(d)))
            rows.append(dict(axis=axis, setting=setting, method=label, inference="spectral",
                             fpr=float((d[f"{key}_spectral"] < alpha).mean()), n=len(d)))
    return pd.DataFrame(rows)


def figure2(cal, diag=None, lib=None):
    tab = fpr_table(cal)
    fig = plt.figure(figsize=(7.4, 5.4))
    gs = fig.add_gridspec(2, 3, hspace=0.75, wspace=0.42, width_ratios=[1, 1, 0.9])
    from matplotlib.patches import Patch
    # A: real cross-donor null, aligned on the array grid (pre-registered A2)
    ax = fig.add_subplot(gs[0, 0:2])
    a2 = tab[tab.axis == "A2"]
    x = np.arange(len(CAL_METHODS))
    for off, inf, alpha_fill in ((-0.19, "naive", 1.0), (0.19, "spectral", 0.45)):
        vals = [a2[(a2.method == lab) & (a2.inference == inf)].fpr.iloc[0] for _, lab in CAL_METHODS]
        cols = [METHOD_COLOR[lab] for _, lab in CAL_METHODS]
        ax.bar(x + off, vals, width=0.36, color=cols, alpha=alpha_fill, edgecolor="white", linewidth=1.5,
               hatch=None if inf == "naive" else "////")
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.015, f"{100 * v:.0f}%" if v >= 0.095 else f"{100 * v:.1f}%",
                    ha="center", va="bottom", fontsize=6.3, color=INK)
    ax.axhspan(0.025, 0.075, color="#dfeaf8", zorder=0, lw=0)
    ax.axhline(0.05, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(x, [lab for _, lab in CAL_METHODS])
    ax.set_ylabel("false-positive rate (α = 0.05)")
    ax.set_ylim(0, 1.0)
    ax.legend(handles=[Patch(facecolor=MUTED, label="usual test (t-test or 999 permutations)"),
                       Patch(facecolor=MUTED, alpha=0.45, hatch="////", label="spectral sign-flip null")],
              frameon=False, loc="center right", fontsize=6.5)
    ax.set_title("Genes from different donors, same array spots (15,000 pairs)", fontweight="normal",
                 fontsize=8)
    hgrid(ax)
    panel(ax, "A")
    # D: library-size diagnostic
    ax = fig.add_subplot(gs[0, 2])
    if lib is not None:
        for k, mode in enumerate(("aligned", "rotated")):
            v = lib[lib["mode"] == mode].libsize_J1_z.to_numpy()
            ax.scatter(np.full(len(v), k) + np.linspace(-0.12, 0.12, len(v)), v, s=14,
                       color=BLUE if mode == "aligned" else MUTED, edgecolor="white", lw=0.5, zorder=3)
        ax.axhspan(-1.96, 1.96, color="#eeeeea", lw=0, zorder=0)
        ax.axhline(0, color=BASE, lw=0.8)
        ax.set_xticks([0, 1], ["same array\nspots", "rotated\n180°"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylabel("z, J between the two\nsections' library sizes")
        ax.set_title("Shared spot capture", fontweight="normal", fontsize=8)
        hgrid(ax)
    panel(ax, "B")
    # C, D: simulated fields
    order = ["white", "matern0.5_range1", "matern0.5_range3", "matern0.5_range10",
             "matern2.5_range1", "matern2.5_range3", "matern2.5_range10"]
    ticks = ["white", "1", "3", "10", "1", "3", "10"]
    a1 = tab[tab.axis == "A1"]
    for k, inf in enumerate(("naive", "spectral")):
        ax = fig.add_subplot(gs[1, k])
        for _, lab in CAL_METHODS:
            v = [a1[(a1.method == lab) & (a1.inference == inf) & (a1.setting == st)].fpr.iloc[0] for st in order]
            ax.plot(range(len(order)), v, marker="o", ms=3.2, lw=1.3, color=METHOD_COLOR[lab], label=lab)
        ax.axhspan(0.025, 0.075, color="#dfeaf8", zorder=0, lw=0)
        ax.axhline(0.05, color=INK2, lw=0.8, ls="--")
        ax.set_xticks(range(len(order)), ticks, fontsize=6.5)
        ax.set_xlabel("field range (spot spacings)")
        ytop = 0.97 if inf == "naive" else 0.29
        ax.axvline(3.5, color=GRID, lw=0.8)
        ax.text(2, ytop, "ν = ½", ha="center", va="top", fontsize=6.5, color=INK2)
        ax.text(5, ytop, "ν = 5/2", ha="center", va="top", fontsize=6.5, color=INK2)
        ax.set_ylabel("false-positive rate")
        ax.set_ylim(0, 1.0 if inf == "naive" else 0.3)
        ax.set_title("Simulated fields: usual tests" if inf == "naive" else "Simulated fields: spectral null",
                     fontweight="normal", fontsize=8)
        hgrid(ax)
        if k == 0:
            ax.legend(frameon=False, fontsize=5.8, loc="upper left", bbox_to_anchor=(0.0, 0.92))
        panel(ax, "CD"[k])
    # E: rotated real-data null (spectral)
    ax = fig.add_subplot(gs[1, 2])
    if diag is not None:
        rot = diag[diag["mode"] == "rotated"]
        keys = [("pearson", "Pearson r"), ("lee", "Lee's L"), ("moran", "Moran's I"), ("J0", "J (scale 0)"),
                ("J1", "J")]
        vals = [float((rot[f"{k}_spectral"] < 0.05).mean()) for k, _ in keys]
        ax.bar(range(len(keys)), vals, color=[METHOD_COLOR[l] for _, l in keys], alpha=0.45, hatch="////",
               edgecolor="white", linewidth=1.2)
        for k, v in enumerate(vals):
            ax.text(k, v + 0.004, f"{100 * v:.1f}%", ha="center", va="bottom", fontsize=6.0, color=INK)
        ax.axhspan(0.025, 0.075, color="#dfeaf8", zorder=0, lw=0)
        ax.axhline(0.05, color=INK2, lw=0.8, ls="--")
        ax.set_xticks(range(len(keys)), ["r", "L", "I", "J₀", "J₁"])
        ax.set_ylim(0, 0.3)
        ax.set_ylabel("false-positive rate")
        ax.set_title("Real genes, rotated 180°:\nspectral null", fontweight="normal", fontsize=8)
        hgrid(ax)
    panel(ax, "E")
    save(fig, "fig2_calibration")
    return tab


# ----------------------------------------------------------------------------- Figure 3
B1_METHODS = [("J1", "J"), ("gradient_coenergy1", "Gradient co-energy"), ("neg_lee", "Lee's L"),
              ("neg_moran", "Moran's I"), ("neg_pearson", "Pearson r")]


def strip(ax, data, methods, col, rng):
    for k, (key, lab) in enumerate(methods):
        v = data[data.method == key][col].to_numpy()
        ax.scatter(k + rng.uniform(-0.12, 0.12, len(v)), v, s=11, color=METHOD_COLOR[lab], alpha=0.9,
                   edgecolor="white", linewidth=0.5, zorder=3)
        ax.plot([k - 0.25, k + 0.25], [np.mean(v)] * 2, color=INK, lw=1.2, zorder=4)
    ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(range(len(methods)), [lab.replace(" co-energy", "\nco-energy") for _, lab in methods])
    ax.set_ylim(0, 1)
    hgrid(ax)


def figure3(b1, b1pairs, b2, c):
    rng = np.random.default_rng(1)
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.32)
    k10 = b1[b1.K == 10]
    ax = fig.add_subplot(gs[0, 0])
    strip(ax, k10, B1_METHODS, "auc_adjacent_vs_distant", rng)
    ax.set_ylabel("AUROC: adjacent vs distant layers")
    ax.set_title("DLPFC, 12 sections (pre-registered primary)", fontweight="normal", fontsize=8)
    panel(ax, "A")
    ax = fig.add_subplot(gs[0, 1])
    order = ["same", "adjacent", "distant"]
    for k, (key, lab, colr) in enumerate([("pearson_r", "Pearson r", YELLOW), ("lee_L", "Lee's L", ORANGE),
                                          ("J1", "J", BLUE)]):
        per_sec = b1pairs.groupby(["section", "pair_class"])[key].mean().unstack()[order]
        m = per_sec.mean()
        ax.plot(range(3), m.values, marker="o", ms=4, lw=1.6, color=colr, label=lab)
        for s in per_sec.index:
            ax.plot(range(3), per_sec.loc[s].values, color=colr, lw=0.4, alpha=0.35)
    ax.axhline(0, color=BASE, lw=0.8)
    ax.set_xticks(range(3), ["same layer", "adjacent\nlayers", "distant\nlayers"])
    ax.set_ylabel("mean value over marker pairs")
    ax.set_title("Adjacent-layer markers overlap (r > 0)", fontweight="normal", fontsize=8)
    ax.legend(frameon=False, fontsize=6.5)
    hgrid(ax)
    panel(ax, "B")
    ax = fig.add_subplot(gs[1, 0])
    for key, lab in [("J1", "J"), ("gradient_coenergy1", "Gradient co-energy"), ("neg_lee", "Lee's L"),
                     ("neg_pearson", "Pearson r")]:
        d = b2[b2.method == key].groupby(["beta", "section"]).auc_adjacent_vs_distant.mean().reset_index()
        m = d.groupby("beta").auc_adjacent_vs_distant.agg(["mean", "min", "max"])
        ax.plot(m.index, m["mean"], marker="o", ms=4, lw=1.6, color=METHOD_COLOR[lab], label=lab)
        ax.fill_between(m.index, m["min"], m["max"], color=METHOD_COLOR[lab], alpha=0.15, lw=0)
    ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
    ax.set_xticks([0.5, 1, 2])
    ax.set_xlabel("layer effect β (log fold change; exclusive markers)")
    ax.set_ylabel("AUROC: adjacent vs distant")
    ax.set_ylim(0, 1)
    ax.set_title("Semi-synthetic, real layer maps (known truth)", fontweight="normal", fontsize=8)
    ax.legend(frameon=False, fontsize=6.3, loc="upper left")
    hgrid(ax)
    panel(ax, "C")
    ax = fig.add_subplot(gs[1, 1])
    meth = [("J1", "J"), ("gradient_coenergy1", "Gradient co-energy"), ("neg_lee", "Lee's L"),
            ("neg_pearson", "Pearson r")]
    w = 0.2
    for k, (key, lab) in enumerate(meth):
        d = c[c.method == key].set_index("embryo")
        for e_i, emb in enumerate(sorted(c.embryo.unique())):
            ax.bar(e_i + (k - 1.5) * w, d.loc[emb, "auc_contact_vs_apart"], width=w * 0.92,
                   color=METHOD_COLOR[lab], edgecolor="white", linewidth=0.8, label=lab if e_i == 0 else None)
    ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(range(3), ["embryo 1", "embryo 2", "embryo 3"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUROC: touching vs separated\ncell types")
    ax.set_title("seqFISH mouse embryo, single cells", fontweight="normal", fontsize=8)
    ax.legend(frameon=False, fontsize=6.3, ncol=2, loc="upper left")
    hgrid(ax)
    panel(ax, "D")
    save(fig, "fig3_accuracy")


# ----------------------------------------------------------------------------- Figure 4
def figure4(crc):
    counts, genes, meta = load_crc()
    lib = np.asarray(counts.sum(axis=1)).ravel()
    counts, meta = counts[lib > 0], meta[lib > 0]
    graph = jx.visium_graph(meta.array_row.values, meta.array_col.values)
    xn = lognorm(counts)
    gi = {g: i for i, g in enumerate(genes)}
    px = meta.array_col.values / 2.0
    py = -meta.array_row.values * np.sqrt(3) / 2
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.55)
    ax = fig.add_subplot(gs[0, 0:2])
    hb = ax.hexbin(crc.lee_L, crc.J1, gridsize=60, cmap=SEQ, bins="log", mincnt=1, linewidths=0)
    ax.axhline(0, color=BASE, lw=0.8)
    ax.axvline(0, color=BASE, lw=0.8)
    ax.set_xlabel("Lee's L")
    ax.set_ylabel("J (scale 1)")
    ax.set_title("44,850 pairs of the 300 most autocorrelated genes", fontweight="normal", fontsize=8)
    cax = ax.inset_axes([0.80, 0.70, 0.025, 0.25])
    cb = fig.colorbar(hb, cax=cax)
    cb.set_label("pairs", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    for gx, gy, lab_dx, lab_dy in [("MGP", "EPCAM", 0.05, 0.03), ("COL1A1", "COL1A2", -0.55, -0.02),
                                   ("TAGLN", "MYL9", -0.5, 0.06)]:
        r = crc[((crc.gene_x == gx) & (crc.gene_y == gy)) | ((crc.gene_x == gy) & (crc.gene_y == gx))].iloc[0]
        ax.scatter(r.lee_L, r.J1, s=22, facecolor="none", edgecolor=RED, lw=1.2)
        ax.annotate(f"{gx}–{gy}", (r.lee_L, r.J1), (r.lee_L + lab_dx, r.J1 + lab_dy), fontsize=6.5,
                    color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
    panel(ax, "A")
    ax = fig.add_subplot(gs[0, 2])
    # v15: pairs called at q < 0.05 under the spectral sign-flip null, normal approximation and Monte Carlo
    # (benchmarks/v3/mc_check: crc_mc.py for J and Pearson's r, crc_lee_mc.py for Lee's L)
    mcd = RESULTS / "mc_check"
    mc = pd.concat([pd.read_csv(mcd / "crc_mc_summary.csv"), pd.read_csv(mcd / "crc_lee_mc_summary.csv")]).set_index("stat")
    rows = [("J1", "J", "#F8766D"), ("pearson", "Pearson", "#00BE67"), ("lee", "Lee", "#C77CFF")]
    for i, (key, lab, col) in enumerate(rows):
        for j, (meth, alpha) in enumerate((("norm_q05", 0.35), ("mc_q05", 1.0))):
            v = float(mc.loc[key, meth])
            y = i + (j - 0.5) * 0.38
            ax.barh(y, v, height=0.36, color=col, alpha=alpha, linewidth=0)
            ax.text(v + 2, y, f"{v:.1f}", va="center", fontsize=6.5, color=INK if j else INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab for _, lab, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Pairs called, q < 0.05 (%)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=MUTED, alpha=0.35, label="Normal approx."),
                       Patch(facecolor=MUTED, label="Monte Carlo")], fontsize=5.5, frameon=False,
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.0, columnspacing=0.8)
    ax.set_title("Spectral sign-flip test", fontweight="normal", fontsize=8, pad=14)
    panel(ax, "B")
    for k, gname in enumerate(["EPCAM", "MGP"]):
        ax = fig.add_subplot(gs[1, k])
        v = np.asarray(xn[:, gi[gname]].todense()).ravel()
        ax.scatter(px, py, c=v, cmap=SEQ, s=1.6, linewidths=0, vmax=np.percentile(v, 98), rasterized=True)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{gname} (log-normalized)", fontweight="normal", fontsize=8)
        panel(ax, "CD"[k])
    ax = fig.add_subplot(gs[1, 2])
    pair = np.column_stack([np.asarray(xn[:, gi["EPCAM"]].todense()).ravel(),
                            np.asarray(xn[:, gi["MGP"]].todense()).ravel()])
    e = jx.score_edges(pair, graph, scale=1, ridge=1.0)
    node = np.zeros(len(px))
    cnt = np.zeros(len(px))
    np.add.at(node, e.endpoints[:, 0], np.nan_to_num(e.j))
    np.add.at(node, e.endpoints[:, 1], np.nan_to_num(e.j))
    np.add.at(cnt, e.endpoints[:, 0], 1)
    np.add.at(cnt, e.endpoints[:, 1], 1)
    node = np.divide(node, cnt, out=np.zeros_like(node), where=cnt > 0)
    vmax = np.percentile(np.abs(node), 98)
    sc = ax.scatter(px, py, c=node, cmap=DIVERGING, norm=TwoSlopeNorm(0, -vmax, vmax), s=1.6, linewidths=0,
                    rasterized=True)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("local J, EPCAM–MGP", fontweight="normal", fontsize=8)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, ticks=[-vmax, 0, vmax])
    cb.ax.set_yticklabels(["concordant", "0", "opposing"], fontsize=6)
    panel(ax, "E")
    save(fig, "fig4_crc")


# ----------------------------------------------------------------------------- Figure 5
def figure5(sp, perm=None):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axes[0]
    for task, lab, colr in [("J1_all_pairs", "J (scale 1)", BLUE), ("J0_all_pairs", "J (scale 0)", VIOLET),
                            ("LeeL_all_pairs", "Lee's L", ORANGE), ("MoranI_all_pairs", "Moran's I", AQUA),
                            ("Pearson_all_pairs", "Pearson r", YELLOW)]:
        d = sp[sp.task == task]
        ax.plot(d.genes, d.seconds, marker="o", ms=3.5, lw=1.5, color=colr, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("genes (all pairs)")
    ax.set_ylabel("seconds (1 thread)")
    ax.set_title("All-pairs statistics, 3,611 spots", fontweight="normal", fontsize=8)
    ax.legend(frameon=False, fontsize=6.3)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    panel(ax, "A")
    ax = axes[1]
    d = sp[sp.task == "J1_spectral_pvalues_all_pairs"]
    eig = float(sp[sp.task == "spectral_eigendecomposition_once"].seconds.iloc[0])
    ax.plot(d.genes, d.seconds + eig, marker="o", ms=3.5, lw=1.5, color=BLUE,
            label="J, spectral null (incl. eigendecomposition)")
    if perm is not None:
        grid = np.array([100, 250, 500, 1000, 2000, 4000])
        for task, lab, colr in [("J1_permutation999_all_pairs", "J, 999 permutations", VIOLET),
                                ("LeeL_permutation999_all_pairs", "Lee's L, 999 permutations", ORANGE)]:
            m = perm[perm.task == task].sort_values("genes")
            ax.plot(m.genes, m.seconds, marker="s", ms=3.5, lw=1.5, color=colr, label=lab)
            last = m.iloc[-1]
            ext = last.seconds * (grid[grid >= last.genes] / last.genes) ** 2
            ax.plot(grid[grid >= last.genes], ext, ls="--", lw=1.1, color=colr)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("genes (all pairs)")
    ax.set_ylabel("seconds (1 thread)")
    ax.set_title("Calibrated vs permutation p-values", fontweight="normal", fontsize=8)
    ax.legend(frameon=False, fontsize=6.0, loc="upper left")
    ax.grid(True, which="major", color=GRID, lw=0.6)
    panel(ax, "B")
    fig.tight_layout()
    save(fig, "fig5_speed")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    if "1" in which:
        print(figure1())
    if "2" in which:
        cal = pd.read_csv(RESULTS / "calibration_all.csv.gz", low_memory=False)
        diag = pd.read_csv(RESULTS / "diagnostic_array_null.csv.gz", low_memory=False)
        lib = pd.read_csv(RESULTS / "diagnostic_libsize.csv")
        figure2(cal, diag, lib).to_csv(RESULTS / "calibration_fpr_table.csv", index=False)
    if "3" in which:
        figure3(pd.read_csv(RESULTS / "dlpfc_b1_auroc.csv"), pd.read_csv(RESULTS / "dlpfc_b1_pairs_K10.csv.gz"),
                pd.read_csv(RESULTS / "dlpfc_b2_semisynthetic.csv"), pd.read_csv(RESULTS / "seqfish_c_auroc.csv"))
    if "4" in which:
        figure4(pd.read_csv(RESULTS / "crc_pairs_top300.csv.gz"))
    if "5" in which:
        perm = RESULTS / "speed_permutation.csv"
        figure5(pd.read_csv(RESULTS / "speed.csv"), pd.read_csv(perm) if perm.exists() else None)
