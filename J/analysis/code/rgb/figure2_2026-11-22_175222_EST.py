"""Replot the original calibration tests with empirical-control and simulated-null labels."""
import argparse
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("project", type=Path, help="J-analysis directory")
parser.add_argument("output", type=Path)
args=parser.parse_args()
sys.path.insert(0,str(args.project.resolve()/"benchmarks/v3"))
from figures_v3 import DEFAULT_TESTS, GRID, INK, INK2, MUTED, PKG, RESULTS, SPECTRAL, hbar, hgrid, panel, wilson

def figure2(output):
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
         [PKG[p] for p in t.package], err=(t.lo.values, t.hi.values), xlabel="control pairs with unadjusted p < 0.05 (%)",
         xmax=100)
    ax.axvspan(2.5, 7.5, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.axvline(5, color=INK2, lw=0.8, ls="--", zorder=0.5)


    ax.set_title("Rotated cross-donor control · 15,000 gene-pair observations", fontweight="normal", fontsize=8)
    panel(ax, "A")
    # B: the same statistics under the spectral null
    ax = fig.add_subplot(gs[1, 0])
    d = cal[(cal.axis == "A2") & (cal.setting == "rotated")]
    vals = [100 * (d[c] < 0.05).mean() for c, _, _ in SPECTRAL]
    ci = [wilson(int((d[c] < 0.05).sum()), len(d)) for c, _, _ in SPECTRAL]
    hbar(ax, [l for _, l, _ in SPECTRAL], vals, [PKG[p] for _, _, p in SPECTRAL],
         err=(100 * np.array([c[0] for c in ci]), 100 * np.array([c[1] for c in ci])),
         xlabel="control pairs with p < 0.05 (%)", xmax=10, fmt="{:.1f}")
    ax.axvspan(2.5, 7.5, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.axvline(5, color=INK2, lw=0.8, ls="--", zorder=0.5)
    ax.set_title("Rotated control, spectral test", fontweight="normal", fontsize=8)
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
    ax.set_ylabel("control rejections (%)")
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_title("Time and rotated-control rejections", fontweight="normal", fontsize=8)
    panel(ax, "E")
    for ext in ("png", "pdf"):
        fig.savefig(output / f"figure2_calibration_2026-11-22_133437_EST.{ext}", bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)
    return tab

figure2(args.output)
