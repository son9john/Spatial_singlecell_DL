"""Plot overlap at matched budgets for the manuscript supplement."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "package_comparison"
FIGURES = ROOT / "figures" / "v2"
EDGE_METHODS = {
    "j_regularized": ("Regularized J", "#F8766D", "o", "-"),
    "unsigned_gradient": ("Gradient-difference magnitude", "#7F7F7F", "s", "-"),
    "liana_pearson": ("LIANA+ weighted Pearson (negative)", "#7CAE00", "^", "--"),
    "spatialdm_morans_positive": ("SpatialDM local Moran (positive)", "#CD9600", "P", "--"),
    "liana_jaccard_positive": ("LIANA+ weighted Jaccard (positive)", "#00BFC4", "X", ":"),
    "opposite_gradient_iou": ("Opposite-gradient weighted Jaccard", "#00A9FF", "D", "-."),
}


def plot():
    sim = pd.read_csv(TABLES / "merfish_edge_iou.csv")
    tissue = pd.read_csv(TABLES / "seqfish_edge_iou.csv")
    pair = pd.read_csv(TABLES / "global_pair_iou.csv")
    with plt.rc_context({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 8, "axes.labelsize": 8, "font.family": "DejaVu Sans"}):
        fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.4))
        fig.subplots_adjust(top=.88, bottom=.38, right=.985, left=.075, wspace=.42)
        panels = (
            (sim, "exact_iou", "A  Exact · simulated MERFISH"),
            (tissue, "exact_iou", "B  Exact · seqFISH"),
            (tissue, "one_hop_iou", "C  One-hop · seqFISH"),
        )
        for ax, (data, metric, title) in zip(axes.flat, panels):
            for key, (label, color, marker, style) in EDGE_METHODS.items():
                values = data[data.method.eq(key)].groupby("budget")[metric].median()
                ax.plot(100 * values.index, values, color=color, marker=marker, linestyle=style,
                        ms=4, lw=1.25, label=label)
            ax.set_xscale("log")
            ax.set_xticks([.1, .25, .5, 1], ["0.10", "0.25", "0.50", "1.00"])
            ax.set(xlabel="Top-ranked edges (%)", ylabel="Median IoU", ylim=(0, None))
            ax.set_title(title, loc="left", weight="bold")
            ax.grid(axis="y", alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=2, loc="lower center", frameon=False,
                   handlelength=2.8, columnspacing=1.4, fontsize=7)
        FIGURES.mkdir(exist_ok=True)
        fig.savefig(FIGURES / "figs1_jaccard_overlap.png", dpi=360, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    plot()
