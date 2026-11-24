"""Rebuild Figure 9 from held-out marker pairs and the original specimen graphs."""
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve

NAME = "figure9_boundary_method_auroc_2026-11-22_142751_EST"
REFERENCE = "metrics_by_unit_2026-11-22_104504_EST.csv"
RUST, TEAL, SLATE = "#F8766D", "#00BA38", "#619CFF"
GRID, MUTED = "#E6EAED", "#758490"
# Score directions are the same in both cohorts and all specimens.
METHODS = [
    ("J1", "Boundary J (1 step)", 1), ("J4", "Boundary J (4 steps)", 1),
    ("neg_codispersion1", "Neg. codispersion (1 step)", 1),
    ("neg_codispersion4", "Neg. codispersion (4 steps)", 1),
    ("pearson", "Pearson", -1), ("smooth_pearson_s1", "Pearson (1 step)", -1),
    ("smooth_pearson_s4", "Pearson (4 steps)", -1),
    ("smooth_pearson_s16", "Pearson (16 steps)", -1),
    ("spearman", "Spearman", -1), ("J_region", "Regional J", -1),
    ("giotto_spatial_cor", "Giotto spatial correlation", -1),
    ("hotspot_Z", "Hotspot Z", -1), ("lee_l", "Lee’s L", -1),
    ("liana_cosine_mean", "LIANA+ cosine", -1), ("liana_morans_R", "LIANA+ Moran", -1),
    ("meringue_moran", "MERINGUE bivariate Moran", -1), ("spatialdm_R", "SpatialDM R", -1),
]
COHORTS = {"B1": "Human DLPFC", "C1": "seqFISH embryos"}


def mean_curve(curves):
    """Average TPR including both ends of the vertical jumps of each specimen ROC."""
    xs = np.unique(np.concatenate([x for x, _ in curves]))
    lower, upper = [], []
    for x, y in curves:
        knots, first = np.unique(x, return_index=True)
        last = np.r_[first[1:] - 1, len(x) - 1]
        before, after = y[first], y[last]
        loc = np.searchsorted(knots, xs)
        between = knots[loc] != xs
        lo, hi = before[loc].copy(), after[loc].copy()
        k = loc[between]
        w = (xs[between] - knots[k - 1]) / (knots[k] - knots[k - 1])
        lo[between] = hi[between] = after[k - 1] + w * (before[k] - after[k - 1])
        lower.append(lo)
        upper.append(hi)
    return np.repeat(xs, 2), np.column_stack([np.mean(lower, axis=0), np.mean(upper, axis=0)]).ravel()


def analyze(project, results):
    sys.path.insert(0, str(project / "benchmarks/v3"))
    from common import jx
    from datasets import load_dlpfc, load_seqfish, lognorm
    reference = pd.read_csv(results / REFERENCE, dtype={"unit": str}).query("task == 'touch_same'")
    reference = reference.set_index(["design", "unit", "method"])
    assert reference.index.is_unique
    seq, seq_genes, seq_meta = load_seqfish()
    records, curves = [], {d: [] for d in COHORTS}
    for design in COHORTS:
        for path in sorted(results.glob(f"{design}_*_pairs_2026-11-22_104504_EST.csv.gz")):
            pairs = pd.read_csv(path, dtype={"unit": str})
            pairs = pairs[pairs.pair_class.isin(["touch", "same"])].copy()
            assert not pairs[["gene_x", "gene_y"]].duplicated().any()
            assert pairs.design.eq(design).all() and pairs.unit.nunique() == 1
            unit = pairs.unit.iloc[0]
            touch = pairs.pair_class.eq("touch").to_numpy()
            if design == "B1":
                np.testing.assert_array_equal(pairs.layer_distance, touch.astype(int))
                counts, genes, meta = load_dlpfc(unit)
                expr = lognorm(counts)
                graph = jx.visium_graph(meta.array_row, meta.array_col)
            else:
                assert (pairs.loc[touch, "contact_z"] >= 3).all()
                assert (pairs.loc[touch, "type_x"].values != pairs.loc[touch, "type_y"].values).all()
                np.testing.assert_array_equal(pairs.loc[~touch, "type_x"], pairs.loc[~touch, "type_y"])
                mask = seq_meta.embryo.eq(unit).to_numpy()
                genes, expr = seq_genes, seq[mask]
                xy = seq_meta.loc[mask, ["x_global_affine", "y_global_affine"]].to_numpy()
                graph = jx.spatial_graph(xy, neighbors=6)
            names = np.union1d(pairs.gene_x, pairs.gene_y)
            columns = np.flatnonzero(np.isin(genes, names))
            lookup = pd.Index(genes[columns])
            assert len(lookup) == len(names) and lookup.is_unique
            x = expr[:, columns].toarray().astype(float)
            i, j = lookup.get_indexer(pairs.gene_x), lookup.get_indexer(pairs.gene_y)
            assert i.min() >= 0 and j.min() >= 0
            # Validate both the archived baseline and the four-step graph preprocessing.
            np.testing.assert_allclose(jx.codispersion_matrix(x, graph, scale=1)[i, j],
                                       pairs.neg_codispersion1, rtol=0, atol=1e-10)
            np.testing.assert_allclose(jx.juxtaposition_matrix(x, graph, scale=4)[i, j],
                                       pairs.J4, rtol=0, atol=1e-10)
            pairs["neg_codispersion4"] = jx.codispersion_matrix(x, graph, scale=4)[i, j]
            n_touch, n_same = int(touch.sum()), int((~touch).sum())
            for key, label, direction in METHODS:
                raw = pairs[key].to_numpy()
                assert np.isfinite(raw).all()
                natural = roc_auc_score(touch, raw)
                oriented = roc_auc_score(touch, direction * raw)
                np.testing.assert_allclose(oriented, natural if direction == 1 else 1-natural, atol=1e-12)
                if key != "neg_codispersion4":
                    target = reference.loc[(design, unit, key)]
                    np.testing.assert_allclose(natural, target.auroc, rtol=0, atol=1e-12)
                    assert (n_touch, n_same) == (target.n_pos, target.n_neg)
                records.append(dict(design=design, cohort=COHORTS[design], unit=unit,
                                    donor_or_embryo=pairs.tissue.iloc[0], n_touch=n_touch, n_same=n_same,
                                    method=key, label=label, touching_ranked_first="higher" if direction==1 else "lower",
                                    auroc_higher_first=natural, auroc_as_plotted=oriented))
            curves[design].append(roc_curve(touch, pairs.J1, drop_intermediate=False)[:2])
    for design, expected in [("B1", (12, 6400, 3420)), ("C1", (3, 450, 149))]:
        units = [r for r in records if r["method"] == "J1" and r["design"] == design]
        assert (len(units), sum(r["n_touch"] for r in units), sum(r["n_same"] for r in units)) == expected
        np.testing.assert_allclose(auc(*mean_curve(curves[design])),
                                   np.mean([r["auroc_as_plotted"] for r in units]), rtol=0, atol=1e-12)
    return pd.DataFrame(records), curves


def draw(table, curves, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.8, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(8.15, 7.65))
    for col, (design, title) in enumerate(COHORTS.items()):
        left = (.125, .60)[col]
        top = fig.add_axes([left, .715, .315, .222])
        top.set_title(f"{'AB'[col]}   {title}", loc="left", fontsize=10.4, weight="bold", pad=7)
        top.plot([0, 1], [0, 1], ls=(0, (3, 3)), color=MUTED, lw=.8)
        for x, y in curves[design]:
            top.plot(x, y, color="#A5B0B9", lw=.7, alpha=.88)
        top.plot(*mean_curve(curves[design]), color=RUST, lw=2.0)
        top.set(xlim=(0, 1), ylim=(0, 1.008), xticks=[0, .25, .5, .75, 1],
                yticks=[0, .25, .5, .75, 1], xlabel="False-positive rate")
        if col == 0:
            top.set_ylabel("True-positive rate", labelpad=3)
        top.tick_params(length=2.6, color=MUTED, labelsize=8)
        top.spines[["bottom", "left"]].set_color("#B3BEC5")
        top.set_axisbelow(True)
        top.grid(color=GRID, lw=.6)

        bottom = fig.add_axes([(.250, .658)[col], .102, .267, .446])
        bottom.set_title(f"{'CD'[col]}   {title}", loc="left", fontsize=10.4, weight="bold", pad=8)
        subset = table[table.design.eq(design)]
        for row, (key, label, _direction) in enumerate(METHODS):
            a = subset[subset.method.eq(key)].sort_values("unit").auroc_as_plotted.to_numpy()
            color = RUST if key in {"J1", "J4"} else TEAL if key == "J_region" else SLATE
            assert np.all((a >= .475) & (a <= .93)), (label, a.min(), a.max())
            bottom.scatter(a, row + np.linspace(-.22, .22, len(a)), facecolor="white",
                           edgecolor=color, s=13, linewidth=.68, alpha=.85, zorder=3)
            bottom.scatter(a.mean(), row, color=color, marker="D", s=32,
                           edgecolor="white", linewidth=.45, zorder=4)
            bottom.text(1.03, row, f"{a.mean():.3f}", transform=bottom.get_yaxis_transform(),
                        ha="left", va="center", fontsize=8.0, color=color)
        bottom.set(xlim=(.475, .93), xticks=[.5, .6, .7, .8, .9], ylim=(len(METHODS)-.45, -.64),
                   yticks=range(len(METHODS)), xlabel="AUROC")
        bottom.set_yticklabels([f"{label}  {'↑' if direction == 1 else '↓'}" for _, label, direction in METHODS],
                              fontsize=8.5)
        bottom.tick_params(axis="y", length=0, pad=4.5, labelleft=col==0)
        bottom.tick_params(axis="x", length=2.5, color=MUTED, labelsize=8)
        bottom.text(1.03, 1.027, "Mean", transform=bottom.transAxes, color=SLATE, fontsize=7.8)
        bottom.axvline(.5, color=MUTED, lw=.8, ls=(0, (3, 3)))
        bottom.set_axisbelow(True)
        bottom.grid(axis="y", color=GRID, lw=.68)
        bottom.spines["left"].set_visible(False)
        bottom.spines["bottom"].set_color("#B3BEC5")
    def line(color, width, label, style="-"):
        return Line2D([], [], color=color, lw=width, ls=style, label=label)
    fig.legend(handles=[line("#A5B0B9", 1, "Section or embryo"), line(RUST, 2, "Mean ROC, boundary J (1 step)"),
                        line(MUTED, .8, "Chance", "--")], loc="center", bbox_to_anchor=(.51, .615),
               ncol=3, frameon=False, columnspacing=2.1, fontsize=8.5)
    dots = [Line2D([], [], marker="o", ls="", mfc="white", mec=SLATE, ms=4.3, label="Section or embryo"),
            Line2D([], [], marker="D", ls="", mfc=SLATE, mec="white", mew=.4, ms=5.6, label="Mean")]
    fig.legend(handles=dots, loc="center", bbox_to_anchor=(.23, .022), ncol=2, frameon=False,
               columnspacing=1.3, handletextpad=.3, fontsize=8.4)
    fig.text(.53, .022, "↑ Higher score first     ↓ Lower score first", va="center",
             color=SLATE, fontsize=8.4)
    output.mkdir(exist_ok=True, parents=True)
    for ext in ["png", "pdf"]:
        fig.savefig(output / f"{NAME}.{ext}", dpi=360, facecolor="white",
                    bbox_inches="tight", pad_inches=.085)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path, help="J-analysis repository")
    parser.add_argument("output", type=Path, help="Figure directory")
    parser.add_argument("--source-output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    source = project / "revisions/v5_2026-11-22_104348_EST/results"
    table, curves = analyze(project, source)
    draw(table, curves, args.output)
    args.source_output.mkdir(exist_ok=True, parents=True)
    table.to_csv(args.source_output / f"{NAME}_source.csv", index=False)
    print(table.pivot_table(index="label", columns="design", values="auroc_as_plotted").round(5))
