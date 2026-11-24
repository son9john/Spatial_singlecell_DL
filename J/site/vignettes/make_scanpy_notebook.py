"""Writes scanpy_juxtaposition.ipynb (unexecuted) from the cells below. Run: python make_scanpy_notebook.py"""
import json
from pathlib import Path

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md("""
# Juxtaposition analysis of spatial transcriptomics with Scanpy

*A complete workflow with the `juxtaposed` Python package: interfaces with boundary J, co-localization with regional J.*

Junho Song, Jihyun Kim, Jaesung Yoo, Dongjoo Lee, Kwangsoo Kim

This notebook starts from a standard Scanpy analysis and adds juxtaposition analysis on top of it. It uses the same 10x Visium
section (mouse brain, sagittal anterior) as the companion Seurat vignette, so the two can be compared line by line.

The package provides two scores, and they answer **different questions**:

| Question | Score | Function | Reads |
|---|---|---|---|
| Do two programs meet at a tissue interface and change in *opposite* directions across it? | **Boundary J** | `jx.juxtaposition()` | J > 0 opposing changes (an interface); J < 0 concordant; J ≈ 0 no shared boundary |
| *Where* is that interface? | **Edge map** | `jx.score_edges()` → `.to_nodes()` | High signed evidence marks the interface edges |
| Are two genes expressed in the *same* neighbourhoods (co-expression, co-localization)? | **Regional J** | `jx.regional_juxtaposition_matrix()` | > 0 shared region; < 0 complementary regions |

Boundary J is not a co-expression statistic: two genes expressed in the same region rise and fall together across that
region's edge, so boundary J is negative for them. For co-localization, use regional J.

**Why boundary J for interfaces?** Lee's L and bivariate Moran's I compare smoothed *levels*, so two complementary domains that
touch and two that sit far apart can score alike. Boundary J compares *changes across the same edges*, and is positive only
when the domains meet. Its p-values come from an exact spectral sign-flip null on the tissue graph, with no permutations.
""")

md("""
## 1. Setup

```bash
pip install "scanpy>=1.10,<1.12" leidenalg igraph matplotlib
pip install "git+https://github.com/son9john/Spatial_singlecell_DL@juxtaposition-index#subdirectory=J/python"
```

`juxtaposed` needs only NumPy and SciPy and works on any AnnData object. The loading and plotting calls below are those of
Scanpy 1.10–1.11; Scanpy 1.11 deprecates `read_visium` and its spatial plots in favour of Squidpy, so on later versions
load with `squidpy.read.visium()` and plot with `squidpy.pl.spatial_scatter()`. The juxtaposition steps do not change.
""")

code("""
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import norm, spearmanr

import juxtaposed as jx

# Default R (ggplot2) colours, matching the R vignette
OPPOSING, REGIONAL, CONCORDANT, NS = "#F8766D", "#00BA38", "#619CFF", "#7F7F7F"
DIV = LinearSegmentedColormap.from_list("j_div", ["#3A3A98", "#619CFF", "#F7F7F7", "#F8766D", "#832424"])
EXPR = LinearSegmentedColormap.from_list("j_expr", ["#F2F2F2", "#56B1F7", "#132B43"])
sc.settings.set_figure_params(dpi=90, frameon=False)
rng = np.random.default_rng(2021)
print("juxtaposed", jx.__version__, "| scanpy", sc.__version__)
""")

md("""
## 2. A standard Scanpy analysis

These steps are kept short; any established preprocessing works. What matters for J is a normalized, log-scale expression
matrix and the spot coordinates.
""")

code("""
adata = sc.datasets.visium_sge(sample_id="V1_Mouse_Brain_Sagittal_Anterior")
adata.var_names_make_unique()
adata.var["mt"] = adata.var_names.str.startswith("mt-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
sc.pp.filter_genes(adata, min_cells=10)
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
sc.pp.pca(adata, n_comps=30)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.8, key_added="leiden", flavor="igraph", n_iterations=2, directed=False)
adata
""")

code("""
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
sc.pl.umap(adata, color="leiden", ax=axs[0], show=False, legend_loc="on data")
# Cluster map drawn directly from the spot coordinates, with the same colours as the UMAP
xy = adata.obsm["spatial"]
palette = dict(zip(adata.obs["leiden"].cat.categories, adata.uns["leiden_colors"]))
axs[1].scatter(xy[:, 0], xy[:, 1], c=adata.obs["leiden"].map(palette).to_list(), s=6, linewidths=0)
axs[1].set_aspect("equal"); axs[1].invert_yaxis(); axs[1].set_axis_off(); axs[1].set_title("leiden (tissue)")
plt.tight_layout()
""")

md("""
## 3. Build the spatial graph

J is defined on a graph whose edges join physically adjacent spots. `jx.from_anndata()` extracts the expression matrix and
builds the graph: Visium hexagonal adjacency when `adata.obs` has `array_row`/`array_col`, otherwise a k-nearest-neighbour
graph on `adata.obsm["spatial"]`.

**Using Squidpy?** Its graph can be passed directly: after `sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=6)`,
use `graph = adata.obsp["spatial_connectivities"]`.
""")

code("""
markers = ["Hpca", "Ttr", "Mbp", "Plp1", "Nrgn", "Calb2", "Pcp4", "Gfap", "Snap25", "Cck"]
hvg = adata.var.query("highly_variable").sort_values("dispersions_norm", ascending=False).index[:200].tolist()
panel = [g for g in dict.fromkeys(hvg + markers) if g in adata.var_names]

values, graph, names = jx.from_anndata(adata, genes=panel)
keep = values.std(axis=0) > 0
values, names = values[:, keep], [n for n, k in zip(names, keep) if k]
degree = np.asarray((graph > 0).sum(axis=1)).ravel()
print(values.shape, "spots x genes")
pd.Series(degree).value_counts().sort_index().rename("spots").rename_axis("degree").to_frame().T
""")

md("""
Most spots should have six neighbours; tissue edges have fewer. For cell-level platforms (Xenium, MERFISH, CosMx) use
`jx.spatial_graph(coords, neighbors=6)` to `10`, or a fixed `radius`.

The panel here is the 200 most variable genes plus a few anatomical markers. All-pairs J scales with the square of the panel
size; 100–500 genes is typical.
""")

md("""
## 4. Boundary J for all pairs

The null distribution depends only on the graph, so compute it once and reuse it.
""")

code("""
null = jx.SpectralNull(graph)
res = jx.juxtaposition(values, graph, features=names, null=null, scale=1, method="auto")
tab = res.table()
print(f"p-values: {res.extra['p_method']}; effective spectral modes per pair: median {tab.modes.median():.0f}, "
      f"fewer than 20 for {(tab.modes < 20).mean():.1%} of pairs")
tab["call"] = np.select([(tab.q < 0.05) & (tab.J > 0), (tab.q < 0.05) & (tab.J < 0)],
                        ["opposing (interface)", "concordant"], "not significant")
print(tab.call.value_counts().to_string())
tab.head(10)
""")

md("""
Each row is one unordered pair: `J` (between −1 and 1), `z` and `p` from the spectral sign-flip null, and
Benjamini–Hochberg `q` over all pairs. `scale=1` applies one step of graph smoothing before differencing, which suppresses
spot-level noise (see section 9).

`modes` counts the spectral modes that carry each pair's null variance. The normal approximation behind `p` is reliable
when many modes do; with few, it is too conservative. `method="auto"` keeps the normal p-values and replaces those of
pairs with fewer than 20 modes by Monte Carlo sign-flip p-values (4,000 draws by default); `method="monte_carlo"` does so
for every pair.
""")

code("""
colours = {"opposing (interface)": OPPOSING, "concordant": CONCORDANT, "not significant": NS}
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for call, d in tab.groupby("call"):
    ax.scatter(d.J, -np.log10(d.q.clip(lower=1e-300)), s=5, alpha=0.7, color=colours[call], label=call)
ax.axhline(-np.log10(0.05), ls="--", color=NS, lw=1)
ax.set(xlabel="boundary J", ylabel="-log10 q", title="All gene pairs")
ax.legend(frameon=False, fontsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
""")

md("""
## 5. The strongest interfaces
""")

code("""
top = res.top(15, direction="opposing", max_q=0.05)
top
""")

md("""
Positive J means that where one gene rises across an edge, the other falls: two complementary programs meeting along a
boundary. A pair can be strongly anticorrelated overall and still have J ≈ 0 if its domains never touch.
""")

md("""
## 6. Map one interface

`jx.score_edges()` computes a local J on every edge, pooled over the edges incident to its two endpoints. These settings
(lazy smoother, 4 steps, ridge 1) are those of the preprint's known-truth localization benchmark; `signed_evidence` ranks strong
opposing transitions first and localized interfaces best there.
""")

code("""
gx, gy = top.iloc[0][["feature_x", "feature_y"]]
ix, iy = names.index(gx), names.index(gy)
edges = jx.score_edges(values[:, [ix, iy]], graph, scale=4, ridge=1.0, smoother_kind="lazy")
adata.obs["interface"] = edges.to_nodes(adata.n_obs, field="signed_evidence")
adata.obs["local_J"] = edges.to_nodes(adata.n_obs, field="j")

def spot_map(ax, values, cmap, title, vmin=None, vmax=None, vcenter=None):
    # Spots coloured by a value; spots without a value (NaN) are drawn in light grey.
    xy = adata.obsm["spatial"]
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    ax.scatter(xy[~ok, 0], xy[~ok, 1], c="#D9D9D9", s=6, linewidths=0)
    norm = TwoSlopeNorm(vcenter, vmin, vmax) if vcenter is not None else None
    sc_ = ax.scatter(xy[ok, 0], xy[ok, 1], c=v[ok], cmap=cmap, norm=norm, s=6, linewidths=0,
                     vmin=None if norm else vmin, vmax=None if norm else vmax)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_axis_off(); ax.set_title(title)
    plt.colorbar(sc_, ax=ax, shrink=0.7)

def expression(gene):
    x = adata[:, gene].X
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x).ravel()

lim = np.nanmax(np.abs(adata.obs["interface"]))
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
spot_map(axs[0], expression(gx), EXPR, gx)
spot_map(axs[1], expression(gy), EXPR, gy)
spot_map(axs[2], adata.obs["interface"], DIV, f"interface: {gx} vs {gy}", -lim, lim, vcenter=0)
plt.tight_layout()
""")

md("""
Red spots lie on edges where the two genes change in opposite directions: the interface between their domains.

### Which clusters does the interface separate?

For each pair of adjacent clusters, the mean signed evidence on the edges that cross between them. The top rows are the
annotated boundaries the interface runs along.
""")

code("""
lab = adata.obs["leiden"].astype(str).to_numpy()
a, b = lab[edges.endpoints[:, 0]], lab[edges.endpoints[:, 1]]
cross = a != b
(pd.DataFrame({"boundary": [" | ".join(sorted(p)) for p in zip(a[cross], b[cross])],
               "signed_evidence": edges.signed_evidence[cross]})
   .groupby("boundary").signed_evidence.agg(edges="size", mean_evidence="mean")
   .query("edges >= 10").sort_values("mean_evidence", ascending=False).head(10))
""")

md("""
### Concordant pairs

Negative J: the two genes change in the same direction across the same edges, on the same side of a boundary.
""")

code("""
res.top(10, direction="concordant", max_q=0.05)
""")

md("""
## 7. Co-localization with regional J

For "are these genes expressed in the same neighbourhoods?", use regional J. It averages an edge-sum concordance over graph
scales 1, 2, 4, 8 and 16 steps. It is a score without a p-value.
""")

code("""
genes50 = [g for g in dict.fromkeys(markers + hvg) if g in names][:50]
idx50 = [names.index(g) for g in genes50]
R = pd.DataFrame(jx.regional_juxtaposition_matrix(values[:, idx50], graph), index=genes50, columns=genes50)

Z = linkage(squareform(np.clip(1 - R.to_numpy(), 0, None), checks=False), method="average")
order = leaves_list(Z)
fig, ax = plt.subplots(figsize=(8.5, 8))
im = ax.imshow(R.iloc[order, order], cmap=DIV, vmin=-1, vmax=1)
ax.set_xticks(range(len(order)), R.columns[order], rotation=90, fontsize=7)
ax.set_yticks(range(len(order)), R.index[order], fontsize=7)
fig.colorbar(im, ax=ax, shrink=0.6, label="regional J")
ax.set_title("Regional J: shared neighbourhood expression")
""")

code("""
modules = pd.Series(fcluster(Z, t=0.6, criterion="distance"), index=genes50)
{k: v.index.tolist() for k, v in modules.groupby(modules) if len(v) > 1}
""")

md("""
Each module is a set of genes that occupy the same neighbourhoods; `sc.tl.score_genes()` turns a module into a score you can
map and compare across samples.

## 8. How boundary J relates to the established statistics
""")

code("""
cmp = jx.compare(values[:, idx50], graph, features=genes50)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.axhline(0, color=NS, lw=0.8); ax.axvline(0, color=NS, lw=0.8)
ax.scatter(cmp.lee_L, cmp.J, s=6, alpha=0.6, color=CONCORDANT)
ax.set(xlabel="Lee's L (level co-localization)", ylabel="boundary J (opposing change across edges)")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
""")

md("""
They answer different questions, so they are not expected to agree. Pairs with strongly negative L and J near zero are
complementary but do not touch; pairs with J > 0 are complementary *and* meet at an interface.

## 9. Checks before reporting

### Calibration on scrambled tissue

Permuting each gene's values over the spots, independently for each gene, destroys all spatial structure and all
dependence between genes, so the p-values should then be uniform. Permute genes independently: permuting whole spots keeps
the genes' ordinary correlation, and correlated genes do change together across edges, so those pairs are correctly called.
""")

code("""
perm = np.column_stack([rng.permutation(values[:, k]) for k in range(60)])
p = jx.juxtaposition(perm, graph, null=null, scale=1).table()["p"]
print(f"share of p < 0.05 on scrambled tissue: {np.mean(p < 0.05):.3f}")
fig, ax = plt.subplots(figsize=(5, 3))
ax.hist(p, bins=np.linspace(0, 1, 21), color=CONCORDANT, edgecolor="white")
ax.set(xlabel="p", title="Should be flat")
""")

md("""
### Robustness to the smoothing scale
""")

code("""
by_scale = pd.DataFrame({f"scale_{s}": jx.juxtaposition(values[:, idx50], graph, features=genes50, scale=s, test=False)
                          .table().sort_values(["feature_x", "feature_y"])["J"].to_numpy() for s in range(4)})
by_scale.corr(method="spearman").round(3)
""")

md("""
High rank correlations mean that conclusions do not hinge on the smoothing choice. Report the scale used.

### Replication across sections

Run the same analysis on the second anterior section and combine per-pair z-scores with Stouffer's method.
""")

code("""
def run_section(sample_id, genes):
    ad = sc.datasets.visium_sge(sample_id=sample_id)
    ad.var_names_make_unique()
    sc.pp.filter_genes(ad, min_cells=10)
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    genes = [g for g in genes if g in ad.var_names]
    v, g, n = jx.from_anndata(ad, genes=genes)
    ok = v.std(axis=0) > 0
    return jx.juxtaposition(v[:, ok], g, features=[x for x, k in zip(n, ok) if k], scale=1).table()

tab2 = run_section("V1_Mouse_Brain_Sagittal_Anterior_Section_2", names)
both = tab.merge(tab2, on=["feature_x", "feature_y"], suffixes=("_1", "_2"))
both["z_combined"] = (both.z_1 + both.z_2) / np.sqrt(2)
both["q_combined"] = jx.bh_qvalues(2 * norm.sf(np.abs(both.z_combined.to_numpy())))
print("correlation of J between sections:", round(both[["J_1", "J_2"]].corr().iloc[0, 1], 3))
(both.query("q_combined < 0.05 and J_1 > 0 and J_2 > 0").sort_values("z_combined", ascending=False)
     [["feature_x", "feature_y", "J_1", "J_2", "z_combined", "q_combined"]].head(10))
""")

md("""
Stouffer's combination assumes independent sections; for serial sections of one animal, treat it as descriptive.

## 10. Export
""")

code("""
tab.to_csv("boundary_J_all_pairs.csv", index=False)
R.to_csv("regional_J_matrix.csv")
adata.write_h5ad("anterior1_with_interface.h5ad")
""")

md("""
## 11. Reporting checklist

- Graph: construction (Visium lattice, k nearest neighbours or radius), *k*, and the degree distribution.
- Expression values: layer and normalization.
- Panel: how genes were chosen, and the number of pairs tested.
- Score and question: boundary J for interfaces; regional J for co-localization.
- Scale (smoothing steps) for boundary J, and the robustness check.
- Inference: spectral sign-flip null (normal approximation, or Monte Carlo for pairs with few effective modes), two-sided,
  with Benjamini–Hochberg q over all pairs tested.
- Edge maps: `score_edges()` settings (smoother, scale, ridge) and the field shown.
- Replication across sections, if available.

**Methods sentence you can adapt.** "Spot adjacency was defined by the hexagonal Visium lattice. For each gene pair, the
juxtaposition index J (juxtaposed v0.5.0) was computed after one step of graph smoothing, and two-sided p-values were obtained
from the spectral sign-flip null of the tissue graph (normal approximation; Monte Carlo p-values from 4,000 sign-flip draws
for pairs with fewer than 20 effective spectral modes), with Benjamini–Hochberg correction over all pairs. Interfaces were
mapped with edge-level J (lazy smoother, four steps, ridge 1). Co-localization was assessed with regional J over graph scales
1–16."
""")

code("""
sc.logging.print_header()
""")

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
out = Path(__file__).with_name("scanpy_juxtaposition.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("wrote", out)
