# juxtaposed

**22 November 2026 · regional expression.** Use `jx.regional_juxtaposition_matrix(X, graph)` to measure whether two genes occupy similar tissue neighborhoods. Positive values capture shared high or low signal. Use the original `juxtaposition_matrix` for opposing changes across edges. The regional score has descriptive output on [−1, 1] with equally weighted heat steps 1, 2, 4, 8 and 16.


**The juxtaposition index J for spatial omics, with spectral-null p-values for every gene pair.**

Most bivariate spatial statistics, such as Pearson's r, bivariate Moran's I, Lee's L and the
Moran's R of SpatialDM and LIANA+, ask whether two genes are *in the same places*. J asks a
different question: **do two genes change across the same tissue edges, and in which direction?**

| J | Meaning |
|---|---|
| J > 0 | opposing changes on shared edges, as for complementary domains that touch |
| J ≈ 0 | no shared transitions, even if the genes are anticorrelated overall |
| J < 0 | concordant changes: the genes rise and fall together across boundaries |

For standardized features x, y on a graph with edge weights w, after `scale` smoothing steps:

```
J = -2 Σ_e w_e Δx_e Δy_e / Σ_e w_e (Δx_e² + Δy_e²)
  = [E(x − y) − E(x + y)] / [E(x − y) + E(x + y)]          (bounded in [-1, 1])
```

where E is the gradient (Dirichlet) energy. On a regular graph, J = (I_xy − r) / (1 − (I_x + I_y)/2):
the excess of neighbour cross-association (bivariate Moran's I) over same-spot association
(Pearson's r). With `laplacian="normalized"`, the same identity holds exactly on any graph.
Lee's L reads the smooth part of an association; J reads the gradient part.

## Install

```bash
python -m pip install .           # from this source directory
python -m pip install ".[tables]" # includes pandas for ranked tables
python -m pip install ".[mlx]"    # enables MLX on supported Apple Silicon Macs
```

## Quick start

```python
import juxtaposed as jx

graph = jx.spatial_graph(coords, neighbors=6)          # any platform (cells or spots)
# graph = jx.visium_graph(array_row, array_col)        # Visium hexagonal neighbours

res = jx.juxtaposition(expr, graph, features=genes)    # J + spectral-null z, p, BH q for all pairs
res.top(20)                                            # most opposed pairs
res.top(20, direction="concordant")                    # strongest shared transitions
jx.compare(expr, graph, features=genes)                # J next to Lee's L, Moran's I and r

# where along the tissue a pair opposes: local map for one pair
edges = jx.score_edges(expr[:, [i, k]], graph, scale=4, ridge=1.0, smoother_kind="lazy")
spot_map = edges.to_nodes(graph.shape[0])              # per-spot average, ready to plot
```

From AnnData, with no hard dependency on anndata:

```python
X, graph, genes = jx.from_anndata(adata, genes=my_genes)   # uses array_row/array_col if present
res = jx.juxtaposition(X, graph, features=genes)
```

`expr` is a matrix of spots or cells by genes holding normalized values. For counts, use for
example `log1p(1e4 * counts / library_size)`.

## Inference under the spectral null

Spatially variable genes are autocorrelated, which breaks the independence that the usual tests
assume. In the benchmark, real genes from **different human donors**, with one section rotated,
share no biology. On these pairs, the default tests of the Scanpy and Seurat ecosystems called many
of them significant at p < 0.05:

- 76–88% for Pearson and Spearman t-tests, LIANA+ permutations and MERINGUE permutations;
- 45–47% for the one-sided SpatialDM, Hotspot and Lee permutation tests, whose ceiling is 50%.

`SpectralNull` implements Moran spectral randomization. It randomizes the signs of each gene's
graph-Fourier coefficients, which keeps the gene's spatial autocorrelation exactly while breaking
its link to the other gene. The spectral test of J rejected 5.3% of the rotated cross-donor pairs
(10.6% at matching array positions). One eigendecomposition per tissue section gives p-values for
**all** pairs, for J, Lee's L, Moran's I and Pearson's r, with no permutations.

The closed-form p-value is a normal tail area. It is accurate when many spectral modes carry the
null variance, as for J (about 51 effective modes in a colorectal section), and too conservative
when few do, as for Pearson's r and Lee's L on smooth maps (about 2). Every test returns `modes`,
and warns when many pairs have fewer than 20. `method="monte_carlo"` samples the null directly;
`method="auto"` does so only for the pairs with few modes. In the colorectal section, Lee's L called
0.4% of pairs at q < 0.05 with the normal approximation and 64% with Monte Carlo p-values; J called
87% either way.

```python
null = jx.SpectralNull(graph)            # once per section (n up to ~20,000 spots)
out_j   = null.test_j(expr, scale=1)     # dict of p x p matrices: stat, sd, z, p, modes
out_lee = null.test_lee(expr, method="monte_carlo", n_draws=1000)   # few modes: sample the null
```

## Speed

- All-pairs J is formed as a symmetric rank-k update after a sparse Cholesky-type transform of
  the graph Laplacian. That is the same cost as Lee's L.
- On one CPU thread, all 8 million pairs of 4,000 genes take about 1.5 s in float64 and 0.8 s
  with `dtype=np.float32`, on 3,611 spots.
- Spectral-null p-values and q-values for all of them take a few seconds after a one-time
  eigendecomposition.
- For score-only work, `backend="auto"` tries MLX on Apple Silicon, then a compatible installed
  PyTorch GPU build (CUDA, ROCm or Intel XPU). It falls back to the CPU when an accelerator is
  unavailable. The accelerator kernels use float32 at power 1 on the combinatorial Laplacian.

```python
J = jx.juxtaposition_matrix(expr, graph, scale=1, backend="auto")
J32 = jx.juxtaposition_matrix(expr, graph, scale=1, dtype=np.float32)
jx.available_backends()
```

## Main functions

| Function | What it returns |
|---|---|
| `juxtaposition(X, graph, features, scale=1, test=True)` | `JResult`: J matrix, z, p, q; `.table()`, `.top()` |
| `juxtaposition_matrix(X, graph, scale=1, power=1, laplacian="combinatorial")` | all-pairs J (`power=0`: negative codispersion) |
| `lee_l_matrix`, `moran_bv_matrix`, `pearson_matrix`, `codispersion_matrix` | comparators on the same graph |
| `SpectralNull(graph).test_j / test_lee / test_moran / test_pearson` | all-pairs tests under the spectral null (normal or Monte Carlo p-values) |
| `permutation_test(x, y, graph, statistic)` | classical permutation test, for comparison only |
| `score_edges(X2, graph, scale, ridge, power, smoother_kind)` | local J, signed cross-gradient energy and joint energy on every edge |
| `juxtaposed_value(dx, dy, weights, power)` | J for given increments |
| `spatial_graph`, `visium_graph`, `smoother`, `lazy_walk`, `laplacian`, `normalized_laplacian` | graph utilities |

`scale` is the number of smoothing steps applied before differencing. The steps use uniform
heat diffusion, `S = I − L/(2 d_max)`; `scale=0` gives raw increments.

## When to use J and when not to

* **Use J to ask whether two genes change in opposite directions across shared edges.**
  Examples are epithelial against stromal programmes, morphogen gradients and boundaries
  between exclusive domains. With exclusive markers it outperformed every co-localization
  statistic in the benchmark. Its spectral-null p-values need no Monte Carlo step, because many
  spectral modes carry its null variance.
* **Use a co-localization statistic to ask whether two domains touch.** Examples are Lee's L
  with `SpectralNull.test_lee(..., method="monte_carlo")`, and SpatialDM or LIANA+ Moran's R. At Visium and seqFISH
  resolution, markers of neighbouring domains overlap. In the benchmark, co-localization
  statistics, used with their natural sign, identified touching layers and cell types better
  than J (AUROC about 0.78 against 0.45). Adding J to a co-localization statistic improved the
  ranking of touching-but-distinct domains against both distant and same-domain pairs in all
  12 cortical sections.
* **To locate where an interface lies, prefer unnormalized gradient scores.** The
  `signed_evidence` field of `score_edges` is one; another is |Δx − Δy|.
* **Check against technical covariates.** J reads spot-to-spot changes, so it also picks up
  spot-level technical structure such as capture efficiency. Check each gene against log
  library size.

## Cite

Song J, Lee D, Yoo J, Kim K. *Boundary and regional juxtaposition scores for spatial gene expression*. Manuscript v5, 22 November 2026. Local preprint draft.

The companion R package is in `../r/juxtaposed` in this revision.
