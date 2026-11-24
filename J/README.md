# Juxtaposition index J

This folder extends the 2021 Lee's L code of this repository (`adata_bsc.py`, `calculate_bsc_all`, first committed on 25 October 2021, commit 9415612) to the juxtaposition index J.

Lee's L reads the smoothed levels of two genes. J reads their changes across the same tissue edges:

    J = -2 Σ_e w_e Δx_e Δy_e / Σ_e w_e (Δx_e² + Δy_e²)

J is positive when one gene rises where the other falls. It lies between −1 and 1, and it decomposes into a map of edge contributions that shows where two expression domains meet. Regional J, its companion score, measures shared neighbourhood expression. Both come with a spectral sign-flip null that keeps each gene's spatial autocorrelation.

## Contents

| Folder | What it holds |
|---|---|
| `python/` | Python package `juxtaposed` 0.5.0 (adds Monte Carlo spectral p-values and effective spectral modes) |
| `r/` | R package `juxtaposed` 0.4.0 |
| `manuscript/` | Preprint manuscript v15 (PDF) |
| `analysis/code/` | Scripts that draw the manuscript figures |
| `analysis/benchmark/` | Benchmark scripts and result tables; Monte Carlo checks in `benchmarks/v3/mc_check`; post hoc experiments of the Supplementary Results in `experiments/` and `revisions/` |
| `site/` | Website: apps for R and Python, Seurat vignette, Scanpy notebook (<https://son9john.github.io/Spatial_singlecell_DL/>) |

## Install

```bash
pip install ./J/python                   # Python
R CMD INSTALL J/r/juxtaposed             # R
```

## Quick start (Python)

```python
import juxtaposed as jx
graph = jx.spatial_graph(coords, neighbors=6)
res = jx.juxtaposition(expr, graph, features=genes, method="auto")
res.top(20)
```

`res.table()` lists J, z, p and q for every gene pair, and `modes`, the number of spectral modes behind each test. With
`method="auto"`, pairs with fewer than 20 modes get Monte Carlo sign-flip p-values instead of the normal approximation.

The 2021 files in the repository root are unchanged. Their original commit dates are in the history (`git log --diff-filter=A --date=iso --name-only`).
