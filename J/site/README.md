# juxtaposed website

The workflow `.github/workflows/pages.yml` builds this folder and publishes it with GitHub Pages at
<https://son9john.github.io/Spatial_singlecell_DL/>.

| Path on the site | Source | What it is |
|---|---|---|
| `/` | `index.html` | Landing page |
| `/r/` | `apps/r/app.R` | Shiny app for R (Shinylive + webR, runs in the browser) |
| `/python/` | `apps/python/app.py`, `pipeline.py` | Shiny app for Python (Shinylive + Pyodide, runs in the browser) |
| `/vignette/seurat_juxtaposition.html` | `vignettes/seurat_juxtaposition.Rmd` | Seurat vignette |
| `/vignette/scanpy_juxtaposition.html` | `vignettes/scanpy_juxtaposition.ipynb` | Scanpy notebook (regenerate with `python vignettes/make_scanpy_notebook.py`) |

The apps use the package sources in `J/python/src/juxtaposed` and `J/r/juxtaposed/R`; the workflow copies them next to each app.

## Run the apps on your own computer

From the repository root:

```bash
# Python app
pip install shiny pandas matplotlib ./J/python
cp -R J/python/src/juxtaposed J/site/apps/python/juxtaposed
shiny run J/site/apps/python/app.py
```

```r
# R app
install.packages(c("shiny", "bslib", "DT", "ggplot2", "Matrix"))
dir.create("J/site/apps/r/juxtaposed", showWarnings = FALSE)
file.copy(list.files("J/r/juxtaposed/R", full.names = TRUE), "J/site/apps/r/juxtaposed", overwrite = TRUE)
shiny::runApp("J/site/apps/r")
```

## Input format for the apps

A CSV or TSV with columns `x` and `y` (spot or cell coordinates) followed by one column per gene, already normalized
(for example log1p of counts per 10,000). Up to 200 genes are scored. From Seurat:

```r
xy <- GetTissueCoordinates(obj)[colnames(obj), 1:2]; names(xy) <- c("x", "y")
genes <- head(VariableFeatures(obj), 100)
write.csv(cbind(xy, t(as.matrix(GetAssayData(obj, layer = "data")[genes, ]))), "for_app.csv", row.names = FALSE)
```

From Scanpy:

```python
import pandas as pd
genes = adata.var_names[adata.var.highly_variable][:100]
df = pd.DataFrame(adata[:, genes].X.toarray(), columns=genes)
df.insert(0, "y", adata.obsm["spatial"][:, 1]); df.insert(0, "x", adata.obsm["spatial"][:, 0])
df.to_csv("for_app.csv", index=False)
```
