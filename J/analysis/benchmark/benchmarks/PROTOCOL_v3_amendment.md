# Protocol amendment v3: comparison against ecosystem gold standards

Written and hashed on 22 November 2026, before any gold-standard result in this amendment was
computed. Every J result referred to here was already known from PROTOCOL_v2. So this amendment
is blind only for the comparator results, not for J's.

The display label for the Lee comparison was updated after the initial freeze. The
permutation settings and analysis definitions are unchanged.

## Why the amendment exists

PROTOCOL_v2 compared J with Lee's L, bivariate Moran's I and Pearson's r. Lee's L is not what
most spatial-transcriptomics analysts use. This amendment adds the bivariate statistics, and the
default significance procedure of each, that ship with the Scanpy/scverse and Seurat/Bioconductor
ecosystems. All endpoints, datasets, gene sets, graphs, seeds and success criteria of
PROTOCOL_v2 are unchanged.

## Comparators (each with its package's documented default or tutorial setting)

| Label | Package convention reproduced | Statistic | Default test |
|---|---|---|---|
| Pearson | Scanpy / Seurat users (scipy.stats.pearsonr, cor.test) | r | t test, two-sided |
| Spearman | same | rho | t test, two-sided |
| SpatialDM | SpatialDM 0.3 `weight_matrix(l=1.2, cutoff=0.2)` + `spatialdm_global` | global Moran's R, RBF weights (186 nearest neighbours, weights below 0.2 set to 0, self-weight 1 for spots, rescaled to sum N) | analytic z, one-sided (tutorial `method='z-score'`); permutation (1,000, one-sided) as secondary |
| LIANA+ | LIANA+ 2.x `spatial_neighbors(bandwidth=200 px, cutoff=0.1, kernel='gaussian', set_diag=True)` + `bivariate(global_name='morans', n_perms=100)` | global Moran's R, Gaussian weights (at most 100 neighbours) | 100 spot-label permutations, two-sided |
| Hotspot | Hotspot `create_knn_graph(weighted_graph=False, n_neighbors=30)`, model `'none'` on z-scored input, `compute_local_correlations` | H = xᵀ(A + Aᵀ)y, conditional symmetric Z (smaller \|Z\| of the two conditionings) | normal, one-sided (as used to build modules) |
| MERINGUE | `spatialCrossCorTest(x, y, w, n=1000)` on the binary first-ring graph | symmetrized bivariate Moran's I, row-normalized weights | 1,000 permutations of x, two-sided, observed value counted in the null |
| Lee | spdep `lee.mc(nsim=999)` | Lee's L, row-standardized first-ring graph | 999 permutations, one-sided "greater" |
| Giotto | `detectSpatialCorFeats(method='network')` | Pearson correlation of neighbour-averaged expression | no test (score only) |
| LIANA+ cosine | `bivariate(local_name='cosine')`, mean over spots | weighted cosine | no global test (score only) |
| J (this work) | `juxtaposed` 0.2 | J at scale 1 | exact spectral sign-flip null, two-sided |

Distances are converted to the units the tutorials used. For Visium we work in units of the
100 µm spot spacing: SpatialDM l = 1.2 spacings (its melanoma tutorial ran on array units with a
spacing of 1), and LIANA+ bandwidth = 200 px at the DLPFC median spacing of 137 px, which is
1.46 spacings. For single-cell seqFISH the same multiples of the median nearest-neighbour
distance are used, and SpatialDM `single_cell=True`, LIANA+ `set_diag=False`.

The calculations are re-implemented from the published formulas. This is necessary because the
packages cannot be installed in the analysis container. A separate runner
(`benchmarks/v3/official/`) repeats the calibration with the official packages. The paper
reports agreement between the two wherever both have been run. MERINGUE agreement with the
bivariate Moran's I used here is already established by the parallel analysis at 1e-12 (45
pairs).

## Endpoints (unchanged definitions, comparators added)

1. **A1 synthetic null.** False-positive rate (FPR) at α = 0.05 with Bradley's liberal interval
   [0.025, 0.075], for every comparator with a test.
2. **A2 cross-donor null.** As in v2 (aligned array positions), plus the 180° rotated variant
   that removes shared array-position effects. The rotated variant was a post-hoc diagnostic in
   v2. Here it is declared a co-primary null for all methods, because it is the only real-tissue
   null that is free of the technical artefact identified in v2.
3. **B1 DLPFC layer adjacency.** AUROC for adjacent versus distant layer-marker pairs, K = 10,
   leave-one-donor-out markers, all methods' scores. Higher score = adjacent for every method,
   as for J.
4. **B2 semi-synthetic exclusive markers.** As in v2, with all methods.
5. **C seqFISH contact.** v2 design (contact z ≥ 3 versus ≤ −3) is primary. The parallel
   analysis's contact-rate design (rate ≥ 0.15 versus ≤ 0.01, top-3 markers) is secondary,
   with each method's direction fixed in advance as above.
6. **CRC real-data illustration.** Fraction of all pairs of the top 300 spatially variable genes
   called significant at BH q < 0.05 by each default test. There is no ground truth, so this
   endpoint only describes how many calls each default test makes.
7. **Run time.** Time to compute p-values for all pairs of a fixed gene panel.

## Local (edge-level) known-truth results

The known-truth average-precision benchmark on the held-out Allen geometries is the design of
`Juxtaposed_preprint_v3`. It was run with the official LIANA+ 2.1.1 and SpatialDM 0.3.1 packages by
the parallel analysis. It is reported as computed there and is not re-run here. Its numbers are
read from `results/package_comparison/local_summary.csv` as snapshotted on 22 November 2026.

## What would count against J

- J's spectral test outside [0.025, 0.075] on A1 or on the rotated A2 null.
- A comparator's default test calibrated on both nulls while also matching or beating J's
  accuracy endpoints. In that case the paper would recommend that comparator.
