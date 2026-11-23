# Benchmark v3: the ecosystem gold standards, Jaccard endpoints and faster kernels

The rules for this benchmark are in `../PROTOCOL_v3_amendment.md`. The amendment was hashed before any
comparator result was computed; the hash is in `../../results/v3/protocol_v3_amendment.sha256`. The
Jaccard endpoints, the complementarity analysis and the kernel optimizations were added afterwards and
are exploratory.

| Script | What it does |
|---|---|
| `gold.py` | Re-implementations of the Scanpy/Seurat-ecosystem statistics and their default tests: Pearson/Spearman t, SpatialDM global R (z-test and permutation), LIANA+ Moran's R (100 permutations), Hotspot conditional Z, MERINGUE and separate Lee permutations, Giotto spatial correlation, LIANA+ cosine |
| `bench_gold_calibration.py` | False-positive rates: Gaussian random fields (A1) and cross-donor genes, aligned and rotated (A2) |
| `bench_gold_accuracy.py` | DLPFC adjacency, semi-synthetic markers, seqFISH (contact z and contact-rate designs), colorectal significance; every statistic with its natural sign |
| `bench_jaccard.py` | Interface IoU (segmentation Jaccard) and the lab's threshold-swept Jaccard AUC; reproducibility and redundancy of top-1% pairs |
| `bench_complementarity.py` | J combined with a co-localization statistic (cross-validated) |
| `bench_gold_speed.py`, `bench_speed_v3.py` | Run time of every test and of the optimized J kernels |
| `figures_v3.py` | Figures 2–7 |
| `official/` | Runner that repeats the calibration with the official packages (LIANA+, SpatialDM, Hotspot in Python; MERINGUE and spdep in R) |

## Official packages

The container used for this analysis could not install the packages, so the statistics are
re-implemented from their published definitions. To repeat the calibration with the packages
themselves:

```sh
python official/export_null_pairs.py                          # writes results/v3/official_inputs/
python official/run_official_python.py ../../results/v3/official_inputs
Rscript official/run_official_R.R ../../results/v3/official_inputs 250
```

The Python runner records package versions and any per-package error, so it can be run as is.
It was written without access to the packages; check its printed versions and error table.

`results/v3/external_parallel/` is a checksummed snapshot of the results that the parallel
analyses produced with the official packages. It covers the known-truth local benchmark with LIANA+
2.1.1 and SpatialDM 0.3.1, the edge IoU, the seqFISH contact-rate benchmark and the Apple-silicon
timings. The manuscript reads those numbers from this snapshot.
