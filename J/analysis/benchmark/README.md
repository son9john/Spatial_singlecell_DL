# Benchmark code and results

This folder holds the scripts and result tables behind the calibration analysis of the preprint (Fig 2,
Table S1) and the other benchmark v3 endpoints. Its layout mirrors the analysis repository, so the
scripts run unchanged:

| Path | Content |
|---|---|
| `benchmarks/PROTOCOL*.md` | The benchmark protocol (v2) and its amendment (v3). Their SHA-256 hashes were recorded before any comparator result was computed: `benchmarks/PROTOCOL_v2.sha256` and `results/v3/protocol_v3_amendment.sha256`. |
| `benchmarks/v3/` | The v3 benchmark scripts. `run_all.sh` lists every step; `README.md` describes each script. |
| `packages/python_v0.3/` | The version of the juxtaposed Python package that the benchmark used (`common.py` imports it from here). |
| `results/v3/` | All v3 outputs: per-pair tables (`*.csv.gz`), summaries, logs and the protocol hash. |
| `download_data.py` | Downloads the 12 public DLPFC Visium sections (Maynard et al. 2021) from their authors into `data/dlpfc/`. |
| `data_checksums.csv` | SHA-256 of every downloaded file at the time of the analysis. |
| `verify_calibration.py` | Recomputes the calibration numbers of the manuscript from `results/v3/gold_calibration_all.csv.gz`. |

## Where the calibration numbers come from

All of them are computed by `benchmarks/v3/bench_gold_calibration.py` and stored, one row per gene pair, in
`results/v3/gold_calibration_all.csv.gz`. A pair counts as called when its p-value is below 0.05.

| Manuscript value | Rows | Column |
|---|---|---|
| Boundary J, spectral test, aligned control: 1,594 of 15,000 pairs (10.6%) | `axis == "A2"`, `setting == "aligned"` | `J1_spectral` |
| Boundary J, spectral test, rotated control: 794 of 15,000 pairs (5.3%) | `axis == "A2"`, `setting == "rotated"` | `J1_spectral` |
| Comparators with the spectral null: rotated 3.0–3.6%, aligned 4.3–5.2% | as above | `pearson_spectral`, `spatialdm_spectral`, `liana_spectral`, `meringue_spectral`, `lee_spectral` |
| Default tests of each package (Fig 2A, Table S1 rows A2) | as above | `pearson_t`, `spearman_t`, `spatialdm_z`, `spatialdm_perm`, `liana_perm`, `hotspot_z`, `meringue_perm`, `lee_perm` |
| Independent random fields (Table S1 rows A1) | `axis == "A1"`, one `setting` per field type | same columns |

`results/v3/gold_calibration_fpr_table.csv` is the summary written by `figures_v3.py` (Fig 2), and
`results/v3/gold_calibration_log.txt` is the run log (matched spots per section pair, run times).

To check the numbers:

```sh
python verify_calibration.py
```

## Reproduce from the raw data

Python 3.11 or later, with numpy, scipy, pandas, scikit-learn, matplotlib and h5py.

```sh
python download_data.py                        # about 135 MB into data/dlpfc/, checksums in data/source_manifest.csv
cd benchmarks/v3
python bench_gold_calibration.py all 2         # A1 and A2 (aligned and rotated); about 10 minutes with 2 processes
python figures_v3.py 2                         # Fig 2 and results/v3/gold_calibration_fpr_table.csv
cd ../..
python verify_calibration.py
```

The random seed is fixed (`SEED = 20261122` in `benchmarks/v3/bench_calibration.py`), so a rerun gives the
same table. `bench_gold_calibration.py` writes `results/v3/gold_calibration_all.csv.gz` and overwrites the
archived copy; keep a copy if you want to compare.

The comparator statistics (SpatialDM, LIANA+, Hotspot, MERINGUE, Lee's L) are re-implemented from their
published definitions in `benchmarks/v3/gold.py`, because the environment of the analysis could not install
the packages. `benchmarks/v3/official/` repeats the calibration with the packages themselves; see
`benchmarks/v3/README.md`.

The data are public: Maynard KR et al., Nat Neurosci 24, 425–436 (2021), spatialLIBD / HumanPilot.
