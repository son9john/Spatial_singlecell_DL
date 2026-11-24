# Monte Carlo check of the spectral sign-flip p-values (JK, 2026-11-22)

Independent re-implementation by JK of the spectral null from the protocol text, run on the shared data.
It first reproduces the archived normal-approximation results, then replaces the normal tail area by explicit
sign-flip randomizations of the same null. `README_JK.md` is the original description with the results.

Only the paths were changed for this deposit: the scripts find the analysis folder from `JBENCH_ROOT` (or two
levels above this folder), and `crc_lee_mc.py` also writes its summary to `crc_lee_mc_summary.csv`.
Outputs are written to the working directory; run them from `results/v3/mc_check`:

```sh
cd results/v3/mc_check
python ../../../benchmarks/v3/mc_check/crc_mc.py             # J and Pearson's r, colorectal, 4,000 flips
python ../../../benchmarks/v3/mc_check/crc_lee_mc.py 1000    # Lee's L, colorectal, 1,000 randomizations
python ../../../benchmarks/v3/mc_check/a2_mc.py              # J and Pearson's r, cross-donor controls
python ../../../benchmarks/v3/mc_check/a2_lee_mc.py          # Lee's L, cross-donor controls
python ../../../benchmarks/v3/mc_check/ci.py                 # cluster-bootstrap intervals
```

`results/v3/mc_check/archived_JK/` keeps the tables of the original run. Rerunning the scripts reproduces them
(seeds are fixed). `h5min.py` reads the HDF5 count matrices with the system HDF5 library when h5py is absent.

| Script | Manuscript values |
|---|---|
| `crc_mc.py`, `crc_lee_mc.py` | Fig 3B, Table 1: pairs called at q < 0.05 with normal and Monte Carlo p-values (J 86.8%; Pearson's r 32.7% and 67.0%; Lee's L 0.4% and 64.2%); effective spectral modes |
| `a2_mc.py`, `a2_lee_mc.py` | Monte Carlo rejection rates on the cross-donor controls (rotated 5.9% for Pearson's r and Lee's L; aligned 10.7% and 10.4%) |
| `ci.py` | Cluster-bootstrap intervals for the rejection rates and AUROC differences, donor-level gains of J + co-localization |
| `eff_scale.py` | Effective modes and agreement of normal and Monte Carlo p-values for J at 1–16 diffusion steps |
