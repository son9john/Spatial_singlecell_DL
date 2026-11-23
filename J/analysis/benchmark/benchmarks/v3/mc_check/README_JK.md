# Monte Carlo check of the spectral sign-flip p-values (JK, 2026-11-22)

Independent re-implementation from the protocol text (PROTOCOL_v2 sections 3-4, v3 amendment), run on the
shared `J_analysis_pipelines` data. It reproduces the archived normal-approximation results
(A2: J1 statistic to 1e-15, J1 p to 0.002; CRC: same top-300 genes, Lee p to 2e-4, 0.44% at q < 0.05),
then replaces the normal tail area by explicit sign-flip randomizations of the same null.

| File | Content |
|---|---|
| `a2_mc.py` | A2 cross-donor control, rotated and aligned, J1 and Pearson; 4,000 sign flips per pair |
| `crc_mc.py` | Colorectal section, J1 and Pearson, 44,850 pairs; 4,000 sign flips |
| `a2_lee_mc.py` | A2 control, Lee's L (independent flips of both genes); 1,000 randomizations |
| `crc_lee_mc.py` | Colorectal section, Lee's L (independent flips of both genes); 1,000 randomizations |
| `a2_mc_summary.csv`, `crc_mc_summary.csv` | Result tables |
| `h5min.py`, `a2_helpers.py` | HDF5 reader (ctypes; h5py was unavailable) and helpers |

Paths point to `/tmp/jpipe/x/J_analysis_pipelines`; edit `ROOT` to rerun. Repeated eigenvalues are not
rotated within eigenspaces (0-5 per graph), which explains the small differences from the archive.

## Results (% of pairs)

A2 control, unadjusted p < 0.05, 15,000 pairs:

| Control | J1 normal | J1 Monte Carlo | Pearson normal | Pearson Monte Carlo |
|---|---|---|---|---|
| Rotated | 5.3 | 5.3 | 2.9 | 5.9 (2.8-10.0 by section pair) |
| Aligned | 10.6 | 10.6 | 5.2 | 10.7 (2.1-18.0 by section pair) |

Lee's L on the same pairs: rotated 2.9 (normal) vs 5.9 (Monte Carlo); aligned 4.3 vs 10.4.

Median effective modes: J1 about 285, Pearson about 2.

Colorectal section, 44,850 pairs:

| Statistic | Normal, p < 0.05 | Normal, BH q < 0.05 | Monte Carlo, p < 0.05 | Monte Carlo, BH q < 0.05 |
|---|---|---|---|---|
| J1 | 87.2 | 86.8 | 87.2 | 86.8 |
| Pearson | 55.9 | 32.7 | 69.4 | 67.0 |
| Lee's L | 51.0 | 0.4 | 67.1 | 64.2 |

Lee's L normal-approximation p-value quantiles (1%, 10%, 25%, 50%): 0.0006, 0.0053, 0.016, 0.048.
The BH line at those ranks is 0.0005, 0.005, 0.0125, 0.025, so the 0.4% is a near miss at every rank.

## Added on the second pass

| File | Content |
|---|---|
| `ci.py` | Cluster bootstrap: rejection rates (section pairs, then genes) and AUROC differences (donors, then sections) |
| `calibration_cluster_ci.csv`, `auroc_cluster_ci.csv` | Their results |
| `eff_scale.py` | J at scales 1-16 in the colorectal section: effective modes 51 to 11, normal and Monte Carlo agree |
| `fig3b.py` | Redrawn Fig 3B and relabelled Fig 7B (image edits; regenerate from the authors' scripts for the final version) |
