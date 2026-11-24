#!/usr/bin/env bash
# Reproduce every table and figure added by amendment v3 (PROTOCOL_v3_amendment.md) and the
# combined manuscript. Data are read from ../../data (J-analysis/data); results go to
# ../../results/v3 and figures to ../../figures/v3. Needs numpy, scipy, pandas, scikit-learn,
# matplotlib (and h5py, or the system libhdf5 for the bundled reader).
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python bench_gold_calibration.py all 2     # A1 + A2 aligned/rotated, every ecosystem default test (~10 min)
python bench_gold_accuracy.py b1           # DLPFC adjacency, natural orientation
python bench_gold_accuracy.py b2           # semi-synthetic exclusive markers
python bench_gold_accuracy.py c            # seqFISH, both designs
python bench_gold_accuracy.py crc          # colorectal: pairs called significant by each test
python bench_jaccard.py m1                 # interface IoU on expert labels (DLPFC, seqFISH)
python bench_jaccard.py m2                 # interface IoU, exclusive markers on DLPFC maps
python bench_jaccard.py m34                # Jaccard reproducibility and redundancy
python bench_complementarity.py            # J + co-localization (exploratory)
python bench_gold_speed.py                 # cost of each ecosystem test, 4,000 genes
python bench_speed_v3.py 1                 # optimized J kernels, one thread
python bench_speed_v3.py 2                 # two threads
python figures_v3.py 2 3 5 6 7             # Figures 2-7 of the combined manuscript
python ../../../manuscript/source_v3_combined/build_manuscript_v3.py
