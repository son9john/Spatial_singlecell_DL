"""Recompute the calibration numbers of the preprint (Fig 2, Table S1) from the archived per-pair table.

Usage: python verify_calibration.py [path to gold_calibration_all.csv.gz]
"""
import sys
from pathlib import Path

import pandas as pd

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results" / "v3" / "gold_calibration_all.csv.gz"
d = pd.read_csv(path, low_memory=False)
alpha = 0.05

default = ["pearson_t", "spearman_t", "spatialdm_z", "spatialdm_perm", "liana_perm", "hotspot_z", "meringue_perm",
           "lee_perm", "J1_spectral"]
spectral = ["J1_spectral", "pearson_spectral", "spatialdm_spectral", "liana_spectral", "meringue_spectral", "lee_spectral"]

a2 = d[d["axis"] == "A2"]
print("Cross-donor controls (A2): pairs per control")
print(a2.groupby("setting").size().to_string(), "\n")

print("Boundary J, two-sided spectral test (J1_spectral < 0.05)")
for setting, g in a2.groupby("setting"):
    k = int((g["J1_spectral"] < alpha).sum())
    print(f"  {setting:8s} {k:5d} of {len(g):,} pairs = {100 * k / len(g):.1f}%")

print("\nPercent called with the spectral null, every statistic")
print(a2.groupby("setting")[spectral].apply(lambda g: (g < alpha).mean() * 100).round(1).to_string())
comp = [c for c in spectral if c != "J1_spectral"]
rng = a2.groupby("setting")[comp].apply(lambda g: (g < alpha).mean() * 100)
for setting, row in rng.iterrows():
    print(f"  comparators, {setting}: {row.min():.1f}-{row.max():.1f}%")

print("\nPercent called by the default tests (Table S1)")
table = d.groupby(["axis", "setting"])[default].apply(lambda g: (g < alpha).mean() * 100).round(1)
print(table.to_string())
