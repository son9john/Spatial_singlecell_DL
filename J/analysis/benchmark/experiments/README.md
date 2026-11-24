# Post hoc experiments (Supplementary Results, "Post hoc extensions")

None of these analyses was part of the benchmark protocol. They were run after the main analyses were fixed, on the
same gene pairs and labels, and are reported as exploratory in the Supplementary Results of manuscript v15.
Paths are relative to `J/analysis/benchmark`. Scripts find this folder through `JBENCH_ROOT` or their own location.

| Folder or script | What it tests | Main output |
|---|---|---|
| `seed_weighted_j/seed_j.py` | Seed-weighted J (first-knee seeds, Gaussian distance weights, σ = 3 spacings) on the observed-anatomy benchmark | `results_seed_j.csv` |
| `seed_weighted_j/seed_j_remedies.py` | Soft-clustered seeds and one map-fixed seed set per section | `results_remedies.csv` |
| `physics_scores/physics_scores.py` | Sliced Wasserstein distance, quantum fidelity after heat diffusion, Flory–Huggins contact energy, lagged mutual information | `results_physics.csv` |
| `physics_scores/combine.py`, `merfish_combine.py` | Cross-validated logistic combinations of within-section ranks | `combine.csv`, `merfish_combine_J1.csv` |
| `seed_j_pilot.py`, `seed_j_remedies.py` | Simulation pilots and null bias of pair-chosen seeds | `seed_j_pilot_results/`, `seed_j_remedies_results/` |
| `seed_j_real_validation.py` | Serial-section reproducibility (151507/151508) and the L6–white-matter screen in 151507 | `seed_j_real_validation_results/` |
| `crc_seed_validation.py` | Map-fixed seeds from EPCAM, ELF3, MGP and VIM in colorectal section GSM8594561 | `crc_seed_validation_results/` |
| `../revisions/clustered_fixed_seeds_2026-11-23_000100_EST` | Clustered and fixed seeds: calibration, weak-panel check | see its README |
| `../revisions/merfish_combo_transfer_2026-11-23_001853_EST` | J4 + regional J on the four MERFISH sections; rotated-control check at matched scale | see its README |
| `../juxtaposed/_seed_extensions.py`, `../tests/test_seed_extensions.py` | Experimental seed-weighting functions used above; not part of the released packages | — |

Pairs and labels come from `../revisions/v5_2026-11-22_104348_EST/results`.

Caveats stated in the manuscript: seeds chosen from the pair being scored bias the spectral null (11–13% at α = 0.05);
in the colorectal section the map-fixed score rejected 7.8% of rotated pairs, outside the 2.5–7.5% interval, so that
result is descriptive; detection rates on anatomically expected pairs are not estimates of power.
