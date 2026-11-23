# J4 and regional J on held-out Allen MERFISH sections

2026-11-23, 00:18–00:23 EST. Exploratory follow-up alongside manuscript v14. [Design](DESIGN.md) · [Transfer figure](merfish_transfer.pdf).

## MERFISH transfer

The human cortex score combines within-section percentile ranks of regional J and boundary J4 and their product in a logistic model. Repeating the archived leave-one-donor-out evaluation gives mean cortical AUROCs 0.7552 for the combination and 0.7270 for J1 + Hotspot when same-layer and distant pairs are pooled. We used the same expression scores and marker lists from archived v5 pairs.

The main transfer freezes coefficients learned exclusively in DLPFC. A second transfer fits the same formula on two MERFISH sections from one mouse and tests on two sections of the other mouse. That fold's marker panel was chosen from the training mouse, and both fit and evaluation used markers from that one panel. All four primary MERFISH pair panels and original score values reproduce the v5 archive. The same-structure markers in the secondary endpoint extend that archive. Within each section, a pair of distinct CCF structures has ten or more graph edges for a positive reference label, zero for a separated negative, and one to nine for exclusion. The primary endpoint includes 345, 366, 33 and 701 positive pairs across four sections. The table gives equal-weight section means. The two sections per mouse have uneven reference support. These comparisons rank anatomical marker labels.

| Task | Regional J | Frozen human model | Other-mouse fit |
|---|---:|---:|---:|
| Contacting vs separated, mean AUROC | 0.5426 | 0.5538 | 0.5760 |
| Contacting vs separated, mean AP | 0.1694 | 0.1680 | 0.1758 |
| Contacting vs separated, top-20% Jaccard | 0.1095 | 0.1230 | 0.1196 |
| Contacting vs separated or same, mean AUROC | 0.5242 | 0.5294 | 0.5632 |
| Contacting vs separated or same, mean AP | 0.1451 | 0.1497 | 0.1566 |
| Contacting vs separated or same, top-20% Jaccard | 0.0975 | 0.0984 | 0.1163 |

The frozen human coefficients raised primary AUROC in all four sections and average precision in two. In the pooled-negative endpoint they improved AUROC in only one of four sections. The largest gain came from the section with 33 positive pairs. The other-mouse fit improved AUROC in all four sections on both endpoints, and its pooled-negative AP and Jaccard also increased on average. With two animals this is exploratory. The interaction coefficients differ by experiment and interpretation of the other-mouse fit is local to that dataset.

[Complete metrics](results/merfish_mean_metrics.csv) · [Section-level paired changes](results/merfish_paired_deltas.csv) · [Animal-level paired changes](results/merfish_animal_deltas.csv) · [Coefficients](results/model_coefficients.csv).

## Audit of the supplied seed-validation results

The 0.164 to 0.280 top-1% Jaccard improvement is for the single pair of serial sections 151507 and 151508. The all-pair Spearman estimate changes from 0.867 to 0.863. The top-1% Lee's L Jaccard is 0.488 for the same panel.

The archived DLPFC L6–WM target pairs verify 6/144 and 28/144 with BH q < 0.05 and positive z. The median signed J increases from 0.0118 to 0.0342, median z from 0.415 to 1.063 and ranking AUROC from 0.855 to 0.868. These are marker-pair discovery rates on one measured boundary. True molecular opposition is unobserved, and the pairs reuse genes. The code calculates the exact conditional variance of the cross-term for independent spectral sign flips in the chosen eigenbasis. Its p-values use a normal approximation.

Recomputing the empirical cross-donor rotated control reproduces the reported 5.24% and 5.80%, which use one smoothing step and distance bandwidth 3. The target itself uses four steps and bandwidth 4. On the same control with matching settings the rates are 4.28% and 4.84%. Removing the six discovery genes from both ranked 50-gene panels gives 3.96% and 4.36% (99/2,500 and 109/2,500). The screen uses seeds located from six reference genes measured on the target tissue, with marker identities chosen on the other donors. The corrected control uses a single rotated cross-donor pair and 50 of the 57 seeds that are present in the overlap. Retained spatial dependence and repeated use of genes should be considered when interpreting the empirical rate.

[Verified L6–WM counts](results/verified_target_counts.csv) · [Rotated-control settings and rates](results/rotated_control_parameter_check.csv).

## Rebuild

Run from the parent of J-analysis using Python with NumPy, SciPy, pandas, scikit-learn, Matplotlib and h5py. The archived MERFISH cache builder uses the Allen raw files already under J-analysis/data/single_cell. If the four matching cache files already exist, reuse that directory.

```sh
run=J-analysis/revisions/merfish_combo_transfer_2026-11-23_001853_EST
cache=/path/to/local/merfish/cache
python J-analysis/revisions/v5_2026-11-22_104348_EST/code/allen_transfer.py --root J-analysis --cache "$cache" --out "$cache" --prepare-only
python "$run/transfer.py" J-analysis "$cache" "$run/results"
python "$run/target_audit.py" J-analysis "$run/results"
python "$run/report.py" "$run"
```
