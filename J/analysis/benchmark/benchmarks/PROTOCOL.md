# Benchmark protocol for a reproducible opposition score

## Primary endpoint

The target is an edge that crosses a known focal interface where two expression signals change in opposite directions. Use non-interpolated average precision (AP) as the primary metric because most candidate edges are negatives. Report the positive prevalence and the secondary recall at a false-positive rate of at most 1%. Report results for each geometry and count level. Binary labels come from the data generator and are absent from graph construction and scoring.

## Data and difficult negatives

Use two anatomically distinct geometries from Allen MERFISH mice Zhuang-ABCA-3 and Zhuang-ABCA-4. These mice were outside the earlier score-development pilot. Choose the 5,000 central registered or experimentally aligned coordinates in the middle label of the available eligible section labels for each mouse. Select sections using only identifiers and coordinate availability. Cell identities and measured expression are excluded from the generator and scoring. Preserve the actual cell positions and simulate two count vectors for each scenario on the same graph.

Each matched replicate contains focal opposing, focal concordant, unilateral, staggered, broad opposing, and independent patterns. A geometrically fixed curved interface, replicate phase, spatial background, cell depth and assay count regime are shared across the six scenarios. An exact binary truth mask identifies the focal opposing edges. Edges from all other scenes, including the broad opposing and staggered scenes, count as negatives for the focal-localization task. The complete graph is evaluated. A separate supplied-interface task ranks the one focal pair among the six candidates using its mean score along the same target edges. That task does not represent blind boundary detection.

## Methods and reporting

Use the same 12-neighbor, 60-micrometer graph, exposure normalization and four lazy diffusion steps for all expression-based competitors. Test plain J, regularized J at a fixed ridge of one, signed and unsigned gradients, joint gradient energy, negative codispersion, negative local Lee/Moran/Pearson/cosine, and edge length alone. Include a zero-diffusion sensitivity analysis. Use 16 matched seeds per count regime and geometry. No tuning against the held-out outcomes. Report all per-replicate scores, both geometry summaries, and the paired median change from plain to regularized J. Do not treat two coordinate donors or simulation replications as biological validation.

The real atlas and the synthetic truth answer different questions. Use the existing DLPFC, CRC, and two-animal atlas analyses as real-data feasibility and limitations. Their anatomy and taxonomy identify tissue compartments and cells. They do not supply a gold standard for bivariate molecular opposition or cell–cell signaling.

For runtime, report graph, normalization and I/O exclusions, hardware, number of tested pairs and warm-repeat median for optimized J and Lee. The dense all-pair kernel is compared with another optimized dense all-pair kernel on the same input. Do not equate descriptive score computation with a permutation-based significance test.

## References

- Saito and Rehmsmeier, PLOS ONE 2015. https://doi.org/10.1371/journal.pone.0118432
- Zhang et al., Nature 2023. https://doi.org/10.1038/s41586-023-06808-9
- Rukhin and Vallejos, Statistics & Probability Letters 2008. https://doi.org/10.1016/j.spl.2007.11.017
