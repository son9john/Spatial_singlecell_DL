# Connected seeds, a fixed discovery panel, and alternative boundary scores

2026-11-23, 00:01–00:13 EST. Exploratory follow-up to manuscript v14 and the first-elbow pilot. [Analysis decisions and timing](DESIGN.md) · [Four-panel comparison](benchmark.pdf).

## Methods

- **Connected seeds.** Compute the existing one-step ridge-1 local J across edges, then average over incident edges at each site. Find the first elbow among positive site scores. On that candidate subgraph discard components of fewer than three sites. Apply the prior distance decay to the entire scoring graph. For seedless cases the scorer uses uniform weights.
- **Fixed map-level seeds.** Compute local J across all 496 distinct pairs from 32 discovery genes and freeze the mean map within the specimen. Compare the signed mean with a mean of positive opposition. Test genes are disjoint from discovery genes. In grid simulations the discovery panel is either amplitude 1 or amplitude matched to the scored genes, using a separate random-noise draw and the same geometry. B2 uses a separate amplitude-matched discovery panel. Observed figures use different genes from the same tissue, so common anatomy and measurement artifacts can still affect both.
- **Wavelet contrast.** Apply the one-step minus the 16-step graph heat operator to both gene maps and calculate negative normalized overlap. This is a bandpass construction inspired by spectral graph wavelets. [Hammond et al.](https://arxiv.org/abs/0912.3848)
- **Resolvent J.** Apply the screened operator (I + 4Δt L) inverse to both maps, then compute boundary J. Here L is the graph Laplacian and Δt is the package's explicit diffusion step. [Smola and Kondor](https://www.cs.columbia.edu/~risi/papers/SmolaKondor.pdf)
- **Anisotropic J.** Freeze edge conductances from the 32-gene panel, slowing smoothing across larger gradients. Smooth both test signals on that graph for four steps, then measure opposition on the original graph. [Perona and Malik](https://authors.library.caltech.edu/records/1p8h5-5x870)
- **Unitary phase contrast.** Compute (exp(−i t L) − I) applied to each signal, using t = π / maximum degree. Take the negative normalized real Hermitian overlap. Algebraically, its spectral weight is 2[1−cos(tλ)]. The implementation uses ordinary sparse matrix products on a classical computer. [Farhi and Gutmann](https://doi.org/10.1103/PhysRevA.58.915)

## What the results say

With focal signals of amplitude 0.10 and 0.25, local J edge AP is 0.0098 and 0.0117. Connected seeds bring these to 0.0100 and 0.0133. A fixed positive map learned from a clean discovery panel raises AP to 0.0189 and 0.0242. With an equally weak discovery panel, the corresponding APs are 0.0056 and 0.0059. The weak panel signed-mean variant gives 0.0074 and 0.0123. This comparison changes only the discovery panel. A real improvement from fixed weights depends on a discovery map that carries the relevant boundary. Geometry-blocked intervals and all amplitudes are in [paired_simulation_intervals.csv](paired_simulation_intervals.csv).

At β = 0.5 on 12 cortical layouts (five replicates per section), wavelet contrast has the highest adjacent-versus-distant AUROC of the new physics-derived filters, 0.568, versus 0.543 for J₁, 0.606 for J₄, 0.640 for Lee's L, 0.657 for archived Hotspot, and 0.662 for regional J. Original J₁ AUROCs reproduce the archived B2 values for every section and replicate. The phase operator improves annotated edge AP in each of the three seqFISH embryos (mean 0.0336 → 0.0398), while cortex AP falls (0.0201 → 0.0185). Anisotropic diffusion does the reverse: cortex 0.0201 → 0.0224 in all three donors, seqFISH 0.0336 → 0.0295. These changes concern local boundary ranking, with the same selected markers and reference edges. Recomputed local references on the same 916 pairs were stronger in both cohorts: LIANA+ cosine AP 0.0281 in cortex and SpatialDM local Moran AP 0.0538 in seqFISH. The weakest B2 localization AP was 0.0165 for J₁, 0.0164 for clustered seeds, 0.0160 for fixed signed seeds, and 0.0195 for anisotropic J.

## The calibration question

In 720 unrelated simulated pairs across three spectral noise settings, the unweighted J₁ mean score was −0.0014. A first-elbow map learned and evaluated on the same measurement gave +0.1160; connected components reduced it to +0.1057. If those same-pair weights were frozen and reused in randomization, the two-sided rejection rate at 5% was 12.9% for the first elbow and 11.1% for the clustered version. Rerunning seed selection within each randomization brought the rejection rates to 5.0% and 4.6%. The observed scores remain shifted because those are the adaptive statistics being calibrated.

Independently fixed signed and positive panel maps give mean scores +0.0009 and approximately zero, with conditional Monte Carlo rejection rates 5.3% and 4.7% across regimes. The signed panel gave 4.2%, 8.3%, and 3.3% separately, with 240 pairs per regime. These are finite synthetic checks under a specified original-graph sign-symmetry model. They leave measured-tissue calibration open.

For each fixed seed map the weighted operator is S L_w S. This generally has nonzero off-diagonal entries in the original graph eigenbasis. The measured fraction of Frobenius norm off that diagonal was 0.373 and 0.268 for the signed and positive masks in the null pilot. Thus the old diagonal variance formula is insufficient. The Monte Carlo test here randomizes one map's signs in the original basis and recomputes the complete weighted score, including the denominator. The fast normal comparison recomputes the coefficient vector for the **weighted numerator**, preserving the original eigenbasis and calculating the correct conditional numerator variance. It gave 4.6% and 4.2% in the same pooled controls. For the bandpass, resolvent and unitary methods the operator remains diagonal in the original graph basis, which this run checked against their direct scores.

## Reproducibility

Use Python with NumPy, SciPy, pandas 2.2+, scikit-learn and Matplotlib. Run from the parent of J-analysis. The earlier seed experiment and archived v5 package/data remain input dependencies.

```sh
run=J-analysis/revisions/clustered_fixed_seeds_2026-11-23_000100_EST
python "$run/experiment.py" J-analysis "$run/results" simulation
python "$run/experiment.py" J-analysis "$run/weak_panel" simulation-weak-panel
python "$run/experiment.py" J-analysis "$run/results" b2
python "$run/experiment.py" J-analysis "$run/results" observed
python "$run/experiment.py" J-analysis "$run/phase" simulation --phase-only
python "$run/experiment.py" J-analysis "$run/phase" b2 --phase-only
python "$run/experiment.py" J-analysis "$run/phase" observed --phase-only
python "$run/experiment.py" J-analysis "$run/results" null
python "$run/local_references.py" J-analysis "$run/results"
python "$run/report.py" "$run" J-analysis/results/v3/gold_b2_semisynthetic.csv
```

[Grid localization](simulation_edge_summary.csv) · [Weak B2 ranking](b2_summary.csv) · [Observed pair ranking](observed_pair_summary.csv) · [Observed edge AP and IoU](observed_edges_summary.csv) · [Synthetic null](null_summary.csv).
