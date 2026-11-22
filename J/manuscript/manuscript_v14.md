---
title: "The juxtaposition index J maps where spatial expression domains meet, with calibrated tests for every gene pair"
---

Junho Song^1^, Jihyun Kim^2^, Jaesung Yoo^3^, Dongjoo Lee^4^, Kwangsoo Kim^5,6,\*^

**Manuscript v14 · Revised 22 November 2026 at 18:25 EST**

^1^ Penn State College of Medicine, Hershey, Pennsylvania, USA

^2^ Department of Statistics, University of Georgia, Athens, Georgia, USA

^3^ Neuroscience Center, University of North Carolina at Chapel Hill, Chapel Hill, North Carolina, USA

^4^ Lunit Inc., Seoul, Republic of Korea

^5^ Department of Transdisciplinary Medicine, Seoul National University Hospital, Seoul, Republic of Korea

^6^ Department of Medicine, Seoul National University College of Medicine, Seoul, Republic of Korea

^\*^ Corresponding author: Kwangsoo Kim

**Keywords:** spatial transcriptomics; gene-pair association; tissue boundaries; Lee's L; spatial autocorrelation; Moran spectral randomization; codispersion; Jaccard index; benchmark

# Abstract

Spatial transcriptomics studies routinely ask whether two genes are spatially associated. The statistics in common use, from Pearson's r to bivariate Moran's I and Lee's L, ask whether two genes occupy the same places. They cannot separate expression domains that meet from domains that lie apart, and their default tests treat neighbouring spots as independent. We introduce the juxtaposition index J, which reads changes instead of levels. Boundary J measures whether one gene rises where the other falls across the same tissue edges. It is bounded between −1 and 1, decomposes into a signed map of edge contributions and comes with an analytical spectral null that preserves each gene's spatial autocorrelation. Regional J, its companion, measures shared neighbourhood expression. On 15,000 cross-donor gene pairs that share no biology, the default tests of seven established methods called 45–88% significant at α = 0.05; the spectral test of J called 5.3%. Under that calibrated null, J called 87% of 44,850 colorectal-tumour gene pairs at q < 0.05, against 0.4% for Lee's L; its most opposed pairs joined epithelial and stromal genes, and its edge map outlined their interface. For strong exclusive markers simulated on 12 measured cortical layer maps, four-step J ranked adjacent above distant layers with AUROC 0.995, against 0.802 for the best co-localization statistic, and its edge maps located the interface better than the local scores of LIANA+ and SpatialDM. The top pairs of J overlapped none of those of the co-localization methods (Jaccard index 0.00), and adding J to a co-localization statistic improved held-out ranking of touching-but-distinct cortical layers in 12 of 12 sections. For anatomical proximity of measured markers, a co-localization question, regional J matched the best established statistic (AUROC 0.805). All 8 million pairs of 4,000 genes take 1.5 s on one CPU thread, with calibrated p-values for every pair in 15 s.

# Introduction

Spatial transcriptomics measures gene expression at known tissue positions [1,2]. Pearson correlation relates expression levels of two genes at the same sites. Spearman correlation relates their ranks. Both can be computed on expression matrices prepared in Scanpy or Seurat [23,25]. Squidpy and Seurat also provide single-gene spatial autocorrelation analyses [24,25]. For pairs of genes, bivariate Moran-type statistics relate the expression of one gene to neighboring values of the other, while Lee’s L compares spatially lagged expression [3]. SpatialDM, LIANA+, MERINGUE, Hotspot and Giotto implement related pairwise analyses with different graphs or weighting rules [6–8,26,27]. Our October 2021 Visium code calculated the self-associations of both genes, $L_{xx}$ and $L_{yy}$, alongside their cross-association, $L_{xy}$ [5]. This led us to examine a particular pattern. Both markers retain spatial structure after neighboring spots are averaged, yet their averaged maps have little overall alignment. The remaining question is whether the expression of one gene gives way to that of the other at a shared edge.

A high score can also arise when two independently varying genes each occupy a few broad regions. An independence test that counts every spot as a separate observation, or freely permutes spot labels, can underestimate that variability [9–14]. A second question concerns local changes. Two genes expressed in complementary domains may increase and decrease across the same interface. Their whole-section correlation can resemble that of two genes expressed far apart. Codispersion and the cross-variogram describe such spatial increments [15].

Here we develop that observation into the juxtaposition index J. Boundary J reads changes rather than levels. Across every edge of the tissue graph it compares the change in one gene with the change in the other, and it is positive when one rises where the other falls. It uses the same inputs as Lee's L and the same symmetric matrix product to compute, but it weights the sharp, high-frequency part of the spatial signal that Lee's L discards. Because it is a sum over edges, it also decomposes into a map that shows where on the tissue two domains meet. Regional J, its companion, adds the two ends of each edge instead of differencing them and so measures shared neighbourhood expression. We pair both scores with a spectral sign-flip null that keeps each gene's spatial autocorrelation. We then compare them with the statistics and default tests of SpatialDM, LIANA+, Hotspot, MERINGUE, Giotto, spdep and the Pearson and Spearman correlations of Scanpy and Seurat workflows (Table 3).

A measured gene pair has no universal binary label for spatial association, so each endpoint has an explicit reference. Independent simulated fields supply known negatives for testing. Annotated anatomy defines positive and negative classes for ranking. Simulations on the observed geometry supply known interfaces for localization. The written benchmark protocol designated cortical layer-pair ranking as the primary accuracy endpoint. The regional extension and additional accuracy comparisons were developed after reviewing the original observed-tissue results. We evaluated that fixed extension on additional MERFISH sections.

# Results

Table 1 summarizes the benchmark. Each row compares J with the strongest other score on the same endpoint, and bold marks the better value.

**Table 1. The benchmark at a glance.** Bold marks the best value in each row; for false-positive rates, the value closest to the nominal 5%. J~s~ is boundary J after s diffusion steps (one step unless stated). The Supplementary Tables list every method.

| Endpoint (reference) | J | Best other score |
|:----------------------------------------|:------------------|:----------------------------|
| *No association exists: % called significant at α = 0.05 (target 5%)* | | |
| 15,000 cross-donor gene pairs, default tests | **5.3** | 44.7 (Hotspot Z, one-sided) |
| Same pairs, every statistic under the spectral null | **5.3** | 3.6 (LIANA+ R) |
| Smoothest independent random fields | **7.8** | 44.3 (Lee's L permutation, one-sided) |
| *Power under the calibrated null (colorectal tumour; descriptive)* | | |
| Gene pairs called at q < 0.05 (%) | **87** | 21 (SpatialDM R); Lee's L 0.4 |
| *Simulated exclusive markers on 12 measured cortical layer maps (AUROC)* | | |
| Adjacent vs distant layers, strong markers (β = 2) | **0.995** (J~4~) | 0.949 (signed gradient Q); 0.802 (Hotspot Z) |
| Adjacent vs distant layers, moderate markers (β = 1) | **0.853** (J~4~) | 0.736 (Hotspot Z) |
| Adjacent vs distant layers, weak markers (β = 0.5) | 0.606 (J~4~) | **0.657** (Hotspot Z) |
| *Locating interface edges* | | |
| Simulated exclusive markers, IoU at true interface size (β = 2) | **0.295** | 0.209 (LIANA+) |
| Known MERFISH interfaces, average precision | 0.345 | **0.383** (gradient magnitude); 0.202 (LIANA+) |
| *Measured markers (AUROC)* | | |
| Touching-but-distinct vs other cortical layer pairs | **0.727** (J + Hotspot) | 0.684 (Hotspot Z) |
| Touching vs same-domain markers, cortex | 0.786 (J~4~) | **0.787** (Pearson, 16 steps, lower first) |
| Touching vs same-domain markers, embryo | 0.775 (J~4~) | **0.785** (Pearson, lower first) |
| Adjacent vs distant cortical layers | **0.805** (regional J) | 0.805 (matched-scale Pearson); 0.799 (Hotspot Z) |
| Contact-enriched vs depleted embryonic cell types | 0.777 (regional J) | **0.779** (Giotto) |
| *Ranking without labels (Jaccard index of top-1% pairs)* | | |
| Largest overlap with a co-localization method (lower = more distinct) | **0.00** | 0.62–0.95 |
| Reproducibility between serial sections | 0.12 | **0.62** (SpatialDM R) |
| *Compute, 4,000 genes on 3,611 spots* | | |
| All pairs, one CPU thread (s) | 1.51; 0.81 in float32 | **0.98** (Pearson r); 1.17 (Lee's L) |
| All pairs, Apple M4 Max GPU (ms) | 60.6 | **48.6** (Lee's L) |

## J reads where two structured maps meet

In the 2021 all-pairs implementation [5], $L_{xy}=\sqrt{L_{xx}L_{yy}}\,r_W$, where $r_W$ correlates the two centered neighborhood-average maps. Each self term is the centered sum of squares of a marker’s neighborhood averages divided by its original centered sum of squares. When both self terms are large, a near-zero $L_{xy}$ implies weak alignment of two individually structured maps. The average can be equally small when the maps meet along an edge and when they lie far apart.

Boundary J uses a different observation from each graph edge. After $s$ diffusion steps, let $\Delta x_e$ and $\Delta y_e$ be the changes in the two standardized expression profiles across the same edge, oriented identically. The boundary score is

$$J_s=\frac{-2\sum_e w_e\,\Delta x_e\,\Delta y_e}{\sum_e w_e\,(\Delta x_e^2+\Delta y_e^2)}.$$

A gene pair that changes in opposite directions across an edge makes a positive contribution. Changes in the same direction contribute negatively. The sum is bounded between −1 and +1 for nonzero total gradient energy. A value near zero can arise from little shared change or from opposing and concordant changes in different places. The edge map distinguishes these possibilities. Under graph heat diffusion, J also measures the initial rate at which covariance changes relative to the loss of the two maps’ spatial variation (Methods).

In a controlled example, both markers occupied eight sites on a 256-site cycle. Moving the second marker from a touching position to a distant one kept both self terms at 0.871 and absolute cross L at 0.032, while one-step boundary J changed from 0.500 to zero (Fig 1A). The same holds on a two-dimensional tissue. On a synthetic section with three domains, Lee's L was −0.34 for markers of touching domains and −0.36 for markers of separated domains. Boundary J was +0.20 and +0.04, and −0.18 for two markers of the same domain (Fig 1B,C). The reason is spectral. Lee's L weights the smooth part of the two maps, where touching and separated domains look alike. J vanishes at zero frequency and weights the sharp part, where a shared edge leaves its signature (Fig 1D). On a regular graph, unsmoothed J also equals the excess of neighbour cross-association over same-spot association, $(I_{xy}-r_{xy})/(1-\tfrac12(I_x+I_y))$, so it can be read directly against Pearson's r and bivariate Moran's I (Supplementary Results).

![**Fig 1. J reads where two structured maps meet.** (A) Two equal-width markers on a cycle either share an edge or have 24 unoccupied sites between them. Both arrangements have the same self L (0.871) and the same absolute cross L (0.032); boundary J is 0.500 for the shared edge and zero for the separated domains. (B) A synthetic 40 × 40 tissue with three domains. Genes A, B and C mark one domain each, and a second gene, A′, also marks domain A (noise SD 0.3). (C) Lee's L is negative and nearly equal for the touching (A–B) and separated (A–C) pairs. Boundary J at two diffusion steps is positive only for the touching pair and negative for two markers of the same domain. (D) Relative weight that each statistic places on graph frequency, computed on the Laplacian spectrum of DLPFC section 151673. Pearson's r weights all frequencies equally, Lee's L the smooth end and boundary J the sharp end.](figures/figure1_concept_2026-11-22_175222_EST.png){width=6.4in}

## Default tests call most unrelated pairs significant; the spectral test of J is calibrated

A test is calibrated when, at α = 0.05, it calls about 5% of truly unrelated pairs significant. To build pairs that cannot share biology, we paired the 50 most spatially autocorrelated genes in one DLPFC section with the 50 in a section from a different donor [2]. Each of six section pairs contributed 2,500 gene-pair observations. In the aligned control, we compared expression at matching Visium array positions. For the rotated control we turned the second coordinate grid by 180° and retained spots present in both sections (3,093–3,631 matched positions). Rotation disrupts registration at identical array addresses while preserving the spatial pattern within each original section. Shared anatomy or technical trends can still produce associations in the overlap. These are empirical controls. The independently generated fields below provide the known-null comparison.

Every pair called significant in the rotated control is therefore a false positive, and a calibrated test would call about 750 of the 15,000. Figure 2A counts how many each procedure called. Pearson’s two-sided t-test called 11,492 (76.6%), the two-sided LIANA+ permutation test 13,175 (87.8%) and the one-sided Lee permutation test 6,740 (44.9%). Spearman and MERINGUE called 76% and 85%. SpatialDM and Hotspot called 45–47%, close to the 50% ceiling of a one-sided test. These percentages count test decisions. The 15,000 observations reuse genes and sections.

The two-sided spectral test for one-step boundary J called 794 rotated-control pairs (5.3%) and 1,594 aligned-control pairs (10.6%). When the comparator statistics used the spectral null, 3.0–3.6% of rotated pairs and 4.3–5.2% of aligned pairs were called significant (Fig 2B and Table S1). Their conventional tests rejected 71–85% of aligned pairs for two-sided tests and 39–42% for one-sided tests. The conditioning model and the alignment both affected the result.

The paired slides showed concordant spot-level library-size patterns at matching array positions. That signal largely disappeared after rotation. On independently generated smooth fields, one-step spectral J rejected 4.7–7.8% across six settings, within the 2.5–7.5% tolerance interval [18] in five. For the smoothest fields, conventional two-sided tests rejected 89–90%. Supplementary Results give the full calibration analysis.

![**Fig 2. Default tests call most unrelated pairs significant; the spectral test of J is calibrated.** (A) False-positive rate of each method's default test. Each of 15,000 gene pairs joins a gene from one donor's section with a gene from a different donor's section turned by 180°, so no pair can be truly associated and every pair called significant (unadjusted p < 0.05) is a false positive. The dashed line marks the nominal 5% and the band marks 2.5–7.5%. Wilson intervals treat pairs as independent and are descriptive because genes and sections recur. Daggers mark one-sided tests, whose ceiling is 50%. (B) The same pairs tested under the spectral null. (C) False-positive rates on independently generated fields. Greater range gives smoother maps. (D) Percentage of 44,850 colorectal pairs called at Benjamini–Hochberg q < 0.05. Biological truth is unknown here. (E) Rotated-control rejection rate against one-thread time to obtain p-values for 4,000 genes. Permutation times are extrapolated. Table 3 specifies the comparator procedures.](figures/figure2_calibration_2026-11-22_175222_EST.png){width=6.15in}

## Under the calibrated null, J keeps power where Lee's L loses it

Spatially smooth maps can align by chance, and a calibrated test must allow for it. In the colorectal section the spectral analysis found 0.4% of Lee's L associations, 18% of Moran's I, 14% of LIANA+ R and 21% of SpatialDM R at q < 0.05. For J the fraction was 87%. The scores weight different parts of the spatial spectrum. A median of 94% of each gene's variance lay in the lower half of the graph spectrum. The null variance of Pearson's r was carried by a median of 2.1 effective spectral modes (participation ratio), and that of a Lee-type smoothing filter by 2.0. For J it was 51 at scale 1. Correlating increments instead of levels is the spatial analogue of differencing time series to avoid spurious regression [19].

The most opposed pairs paired epithelial genes (EPCAM, ELF3, CEACAM5, KLF5, CLDN7, LGALS4) with stromal genes (MGP, C1S, C1R, COL1A1, COL3A1, VIM, MMP2). For MGP–EPCAM, boundary J was +0.539, indicating opposite changes across neighboring spots. The conditional spectral test gave z = 13.25 and an approximate two-sided p of $4.6 \times 10^{-40}$. After adjustment across 44,850 pairs, q was $3.4 \times 10^{-39}$ and the pair met the 0.05 discovery threshold. Lee’s L was −0.877 for the same measured maps, with spectral p = 0.0177 and q = 0.0658. Its large negative effect did not meet that adjusted threshold. The edge-level map of boundary J outlined the epithelial–stromal interface (Fig 3C–E). Removing each gene's library-size component in the gradient domain left the proportion of significant J pairs essentially unchanged (87% before, 87% after). This analysis is descriptive, because the section has no independent boundary annotation.

![**Fig 3. Under the calibrated null, J keeps power where Lee's L loses it.** Colorectal cancer section GSM8594561. (A) Lee's L against J for all pairs of the 300 most autocorrelated genes; circles mark examples. (B) Absolute spectral-null z-scores of Lee's L and J for the same pairs; lines mark |z| = 1.96. (C,D) Log-normalized expression of EPCAM and MGP (colour capped at the 98th percentile). (E) Local J (scale 1, ridge 1) averaged over the edges of each spot; red, opposing; blue, concordant.](figures/fig4_crc_2026-11-22_175222_EST.png){width=100%}

## Boundary J recovers adjacent exclusive domains

We generated counts for layer-restricted markers on the 12 measured DLPFC layer maps, using the observed spot library sizes. The log-mean effect β multiplies a layer indicator softened over neighboring spots. From indicator 0 to 1, β = 2 gives a factor of $e^2$ (about 7.4) in expected intensity at equal depth and background. Graded boundary spots have intermediate effects. The simulated genes retain background expression. By contrast, the subscript in $J_4$ denotes four diffusion steps applied to the resulting expression before scoring. β controls signal generation and $s$ controls analysis. The observed gene panels have no assigned β.

For adjacent versus distant layers, four-step J had mean AUROC 0.606, 0.853 and 0.995 at β = 0.5, 1 and 2 (Fig 4D). Hotspot led at β = 0.5 (0.657). At β = 1 the strongest co-localization score, Hotspot, reached 0.736. At β = 2, four-step J exceeded the strongest non-J score in each of the 12 section means. At that effect size, two-step J reached 0.990 and one-step J reached 0.94881, close to negative codispersion (0.94892) and signed gradient Q (0.94851). Hotspot was the strongest co-localization score (0.802). Four-step J and several co-localization scores oriented with lower values first reached 1.000 for adjacent versus same-layer pairs. Figure 4A–C illustrates the strongest simulated effect. The scale comparison and the emphasis on β = 2 were exploratory. Table S13 gives all method-by-effect combinations.

![**Fig 4. Boundary J recovers adjacent exclusive domains.** Pair-level accuracy with simulated exclusive markers on 12 measured DLPFC layer maps. β is the log-fold effect of a marker in its own layer (β = 2, about 7.4-fold); J~s~ is boundary J after s diffusion steps. (A) All methods at β = 2 for adjacent versus distant pairs in each section (points) and across sections (diamonds). (B) Paired section-level comparisons of J at scales 1 and 4 against the best non-J method in the same section (dashed line, equal AUROC). (C) Adjacent versus same-layer pairs at β = 2 for selected scores. A minus sign denotes lower co-localization ranked first. (D) Adjacent versus distant AUROC across β = 0.5, 1 and 2 on a base-2 axis. Each section contributes five simulations per effect.](figures/figure4_pair_accuracy_2026-11-22_175222_EST.png){width=6.5in}

## Signed edge maps locate the interface

For each edge, local boundary J pools changes on neighboring edges and adds a ridge to stabilize windows with little expression change. In 64 known-interface simulations on observed MERFISH coordinates, regularized J exceeded LIANA+ weighted Pearson on average precision in every paired replicate. Scores based only on the magnitude of gradients were competitive or stronger (Fig 5A). In simulated exclusive cortical markers, intersection over union at the true interface size reached 0.19 and 0.29 at β = 1 and 2. At β = 2 the strongest tested package score reached 0.21 (Fig 5B). These comparisons quantify localization when the target is known.

Measured whole-section boundary detection was harder. On the original cortical and embryo marker panels, mean interface overlap ranged from 0.000 to 0.022 across methods (Fig 5C). A signed boundary map is most informative when interpreted with the measured expression and tissue anatomy, as in Fig 10. Supplementary Tables S3, S5–S6 and S9–S12 report local controls, alternate metrics and the measured MERFISH pilot.

![**Fig 5. Signed edge maps locate the interface.** (A) Known truth on two held-out MERFISH geometries: median average precision over four strata (16 replicates each) for edges crossing a focal opposing interface; LIANA+ and SpatialDM scores from the official packages. (B) Jaccard index (IoU) at the true interface size for exclusive markers simulated on the real DLPFC layer maps (12 sections, 2 replicates, adjacent-layer boundaries). (C) The same metric for real marker pairs against expert-annotated DLPFC layer boundaries and seqFISH cell-type contacts; dashed and dotted lines, chance. (D) Example: the annotated L6–white-matter edges and the top-|T| edges of regularized J and of SpatialDM's local score.](figures/fig5_local_jaccard_2026-11-22_175222_EST.png){width=100%}

## J carries information that co-localization statistics do not

To compare rankings without labels we used Jaccard indices on the same 300-gene panel per donor. The co-localization statistics largely measure one property. Within a section, the top 1% of pairs of SpatialDM, LIANA+, MERINGUE and Lee’s L overlapped 0.87–0.95 with one another and 0.67 with Pearson's r. The top pairs of J overlapped none of them (≤ 0.00), and concordant J (most negative) overlapped them weakly (0.09–0.18; Fig 6B).

That information is complementary. Touching but distinct domains should co-localize at the scale of a neighbourhood and change in opposite directions across their shared edges. We fitted a logistic model on within-section percentile ranks of a co-localization statistic, J and their product, on two donors, and evaluated it on the third. With distant-layer and same-layer pairs pooled as negatives, the combined model reached AUROC 0.727 with Hotspot (0.684 alone) and 0.722 with Lee’s L (0.677 alone). It improved on the co-localization statistic in 12 of 12 sections for every co-localization statistic tested. In the seqFISH embryos, where markers of touching types overlap more, the gain was negligible (0.74 against 0.74).

Reading sharp transitions has a cost in reproducibility. Between serial sections of the same donor, 10 µm or 300 µm apart, co-localization statistics shared 0.51–0.62 and 0.45–0.47 of their top 1% of pairs (chance 0.005). J shared 0.12–0.18, far above chance but much less, and its overlap rose with the smoothing scale, consistent with spot-level noise in its high-frequency components (Fig 6A).

![**Fig 6. J carries information that co-localization statistics do not.** Jaccard overlaps of top-ranked pairs. (A) Reproducibility: Jaccard index of each method's top 1% of the 44,850 pairs of a donor's 300-gene panel between serial sections 10 µm apart (solid) and 300 µm apart (light); six section pairs each; dashed line, chance. (B) Redundancy: mean Jaccard index between methods' top-1% pairs within a section (12 sections).](figures/fig6_jaccard_repro_redundancy_2026-11-22_175222_EST.png){width=100%}

## All pairs at the cost of Lee's L

J needs one Gram matrix $G=\tilde X^{\top}\mathcal{L}\tilde X$. A general matrix product costs twice as much as the symmetric product $(W\tilde X)^{\top}(W\tilde X)$ behind Lee's L. Because $\mathcal{L}$ is positive semidefinite and sparse, the package factors $\mathcal{L}+\varepsilon I=R^{\top}R$ once per section with a symmetric sparse LU (ε = 10^−13^ d~max~). It then forms $G=(R\tilde X)^{\top}(R\tilde X)$ as a symmetric rank-k update, the same BLAS operation as Lee's L. The relative error in G is about 10^−13^. The spectral test uses the same device: the filter h is non-negative, so both the J Gram and its null variance are symmetric rank-k updates.

On 3,611 spots and one CPU thread, all 7,998,000 pairs of 4,000 genes took 1.51 s for J, 0.81 s in float32, 1.17 s for Lee's L, 2.02 s for bivariate Moran's I and 0.98 s for Pearson's r, against 2.37 s for the previous J kernel. J with approximate spectral p-values and Benjamini–Hochberg q-values for every pair took 6.7 s (4.7 s in float32) after a one-time eigendecomposition of 7.8 s, whereas the permutation tests of the ecosystem packages take an extrapolated 3–40 minutes (Fig 7). On an Apple M4 Max with custom Metal kernels through MLX, all pairs of 4,000 genes took 60.6 ms for J and 48.6 ms for an equally optimized Lee's L, against 486.9 ms for our original 2021 implementation on the CPU. The largest float32 deviation from the float64 J was 1.25 × 10^−6^. Regional J, which averages five scales, took 2.19 s on the CPU of the same machine and 351 ms on its GPU.

The Python and R packages implement both scores. For a two-column matrix of normalized expression aligned to the rows of a spatial graph, the Python calls below return the pair’s regional concordance, its signed boundary score and a local edge map. The package also accepts precomputed increments through `juxtaposed_value`.

```python
import juxtaposed as jx

regional = jx.regional_juxtaposition_matrix(pair, graph)[0, 1]
boundary = jx.juxtaposition_score(pair[:, 0], pair[:, 1], graph)
edges = jx.score_edges(pair, graph, scale=1, ridge=1)
```

The rows of `edges.endpoints` identify the neighboring sites and `edges.j` gives their local scores. Boundary and regional all-pairs calculations also accept accelerator backends. CPU is available across platforms. The tests and hardware used for the timing comparisons are given in Methods and Table S8.

![**Fig 7. All pairs at the cost of Lee's L.** (A) All-pairs statistics for 4,000 genes on 3,611 spots from raw values (one and two CPU threads; median of three warm runs); 'v0.2 kernel' is the previous general-product J. (B) Time to p-values for all pairs on one thread: J with the spectral null (including the one-time eigendecomposition) and each ecosystem default test; permutation tests are extrapolated from one all-pairs evaluation. Calibration is shown in Fig 2.](figures/fig7_speed_2026-11-22_175222_EST.png){width=100%}

## Choosing between the two scores

The two juxtaposition scores answer different questions (Table 2). Regional J asks whether two genes share neighbourhoods. Boundary J asks whether they change in opposite directions across the same edges, and where.

**Table 2. Choosing a juxtaposition score from the biological question.**

| Question about a pair of genes | Score and readout | Observed example |
|-------------------------|------------------------------|----------------------------|
| Do the markers occupy the same neighborhoods? | Regional J averages concordance across five spatial scales. Positive values indicate shared regional expression or depletion. | Eng and Aplnr in the mouse embryo, +0.416 (Fig 11). |
| Does one marker rise where the other falls? | Boundary J measures signed changes across the same graph edges. Positive values indicate net opposition. Inspect the edge map to locate the contributions. | CST3 and PPP3CA in human cortex, +0.405 on the annotated layer 1–2 interface (Fig 10). |
| Both genes are spatially patterned, but cross L is near zero. Do their domains meet? | Inspect both self terms, then evaluate signed boundary J and the expression maps at a stated scale. | Touching and separated domains with identical self and cross L (Fig 1A). |

## Use regional J for shared neighborhoods and anatomical proximity

Markers assigned to adjacent layers or contacting cell types can share a broad expression domain. In 6,400 cortical and 450 embryo positive pair observations, the annotated interface accounted for only 1.66% and 1.20% of total gradient energy, respectively (means of within-specimen medians). For those pairs, ordinary same-site Pearson correlation was positive in 83% of cortical and 78% of embryonic observations. Anatomical proximity of measured markers is therefore a co-localization question, and regional J is the score for it (Table 2).

Regional J uses the sum of expression at the two ends of each edge and averages concordance over 1, 2, 4, 8 and 16 diffusion steps. In human cortex, it ranked adjacent above distant layers with mean AUROC 0.8054. Matched-scale Pearson reached 0.8048 and Hotspot 0.7988. The two smoothed scores captured much the same signal. In embryos, regional J reached 0.7773 for contact enrichment and 0.7451 for high versus low contact rates. Giotto reached 0.7786 for enrichment and Hotspot reached 0.7606 for contact rate (Fig 8). These comparisons use markers chosen in other specimens for the cortical and embryo enrichment endpoints. On four additional MERFISH sections from two mice, anatomical transfer was weak for every score, including regional J (0.543) and Hotspot (0.547; Table S14 and Supplementary Results).

![**Fig 8. Use regional J for anatomical proximity.** Regional J and comparator scores for anatomical proximity in measured expression. (A) Adjacent versus distant cortical layers. (B) Embryonic cell types with enriched versus depleted contact. (C) High versus low cell-type contact rate, with markers selected within each embryo. Open points represent specimens and diamonds are equal-weight means. Higher scores predict proximity. The matched Pearson control uses the same graph and five diffusion scales as regional J. The complete comparator panel appears in Fig S1 and Tables S2 and S14.](figures/figure8_observed_accuracy_2026-11-22_175222_EST.png){width=6.35in}

## Use boundary J for opposing transitions and for touching versus same-domain markers

Changing the comparison group changes the biological question. In Fig 9, positives are pairs of markers assigned to distinct adjacent cortical layers or to distinct embryonic cell types with enriched contact. Negatives are two different genes marking the same layer or the same type. Each specimen contributed its own rankings. The 12 cortical sections yielded 6,400 adjacent and 3,420 same-layer pair observations. The three embryos yielded 450 enriched-contact and 149 same-type observations. One-step boundary J gave mean AUROC 0.727 in cortex (section range 0.687–0.770) and 0.770 in embryos (range 0.736–0.815). An AUROC of 0.727 means that, averaged over sections, an adjacent-layer pair ranked above a same-layer pair in about 73% of such comparisons, giving ties half credit.

The median one-step boundary J was negative in both classes in all 15 specimens. Touching pairs tended to have weaker concordance across edges, so this ranking did not require a positive J. In the exploratory four-step comparison, boundary J reached mean AUROC 0.786 in cortex and 0.775 in embryos (Fig 9C,D). Negative codispersion at the same scale differed from J by less than 0.001 in each cohort. Co-localization scores reach the same task in reverse, because markers of the same domain co-express strongly: ranked with lower co-expression first, four-step Pearson reached 0.777 in cortex and 0.773 in embryos, 16-step Pearson 0.787 in cortex and unsmoothed Pearson 0.785 in embryos. Boundary J came within 0.001 of the best of them in cortex and 0.010 in embryos without reversing its sign, and it adds the edge map. Table S15 shows the effect of score direction and diffusion scale.

![**Fig 9. Identifying markers of touching versus shared anatomical domains.** (A, B) ROC curves for one-step boundary J in 12 human cortical sections from three donors and three mouse embryos. Gray lines show individual specimens. The red line averages their true-positive rates at each false-positive rate. (C, D) AUROC for each method on the same pairs. Open circles show specimens and diamonds show their equally weighted means. Arrows indicate which scores rank first. Higher boundary J or negative codispersion predicts touching, as does lower co-expression. Cortical positives mark adjacent layers. Embryonic positives mark distinct cell types with a contact enrichment z score of at least 3. Negatives mark the same layer or cell type. Markers were selected in other donors or embryos.](figures/figure9_boundary_method_auroc_2026-11-22_175222_EST.png){width=6.35in}

## How to read both scores on the same tissue section

The cortical maps show the difference on the same section. CST3 and PPP3CA were selected using expression from the other two donors. On the 204 edges that cross between layers 1 and 2 in the illustrated section, one-step boundary J reached +0.405. The published layer labels identify these crossing edges. They carried 2.26% of the genes’ total gradient energy, so the map illustrates the contribution at a known interface. Whole-section boundary J was +0.052 and regional J was −0.358 (Fig 10). A second pair chosen in the training donors, NEFL and TUBB2A, broadly occupied the same territory. In the held-out cortex its regional J was +0.763 and boundary J was −0.108 (Fig S3). With the same section chosen on anatomy alone and the highest-ranked marker for each of the two most frequently contacting neighboring layers, MALAT1 and HPCAL1 gave weak whole-tissue scores on both channels (Fig S4).

![**Fig 10. Opposing cortical expression changes for CST3 and PPP3CA in human DLPFC section 151507.** A shows the layer annotation. B and C show the two measured expression profiles on the same spots. D enlarges the boxed region and marks the reference edges crossing layers 1 and 2. E and F show signed edge contributions to whole-tissue boundary J and regional J. Each edge color uses its own panel’s scale. Expression colors are capped for display only. Pair selection used the two other donors.](figures/dlpfc_training_boundary_observed_tissue_2026-11-22_175222_EST.png){width=6.35in}

Eng and Aplnr illustrate regional concordance in single cells. We chose the most frequently contacting annotated cell types among eligible positive type pairs in embryo 1, then chose one marker per type from the other embryos. The 724 edges connecting endothelium and haematoendothelial progenitors had a contact-enrichment z-score of 40.38. Eng and Aplnr occupied similar areas of the embryo. Regional J was +0.416, while boundary J was −0.216 and remained negative on the reference edges themselves (Fig 11). These maps locate shared expression and opposing transitions. Static tissue sections do not determine signaling direction or a causal interaction.

![**Fig 11. Shared regional expression of Eng and Aplnr in the seqFISH mouse embryo.** A shows the annotated cell populations in embryo 1. B and C show measured expression at cell coordinates. D shows graph edges connecting the annotated types in the boxed region. E and F show the contribution of each edge to the two global scores. In F a positive contribution can arise from shared elevation or depletion. Anatomical contacts are defined by the spatial graph.](figures/seqfish_observed_tissue_2026-11-22_175222_EST.png){width=6.35in}

# Discussion

J turns a 2021 observation into a statistic. Where Lee's L reads the smooth part of two maps, J reads their changes across shared edges, at the same computational cost and with an exact decomposition into edge contributions. Three properties follow from that change of view. First, an honest test: the default tests of established methods called most unrelated pairs significant, whereas the spectral test of J stayed near 5% on the rotated control and in most simulated settings, and the same null calibrated every other statistic. Second, information that co-localization does not carry: the top pairs of J overlapped none of theirs, its edge maps located simulated interfaces better than the local scores of LIANA+ and SpatialDM, and combining J with a co-localization statistic improved held-out ranking of touching-but-distinct layers in every cortical section. Third, power: under the same calibrated null J called 87% of colorectal pairs where Lee's L called 0.4%, consistent with the 51 effective spectral modes behind the null of J, against about 2 for Pearson's r and Lee-type smoothing.

The benchmark also shows where each score belongs. Anatomical proximity of measured markers is a co-localization question. Markers of adjacent layers share broad domains, and the interface carries under 2% of their gradient energy. There regional J matched the best established statistics. Touching versus same-domain markers is a contrast question, and boundary J came within 0.01 of the best reversed co-localization statistics while keeping its natural sign. Boundary J is most useful when the question is where and how two domains meet. Its edge map located the CST3–PPP3CA opposition on the layer 1–2 interface (+0.405) while the whole-section score was +0.052, and in the colorectal tumour it outlined the epithelial–stromal boundary. The 2021 observation, two structured maps with little overall alignment, describes exactly the pairs whose edge map is worth reading (Fig 1A).

Limitations. The spectral null assumes that each gene's graph-Fourier coefficients are exchangeable in sign. It therefore does not model spatially heteroscedastic noise or separate biological from technical association. The aligned-array control exceeded the protocol’s calibration limit for J, and shared library-size structure at matching array positions accompanied that excess. Computing the full spectral variance requires a dense eigendecomposition, which is practical up to about 20,000 spots; larger sections need binning or polynomial approximations. Because J reads sharp transitions, its top pairs were less reproducible across serial sections than those of co-localization statistics. The comparator statistics and tests were re-implemented from their published definitions and documented defaults. The local benchmark used the official packages, and a runner that repeats the calibration with the official packages is provided. The observed cortical and embryo benchmarks each included only three independent biological units. Cortex and embryo reference labels measure anatomical relations among marker assignments. Their recovery does not establish molecular binding, regulation or signaling. The additional MERFISH transfer tested two mice, with substantial variation in region coverage. The array-position signal was observed in slides from a single study. Expert labels in both atlases were informed by expression. Interface statistics describe association and do not establish cell–cell communication.

# Methods

## Data

DLPFC: raw filtered Visium count matrices, spot positions and manual layer labels for the 12 HumanPilot sections of three donors [2]. Spots in tissue with a layer label (L1–L6, WM) and a positive library size were retained. Colorectal cancer: Visium CytAssist section GSM8594561, all in-tissue barcodes [21]. seqFISH: the 28,980 cells of z-slice 2 in three embryos with normalized expression of 351 genes and refined cell-type labels [20]. MERFISH geometries: 5,000 cells nearest the coordinate median of one section from each of two animals (Zhuang-ABCA-3.012 and Zhuang-ABCA-4.002) [22]; only coordinates were used. Download URLs and checksums are recorded in the repository.

## Preprocessing and graphs

Counts were transformed as log(1 + 10^4^ × count / library size) and each gene was standardized within a section to mean 0 and population SD 1. Visium graphs connect first-ring hexagonal neighbours (binary weights). seqFISH graphs are the symmetric union of six nearest neighbours per cell. MERFISH graphs are the symmetric union of up to 12 nearest neighbours within 60 µm. Heat smoothing uses $S=I-\mathcal{L}/(2d_{\max})$, which is symmetric, doubly stochastic, positive semidefinite and a polynomial in $\mathcal{L}$. Local maps can instead use the lazy random walk $(I+D^{-1}A)/2$.

## Statistics

For standardized $x$, $y$ after $s$ smoothing steps, with $A=\tilde x^{\top}\mathcal{L}\tilde x$, $B=\tilde y^{\top}\mathcal{L}\tilde y$ and $C=\tilde x^{\top}\mathcal{L}\tilde y=\sum_e w_e\Delta\tilde x_e\Delta\tilde y_e$, the power-mean family is $J_{s,p}=-C/M_p(A,B)$. Here $M_1=(A+B)/2$ gives J, $M_0=\sqrt{AB}$ gives negative codispersion, and $|J_{s,p}|\le 1$ for $p\ge0$ by Cauchy–Schwarz. Because $E(x\mp y)=A+B\mp2C$, $J_{s,1}=[E(\tilde x-\tilde y)-E(\tilde x+\tilde y)]/[E(\tilde x-\tilde y)+E(\tilde x+\tilde y)]$. On a $d$-regular graph $x^{\top}\mathcal{L}y=dn(r_{xy}-I_{xy})$ and $x^{\top}\mathcal{L}x=dn(1-I_x)$, which gives the identity in Results. For the normalized Laplacian $\mathcal{L}_{sym}=I-D^{-1/2}AD^{-1/2}$, $x^{\top}\mathcal{L}_{sym}y=n(r_{xy}-I^{sym}_{xy})$ on any graph, with $I^{sym}_{xy}=x^{\top}D^{-1/2}AD^{-1/2}y/n$. Lee's L uses row-standardized weights $W$: $L=[n/\sum_i(\sum_j w_{ij})^2](W\tilde x)^{\top}(W\tilde y)/(\|\tilde x\|\|\tilde y\|)$ [3]. Bivariate Moran's I is $(x^{\top}Wy+y^{\top}Wx)/(2S_0)$ [4]. Q, the signed cross-gradient energy, is the numerator $-2C$ divided by the edge weight in the window.

Local J for edge $e$ pools the edges incident to either endpoint, so the focal edge counts twice. It is $J_e=-C_e/(M_p(A_e,B_e)+\lambda w_e m/2)$, where $w_e$ is the pooled edge weight, $m$ the section-wide mean joint gradient energy and $\lambda$ the ridge. Unless stated otherwise, local J used four lazy-walk steps and $\lambda=1$.

## Regional concordance, interface attribution and external transfer

Let $A$ be the symmetric weighted adjacency matrix and $D$ its degree matrix. The signless Laplacian $H=D+A$ [32] gives an edge-sum inner product. For centered, unit-variance signals $x,y$ and the same heat operator $S$ used for boundary J, write $u_t=S^t x$ and $v_t=S^t y$. The regional score is

$$J_{\mathrm{reg}}(x,y)=\frac{1}{5}\sum_{t\in\{1,2,4,8,16\}} \frac{2u_t^\top H v_t}{u_t^\top H u_t+v_t^\top H v_t}.$$

For an undirected edge $(i,j)$, the cross term is $a_{ij}(u_{ti}+u_{tj})(v_{ti}+v_{tj})$. Each scale is normalized by its own pair of self energies before averaging. Cauchy–Schwarz bounds the result between −1 and 1. Constant features and zero pair denominators yield an undefined score. Edge contributions use the global denominator for that pair and scale, so they sum to the reported whole-tissue score. The interface-restricted boundary score instead recomputes both numerator and denominator on annotation-crossing edges. The manuscript’s spectral P values apply to the original boundary score. No calibration experiment was performed for regional J. Signless graph energy and multiscale spatial co-occurrence both have established precedents [32,33].

The DLPFC and seqFISH marker lists, positive labels, negative labels and score orientations were inherited from the original observational analyses. The contact-rate extension used its original eight-neighbor weighted graph for regional J. Comparator-specific graphs are documented above. Scale-matched Pearson is the arithmetic mean of within-specimen correlations of $u_t,v_t$ on the same graph and at the same five steps. A site-only control replaced $H$ by $D$. Single-scale results are available for all five steps. These analyses were designed after examining the original results, although the new scores use no labels to fit weights. The reported unit is a section or embryo and the four sections per cortical donor are repeated measurements. We give observed differences at that level. The pair counts do not imply independent biological replicates.

For the Allen transfer, we selected the first two entries for each mouse among previously unselected sections of the archived inventory, which orders sections by anatomical edge count. The rule, eligibility thresholds, score family and evaluation criteria were written before scoring the transfer endpoint. Gene counts were normalized per 1,000 counts and log-transformed with an offset of one. Using two sections of the other mouse, we assigned each gene to its highest-mean CCF structure and selected up to three markers per structure by the excess of that mean above the runner-up. We required a positive excess, mean expression of at least 0.1 and at least 100 cells per eligible structure in training and test. Test truth counted undirected edges in the union of six-nearest-neighbor graphs within each section. Structure pairs with 1–9 connecting edges were omitted. We evaluated different-structure pairs only. For all four endpoints, retrieved gene pairs were the top 20% rounded up. Jaccard was the intersection divided by the union of retrieved and positive pairs.

For the illustrated DLPFC section we ranked 600 eligible adjacent-layer pairs in eight training sections, aggregating within each donor and then equally across donors. We display the highest training pair under each score in the first held-out accession. A separate anatomy-only illustration in each dataset took the highest cross-label edge count among eligible positive classes, followed by the highest-ranked external marker per class. The field of view maximized the number of target crossing-edge midpoints in a square spanning one quarter of the shorter tissue axis. Measured expression was displayed with a gene-specific cap at the 95th percentile of positive values. Edge colors were capped at each panel’s 98th percentile of absolute contributions. All computations used uncapped values.

Native CrossExpression 1.0.0 [16] used its first-nearest-neighbor graph on measured seqFISH cell coordinates. We evaluated its signed correlation, unadjusted cross-expression P value and adjusted P value on the frozen gene pairs. Smaller P values ranked first. P-value rankings and effect-size rankings are listed separately in the source tables.

## Interpreting a score, a test and an anatomical label

For a specified pair, the sign of boundary J describes the predominant direction of change across graph edges and its magnitude describes the normalized cross-gradient. To test the pair, we ask how often the conditional null would produce a statistic at least as extreme as the observed one in the specified tail. For boundary J we use two tails and unadjusted p < 0.05 for the calibration experiment. Values above the threshold give insufficient evidence to reject that null. For the colorectal screen, we apply Benjamini–Hochberg across the 44,850 tested pairs and call discoveries at adjusted p (q) < 0.05. The fraction called in that screen cannot estimate the fraction of true nulls rejected without known biological labels. Regional J is reported as an effect score.

The anatomical benchmark assigns positives before ranking pairs in each test specimen. The cortical labels come from histology and expression-informed layer annotation [2]. The embryo atlas provides annotated single-cell positions [20]. Permuting type labels across the fixed cell graph supplies the expected number and variability of contacts for each type pair. A contact z-score of at least 3 defines enriched contact and a z-score of at most −3 defines depleted contact. These z-scores label the types. They are distinct from the spectral z-score for the two expression maps. AUROC compares the ranks of the reference classes without a significance cutoff. Edge Jaccard compares predicted and reference edge sets as the number in both divided by the number in either, at the specified prediction budget.

## Historical lag components and the graph-diffusion interpretation

For the archived all-pairs implementation [5], let $P$ remove the column mean and let $W$ be the row-standardized neighbor matrix. For each nonconstant expression map $x$ define $a_x=PWx/\|Px\|$. Then $L_{xx}=a_x^\top a_x$, $L_{yy}=a_y^\top a_y$ and $L_{xy}=a_x^\top a_y$. The factorization in Results follows from the cosine of these two centered lag vectors. Centering the lag around its own mean reproduces the 25 October 2021 vectorized function and the later archived notebook. The dated code establishes computation of all three quantities in this project.

For the heat interpretation, take $u=S^s x$ and $v=S^s y$ on the symmetric graph with Laplacian $\mathcal L$. Add continuous diffusion of duration $h$, so $u_h=\exp(-h\mathcal L)u$ and likewise for $v_h$. Write $C(h)=u_h^\top v_h$, $V_x(h)=\|u_h\|^2$, and $V_y(h)=\|v_h\|^2$. Differentiating at zero gives $C'(0)=-2u^\top\mathcal Lv$ and $V'_x(0)=-2u^\top\mathcal Lu$. For nonzero total gradient energy,

$$J_s=\frac{2C'(0)}{-V'_x(0)-V'_y(0)}.$$

Thus, positive J means that an additional infinitesimal smoothing step increases the pair’s cross-product. Normalization expresses that increase relative to the variation lost from the two maps. The variable $h$ indexes spatial smoothing on the tissue graph. A centered finite-difference check using matrix exponentials agreed with the implemented J within $2.7\times 10^{-10}$ for 66 marker pairs at scales 1 and 4 in each of one cortical section and one embryo. The relationship to codispersion, Moran’s I and the graph-frequency response is developed in Supplementary Results and Fig 1D.

## Spectral sign-flip null for boundary J

Let $\mathcal{L}=U\,\mathrm{diag}(\mu)\,U^{\top}$ and $\hat x=U^{\top}x$. The null replaces $\hat x_k$ by $\varepsilon_k\hat x_k$ and $\hat y_l$ by $\eta_l\hat y_l$ with independent random signs. This preserves $\hat x_k^2$ and any quadratic form $x^{\top}h(\mathcal{L})x$, including the Dirichlet energy of the fixed graph. It removes alignment between the signs of the two maps [13]. For any statistic $x^{\top}My$ the null mean is zero and the variance is $\sum_{k,l}\hat x_k^2\hat y_l^2(u_k^{\top}Mu_l)^2$. For J, $M=h(\mathcal{L})$ is diagonal and $\mathrm{Var}=\sum_k h(\mu_k)^2\hat x_k^2\hat y_k^2$. The energies $A$ and $B$ are invariant under the randomization, so every member of the power-mean family and Q share one z-score, $z=-C/\sqrt{\mathrm{Var}}$. Repeated eigenvalues make the eigenbasis arbitrary. Within each repeated eigenspace the null therefore randomizes by a uniform rotation, which replaces $\hat x_k^2$ by its eigenspace average and makes the variance basis-invariant. Two-sided normal tail areas approximate p-values. Monte Carlo randomizations match the closed-form variance in package tests.

## Ecosystem comparators

Each comparator follows the package's documented definition and the default or tutorial setting of its test (Table 3). Distances were converted to the units of the tutorials. For Visium, SpatialDM's RBF length scale was 1.2 spot spacings, with weights below 0.2 set to zero, self-weight 1 and weights rescaled to sum N. LIANA+ used a Gaussian kernel with bandwidth 200 px at the DLPFC median spacing of 137 px (1.46 spacings), cutoff 0.1, at most 100 neighbours and self-weight 1. For seqFISH the same multiples of the median nearest-neighbour distance were used, without self-weights. Hotspot used a binary 30-nearest-neighbour graph on z-scored values (model 'none') and the conditional symmetric Z (the smaller |Z| of the two conditionings). Giotto's spatial correlation is the Pearson correlation of neighbour-averaged expression. LIANA+ cosine is the spot-wise weighted cosine averaged over spots. The MERINGUE statistic matched MERINGUE 1.0 `spatialCrossCor` to 6.4 × 10^−14^ on 45 DLPFC pairs. Local comparisons on the MERFISH geometries and the five seqFISH interfaces used the official LIANA+ 2.1.1 and SpatialDM 0.3.1 packages on a common row-standardized graph. Signed outputs were evaluated both negated and as reported, and the better orientation is shown.

**Table 3.** Comparators and their default tests.

| Package | Statistic | Default test |
|---|---|---|
| Scanpy / Seurat workflows | Pearson r; Spearman ρ | t test, two-sided |
| SpatialDM 0.3 | global Moran's R (RBF weights) | analytical z, one-sided; 1,000 permutations, one-sided |
| LIANA+ 2.x | global Moran's R (Gaussian weights) | 100 permutations, two-sided |
| Hotspot | H = xᵀ(A + Aᵀ)y on a 30-NN graph | conditional Z, one-sided |
| MERINGUE 1.0 | spatial cross-correlation | 1,000 permutations, two-sided |
| spdep | Lee's L | lee.mc, 999 permutations, 'greater' |
| Giotto | correlation of neighbour-averaged expression | none (score only) |
| This work | J (scale 1) | spectral sign-flip null, two-sided |

## Benchmark protocol

Exploratory work on MERFISH boundary localization and a seven-gene cortical interface preceded the pair-level benchmark. The benchmark protocol designated cortical layer-pair accuracy as the primary endpoint and the embryo analysis as exploratory. We added package comparators after reviewing the initial J results. Jaccard overlap, combined-score models and additional timing comparisons were later exploratory analyses. Calibration (axis A) used one known-null simulation and one empirical control.

- A1, independent Gaussian random fields (Matérn ν = ½ and 5/2; ranges 1, 3 and 10 spot spacings; white noise) on sections 151507, 151669 and 151673, with 1,000 pairs per setting and section;
- A2, cross-donor real genes on six section pairs (151507×151673, 151508×151674, 151509×151669, 151510×151670, 151671×151675, 151672×151676), using the 50 highest-Moran's-I genes per section and all 2,500 cross pairs, aligned or rotated (row r → 77 − r, column c → 127 − c).

We compared the rejection rate at α = 0.05 with Bradley’s tolerance interval [0.025, 0.075] [18]. Only A1 specifies independence by construction. Axis B (DLPFC) used leave-one-donor-out markers. A gene's specificity margin is its mean log-expression in a layer minus its highest mean in any other layer, averaged over training sections. Genes with a positive margin and mean ≥ 0.1 were assigned to their best layer, and the top K = 10 per layer were kept (sensitivity analyses used K = 5 and 20). Pairs were classed by layer-index difference as same (0), adjacent (1) or distant (≥ 2). The protocol required one-step J to exceed AUROC 0.5 in at least 10 of 12 sections and to exceed the AUROC of sign-reversed Lee’s L in at least 10 of 12. The amendment uses each statistic's natural sign. The archived endpoint values are listed in Table S16. Because anatomical proximity of measured markers is a co-localization question, the primary endpoint was not met by one-step boundary J (mean AUROC 0.451) and is evaluated with regional J in this manuscript. B2 simulated ten genes per present layer with log-mean β·(S·layer indicator) + 0.3 × Matérn(ν = 3/2, range 3), scaled to two counts per spot with the real library sizes, with Poisson counts, β = 0.5, 1 or 2 and five replicates. Axis C (seqFISH) used contact z-scores from 1,000 label permutations and leave-one-embryo-out markers (K = 3; types with ≥ 50 cells). The contact-rate design followed the parallel analysis: a Gaussian-weighted 8-NN graph, contact rate = contacts/√(n_a n_b), touching ≥ 0.15, separated ≤ 0.01, and top-3 within-embryo markers with mean z difference > 1.

## Jaccard endpoints

Interface localization was scored with three measures:

- IoU at the true interface size: the Jaccard index between the top-|T| edges of a local map and the set T of annotated interface edges, where |T| is the number of interface edges;
- the threshold-swept Jaccard AUC of the lab's earlier evaluation: scores min–max rescaled, binarized at 19 thresholds from 0.05 to 0.95, and Jaccard averaged;
- average precision.

Interfaces were the Visium edges joining adjacent annotated layers (DLPFC) and the 6-NN edges joining two cell types with contact z ≥ 3 (seqFISH), scored for all marker pairs of the two domains. The semi-synthetic version simulated three exclusive markers per layer (β = 1, 2 or 3, two replicates, no background) on the real maps. Reproducibility was the Jaccard index of the top 1% of the 44,850 pairs of a donor's panel between serial sections. The panel was the 300 genes with the highest mean Moran's I among genes detected in ≥ 10% of spots in all four of the donor's sections. Redundancy was the Jaccard index between methods' top-1% sets within a section. The lab's Jaccard similarity of two expression maps (threshold-swept, as above) was included as a co-localization comparator.

## Controlled contrasts and measured MERFISH pilot

The controlled contrast experiment used 42 × 46 grids, six expression scenarios and 40 matched draws per scenario at Gaussian noise SDs of 0.15, 0.35 and 0.60. Pair AUROC pooled the 40 draws at each noise level. Interface scores used the supplied reference edges. Whole-graph scores used all edges. In the amplitude analysis the second contrast was 1, 0.5, 0.25 or 0.1 times the first. Let $A$ and $B$ be the energies of the two edge-increment maps and $C$ their cross-product sum. We used $J_p=-C/M_p(A,B)$ with $M_p(A,B)=[(A^p+B^p)/2]^{1/p}$ and the limit $M_0=\sqrt{AB}$. Pair AUROC was averaged over matched draws. Local AP ranked the true focal transitions against nearby within-domain edges.

For the measured MERFISH pilot we used the symmetric union of 12-nearest-neighbour edges within 60 µm. Reference edges crossed the cortical layer 6a/b–corpus callosum annotation. Controls were within-compartment edges within 75 µm of a reference-edge midpoint. The two assayed cross-compartment marker contrasts were Slc17a7–Mog and Syt6–Mog. Scores used four graph-diffusion steps and a ridge of 1 for regularized J. We report animal-level medians across three sections and two contrasts per animal. The anatomical annotations define the benchmark labels.

## Matched-input package comparisons

The local comparison uses 64 matched simulations on two Allen MERFISH sections (Zhuang-ABCA-3.012 and Zhuang-ABCA-4.002, 5,000 positions per section) and a separate real-expression test on five organ interfaces per seqFISH embryo. At each coordinate graph, all local methods receive the same standardized expression after four diffusion steps. The simulation uses the archived lazy random walk and a radius-limited 12-neighbor graph. The embryo data use the uniform heat smoother defined above and a binary six-neighbor union. Local J sums the incident-edge moments at both endpoints, so the candidate edge enters both sums. A ridge of one adds the pooled edge-weight sum times the section-wide mean joint gradient energy for that pair to the denominator. Raw J uses a ridge of zero. The two gradient baselines average the signed cross-increment and the absolute difference of increments over the same edges.

LIANA+ 2.1.1 scores were obtained with its native weighted Pearson, Spearman, masked Spearman, cosine and local Moran functions. SpatialDM 0.3.1 supplied both directional local Moran terms, which were summed. The packages received the shared row-standardized graph and supplied pairs. Signed cell scores were negated and averaged at the two edge endpoints for opposition ranking. Both unnegated local Moran outputs were also evaluated because cross-neighbor co-expression can increase at interfaces. Average precision evaluated the descending scores across all six pair scenarios on simulated counts and across all graph edges for each supplied seqFISH interface. Expression selected markers in training embryos only. Test labels and positions determined edge targets. The comparison concerns ranking under common preprocessing, with no significance threshold.

We verified that the full DLPFC baseline formula matches MERINGUE 1.0 spatial cross-correlation on 45 pairs from three DLPFC sections (one per donor), using the same expression and weights. The maximum absolute difference was 6.4e-14.

## Evaluation by intersection over union

The laboratory's spatial-pattern studies proposed applying image-segmentation Jaccard to expression maps [29,30]. We used fixed reference sets and selected the top 0.10%, 0.25%, 0.50% or 1.00% of candidate edge scores, with 0.50% as the main descriptive operating point. Both directions of all native package scores were evaluated in this extension. A common label-blind permutation broke score ties. Exact IoU is the intersection size divided by the union size of the predicted and reference edge sets. In simulations each candidate is an edge–pair combination, and annotated seqFISH is evaluated within each supplied pair.

Inspired by boundary-sensitive segmentation evaluation [31], the one-hop extension allows predicted and reference edges from the same pair to match if they share an endpoint. A maximum bipartite matching assigns each edge at most once. One-hop IoU is the number of matches divided by the sum of both set sizes minus that number. The graph determines the physical tolerance. The native LIANA+ weighted Jaccard score uses positive expression supports. A complementary scoring baseline uses weighted Jaccard of the two opposing-gradient channels, pairing positive increments in one gene with negative increments in the other and summing minima and maxima over the two endpoint neighborhoods. The seqFISH comparison repeats the signed gradient at two diffusion steps as a sensitivity check. For global rankings, we computed exact IoU at top 10% and 20% of eligible marker pairs and retained both directions of each co-occupancy statistic. The 2021 BSC was computed as a Gram matrix of column-centred spatial lags divided by the original expression norms, using the same input and graph as the other global scores. All overlap analyses are post hoc.

## Colorectal analysis, effective modes and speed

Genes detected in ≥ 10% of spots (excluding MT- genes) were ranked by Moran's I and the top 300 were kept. The effective number of modes behind a null variance $\sum_k a_k^2$ is the participation ratio $(\sum_k a_k^2)^2/\sum_k a_k^4$ with $a_k=h(\mu_k)\hat x_k\hat y_k$, computed over 5,000 random pairs. Timings used section 151673, the 4,000 most highly expressed genes, one BLAS thread and the median of three warm runs. The cost of a permutation test is the cost of one all-pairs evaluation times the number of permutations.

## Apple silicon scoring benchmark

Dense all-pairs scoring was benchmarked with 64–4,000 genes from DLPFC Visium section 151673 and seqFISH embryo 1, using seven warm repeats per workload. Input expression and the graph were in memory. Timing includes expression standardization and dense score output. The MLX call synchronizes device work before returning. Graph construction, file I/O and inferential tests are outside the scoring interval. The historical comparator is our lab's all-pairs `calculate_bsc_all` function from the October 2021 repository [5]. It runs on the CPU with a dense row-standardized weight matrix and computes its original auxiliary Pearson matrices. We verified its BSC matrix against the corresponding normalized Gram of centred lagged expression. The same benchmark includes optimized lag/Gram Lee scoring on both CPU and MLX.

The Apple benchmark used tiled float64 CPU J alongside the Metal path. The MLX path performs float32 standardization, sparse graph propagation and multiplication on the GPU, then normalizes symmetric blocks with a Metal kernel. Across the complete output matrices, the largest absolute MLX-versus-float64 J difference was 1.25e-06. The float64 tiled J matched the untiled reference. Spectral inference uses the float64 CPU implementation.

## Reproducibility

The initial v3 analyses used Python 3.13. The matched package comparisons, 2021 BSC timings, and v4 figures were produced with Python 3.12. Scripts write per-replicate, per-section and per-pair results. We checked the shared J scales across both semi-synthetic tables before combining methods. The timestamped Fig 4 source and its matched input table are archived with the analysis. For v12, we rebuilt both self terms and the cross term across the full frozen marker panels and compared the lag matrix against the public 2021 all-pairs function at commit 9415612. Maximum absolute cross-term error was below 1.3 × 10^−13^. Recomputed boundary J at scales 1 and 4 agreed with the frozen per-pair results within 7.1 × 10^−12^. For Fig 9 we checked the pair labels and reproduced the archived per-specimen AUROCs from the gene-pair scores. We computed four-step negative codispersion on the same expression matrices and graphs, checking the implementation against archived one-step negative codispersion and four-step boundary J scores. The figure source table contains class counts, ranking directions and per-specimen AUROCs in both the original and plotted directions. Public regional J reproduced the real-matrix benchmark to a maximum absolute difference of 2.11 × 10^−15^. Cross-language comparisons on a full matrix from each platform agreed to 8.26 × 10^−14^. Recomputing the spatial lag statistic from the most recently edited 2024 notebook, `YES_boundary_traveler.ipynb`, reproduced 55 archived values to within 8.9 × 10^−14^. The matched subset of the published colorectal counts is documented in PROVENANCE.md. For v13, every figure was redrawn from the archived result tables with only its colour constants changed to the default R (ggplot2) palette, and Fig 1B–D was recomputed with the package. The redrawn Fig 9 source table and the tissue-map summaries reproduced the archived values to within 9 × 10^−15^.

# Data availability

DLPFC: https://github.com/LieberInstitute/HumanPilot. Colorectal cancer: GEO GSM8594561. seqFISH mouse embryo: https://marionilab.cruk.cam.ac.uk/SpatialMouseAtlas/. MERFISH coordinates: Allen Brain Cell Atlas [22]. All result tables are archived with the code.

# Code availability

Local release candidates for Python and R (`juxtaposed` 0.4.0) implement both scores. The timestamped analysis deposit contains the per-pair results, tissue-map contributions and benchmark scripts. Project repository: https://github.com/Spatial-Transcriptomics-L/J-analysis.

# References

1. Ståhl PL, Salmén F, Vickovic S, et al. Visualization and analysis of gene expression in tissue sections by spatial transcriptomics. Science. 2016;353:78–82. [DOI](https://doi.org/10.1126/science.aaf2403)
2. Maynard KR, Collado-Torres L, Weber LM, et al. Transcriptome-scale spatial gene expression in the human dorsolateral prefrontal cortex. Nat Neurosci. 2021;24:425–436. [DOI](https://doi.org/10.1038/s41593-020-00787-0)
3. Lee SI. Developing a bivariate spatial association measure: an integration of Pearson's r and Moran's I. J Geogr Syst. 2001;3:369–385. [DOI](https://doi.org/10.1007/s101090100064)
4. Wartenberg D. Multivariate spatial correlation: a method for exploratory geographical analysis. Geogr Anal. 1985;17:263–283. [DOI](https://doi.org/10.1111/j.1538-4632.1985.tb00849.x)
5. Song JJ, Yoo J. Spatial_singlecell_DL. GitHub. 2021. [24 October notebook](https://github.com/son9john/Spatial_singlecell_DL/blob/224eb71081521c4e5a2525d16ce48bb2fe92b4ae/adata_bsc.ipynb) and [25 October all-pairs implementation](https://github.com/son9john/Spatial_singlecell_DL/blob/94156124a90602450cef30d428b849ab4f48048d/adata_bsc.py).
6. Miller BF, Bambah-Mukku D, Dulac C, Zhuang X, Fan J. Characterizing spatial gene expression heterogeneity in spatially resolved single-cell transcriptomic data with nonuniform cellular densities. Genome Res. 2021;31:1843–1855. [DOI](https://doi.org/10.1101/gr.271288.120)
7. Li Z, Wang T, Liu P, Huang Y. SpatialDM for rapid identification of spatially co-expressed ligand–receptor and revealing cell–cell communication patterns. Nat Commun. 2023;14:3995. [DOI](https://doi.org/10.1038/s41467-023-39608-w)
8. Dimitrov D, Schäfer PSL, Farr E, et al. LIANA+ provides an all-in-one framework for cell–cell communication inference. Nat Cell Biol. 2024;26:1613–1622. [DOI](https://doi.org/10.1038/s41556-024-01469-w)
9. Clifford P, Richardson S, Hémon D. Assessing the significance of the correlation between two spatial processes. Biometrics. 1989;45:123–134. [DOI](https://doi.org/10.2307/2532039)
10. Dutilleul P. Modifying the t test for assessing the correlation between two spatial processes. Biometrics. 1993;49:305–314. [DOI](https://doi.org/10.2307/2532625)
11. Alexander-Bloch AF, Shou H, Liu S, et al. On testing for spatial correspondence between maps of human brain structure and function. NeuroImage. 2018;178:540–551. [DOI](https://doi.org/10.1016/j.neuroimage.2018.05.070)
12. Burt JB, Helmer M, Shinn M, Anticevic A, Murray JD. Generative modeling of brain maps with spatial autocorrelation. NeuroImage. 2020;220:117038. [DOI](https://doi.org/10.1016/j.neuroimage.2020.117038)
13. Wagner HH, Dray S. Generating spatially constrained null models for irregularly spaced data using Moran spectral randomization methods. Methods Ecol Evol. 2015;6:1169–1178. [DOI](https://doi.org/10.1111/2041-210X.12407)
14. Zhao J, Shan X, Wang G, et al. Spatial co-expression and cell-cell communication inference from spatially resolved transcriptomics with CONCISE. bioRxiv. 2026. [DOI](https://doi.org/10.64898/2026.06.22.733860)
15. Rukhin AL, Vallejos R. Codispersion coefficients for spatial and temporal series. Stat Probab Lett. 2008;78:1290–1300. [DOI](https://doi.org/10.1016/j.spl.2007.11.017)
16. Sarwar A, Rue M, French L, Cross H, Choi S, Chen X, Gillis J. Cross-expression meta-analysis of mouse brain slices reveals coordinated gene expression across spatially adjacent cells. Genome Biol. 2025;26:373. [DOI](https://doi.org/10.1186/s13059-025-03747-8)
17. Lin LI. A concordance correlation coefficient to evaluate reproducibility. Biometrics. 1989;45:255–268. [DOI](https://doi.org/10.2307/2532051)
18. Bradley JV. Robustness? Br J Math Stat Psychol. 1978;31:144–152. [DOI](https://doi.org/10.1111/j.2044-8317.1978.tb00581.x)
19. Granger CWJ, Newbold P. Spurious regressions in econometrics. J Econom. 1974;2:111–120. [DOI](https://doi.org/10.1016/0304-4076(74)90034-7)
20. Lohoff T, Ghazanfar S, Missarova A, et al. Integration of spatial and single-cell transcriptomic data elucidates mouse organogenesis. Nat Biotechnol. 2022;40:74–85. [DOI](https://doi.org/10.1038/s41587-021-01006-2)
21. Oliveira MF, Romero JP, Chung M, et al. High-definition spatial transcriptomic profiling of immune cell populations in colorectal cancer. Nat Genet. 2025;57:1512–1523. [DOI](https://doi.org/10.1038/s41588-025-02193-3)
22. Zhang M, Pan X, Jung W, et al. Molecularly defined and spatially resolved cell atlas of the whole mouse brain. Nature. 2023;624:343–354. [DOI](https://doi.org/10.1038/s41586-023-06808-9)
23. Wolf FA, Angerer P, Theis FJ. SCANPY: large-scale single-cell gene expression data analysis. Genome Biol. 2018;19:15. [DOI](https://doi.org/10.1186/s13059-017-1382-0)
24. Palla G, Spitzer H, Klein M, et al. Squidpy: a scalable framework for spatial omics analysis. Nat Methods. 2022;19:171–178. [DOI](https://doi.org/10.1038/s41592-021-01358-2)
25. Hao Y, Stuart T, Kowalski MH, et al. Dictionary learning for integrative, multimodal and scalable single-cell analysis. Nat Biotechnol. 2024;42:293–304. [DOI](https://doi.org/10.1038/s41587-023-01767-y)
26. DeTomaso D, Yosef N. Hotspot identifies informative gene modules across modalities of single-cell genomics. Cell Syst. 2021;12:446–456. [DOI](https://doi.org/10.1016/j.cels.2021.04.005)
27. Dries R, Zhu Q, Dong R, et al. Giotto: a toolbox for integrative analysis and visualization of spatial expression data. Genome Biol. 2021;22:78. [DOI](https://doi.org/10.1186/s13059-021-02286-2)
28. Jaccard P. The distribution of the flora in the alpine zone. New Phytol. 1912;11:37–50. [DOI](https://doi.org/10.1111/j.1469-8137.1912.tb05611.x)
29. Kim K. Applying Deep Learning in Finding Spatial Gene Patterns. Laboratory presentation. Slides 9–11.
30. Song J. Spatial Bivariate Correlation. Laboratory presentation. Page 20.
31. Cheng B, Girshick R, Dollár P, Berg AC, Kirillov A. Boundary IoU. Proc CVPR. 2021:15334–15342. [Proceedings](https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html)

32. Cvetković D, Rowlinson P, Simić SK. Signless Laplacians of finite graphs. Linear Algebra Appl. 2007;423:155–171. [Article](https://www.sciencedirect.com/science/article/pii/S0024379507000316)
33. Zhang S, Saeki K, Haeno H. geneSCOPE: gene spatial co-occurrence of pairwise expression. Brief Bioinform. 2026;27:bbag302. [DOI](https://doi.org/10.1093/bib/bbag302)

# Supplementary information

## Supplementary Table S1. Percent rejected at α = 0.05 for known-null fields and cross-donor controls

| Setting | Pearson t | Spear. t | Sp. DM z † | Sp. DM perm. † | LIA+ perm. | Hotspot Z † | MER. perm. | Lee perm. † | J spectral |
|----------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| A1 white noise | 5.5 | 5.6 | 4.9 | 4.7 | 5.2 | 4.3 | 4.9 | 5.2 | 5.4 |
| A1 ν=½, range 1 | 18.3 | 17.2 | 34.1 | 21.9 | 38.3 | 24.7 | 31.8 | 21.5 | **5.0** |
| A1 ν=½, range 3 | 62.2 | 61.3 | 47.7 | 39.1 | 71.6 | 39.5 | 68.3 | 39.0 | **5.8** |
| A1 ν=½, range 10 | 85.0 | 84.9 | 48.5 | 44.5 | 86.6 | 45.0 | 86.1 | 44.6 | **5.7** |
| A1 ν=5/2, range 1 | 26.1 | 25.4 | 37.2 | 24.2 | 40.9 | 24.4 | 36.3 | 23.4 | **4.7** |
| A1 ν=5/2, range 3 | 69.0 | 68.4 | 46.1 | 37.2 | 71.2 | 38.1 | 70.4 | 37.1 | **6.3** |
| A1 ν=5/2, range 10 | 90.0 | 88.5 | 47.5 | 44.4 | 89.6 | 44.5 | 89.9 | 44.3 | **7.8** |
| A2 aligned | 73.0 | 71.0 | 42.3 | 39.5 | 85.3 | 39.0 | 82.3 | 39.5 | **10.6** |
| A2 rotated | 76.6 | 75.6 | 47.5 | 45.0 | 87.8 | 44.7 | 85.0 | 44.9 | **5.3** |

† one-sided test. Sp. DM denotes SpatialDM, LIA+ denotes LIANA+, and MER. denotes MERINGUE. Bold marks J when it is closest to the nominal 5% in its row. A1 rows pool three sections (3,000 pairs); A2 rows pool six section pairs (15,000 pairs).

\Needspace{10\baselineskip}

## Supplementary Table S2. Co-localization statistics on observed anatomical proximity

Bold marks the highest value in each row.

| task | Pear. r | Spear. ρ | Sp. DM R | LIA+ R | LIA+ cos. | Hotsp. Z | MER. I | Lee's L | Giotto |
|------------------|------|------|------|------|------|------|------|------|------|
| DLPFC adjacent vs distant | 0.761 | 0.779 | 0.785 | 0.788 | 0.454 | **0.799** | 0.782 | 0.785 | 0.791 |
| DLPFC adjacent vs distant or same | 0.653 | 0.666 | 0.678 | 0.681 | 0.448 | **0.684** | 0.676 | 0.677 | 0.676 |
| seqFISH contact z design | 0.763 | 0.749 | 0.767 | 0.766 | 0.591 | 0.772 | 0.776 | 0.774 | **0.779** |
| seqFISH contact-rate design | 0.690 | 0.688 | 0.729 | 0.736 | 0.544 | **0.761** | 0.734 | 0.738 | 0.734 |
| seqFISH contact rate, ρ | 0.443 | 0.451 | 0.518 | 0.532 | 0.207 | **0.581** | 0.529 | 0.536 | 0.529 |

Package abbreviations follow Table S1. Rows report mean within-specimen AUROC using the scores as defined, except for the contact-rate correlation (Spearman ρ). Additional gradient methods, scale sensitivities and task-oriented comparisons appear in the source tables and Table S15. Regional J on the same endpoints is in Table S14; touching versus same-domain markers are in Table S15 and Fig 9.

## Supplementary Table S3. Local maps

Average precision with known truth on the MERFISH geometries (median of 16 replicates per stratum; LIANA+ and SpatialDM from the official packages):

Bold marks the highest value in each row.

| stratum | J (ridge 1) | J (no ridge) | Q | Absolute gradient difference | LIANA+ Pearson | LIANA+ masked Spearman | LIANA+ Spearman | LIANA+ cosine | SpatialDM local Moran |
|--------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Zhuang-ABCA-3.012 moderate | 0.379 | 0.250 | 0.404 | **0.423** | 0.263 | 0.191 | 0.128 | 0.001 | 0.001 |
| Zhuang-ABCA-3.012 sparse | 0.329 | 0.187 | 0.363 | **0.373** | 0.201 | 0.136 | 0.081 | 0.001 | 0.001 |
| Zhuang-ABCA-4.002 moderate | 0.360 | 0.226 | 0.378 | **0.393** | 0.203 | 0.136 | 0.094 | 0.001 | 0.001 |
| Zhuang-ABCA-4.002 sparse | 0.301 | 0.125 | 0.339 | **0.346** | 0.144 | 0.090 | 0.063 | 0.001 | 0.001 |

Jaccard (IoU) at the true interface size, and the lab's threshold-swept Jaccard AUC:

| setting | J (lazy 4, ridge 1) | J (heat 1, ridge 1) | J (heat 1) | Q | Absolute gradient difference | joint energy | Sp. DM local Moran | LIANA+ local Pearson (−) | LIANA+ local cosine |
|----------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| IoU, simulated β = 1 | **0.187** | 0.013 | 0.026 | 0.012 | 0.006 | 0.004 | 0.002 | 0.068 | 0.000 |
| IoU, simulated β = 2 | **0.295** | 0.189 | 0.135 | 0.183 | 0.148 | 0.064 | 0.031 | 0.209 | 0.000 |
| IoU, simulated β = 3 | 0.320 | 0.325 | 0.204 | 0.354 | **0.372** | 0.271 | 0.104 | 0.227 | 0.000 |
| IoU, DLPFC real | 0.014 | 0.012 | 0.009 | 0.011 | 0.007 | 0.004 | **0.019** | 0.007 | 0.005 |
| IoU, seqFISH real | 0.007 | 0.011 | 0.008 | 0.013 | 0.010 | 0.007 | **0.022** | 0.008 | 0.002 |
| Jaccard AUC, DLPFC real | 0.014 | 0.012 | 0.013 | 0.010 | 0.009 | 0.005 | 0.013 | 0.012 | **0.016** |
| Jaccard AUC, seqFISH real | 0.008 | 0.008 | 0.008 | 0.007 | 0.009 | 0.007 | **0.011** | 0.007 | 0.010 |

## Supplementary Table S4. Top 25 opposing pairs in the colorectal section

| gene x | gene y | J (scale 1) | z (J) | Lee's L | q (L, spectral) | Pearson r |
|---|---|---|---|---|---|---|
| MGP | ELF3 | 0.548 | 13.7 | −0.875 | 0.086 | −0.910 |
| MGP | EPCAM | 0.539 | 13.2 | −0.877 | 0.066 | −0.914 |
| MGP | CEACAM5 | 0.532 | 13.2 | −0.866 | 0.068 | −0.899 |
| MGP | MUC12 | 0.531 | 12.1 | −0.868 | 0.065 | −0.905 |
| MGP | LGALS4 | 0.526 | 12.0 | −0.794 | 0.055 | −0.832 |
| MGP | KLF5 | 0.512 | 13.5 | −0.853 | 0.062 | −0.893 |
| C1S | KLF5 | 0.511 | 15.4 | −0.794 | 0.061 | −0.850 |
| MGP | CLDN7 | 0.501 | 12.8 | −0.815 | 0.070 | −0.852 |
| LGALS4 | C1R | 0.499 | 14.0 | −0.754 | 0.053 | −0.807 |
| LGALS4 | COL3A1 | 0.498 | 13.5 | −0.683 | 0.052 | −0.735 |
| MGP | ST14 | 0.495 | 13.4 | −0.819 | 0.062 | −0.854 |
| VIM | PRR15L | 0.495 | 16.9 | −0.702 | 0.059 | −0.780 |
| LGALS4 | COL1A1 | 0.491 | 12.9 | −0.654 | 0.052 | −0.704 |
| PYGB | VIM | 0.491 | 15.4 | −0.666 | 0.052 | −0.737 |
| LGALS4 | C1S | 0.490 | 13.9 | −0.746 | 0.053 | −0.796 |
| LGALS4 | MMP2 | 0.488 | 12.6 | −0.647 | 0.052 | −0.697 |
| MGP | PHGR1 | 0.486 | 11.2 | −0.823 | 0.057 | −0.856 |
| COL3A1 | PRR15L | 0.484 | 15.3 | −0.734 | 0.064 | −0.796 |
| KLF5 | C1R | 0.484 | 14.7 | −0.784 | 0.061 | −0.836 |
| KLF5 | VIM | 0.483 | 15.9 | −0.715 | 0.055 | −0.782 |
| DCN | KLF5 | 0.481 | 14.0 | −0.732 | 0.062 | −0.782 |
| LGALS4 | COL1A2 | 0.480 | 12.9 | −0.657 | 0.052 | −0.706 |
| EPCAM | GREM1 | 0.480 | 12.8 | −0.758 | 0.053 | −0.800 |
| LGALS4 | COL6A3 | 0.478 | 13.2 | −0.739 | 0.053 | −0.788 |
| MMP2 | KLF5 | 0.477 | 13.6 | −0.681 | 0.057 | −0.731 |

![**Fig S1. Regional J and the full comparator panel for anatomical proximity.** A shows cortical layer adjacency. B shows embryonic cell-type contact enrichment with markers selected in the other embryos. CrossExpression used its native single-neighbor graph. C shows the within-embryo contact-rate design; open points connected to filled diamonds mark the sign-reversed orientation used in an earlier analysis. Diamonds show mean natural-orientation AUROC. Small points show specimens for methods without reversed scores. The raw and adjusted CrossExpression P values rank smaller values first.](figures/figure_s1_observed_pair_accuracy_2026-11-22_175222_EST.png){width=6.4in}

## Supplementary Table S5. Average precision of local edge rankings

Bold marks the highest value in each column.

| Method | 3.012 mod. | 3.012 sparse | 4.002 mod. | 4.002 sparse | seqFISH |
|-----------------------------|--------------|--------------|--------------|--------------|------------|
| Raw J | 0.2497 | 0.1865 | 0.2259 | 0.1249 | 0.0262 |
| Regularized J | 0.3792 | 0.3291 | 0.3604 | 0.3010 | 0.0761 |
| Negative codispersion | 0.2411 | 0.1819 | 0.1951 | 0.1269 | — |
| Signed gradient | 0.4036 | 0.3633 | 0.3784 | 0.3387 | 0.0852 |
| Gradient-difference magnitude | **0.4231** | **0.3725** | **0.3926** | **0.3461** | **0.1409** |
| LIANA+ weighted Pearson | 0.2629 | 0.2012 | 0.2032 | 0.1441 | 0.0319 |
| LIANA+ weighted Spearman | 0.1280 | 0.0810 | 0.0936 | 0.0630 | 0.0092 |
| LIANA+ masked Spearman | 0.1913 | 0.1362 | 0.1362 | 0.0899 | 0.0102 |
| LIANA+ weighted cosine | 0.0012 | 0.0013 | 0.0012 | 0.0013 | 0.0040 |
| LIANA+ local Moran (negative) | 0.0010 | 0.0010 | 0.0011 | 0.0011 | 0.0033 |
| LIANA+ local Moran (positive) | 0.0010 | 0.0010 | 0.0011 | 0.0011 | 0.0395 |
| SpatialDM local Moran (negative) | 0.0010 | 0.0010 | 0.0011 | 0.0011 | 0.0033 |
| SpatialDM local Moran (positive) | 0.0010 | 0.0010 | 0.0011 | 0.0011 | 0.0395 |

Each MERFISH column is the median over 16 simulated count replicates. Columns 3.012 and 4.002 use Allen sections Zhuang-ABCA-3.012 and Zhuang-ABCA-4.002, respectively. Mod. indicates moderate counts. The seqFISH column is the median over five supplied marker pairs in each of three embryos. Negative and positive indicate whether edge scores rank the negated or original local Moran values.

```{=latex}
\Needspace{15\baselineskip}
```

## Supplementary Table S6. Exact and one-hop overlap at the top 0.5% of edges

Bold marks the highest value in each column.

| Method | MERFISH exact | MERFISH one-hop | seqFISH exact | seqFISH one-hop |
|-----------------------------------|----------------|----------------|----------------|----------------|
| Raw J | 0.146 | 0.188 | 0.036 | 0.085 |
| Regularized J | 0.197 | 0.221 | 0.080 | 0.110 |
| Signed gradient | **0.210** | **0.226** | 0.081 | 0.128 |
| Gradient-difference magnitude | 0.201 | 0.220 | **0.114** | **0.160** |
| LIANA+ weighted Pearson (negative) | 0.148 | 0.194 | 0.043 | 0.091 |
| SpatialDM local Moran (positive) | 0.000 | 0.000 | 0.014 | 0.042 |
| LIANA+ weighted Jaccard (positive) | 0.000 | 0.000 | 0.010 | 0.066 |
| Opposite-gradient weighted Jaccard | 0.149 | 0.190 | 0.027 | 0.081 |
| Signed gradient (two steps) | — | — | 0.087 | 0.110 |

Medians across 64 matched simulations or 15 annotated interfaces. Table S5 shows the original average-precision analysis. The full overlap tables include precision, recall and ties by budget.

## Supplementary Table S7. Global pair ranking with both score directions

Bold marks the highest value in each column.

| Method | DLPFC AUROC | DLPFC IoU | seqFISH AUROC | seqFISH IoU |
|-----------------------------|-----------------|------------------|-----------------|------------------|
| 2021 BSC (positive) | **0.785** | **0.316** | 0.774 | 0.268 |
| 2021 BSC (negative) | 0.215 | 0.029 | 0.226 | 0.011 |
| Lee's L (positive) | **0.785** | **0.316** | 0.774 | 0.268 |
| Lee's L (negative) | 0.215 | 0.029 | 0.226 | 0.011 |
| Bivariate Moran (positive) | 0.782 | **0.316** | **0.776** | **0.274** |
| Bivariate Moran (negative) | 0.218 | 0.030 | 0.224 | 0.012 |
| Pearson (positive) | 0.761 | 0.297 | 0.763 | 0.262 |
| Pearson (negative) | 0.239 | 0.034 | 0.237 | 0.013 |

Means across 12 DLPFC sections and three embryos for touching versus separated marker pairs. IoU is measured at the top 20% of eligible pairs. The supplementary data include the 10% budget and the comparison with same-type pairs included among negatives.

![**Fig S2. Jaccard overlap under equal prediction budgets.** (A–C) Local exact and one-hop IoU curves on simulated MERFISH counts and annotated seqFISH cell-type contacts.](figures/figs1_jaccard_overlap_2026-11-22_175222_EST.png){width=5.9in}

```{=latex}
\Needspace{14\baselineskip}
```

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

## Supplementary Table S8. End-to-end scoring time on Apple M4 Max

| Dataset | Cells | Genes | Untiled J CPU | Tiled J CPU | J MLX | Lee CPU | Lee MLX | 2021 BSC |
|-------------------|---------|-----------|------------|-----------|----------|----------|----------|-----------|
| DLPFC 151673 | 3611 | 64 | 2.04 | 2.39 | 1.41 | 1.33 | 1.91 | 3.98 |
| DLPFC 151673 | 3611 | 500 | 14.12 | 13.83 | **4.31** | 8.38 | 4.51 | 27.92 |
| DLPFC 151673 | 3611 | 1000 | 37.33 | 36.16 | 10.43 | 20.04 | 8.63 | 62.72 |
| DLPFC 151673 | 3611 | 2000 | 102.06 | 103.01 | **14.60** | 55.26 | 19.68 | 161.46 |
| DLPFC 151673 | 3611 | 4000 | 421.67 | 301.24 | 60.63 | 161.54 | 48.62 | 486.87 |
| seqFISH embryo 1 | 10150 | 64 | 6.61 | 6.14 | 5.83 | 4.23 | 3.53 | 27.02 |
| seqFISH embryo 1 | 10150 | 351 | 23.64 | 24.60 | 8.33 | 15.86 | 7.88 | 121.18 |

All times are milliseconds, with medians over seven warm runs. Bold marks J when it has the shortest unrounded median in its row. The 2021 BSC column gives the time for the lab's original CPU all-pairs calculation [5]. Lee CPU and Lee MLX denote the matched lag/Gram implementation.

## Supplementary Table S9. Local edge-ranking AUROC in the expanded panel

Bold marks the highest value in each column.

| Method | 3.012 mod. | 3.012 sparse | 4.002 mod. | 4.002 sparse | seqFISH |
|------------------------------|--------------|--------------|--------------|--------------|------------|
| Raw J | 0.9967 | 0.9857 | 0.9956 | 0.9782 | 0.6544 |
| Regularized J | 0.9986 | 0.9972 | 0.9984 | 0.9901 | 0.6569 |
| Negative codispersion | 0.9965 | 0.9866 | 0.9957 | 0.9789 | — |
| Signed gradient | **0.9988** | **0.9982** | **0.9986** | 0.9948 | 0.6513 |
| Gradient-difference magnitude | **0.9988** | 0.9977 | 0.9984 | **0.9966** | **0.8792** |
| LIANA+ weighted Pearson (negative) | 0.9964 | 0.9873 | 0.9955 | 0.9805 | 0.6606 |
| LIANA+ weighted Spearman (negative) | 0.9925 | 0.9817 | 0.9897 | 0.9736 | 0.6389 |
| LIANA+ masked Spearman (negative) | 0.9920 | 0.9801 | 0.9878 | 0.9700 | 0.6561 |
| LIANA+ weighted cosine (negative) | 0.5830 | 0.6204 | 0.5581 | 0.5911 | 0.5865 |
| LIANA+ local Moran (negative) | 0.5155 | 0.5110 | 0.5120 | 0.5141 | 0.3752 |
| LIANA+ local Moran (positive) | 0.4845 | 0.4890 | 0.4880 | 0.4859 | 0.6248 |
| SpatialDM local Moran (negative) | 0.5155 | 0.5110 | 0.5120 | 0.5141 | 0.3752 |
| SpatialDM local Moran (positive) | 0.4845 | 0.4890 | 0.4880 | 0.4859 | 0.6248 |
| LIANA+ weighted Pearson (positive) | 0.0036 | 0.0127 | 0.0045 | 0.0195 | 0.3394 |
| LIANA+ weighted Spearman (positive) | 0.0075 | 0.0183 | 0.0103 | 0.0264 | 0.3611 |
| LIANA+ masked Spearman (positive) | 0.0080 | 0.0199 | 0.0122 | 0.0300 | 0.3439 |
| LIANA+ weighted cosine (positive) | 0.4170 | 0.3796 | 0.4419 | 0.4089 | 0.4135 |
| LIANA+ weighted Jaccard (positive) | 0.6634 | 0.7240 | 0.6911 | 0.7179 | 0.8363 |
| Opposite-gradient weighted Jaccard | 0.9967 | 0.9844 | 0.9953 | 0.9757 | 0.6610 |
| Signed gradient (two steps) | — | — | — | — | 0.6210 |
| LIANA+ weighted Jaccard (negative) | 0.3366 | 0.2760 | 0.3089 | 0.2821 | 0.1637 |

Median full-ranking AUROC across 16 simulated count replicates per MERFISH stratum and 15 annotated seqFISH interfaces. MERFISH section labels and count regimes are as in Table S5. Each score is evaluated on the same candidate edges and labels as the overlap analysis. AUROC here is independent of the overlap budget.

```{=latex}
\Needspace{7\baselineskip}
```

## Supplementary Table S10. Pair AUROC in the controlled contrast simulation

| Method | Reference interface | SD 0.15 | SD 0.35 | SD 0.60 |
|--------------------------------------|---------------------|-------------|-------------|-------------|
| J on supplied interface | Supplied | **1.000** | **1.000** | **1.000** |
| Negative codispersion on interface | Supplied | 1.000 | 1.000 | 1.000 |
| Whole-graph J | None | **1.000** | **0.998** | **0.933** |
| J interface minus interior | Supplied | 0.979 | 0.999 | 0.999 |
| Negative Pearson | None | 0.916 | 0.903 | 0.883 |
| Negative Lee | None | 0.865 | 0.863 | 0.860 |
| Negative bivariate Moran | None | 0.886 | 0.875 | 0.866 |
| Negative legacy BSC | None | 0.865 | 0.863 | 0.860 |
| Legacy Sobel total | None | 0.261 | 0.220 | 0.346 |
| Legacy unsigned contrast | Supplied | 0.535 | 0.529 | 0.551 |

Supplied-interface scores use the simulated reference geometry. Whole-graph J, Pearson, Lee and Moran use all vertices or edges. The two groups represent different information settings. The legacy BSC calculation reproduces the archived spatial lag statistic in this experiment.

## Supplementary Table S11. Contrast-imbalance sensitivity at noise SD 0.35

| Second-to-first step | Normalization (p) | Pair AUROC | Local AP |
|-----------------------|-----------------------------------|------------------|------------------|
| 1 | J (1) | **1.000** | **0.301** |
| 1 | J (½) | **1.000** | 0.301 |
| 1 | J (¼) | **1.000** | 0.298 |
| 1 | Negative codispersion (0) | 1.000 | 0.296 |
| 0.5 | J (1) | 0.870 | **0.284** |
| 0.5 | J (½) | 0.935 | 0.281 |
| 0.5 | J (¼) | 0.945 | 0.277 |
| 0.5 | Negative codispersion (0) | 0.965 | 0.272 |
| 0.25 | J (1) | 0.795 | **0.239** |
| 0.25 | J (½) | **0.800** | 0.231 |
| 0.25 | J (¼) | **0.800** | 0.225 |
| 0.25 | Negative codispersion (0) | 0.800 | 0.217 |
| 0.1 | J (1) | 0.590 | **0.173** |
| 0.1 | J (½) | 0.650 | 0.164 |
| 0.1 | J (¼) | 0.665 | 0.159 |
| 0.1 | Negative codispersion (0) | 0.715 | 0.152 |

Pair AUROC is the mean across 40 draws of one focal and five distractor scenarios. Local AP is computed for the focal contrast against nearby within-domain edges. Data at noise SDs 0.15 and 0.60 are included with the benchmark code.

```{=latex}
\Needspace{23\baselineskip}
```

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

## Supplementary Table S12. Measured MERFISH near-interface ranking

Bold marks the highest value in each column.

| Method | Mouse 1 AP | Mouse 1 AUROC | Mouse 2 AP | Mouse 2 AUROC |
|--------------------------------|----------------|----------------|----------------|----------------|
| Edge length | 0.162 | **0.643** | **0.174** | **0.642** |
| Raw J | 0.131 | 0.555 | 0.108 | 0.539 |
| Regularized J | 0.155 | 0.581 | 0.118 | 0.557 |
| Tempered J (½) | 0.132 | 0.552 | 0.108 | 0.546 |
| Tempered J (¼) | 0.128 | 0.549 | 0.108 | 0.549 |
| Joint gradient energy | **0.196** | **0.643** | 0.153 | 0.593 |
| Negative codispersion | 0.124 | 0.545 | 0.112 | 0.546 |
| Negative local cosine | 0.081 | 0.402 | 0.102 | 0.434 |
| Negative local Lee | 0.081 | 0.386 | 0.105 | 0.425 |
| Negative local Moran | 0.080 | 0.384 | 0.104 | 0.422 |
| Negative local Pearson | 0.127 | 0.557 | 0.118 | 0.559 |
| Signed gradient | 0.169 | 0.594 | 0.138 | 0.569 |
| Gradient-difference magnitude | 0.177 | 0.636 | 0.156 | 0.582 |

Each value is the animal median across three measured sections and two marker contrasts. Scores rank reference crossings against nearby within-compartment edges. Local correlations in this pilot are formula baselines. Tables S5 and S9 report native package scores on the separate simulated-count and seqFISH benchmarks.

## Supplementary Table S13. Matched semi-synthetic pair benchmarking

β = 0.5

| Method | Adjacent vs distant | Adjacent vs same (natural) | Adjacent vs same (oriented) |
|-------------------------------|--------------------|--------------------|-----------------------|
| Hotspot Z | 0.65718 | 0.21657 | 0.78343 |
| LIANA+ Moran | 0.64653 | 0.23506 | 0.76494 |
| Lee’s L | 0.63997 | 0.22602 | 0.77398 |
| Giotto correlation | 0.63843 | 0.20591 | 0.79409 |
| SpatialDM R | 0.63771 | 0.22509 | 0.77491 |
| MERINGUE | 0.63041 | 0.22204 | 0.77796 |
| J~4~ | 0.60594 | **0.79716** | 0.79716 |
| Pearson r | 0.60073 | 0.23639 | 0.76361 |
| Spearman ρ | 0.57489 | 0.14045 | 0.85955 |
| J~2~ | 0.57357 | 0.70558 | 0.70558 |
| Signed gradient Q | 0.54373 | 0.61755 | 0.61755 |
| Negative codispersion | 0.54294 | 0.61721 | 0.61721 |
| J~1~ | 0.54293 | 0.61725 | 0.61725 |
| Gradient co-energy | 0.52874 | 0.45906 | 0.45906 |
| J~0~ | 0.51505 | 0.53657 | 0.53657 |
| LIANA+ cosine | 0.39720 | 0.39013 | 0.60987 |

β = 1

| Method | Adjacent vs distant | Adjacent vs same (natural) | Adjacent vs same (oriented) |
|-------------------------------|--------------------|--------------------|-----------------------|
| J~4~ | **0.85314** | **0.99262** | **0.99262** |
| J~2~ | 0.77897 | 0.96811 | 0.96811 |
| Hotspot Z | 0.73560 | 0.03446 | 0.96554 |
| LIANA+ Moran | 0.71823 | 0.05247 | 0.94753 |
| Lee’s L | 0.70583 | 0.04296 | 0.95704 |
| SpatialDM R | 0.70184 | 0.04429 | 0.95571 |
| Giotto correlation | 0.70086 | 0.02813 | 0.97187 |
| MERINGUE | 0.69073 | 0.03937 | 0.96063 |
| J~1~ | 0.67523 | 0.88162 | 0.88162 |
| Negative codispersion | 0.67522 | 0.88147 | 0.88147 |
| Signed gradient Q | 0.67490 | 0.88069 | 0.88069 |
| Pearson r | 0.65630 | 0.03682 | 0.96318 |
| Spearman ρ | 0.63354 | 0.01259 | 0.98741 |
| Gradient co-energy | 0.59649 | 0.33035 | 0.33035 |
| J~0~ | 0.55868 | 0.65984 | 0.65984 |
| LIANA+ cosine | 0.37072 | 0.32945 | 0.67055 |

β = 2

| Method | Adjacent vs distant | Adjacent vs same (natural) | Adjacent vs same (oriented) |
|-------------------------------|--------------------|--------------------|-----------------------|
| J~4~ | **0.99508** | **1.00000** | **1.00000** |
| J~2~ | 0.98972 | 0.99999 | 0.99999 |
| Negative codispersion | 0.94892 | 0.99915 | 0.99915 |
| J~1~ | 0.94881 | 0.99915 | 0.99915 |
| Signed gradient Q | 0.94851 | 0.99912 | 0.99912 |
| Hotspot Z | 0.80224 | 0.00000 | 1.00000 |
| LIANA+ Moran | 0.76809 | 0.00000 | 1.00000 |
| Gradient co-energy | 0.75557 | 0.13245 | 0.13245 |
| Giotto correlation | 0.75302 | 0.00000 | 1.00000 |
| Lee’s L | 0.75044 | 0.00000 | 1.00000 |
| SpatialDM R | 0.74310 | 0.00000 | 1.00000 |
| J~0~ | 0.73092 | 0.93401 | 0.93401 |
| MERINGUE | 0.72773 | 0.00000 | 1.00000 |
| Spearman ρ | 0.68463 | 0.00000 | 1.00000 |
| Pearson r | 0.68170 | 0.00000 | 1.00000 |
| LIANA+ cosine | 0.36825 | 0.28309 | 0.71691 |

Each value averages 12 sections and five replicates per section. In the oriented same-layer comparison, lower co-localization scores rank adjacent ahead of same-layer pairs. The natural sign is retained for J and all gradient-derived statistics. Bold denotes J only when it reaches the best unrounded score within its column and β, including exact ties.


## Supplementary Table S14. Observed anatomical contacts with a fixed regional score

Bold marks the highest value in each row.

| Dataset and endpoint | Regional J | Matched Pearson | Hotspot Z | Giotto |
|----------------------|----------|----------|---------|---------|
| DLPFC, AUROC | **0.8054** | 0.8048 | 0.7988 | 0.7909 |
| DLPFC, AP | **0.6146** | 0.6092 | 0.5973 | 0.5821 |
| DLPFC, Jaccard | **0.3366** | 0.3361 | 0.3264 | 0.3217 |
| seqFISH z, AUROC | 0.7773 | 0.7778 | 0.7716 | **0.7786** |
| seqFISH z, AP | 0.4124 | **0.4137** | 0.3890 | 0.4120 |
| seqFISH z, Jaccard | 0.2783 | **0.2844** | 0.2616 | 0.2831 |
| seqFISH rate, AUROC | 0.7451 | 0.7415 | **0.7606** | 0.7342 |
| seqFISH rate, AP | 0.8357 | 0.8345 | **0.8403** | 0.8306 |
| seqFISH rate, Jaccard | **0.2480** | 0.2470 | **0.2480** | 0.2470 |
| MERFISH, AUROC | 0.5426 | 0.5422 | **0.5465** | 0.5388 |
| MERFISH, AP | 0.1694 | 0.1695 | **0.1697** | 0.1680 |
| MERFISH, Jaccard | **0.1095** | 0.1035 | 0.1083 | 0.1029 |

Values are equal-weight means over specimens. AP denotes average precision. Jaccard uses a 20% prediction budget. Columns show regional J, its matched-scale Pearson control and the two leading ecosystem comparators from Fig 8. The regional score at 16 steps reached cortical AUROC 0.8163 and contact-rate AUROC 0.7643, beside 0.8161 and 0.7631 for Pearson at exactly 16 steps. On embryo contact enrichment the corresponding AUROCs were 0.7742 and 0.7756. The results do not establish a single optimal diffusion scale. The archived supplementary CSV includes each specimen, scale and comparator.

## Supplementary Table S15. Which score directions identify touching versus same-domain markers?

The positive class is a marker pair from distinct touching domains. Boundary J and negative codispersion rank a larger score first. Pearson and regional J are shown in both directions. Ranking lower co-expression first tests the expectation that markers of the same domain co-express more strongly. Each value is a mean within-specimen AUROC. The arrows indicate the direction used for the entire column in both datasets.

Bold marks the highest value in each row.

| Tissue | Boundary J~1~ | Negative codispersion | Pearson ↑ | Pearson ↓ | Regional J ↑ | Regional J ↓ |
|-------------|---------|---------------|---------|---------|-----------|-----------|
| Human cortex | 0.7272 | 0.7271 | 0.2394 | 0.7606 | 0.2173 | **0.7827** |
| Mouse embryo | 0.7696 | 0.7702 | 0.2153 | **0.7847** | 0.2284 | 0.7716 |

\Needspace{10\baselineskip}

**Diffusion sensitivity on the same measured pairs.** The upper panel uses the protocol’s one-step boundary score and the fixed multiscale regional score. The panel below compares alternative boundary scales with Pearson on identically diffused expression at two and four steps. The last column gives a longer-range smoothing control. All entries report within-specimen AUROC averaged over specimens.

| Tissue | Boundary J~2~ ↑ | Pearson, 2 steps ↓ | Boundary J~4~ ↑ | Pearson, 4 steps ↓ | Pearson, 16 steps ↓ |
|--------------|-----------|-----------|-----------|-----------|-----------|
| Human cortex | 0.7543 | 0.7690 | 0.7855 | 0.7768 | **0.7875** |
| Mouse embryo | 0.7755 | **0.7759** | 0.7748 | 0.7731 | 0.7613 |

![**Fig S3. Regional expression in held-out cortex.** NEFL and TUBB2A ranked highest for regional J across the training donors. Their measured profiles in section 151507 share a broad territory across layers 4 and 5. Panels follow Fig 10.](figures/dlpfc_training_regional_observed_tissue_2026-11-22_175222_EST.png){width=6.35in}

![**Fig S4. Anatomy-selected cortical pair with weak whole-tissue scores.** MALAT1 and HPCAL1 were the top external markers of layers 1 and 2. The layer pair was selected by the number of reference crossing edges. It therefore provides a score-independent comparison with Fig 10 in the same section. Panels follow Fig 10.](figures/dlpfc_observed_tissue_2026-11-22_175222_EST.png){width=6.35in}


## Supplementary Results. Graph identities and calibration

For standardized expression vectors $x$ and $y$ on a symmetric tissue graph with edge weights $w_e$, J at smoothing scale $s$ is

$$J_{s}(x,y)=\frac{-2\sum_{e}w_{e}\,\Delta\tilde x_{e}\,\Delta\tilde y_{e}}{\sum_{e}w_{e}\left(\Delta\tilde x_{e}^{2}+\Delta\tilde y_{e}^{2}\right)}=\frac{E(\tilde x-\tilde y)-E(\tilde x+\tilde y)}{E(\tilde x-\tilde y)+E(\tilde x+\tilde y)},\qquad \tilde x=S^{s}x,$$

Here $\Delta\tilde x_e$ is the difference across edge $e$, and $E(v)=\sum_e w_e\Delta v_e^2$ is the Dirichlet (gradient) energy. $S=I-\mathcal{L}/(2d_{\max})$ is one step of uniform heat diffusion, with $\mathcal{L}$ the graph Laplacian and $d_{\max}$ the maximum degree. J lies in [−1, 1]. Positive values mean that the genes change in opposite directions across the same edges, as for complementary domains that touch. Negative values mean concordant change. Scores near zero can reflect weak overlap or cancellation of opposing and concordant changes in different parts of the tissue. Replacing the arithmetic mean of the two energies by their geometric mean gives negative codispersion [15]. Intermediate power means interpolate between the two and keep pairs with unequal gradient amplitudes on a comparable scale. In a simulation with one marker's amplitude halved (noise SD 0.35), pair-level AUROC rose from 0.87 (power 1) to 0.96 (power 0). With the two orientations of every edge included, J is also the negative of Lin's concordance correlation of the increments [17]. These variants share one numerator, so they share one test (Methods).

Two identities connect J with quantities analysts already compute. On a regular graph with row-standardized weights

$$J_{0}(x,y)=\frac{I_{xy}-r_{xy}}{1-\tfrac12\left(I_{x}+I_{y}\right)},$$

where $r_{xy}$ is Pearson's correlation, $I_{xy}$ the bivariate Moran's I and $I_x$, $I_y$ the univariate Moran's I. With the symmetric normalized Laplacian the same identity holds exactly on any graph, with Moran-type coefficients defined on $D^{-1/2}AD^{-1/2}$ (Methods). J is therefore positive when a gene is more associated with its partner in neighbouring spots than in the same spot. For filters diagonal in the eigenbasis of $\mathcal{L}$, the cross term is a weighted sum $\sum_k h(\mu_k)\hat x_k\hat y_k$ over graph frequencies $\mu_k$. Pearson's r weights all frequencies equally. Lee's L, Moran's I and the kernel-weighted R of SpatialDM and LIANA+ emphasize low frequencies. J uses $h(\mu)=\mu(1-\mu/2d_{\max})^{2s}$, which vanishes at zero frequency (Fig 1D). In a synthetic tissue, markers of touching and of separated domains had similar Lee's L, whereas J separated them (Fig 1B,C).


The excess rejections for J on the aligned control were associated with shared array positions. Two thirds of the significant J values were concordant, which pointed to a shared signal rather than a variance error. The log library sizes of the paired slides were strongly concordant at the scale of single spots in all six pairs (J −0.35 to −0.28, z −17.1 to −14.8). They were not concordant at the smooth scale (Lee's L z ≤ 2.2), and the signal vanished after rotation (z −1.7 to 1.3). This is consistent with reproducible spot-to-spot differences in capture efficiency between slides. Regressing each gene on its log library size did not remove it (J 12.7%). J is sensitive to spot-level technical structure in a way that smooth statistics are not (Discussion).

Independent Gaussian random fields on real spot layouts provide 3,000 known-null pairs per setting. With white noise, rejection rates ranged from 4.3% to 5.6%. As the fields became smoother, conventional tests increasingly rejected independence (Fig 2C). For the smoothest fields (Matérn ν = 5/2, range 10 spots), the conventional two-sided tests rejected 89–90% and the one-sided tests 44–48%. The boundary J spectral test rejected 4.7–7.8% across the six autocorrelated settings, within the protocol’s 2.5–7.5% tolerance interval [18] in five of six. Free spot permutations of J also rejected too often. Conditioning on spatial structure accounted for much of the improvement.

In a real section the consequence is direct. Among the 44,850 pairs of the 300 most autocorrelated genes of a colorectal tumour [21], the two-sided default tests declared 98% of pairs significant at q < 0.05. The one-sided tests declared 48%–49%, which is nearly every positively associated pair (Fig 2D). Computing those p-values for all 8 million pairs of 4,000 genes took 2–5 s for analytical tests. Permutation tests took an extrapolated 3–40 min, even when each permutation is shared across all pairs. The spectral null took 15 s and yielded near-nominal rejection on the rotated control (Fig 2E).

## Supplementary Results. Additional comparators and anatomical transfer

**CrossExpression.** A published single-cell comparator, CrossExpression [16], provided an independent comparison on the contact-enrichment endpoint. The native R implementation yielded mean AUROC 0.469 for its correlation effect and 0.641 for ranking by unadjusted cross-expression P value. Every eligible pair tied on the adjusted P value in each embryo, giving AUROC 0.500. The method evaluates exclusive expression across first-nearest neighbors. The benchmark labels describe contact enrichment among cell types. The two definitions address different biological questions.

**MERFISH transfer.** On four additional MERFISH sections from two mice [22], we tested the fixed scores on pairs of registered brain structures with at least ten connecting edges versus structures with none. Markers came from the other animal. Regional J achieved mean AUROC 0.543, matched-scale Pearson 0.542 and Hotspot 0.547. Regional J reached mean top-20% Jaccard 0.110, compared with 0.108 for Hotspot. One section supplied only seven eligible structures and 33 positive marker pairs. Anatomical transfer was weak for the entire panel (Table S14).

## Supplementary Table S16. Archived endpoints of the original benchmark protocol

**Table S16.** Endpoints of the original benchmark protocol, kept as its record. Rows on anatomical proximity (DLPFC adjacent versus distant layers, seqFISH touching versus separated types) apply boundary J outside its indicated use; regional J is the score for those endpoints (Fig 8). Bold marks J when it reaches the best unrounded value in its row. For calibration rows, best means closest to the nominal 5%. Sp. DM denotes SpatialDM, and MER. denotes MERINGUE. Pearson is the Scanpy and Seurat reference. Gradient denotes the unnormalized gradient baseline (|Δx − Δy| or Q); J + co-loc., J combined with a co-localization statistic. The SpatialDM and Hotspot tests and the Lee permutation test are one-sided. CPU timings for bivariate Moran's I and Lee's L use the in-house implementations, measured alongside J with 4,000 genes. Known-truth IoU uses the top 0.5% of edges; DLPFC IoU uses the true interface size, with β = 2 for the simulated markers. The GPU row uses matched MLX implementations. Table S13 gives the complete β-specific rankings. J + co-localization is the cross-validated combination with Hotspot (DLPFC) or Lee's L (seqFISH).

| Endpoint | J | Pearson | Sp. DM | LIANA+ | Hotspot | MER. | Lee | Giotto | Gradient | J + co-loc. |
|--------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| Rotated control rejections (%) | **5.3** | 76.6 | 47.5 | 87.8 | 44.7 | 85.0 | 44.9 | — | — | — |
| FPR, smoothest random fields (%) | **7.8** | 90.0 | 47.5 | 89.6 | 44.5 | 89.9 | 44.3 | — | — | — |
| Rotated control, spectral test (%) | **5.3** | 3.0 | 3.5 | 3.6 | — | 3.1 | 3.0 | — | — | — |
| Time to default p-values, 4,000 genes (s) | 15 | 4 | 2 | 186 | 3 | 1,745 | 883 | — | — | — |
| DLPFC adjacent vs distant (AUROC) | 0.451 | 0.761 | 0.785 | 0.788 | 0.799 | 0.782 | 0.785 | 0.791 | — | — |
| DLPFC adjacent vs distant or same (AUROC) | 0.509 | 0.653 | 0.678 | 0.681 | 0.684 | 0.676 | 0.677 | 0.676 | — | **0.727** |
| seqFISH touching vs separated (AUROC) | 0.389 | 0.763 | 0.767 | 0.766 | 0.772 | 0.776 | 0.774 | 0.779 | — | 0.771 |
| Exclusive markers, β = 1 (AUROC) | 0.675 | 0.656 | 0.702 | 0.718 | 0.736 | 0.691 | 0.706 | 0.701 | — | — |
| Exclusive markers, β = 2, J~1~ (AUROC) | 0.949 | 0.682 | 0.743 | 0.768 | 0.802 | 0.728 | 0.750 | 0.753 | 0.949 | — |
| Exclusive markers, β = 2, J~4~ (AUROC) | **0.995** | 0.682 | 0.743 | 0.768 | 0.802 | 0.728 | 0.750 | 0.753 | 0.949 | — |
| Known-truth local maps, MERFISH (AP) | 0.345 | — | 0.001 | 0.202 | — | — | — | — | 0.383 | — |
| Known-truth local maps, MERFISH (IoU) | 0.197 | — | 0.000 | 0.148 | — | — | — | — | 0.210 | — |
| Exclusive markers on DLPFC maps (IoU) | **0.295** | — | 0.031 | 0.209 | — | — | — | — | 0.148 | — |
| Real DLPFC layer boundaries (IoU) | 0.014 | — | 0.019 | 0.007 | — | — | — | — | 0.007 | — |
| Reproducibility, serial sections (Jaccard) | 0.121 | 0.534 | 0.624 | 0.622 | 0.596 | 0.604 | 0.621 | 0.550 | — | — |
| Largest overlap with another method (Jaccard) | 0.00 | 0.67 | 0.95 | 0.95 | 0.62 | 0.89 | 0.93 | 0.67 | — | — |
| All-pairs statistic, 1 CPU thread (s) | 1.51 | 0.98 | — | — | — | 2.02 | 1.17 | — | — | — |
| All pairs, Apple M4 Max GPU (ms) | 60.6 | — | — | — | — | — | 48.6 | — | — | — |
