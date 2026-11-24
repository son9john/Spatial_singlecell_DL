# J benchmark protocol v2: objective, field-standard evaluation of a pair-level spatial association index

Frozen: 2026-11-22, before any outcome in this protocol was computed. The SHA-256 of this
file is recorded in `results/protocol_v2.sha256` and in the manuscript's Methods.
Earlier pilots (edge-level boundary localization in Allen MERFISH, seven-gene DLPFC
interface contrast) are reported separately as completed exploratory work.

## 0. What J is evaluated for

J is a **pair-level** index: for genes x and y on a tissue graph, it measures whether their
expression changes across the **same edges** and in which direction (positive J = opposing
changes, i.e. complementary domains that touch; negative J = concordant changes). The
established indices (Pearson r, bivariate Moran's I, Lee's L) measure whether two genes
occupy the same places. The benchmark therefore asks questions that every analyst of
bivariate spatial statistics already accepts as objective:

* **A. Calibration.** When two genes are independent but each is spatially autocorrelated,
  does the test reject at the nominal rate? (Type I error / false-positive rate.)
* **B. Accuracy against expert annotation.** Using the manually annotated cortical layers
  of the 12-section human DLPFC Visium dataset (Maynard et al., *Nat Neurosci* 2021), can
  the index tell apart gene pairs whose domains share a boundary from pairs whose domains
  are separated, and from pairs that occupy the same domain? (AUROC.)
* **C. Single-cell downstream.** With expert-curated cell-type labels (seqFISH mouse
  embryo, Lohoff et al., *Nat Biotechnol* 2022), can it separate markers of cell types that
  touch from markers of cell types that do not, and from markers of the same cell type?
* **D. Speed.** Wall-clock time for all gene pairs and for calibrated p-values, on the same
  hardware and inputs for every method.

## 1. Shared inputs (identical for every method)

* Expression: log1p(10^4 × count / spot library size); each gene standardized within section
  (mean 0, population SD 1). seqFISH uses the authors' normalized log-expression, standardized
  within embryo. Genes with zero variance in a section are dropped (reported).
* Graph (Visium): binary first-ring hexagonal adjacency from array coordinates (6 neighbours
  in the interior). seqFISH: symmetric union of k = 6 nearest neighbours per embryo.
* Smoothing for J at scale s: S = I − 𝓛/(2·d_max), applied s times (𝓛 = D − A, combinatorial
  Laplacian). For a regular hexagonal lattice this equals the lazy random walk used in v0.1.
* Lee's L: Lee (2001) definition with the row-standardized W. Bivariate Moran's I: symmetrized
  (xᵀWy + yᵀWx)/(2n) with row-standardized W and standardized x, y.

## 2. Index definitions (signs aligned so that larger = more "complementary")

* J_s = −2 xᵀM_s y / (xᵀM_s x + yᵀM_s y), M_s = S^s 𝓛 S^s. **Primary: s = 1** (one smoothing
  step, matching the single spatial lag inside Lee's L). Sensitivity: s = 0 (original v0.1
  statistic), 2, 4.
* Negative codispersion_s = −xᵀM_s y / sqrt(xᵀM_s x · yᵀM_s y) (s = 1).
* Unsigned gradient co-energy_s = Σ_e |Δz_x,e||Δz_y,e| / sqrt(Σ_e Δz_x,e² Σ_e Δz_y,e²), z = S^s x (s = 1).
* −L (Lee), −I_xy (bivariate Moran), −r (Pearson).

## 3. Inference compared in Axis A

* Pearson: t-test (independent spots) and spectral sign-flip.
* Lee's L, bivariate Moran's I, J_0, J_1: 999 random permutations of one gene across spots
  (two-sided, (b+1)/(B+1)), and spectral sign-flip.
* Spectral sign-flip (Moran spectral randomization with a closed-form variance): with
  𝓛 = U diag(μ) Uᵀ and x̂ = Uᵀx, flipping the signs of x̂ preserves the graph power spectrum
  (hence Moran's I and Geary's c) of x exactly. For a statistic xᵀMy, the null variance when
  the signs of both x̂ and ŷ are flipped independently is Σ_k,l x̂_k² ŷ_l² (u_kᵀ M u_l)²,
  which reduces to Σ_k h(μ_k)² x̂_k² ŷ_k² when M = h(𝓛). Two-sided normal p-values.

## 4. Axis A: calibration

* A1 (simulated): independent Gaussian random fields on the real spot coordinates of sections
  151507, 151669, 151673 (one per donor). Matérn ν = 0.5 and ν = 2.5, ranges 1, 3, 10 spot
  spacings, plus white noise. 1,000 independent (x, y) pairs per setting. Seed 20261122.
* A2 (real data, provably independent): gene x from a section of one donor and gene y from a
  section of a different donor, aligned on the shared Visium array grid and restricted to spots
  in tissue in both. Section pairs: (151507, 151673), (151508, 151674), (151509, 151669),
  (151510, 151670), (151671, 151675), (151672, 151676). Genes: the 50 genes with the highest
  Moran's I in each section (computed on that section alone). All 50 × 50 cross pairs.
* Endpoint: false-positive rate at α = 0.05 (also 0.01) with Wilson 95% CI. A test is called
  calibrated if the rate lies in Bradley's liberal interval [0.025, 0.075].

## 5. Axis B: DLPFC accuracy against the expert layer annotation (primary)

* Data: 12 sections, 3 donors; spots with a manual layer label (L1–L6, WM).
* Markers by leave-one-donor-out: for a test donor, use the 8 sections of the other two donors.
  Per section, compute each gene's mean log-expression in each layer. Specificity margin
  m(g, ℓ) = mean in ℓ − maximum mean in any other layer present, averaged over training
  sections containing ℓ. Each gene belongs to its best layer if that margin is > 0 and its
  mean in that layer is ≥ 0.1. Mitochondrial (MT-) genes are excluded. Top K = 10 genes per
  layer (sensitivity K = 5, 20).
* Test pairs: all pairs of markers whose layers are present in the test section with ≥ 30
  spots. Class by layer-index difference in the order L1 < … < L6 < WM: same (0),
  adjacent (1), distant (≥ 2).
* **Primary endpoint:** per-section AUROC of J_1 for adjacent (positive) versus distant pairs.
* **Primary comparison:** J_1 versus −L. Pre-specified success: AUROC(J_1) > 0.5 in at least 10
  of 12 sections and AUROC(J_1) > AUROC(−L) in at least 10 of 12 sections (one-sided sign test,
  P = 0.019).
* Secondary: AUROC adjacent versus same; ordinal concordance for adjacent > distant > same;
  results by donor and by layer distance; all comparators of §2; K and s sensitivity.
* B2 (semi-synthetic, known truth): on each section's real layer map, simulate 10 marker genes
  per present layer with Poisson counts (spot library sizes from the real section), a layer
  effect β ∈ {0.5, 1, 2} on the log scale softened by one smoothing step, and an independent
  Matérn (ν = 1.5, range 3) background with SD 0.3; mean count about 2 per spot. Five
  replicates per section and β. Same endpoints as B.

## 6. Axis C: single-cell downstream (seqFISH mouse embryo, exploratory)

* Cell types: `celltype_mapped_refined`; types with < 50 cells in an embryo are excluded there.
* Ground truth without expression: label-based contact enrichment z between cell types A ≠ B
  (observed A–B edges vs 1,000 label permutations). Contact if z ≥ 3; apart if z ≤ −3.
* Markers by leave-one-embryo-out: specificity margin as in §5 over the other two embryos,
  K = 3 genes per type.
* Classes: same-type marker pairs; contact-type pairs; apart-type pairs.
* Endpoints: AUROC contact versus apart; AUROC contact versus same. Methods as in §2.

## 7. Axis D: speed

* Section 151673; p ∈ {100, 250, 500, 1000, 2000, 4000} most highly expressed genes.
* Time all-pairs scores for J_1, J_0, Lee's L, bivariate Moran, Pearson from a prebuilt graph
  and standardized matrix; time spectral p-values for all pairs (including the one-time
  eigendecomposition) and 999-permutation p-values (measured directly at p = 100).
* One BLAS thread; median of 3 warm repeats; hardware reported.

## 8. Reporting rules

All per-section, per-replicate and per-pair outputs are written to `results/`. No parameter is
tuned on Axis A–C outcomes. Unfavourable results are reported in the main text.
