# juxtaposed 0.4.0

* 22 November 2026. Adds regional concordance across graph diffusion scales and improves numerical centering of constant features.

# juxtaposed 0.3.0

* `juxtaposition_matrix()`, `juxtaposition()` and `spectral_test_j()` gain `power` (power mean of
  the two gradient energies; `power = 0` is negative codispersion). The spectral z and p do not
  depend on `power`.
* `laplacian = "normalized"` (symmetric normalized Laplacian) for J and `spectral_null()`, for which
  J = 2 (I_xy - r) / ((1 - I_x) + (1 - I_y)) holds exactly on any graph.
* `score_edges()` gains `power` and `smoother = "lazy"` (the lazy random walk of the known-truth
  benchmark); `edges_to_nodes()` averages edge scores per spot for plotting.
* `lazy_walk()` and `power_mean()` are exported.

# juxtaposed 0.2.0

* All-pairs J, Lee's L, bivariate Moran's I, Pearson r, codispersion and gradient co-energy
  on one graph.
* Exact spectral sign-flip null (`spectral_null()`, `spectral_test_*()`) with closed-form
  variances: calibrated p-values for every pair without permutations.
* Smoothing uses uniform heat diffusion S = I - L/(2 d_max), so smoothed J is an exact
  spectral filter. Results at `scale = 0` are unchanged from 0.1.
* Numerical parity with the Python package `juxtaposed` 0.2.0 is tested against shared fixtures.
