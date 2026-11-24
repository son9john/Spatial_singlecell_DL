\name{juxtaposed-package}
\alias{juxtaposed-package}
\alias{juxtaposed}
\docType{package}
\title{The Juxtaposition Index J for Spatial Omics}
\description{
J measures whether two features change across the \emph{same} edges of a spatial graph and in
which direction. For standardized features smoothed by \code{scale} diffusion steps,
\deqn{J = -2 \sum_e w_e \Delta x_e \Delta y_e / \sum_e w_e (\Delta x_e^2 + \Delta y_e^2).}
J lies in [-1, 1]: positive values indicate opposing changes (complementary domains that
touch), negative values concordant changes, and values near zero no shared transitions.
On a regular graph \eqn{J = (I_{xy} - r)/(1 - (I_x + I_y)/2)}, where \eqn{I} is Moran's I with
row-standardized weights and \eqn{r} is Pearson correlation.

Calibrated p-values for all pairs come from an exact spectral sign-flip (Moran spectral
randomization) null with closed-form variances; see \code{\link{spectral_null}}.
}
\seealso{\code{\link{juxtaposition}}, \code{\link{juxtaposition_matrix}},
\code{\link{spectral_null}}, \code{\link{spatial_graph}}}
\keyword{package}
