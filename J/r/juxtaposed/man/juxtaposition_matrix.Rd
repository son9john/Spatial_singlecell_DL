\name{juxtaposition_matrix}
\alias{juxtaposition_matrix}
\alias{juxtaposition_score}
\alias{codispersion_matrix}
\alias{gradient_coenergy_matrix}
\alias{lee_l_matrix}
\alias{moran_bv_matrix}
\alias{pearson_matrix}
\alias{standardize_cols}
\alias{power_mean}
\title{All-pairs spatial association indices on one graph}
\description{
\code{juxtaposition_matrix} returns J for every pair of columns after \code{scale} smoothing
steps. \code{codispersion_matrix} returns negative graph codispersion (same sign as J);
\code{gradient_coenergy_matrix} the sign-free overlap of gradient magnitudes;
\code{lee_l_matrix} Lee's L (Lee 2001) with row-standardized weights; \code{moran_bv_matrix}
the symmetrized bivariate Moran's I; \code{pearson_matrix} Pearson correlation.
\code{standardize_cols} centres and scales columns to unit population variance (zero-variance
columns become zero and give \code{NA} statistics). \code{power_mean} is the power mean of two
nonnegative energies used in the denominator of J (\code{power = 1}: arithmetic mean, J;
\code{power = 0}: geometric mean, negative codispersion). With
\code{laplacian = "normalized"} the increments are taken on the symmetric normalized
Laplacian, for which \eqn{J = 2(I_{xy} - r)/((1 - I_x) + (1 - I_y))} holds exactly on any graph.
}
\usage{
juxtaposition_matrix(X, graph, scale = 1L, standardized = FALSE, power = 1,
                     laplacian = c("combinatorial", "normalized"))
juxtaposition_score(x, y, graph, scale = 1L, power = 1,
                    laplacian = c("combinatorial", "normalized"))
codispersion_matrix(X, graph, scale = 1L, standardized = FALSE)
gradient_coenergy_matrix(X, graph, scale = 1L, standardized = FALSE)
lee_l_matrix(X, graph, standardized = FALSE)
moran_bv_matrix(X, graph, standardized = FALSE)
pearson_matrix(X)
standardize_cols(X)
power_mean(a, b, power = 1)
}
\arguments{
\item{X}{numeric matrix, observations (spots or cells) by features (genes).}
\item{x, y}{numeric vectors for a single pair.}
\item{graph}{symmetric adjacency matrix, e.g. from \code{\link{spatial_graph}}.}
\item{scale}{number of smoothing steps before differencing (0 = raw increments).}
\item{standardized}{set \code{TRUE} if \code{X} is already column-standardized.}
\item{power}{nonnegative power-mean exponent for the two gradient energies (default 1).}
\item{laplacian}{\code{"combinatorial"} (\eqn{D - A}, default) or \code{"normalized"}.}
\item{a, b}{nonnegative numeric vectors or matrices of energies.}
}
\value{A feature-by-feature numeric matrix (a single number for \code{juxtaposition_score}).}
\examples{
xy <- as.matrix(expand.grid(1:12, 1:12))
g <- spatial_graph(xy, neighbors = 4)
x <- as.numeric(xy[, 1] > 6)
X <- cbind(a = x, b = 1 - x, c = x + rnorm(144, sd = 0.1))
juxtaposition_matrix(X, g, scale = 0)
lee_l_matrix(X, g)
}
