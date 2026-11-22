\name{spectral_null}
\alias{spectral_null}
\alias{spectral_test_j}
\alias{spectral_test_lee}
\alias{spectral_test_moran}
\alias{spectral_test_pearson}
\alias{permutation_test}
\title{Calibrated tests for spatially autocorrelated features}
\description{
\code{spectral_null} eigendecomposes the graph Laplacian once. The \code{spectral_test_*}
functions then return statistics, null standard deviations, z-scores and two-sided p-values
for all feature pairs under Moran spectral randomization: the signs of the features'
graph-Fourier coefficients are flipped, which preserves each feature's graph power spectrum
(and so its Moran's I) exactly. The null variance of a quadratic form \eqn{x^T M y} is
\eqn{\sum_{k,l} \hat x_k^2 \hat y_l^2 (u_k^T M u_l)^2}, which needs no permutations.
\code{permutation_test} is the classical spot-permutation test for one pair, provided for
comparison; it is not calibrated for autocorrelated features.
}
\usage{
spectral_null(graph, max_nodes = 20000L, laplacian = c("combinatorial", "normalized"))
spectral_test_j(null, X, scale = 1L, standardized = FALSE, power = 1)
spectral_test_lee(null, X, standardized = FALSE)
spectral_test_moran(null, X, standardized = FALSE)
spectral_test_pearson(null, X, standardized = FALSE)
permutation_test(x, y, graph, statistic = c("J", "L", "I", "r"), scale = 1L,
                 n_perm = 999L, seed = 0L)
}
\arguments{
\item{graph}{symmetric adjacency matrix.}
\item{max_nodes}{maximum number of nodes for the dense eigendecomposition.}
\item{laplacian}{\code{"combinatorial"} or \code{"normalized"}; must match the J being tested.}
\item{power}{power-mean exponent of J's denominator; z and p do not depend on it, because the
energies are invariant under the sign-flip null.}
\item{null}{object returned by \code{spectral_null} for the same graph.}
\item{X}{numeric matrix, observations by features.}
\item{scale}{number of smoothing steps for J.}
\item{standardized}{set \code{TRUE} if \code{X} is already column-standardized.}
\item{x, y}{numeric vectors for one pair.}
\item{statistic}{one of \code{"J"}, \code{"L"}, \code{"I"}, \code{"r"}.}
\item{n_perm}{number of permutations.}
\item{seed}{random seed, or \code{NULL} to leave the RNG state alone.}
}
\value{For \code{spectral_null}, an object of class \code{juxtaposed_null}. For the tests, a
list of matrices \code{stat}, \code{sd}, \code{z} and \code{p}. For \code{permutation_test}, a
list with \code{statistic} and \code{p_value}.}
\references{
Wagner HH, Dray S (2015). Generating spatially constrained null models for irregularly spaced
data using Moran spectral randomization methods. Methods Ecol Evol 6:1169-1178.
}
\examples{
xy <- as.matrix(expand.grid(1:10, 1:10))
g <- spatial_graph(xy, neighbors = 4)
X <- cbind(a = sin(xy[, 1] / 2), b = cos(xy[, 2] / 3), c = rnorm(100))
nl <- spectral_null(g)
spectral_test_j(nl, X)$p
spectral_test_lee(nl, X)$p
}
