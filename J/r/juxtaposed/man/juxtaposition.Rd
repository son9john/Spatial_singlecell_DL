\name{juxtaposition}
\alias{juxtaposition}
\alias{compare_indices}
\title{Ranked table of J for all feature pairs}
\description{
\code{juxtaposition} computes J for every pair of features and, by default, spectral-null
z-scores, p-values and Benjamini-Hochberg q-values. \code{compare_indices} returns J alongside
Lee's L, bivariate Moran's I, Pearson r and negative codispersion (the L-J map).
}
\usage{
juxtaposition(X, graph, features = colnames(X), scale = 1L, test = TRUE,
              null = NULL, max_nodes = 20000L, power = 1,
              laplacian = c("combinatorial", "normalized"))
compare_indices(X, graph, features = colnames(X), scale = 1L)
}
\arguments{
\item{X}{numeric matrix, observations by features (e.g. log-normalized counts).}
\item{graph}{symmetric adjacency matrix.}
\item{features}{feature names (defaults to column names).}
\item{scale}{number of smoothing steps before differencing.}
\item{test}{compute calibrated p-values with \code{\link{spectral_null}}.}
\item{null}{optional precomputed \code{spectral_null} for the same graph.}
\item{max_nodes}{maximum graph size for the exact spectral null.}
\item{power}{power-mean exponent of J's denominator (effect-size scale only).}
\item{laplacian}{\code{"combinatorial"} (default) or \code{"normalized"}.}
}
\value{A data frame with one row per unordered feature pair, sorted by decreasing J.}
\examples{
xy <- as.matrix(expand.grid(1:12, 1:12))
g <- spatial_graph(xy, neighbors = 4)
x <- as.numeric(xy[, 1] > 6)
X <- cbind(a = x + rnorm(144, sd = 0.2), b = 1 - x + rnorm(144, sd = 0.2),
           c = rnorm(144))
head(juxtaposition(X, g))
compare_indices(X, g)
}
