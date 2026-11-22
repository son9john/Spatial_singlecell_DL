\name{regional_juxtaposition_matrix}
\alias{regional_juxtaposition_matrix}
\title{Regional concordance of two spatial expression patterns}
\description{Average energy-normalized concordance of edge sums across diffusion scales. Positive values can reflect two genes being jointly above or jointly below their means.}
\usage{regional_juxtaposition_matrix(X, graph, scales = c(1L, 2L, 4L, 8L, 16L))}
\arguments{
\item{X}{Observation-by-feature expression matrix.}
\item{graph}{Symmetric, nonnegative spatial adjacency matrix with zero diagonal.}
\item{scales}{Distinct nonnegative integer heat-diffusion steps.}
}
\value{Feature-by-feature matrix with entries in [-1, 1]. Comparisons involving a constant gene or a zero denominator return NA.}
\seealso{\code{\link{juxtaposition_matrix}} for signed opposing transitions.}
\examples{
g <- spatial_graph(cbind(1:8, rep(0, 8)), neighbors = 2)
X <- cbind(a = 1:8, b = 8:1)
regional_juxtaposition_matrix(X, g)
}
