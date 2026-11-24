\name{spatial_graph}
\alias{spatial_graph}
\alias{visium_graph}
\alias{graph_laplacian}
\alias{graph_smoother}
\alias{lazy_walk}
\title{Spatial graphs, Laplacian and smoothing operator}
\description{
\code{spatial_graph} builds the symmetric union of k-nearest-neighbour edges (binary weights),
optionally dropping edges longer than \code{radius}. \code{visium_graph} builds first-ring
hexagonal adjacency from Visium array coordinates. \code{graph_laplacian} returns
\eqn{L = D - A}; \code{graph_smoother} returns the uniform heat-diffusion step
\eqn{S = I - L/(2 d_{max})}, which is symmetric, doubly stochastic and a polynomial in \eqn{L}.
With \code{laplacian = "normalized"} they use \eqn{I - D^{-1/2} A D^{-1/2}} and
\eqn{I - L_{sym}/2}. \code{lazy_walk} returns \eqn{(I + D^{-1}A)/2} for local edge maps.
}
\usage{
spatial_graph(coords, neighbors = 6L, radius = NULL)
visium_graph(array_row, array_col)
graph_laplacian(graph, laplacian = c("combinatorial", "normalized"))
graph_smoother(graph, laplacian = c("combinatorial", "normalized"))
lazy_walk(graph)
}
\arguments{
\item{coords}{numeric matrix of coordinates (observations by spatial axes).}
\item{neighbors}{number of nearest neighbours per observation.}
\item{radius}{optional maximum edge length in coordinate units.}
\item{array_row, array_col}{integer Visium array coordinates.}
\item{graph}{symmetric adjacency matrix with zero diagonal (base or \pkg{Matrix}).}
\item{laplacian}{\code{"combinatorial"} (\eqn{D - A}) or \code{"normalized"}.}
}
\value{A sparse \code{dgCMatrix} adjacency matrix; for \code{graph_laplacian} a list with
\code{L} and \code{degree}; for \code{graph_smoother} a sparse matrix.}
\examples{
xy <- as.matrix(expand.grid(1:10, 1:10))
g <- spatial_graph(xy, neighbors = 4)
g2 <- visium_graph(c(0, 0, 1), c(0, 2, 1))
S <- graph_smoother(g)
}
