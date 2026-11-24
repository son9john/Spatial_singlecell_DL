\name{juxtaposed_value}
\alias{juxtaposed_value}
\alias{score_edges}
\alias{edges_to_nodes}
\title{J for increments and local J on graph edges}
\description{
\code{juxtaposed_value} computes \eqn{-C/M_p(A, B)} from paired increments, where
\eqn{A = \sum w dx^2}, \eqn{B = \sum w dy^2}, \eqn{C = \sum w dx dy} and \eqn{M_p} is the
power mean (\code{power = 1} gives J, \code{power = 0} signed codispersion).
\code{score_edges} returns local J for one pair on every undirected edge, pooling the edges
incident to both endpoints; \code{ridge} shrinks low-energy windows toward zero, and
\code{smoother = "lazy"} diffuses with \eqn{(I + D^{-1}A)/2} instead of the heat step (the
setting of the known-truth benchmark is \code{scale = 4, ridge = 1, smoother = "lazy"}).
The \code{signed_evidence} column is the unnormalized signed cross-gradient energy, which
localized interfaces better than the bounded J in known-truth tests.
\code{edges_to_nodes} averages an edge column over the edges incident to each node.
}
\usage{
juxtaposed_value(dx, dy, weights = NULL, power = 1)
score_edges(X, graph, scale = 0L, ridge = 0, standardized = FALSE, power = 1,
            smoother = c("heat", "lazy"))
edges_to_nodes(edges, n, field = "j")
}
\arguments{
\item{dx, dy}{numeric increments of the two features on the same oriented edges.}
\item{weights}{optional nonnegative edge weights.}
\item{power}{power-mean exponent (nonnegative).}
\item{X}{two-column numeric matrix.}
\item{graph}{symmetric adjacency matrix.}
\item{scale}{number of smoothing steps.}
\item{ridge}{nonnegative shrinkage toward zero for low-energy windows.}
\item{standardized}{set \code{TRUE} if \code{X} is already column-standardized.}
\item{smoother}{diffusion applied \code{scale} times: \code{"heat"} or \code{"lazy"}.}
\item{edges}{data frame returned by \code{score_edges}.}
\item{n}{number of nodes in the graph.}
\item{field}{edge column to average (\code{"j"}, \code{"signed_evidence"} or \code{"energy"}).}
}
\value{A number for \code{juxtaposed_value}; a data frame with edge endpoints (one-based),
local J, signed evidence and energy for \code{score_edges}.}
\examples{
juxtaposed_value(c(1, -3, 5), c(-1, 3, -5))
xy <- as.matrix(expand.grid(1:6, 1:6))
g <- spatial_graph(xy, neighbors = 4)
head(score_edges(cbind(xy[, 1], -xy[, 1]), g))
}
