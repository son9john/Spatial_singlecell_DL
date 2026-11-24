# Shared above- or below-average expression on edges of a spatial graph.
regional_juxtaposition_matrix <- function(X, graph, scales = c(1L, 2L, 4L, 8L, 16L)) {
  if (!is.numeric(scales) || !length(scales) || any(!is.finite(scales)) ||
      any(scales < 0 | scales > .Machine$integer.max | scales != floor(scales)) || anyDuplicated(scales))
    stop("scales must contain distinct nonnegative integers")
  scales <- sort(as.integer(scales))
  Z <- standardize_cols(X)
  valid <- attr(Z, "ok")
  a <- .check_graph(graph, nrow(Z))
  S <- graph_smoother(a)
  Q <- Matrix::Diagonal(x = Matrix::rowSums(a)) + a
  total <- matrix(0, ncol(Z), ncol(Z))
  previous <- 0L
  for (t in scales) {
    for (k in seq_len(t - previous)) Z <- as.matrix(S %*% Z)
    G <- as.matrix(crossprod(Z, as.matrix(Q %*% Z)))
    energy <- pmax(diag(G), 0)
    denominator <- outer(energy, energy, "+")
    V <- (G + t(G)) / denominator
    V[denominator <= 0] <- NA_real_
    total <- total + V
    previous <- t
  }
  .mask(pmin(pmax(total / length(scales), -1), 1), valid, colnames(X))
}
