# Pair-level spatial association indices on a shared graph.
# Signs: J > 0 = opposing changes across the same edges; J < 0 = concordant changes.

#' @rdname juxtaposition_matrix
standardize_cols <- function(X) {
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  if (nrow(X) < 2L) stop("values must have at least two observations")
  if (any(!is.finite(X))) stop("values must be finite")
  n <- nrow(X)
  # Remove a fixed column offset first to preserve truly constant inputs.
  X <- sweep(X, 2L, X[1L, ], "-")
  mu <- colMeans(X)
  sdv <- sqrt(colMeans((X - rep(mu, each = n))^2))
  ok <- sdv > 0
  Z <- matrix(0, n, ncol(X), dimnames = dimnames(X))
  if (any(ok)) {
    Z[, ok] <- (X[, ok, drop = FALSE] - rep(mu[ok], each = n)) / rep(sdv[ok], each = n)
  }
  attr(Z, "ok") <- ok
  Z
}

#' @rdname juxtaposition_matrix
power_mean <- function(a, b, power = 1) {
  if (!is.finite(power) || power < 0) stop("power must be finite and nonnegative")
  if (power == 1) return((a + b) / 2)
  if (power == 0) return(sqrt(a) * sqrt(b))
  hi <- pmax(a, b)
  lo <- pmin(a, b)
  rel <- ifelse(hi > 0, lo / hi, 0)
  hi * exp(log1p(0.5 * expm1(power * log(rel))) / power)
}

.prep <- function(X, graph, scale, standardized, laplacian = "combinatorial") {
  if (standardized) {
    Z <- as.matrix(X)
    storage.mode(Z) <- "double"
    ok <- rep(TRUE, ncol(Z))
  } else {
    Z <- standardize_cols(X)
    ok <- attr(Z, "ok")
    attr(Z, "ok") <- NULL
  }
  a <- .check_graph(graph, nrow(Z))
  scale <- as.integer(scale)
  if (length(scale) != 1L || is.na(scale) || scale < 0L) {
    stop("scale must be a nonnegative integer (number of smoothing steps)")
  }
  if (scale > 0L) {
    S <- graph_smoother(a, laplacian)
    for (k in seq_len(scale)) Z <- as.matrix(S %*% Z)
  }
  list(Z = Z, graph = a, ok = ok)
}

.graph_gram <- function(Z, graph, laplacian = "combinatorial") {
  lp <- graph_laplacian(graph, laplacian)
  G <- as.matrix(Matrix::crossprod(Z, as.matrix(lp$L %*% Z)))
  (G + t(G)) / 2
}

.mask <- function(M, ok, names = NULL) {
  M <- as.matrix(M)
  if (!all(ok)) {
    M[!ok, ] <- NA_real_
    M[, !ok] <- NA_real_
  }
  if (!is.null(names)) dimnames(M) <- list(names, names)
  M
}

.upper_edges <- function(a) {
  tr <- methods::as(Matrix::triu(a, k = 1), "TsparseMatrix")
  o <- order(tr@i, tr@j)
  list(i = tr@i[o] + 1L, j = tr@j[o] + 1L, x = tr@x[o])
}

.row_standardize <- function(a) {
  d <- Matrix::rowSums(a)
  inv <- ifelse(d > 0, 1 / d, 0)
  Matrix::Diagonal(x = inv) %*% a
}

#' @rdname juxtaposition_matrix
juxtaposition_matrix <- function(X, graph, scale = 1L, standardized = FALSE, power = 1,
                                 laplacian = c("combinatorial", "normalized")) {
  laplacian <- .laplacian_kind(laplacian)
  p <- .prep(X, graph, scale, standardized, laplacian)
  G <- .graph_gram(p$Z, p$graph, laplacian)
  e <- pmax(diag(G), 0)
  den <- outer(e, e, function(a, b) power_mean(a, b, power))
  J <- matrix(NA_real_, nrow(G), ncol(G))
  pos <- den > 0
  J[pos] <- -G[pos] / den[pos]
  J <- pmin(pmax(J, -1), 1)
  .mask(J, p$ok, colnames(as.matrix(X)))
}

#' @rdname juxtaposition_matrix
juxtaposition_score <- function(x, y, graph, scale = 1L, power = 1,
                                laplacian = c("combinatorial", "normalized")) {
  juxtaposition_matrix(cbind(x, y), graph, scale = scale, power = power,
                       laplacian = laplacian)[1, 2]
}

#' @rdname juxtaposition_matrix
codispersion_matrix <- function(X, graph, scale = 1L, standardized = FALSE) {
  p <- .prep(X, graph, scale, standardized)
  G <- .graph_gram(p$Z, p$graph)
  e <- pmax(diag(G), 0)
  den <- sqrt(outer(e, e))
  C <- matrix(NA_real_, nrow(G), ncol(G))
  pos <- den > 0
  C[pos] <- -G[pos] / den[pos]
  C <- pmin(pmax(C, -1), 1)
  .mask(C, p$ok, colnames(as.matrix(X)))
}

#' @rdname juxtaposition_matrix
gradient_coenergy_matrix <- function(X, graph, scale = 1L, standardized = FALSE) {
  p <- .prep(X, graph, scale, standardized)
  up <- .upper_edges(p$graph)
  D <- abs(p$Z[up$j, , drop = FALSE] - p$Z[up$i, , drop = FALSE]) * sqrt(up$x)
  G <- crossprod(D)
  e <- diag(G)
  den <- sqrt(outer(e, e))
  C <- matrix(NA_real_, nrow(G), ncol(G))
  pos <- den > 0
  C[pos] <- G[pos] / den[pos]
  .mask(C, p$ok, colnames(as.matrix(X)))
}

#' @rdname juxtaposition_matrix
lee_l_matrix <- function(X, graph, standardized = FALSE) {
  p <- .prep(X, graph, 0L, standardized)
  W <- .row_standardize(p$graph)
  n <- nrow(p$Z)
  rows <- Matrix::rowSums(W)
  WZ <- as.matrix(W %*% p$Z)
  L <- (n / sum(rows^2)) * crossprod(WZ) / n
  .mask(L, p$ok, colnames(as.matrix(X)))
}

#' @rdname juxtaposition_matrix
moran_bv_matrix <- function(X, graph, standardized = FALSE) {
  p <- .prep(X, graph, 0L, standardized)
  W <- .row_standardize(p$graph)
  s0 <- sum(W)
  M <- as.matrix(crossprod(p$Z, as.matrix(W %*% p$Z)))
  .mask(0.5 * (M + t(M)) / s0, p$ok, colnames(as.matrix(X)))
}

#' @rdname juxtaposition_matrix
pearson_matrix <- function(X) {
  Z <- standardize_cols(X)
  ok <- attr(Z, "ok")
  attr(Z, "ok") <- NULL
  .mask(crossprod(Z) / nrow(Z), ok, colnames(as.matrix(X)))
}
