# Calibrated inference: exact spectral sign-flip (Moran spectral randomization) null with
# closed-form variances, plus a classical spot-permutation test for comparison.

#' @rdname spectral_null
spectral_null <- function(graph, max_nodes = 20000L,
                          laplacian = c("combinatorial", "normalized")) {
  laplacian <- .laplacian_kind(laplacian)
  a <- .check_graph(graph)
  n <- nrow(a)
  if (n > max_nodes) {
    stop(sprintf(paste0("%d nodes exceeds max_nodes = %d for the exact spectral null; ",
                        "aggregate cells into spatial bins or raise max_nodes"), n, max_nodes))
  }
  lp <- graph_laplacian(a, laplacian)
  ev <- eigen(as.matrix(lp$L), symmetric = TRUE)
  ord <- rev(seq_len(n))
  mu <- pmax(ev$values[ord], 0)
  # Repeated eigenvalues: randomize by rotation inside the eigenspace (basis-invariant).
  tol <- 1e-8 * max(1, max(mu))
  grp <- cumsum(c(TRUE, diff(mu) > tol))
  structure(list(mu = mu, U = ev$vectors[, ord, drop = FALSE], graph = a,
                 dmax = if (!n) 0 else if (laplacian == "normalized") 1 else max(lp$degree),
                 laplacian = laplacian, group = grp,
                 degenerate = anyDuplicated(grp) > 0, mhat = new.env()),
            class = "juxtaposed_null")
}

.sq <- function(null, Xh) {
  A2 <- Xh^2
  if (!isTRUE(null$degenerate)) return(A2)
  s <- rowsum(A2, null$group, reorder = TRUE)
  m <- as.vector(table(null$group))
  (s / m)[null$group, , drop = FALSE]
}

.spectral_filter <- function(null, scale) {
  if (null$dmax == 0) return(rep(0, length(null$mu)))
  null$mu * (1 - null$mu / (2 * null$dmax))^(2L * as.integer(scale))
}

.coefficients <- function(null, X, standardized) {
  if (standardized) {
    Z <- as.matrix(X)
    ok <- rep(TRUE, ncol(Z))
  } else {
    Z <- standardize_cols(X)
    ok <- attr(Z, "ok")
    attr(Z, "ok") <- NULL
  }
  if (nrow(Z) != nrow(null$U)) stop("values and graph have different numbers of observations")
  list(Xh = crossprod(null$U, Z), ok = ok, names = colnames(as.matrix(X)))
}

.finish <- function(stat, sdv, ok, names) {
  z <- matrix(NA_real_, nrow(stat), ncol(stat))
  pos <- is.finite(sdv) & sdv > 0
  z[pos] <- stat[pos] / sdv[pos]
  p <- 2 * stats::pnorm(-abs(z))
  out <- list(stat = stat, sd = sdv, z = z, p = p)
  lapply(out, function(m) {
    m <- .mask(m, ok, names)
    diag(m) <- NA_real_
    m
  })
}

#' @rdname spectral_null
spectral_test_j <- function(null, X, scale = 1L, standardized = FALSE, power = 1) {
  cf <- .coefficients(null, X, standardized)
  h <- .spectral_filter(null, scale)
  G <- crossprod(cf$Xh, h * cf$Xh)
  e <- pmax(diag(G), 0)
  den <- outer(e, e, function(a, b) power_mean(a, b, power))
  A2 <- .sq(null, cf$Xh)
  V <- crossprod(A2, (h^2) * A2)
  J <- matrix(NA_real_, nrow(G), ncol(G))
  sdv <- matrix(NA_real_, nrow(G), ncol(G))
  pos <- den > 0
  J[pos] <- -G[pos] / den[pos]
  sdv[pos] <- sqrt(pmax(V[pos], 0)) / den[pos]
  .finish(J, sdv, cf$ok, cf$names)
}

.quadratic_test <- function(null, X, key, M, norm, standardized) {
  cf <- .coefficients(null, X, standardized)
  if (!exists(key, envir = null$mhat, inherits = FALSE)) {
    Mh <- crossprod(null$U, as.matrix(M %*% null$U))
    assign(key, (Mh + t(Mh)) / 2, envir = null$mhat)
  }
  Mh <- get(key, envir = null$mhat)
  stat <- crossprod(cf$Xh, Mh %*% cf$Xh) / norm
  A2 <- .sq(null, cf$Xh)
  V <- crossprod(A2, (Mh * Mh) %*% A2)
  .finish(stat, sqrt(pmax(V, 0)) / norm, cf$ok, cf$names)
}

#' @rdname spectral_null
spectral_test_lee <- function(null, X, standardized = FALSE) {
  W <- .row_standardize(null$graph)
  rows <- Matrix::rowSums(W)
  .quadratic_test(null, X, "lee", Matrix::crossprod(W), sum(rows^2), standardized)
}

#' @rdname spectral_null
spectral_test_moran <- function(null, X, standardized = FALSE) {
  W <- .row_standardize(null$graph)
  .quadratic_test(null, X, "moran", (W + Matrix::t(W)) / 2, sum(W), standardized)
}

#' @rdname spectral_null
spectral_test_pearson <- function(null, X, standardized = FALSE) {
  cf <- .coefficients(null, X, standardized)
  n <- nrow(null$U)
  stat <- crossprod(cf$Xh) / n
  A2 <- .sq(null, cf$Xh)
  .finish(stat, sqrt(pmax(crossprod(A2), 0)) / n, cf$ok, cf$names)
}

#' @rdname spectral_null
permutation_test <- function(x, y, graph, statistic = c("J", "L", "I", "r"), scale = 1L,
                             n_perm = 999L, seed = 0L) {
  statistic <- match.arg(statistic)
  a <- .check_graph(graph)
  zx <- as.vector(standardize_cols(x))
  zy <- as.vector(standardize_cols(y))
  n <- length(zx)
  if (!is.null(seed)) set.seed(seed)
  perms <- replicate(n_perm, sample.int(n))
  yp <- matrix(zy[perms], nrow = n)
  if (statistic == "J") {
    S <- graph_smoother(a)
    lap <- graph_laplacian(a)$L
    for (k in seq_len(as.integer(scale))) {
      zx <- as.vector(S %*% zx)
      zy <- as.vector(S %*% zy)
      yp <- as.matrix(S %*% yp)
    }
    lzx <- as.vector(lap %*% zx)
    ex <- sum(zx * lzx)
    obs <- -2 * sum(zy * lzx) / (ex + sum(zy * as.vector(lap %*% zy)))
    nul <- -2 * as.vector(crossprod(yp, lzx)) / (ex + colSums(yp * as.matrix(lap %*% yp)))
  } else if (statistic == "L") {
    W <- .row_standardize(a)
    v <- as.vector(Matrix::crossprod(W, W %*% zx)) / sum(Matrix::rowSums(W)^2)
    obs <- sum(zy * v)
    nul <- as.vector(crossprod(yp, v))
  } else if (statistic == "I") {
    W <- .row_standardize(a)
    v <- 0.5 * as.vector(W %*% zx + Matrix::crossprod(W, zx)) / sum(W)
    obs <- sum(zy * v)
    nul <- as.vector(crossprod(yp, v))
  } else {
    obs <- sum(zx * zy) / n
    nul <- as.vector(crossprod(yp, zx)) / n
  }
  list(statistic = obs, p_value = (1 + sum(abs(nul) >= abs(obs) - 1e-15)) / (length(nul) + 1))
}
