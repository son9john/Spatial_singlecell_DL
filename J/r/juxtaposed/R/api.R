# High-level entry points and local (edge-level) values.

.pair_index <- function(p) {
  iu <- which(upper.tri(matrix(0, p, p)), arr.ind = TRUE)
  iu[order(iu[, 1], iu[, 2]), , drop = FALSE]
}

#' @rdname juxtaposition
juxtaposition <- function(X, graph, features = colnames(X), scale = 1L, test = TRUE,
                          null = NULL, max_nodes = 20000L, power = 1,
                          laplacian = c("combinatorial", "normalized")) {
  laplacian <- .laplacian_kind(laplacian)
  X <- as.matrix(X)
  if (is.null(features)) features <- paste0("f", seq_len(ncol(X)) - 1L)
  if (length(features) != ncol(X)) stop("features must have one name per column")
  J <- juxtaposition_matrix(X, graph, scale = scale, power = power, laplacian = laplacian)
  iu <- .pair_index(ncol(X))
  out <- data.frame(feature_x = features[iu[, 1]], feature_y = features[iu[, 2]],
                    J = J[iu], stringsAsFactors = FALSE)
  if (isTRUE(test)) {
    if (is.null(null)) null <- spectral_null(graph, max_nodes = max_nodes, laplacian = laplacian)
    if (!identical(null$laplacian, laplacian)) stop("the supplied null uses a different laplacian")
    tt <- spectral_test_j(null, X, scale = scale, power = power)
    out$z <- tt$z[iu]
    out$p <- tt$p[iu]
    out$q <- stats::p.adjust(out$p, method = "BH")
  }
  out <- out[order(-out$J, na.last = TRUE), , drop = FALSE]
  rownames(out) <- NULL
  out
}

#' @rdname juxtaposition
compare_indices <- function(X, graph, features = colnames(X), scale = 1L) {
  X <- as.matrix(X)
  if (is.null(features)) features <- paste0("f", seq_len(ncol(X)) - 1L)
  iu <- .pair_index(ncol(X))
  out <- data.frame(
    feature_x = features[iu[, 1]], feature_y = features[iu[, 2]],
    J = juxtaposition_matrix(X, graph, scale = scale)[iu],
    lee_L = lee_l_matrix(X, graph)[iu],
    moran_I = moran_bv_matrix(X, graph)[iu],
    pearson_r = pearson_matrix(X)[iu],
    neg_codispersion = codispersion_matrix(X, graph, scale = scale)[iu],
    stringsAsFactors = FALSE)
  out <- out[order(-out$J, na.last = TRUE), , drop = FALSE]
  rownames(out) <- NULL
  out
}

#' @rdname juxtaposed_value
juxtaposed_value <- function(dx, dy, weights = NULL, power = 1) {
  dx <- as.numeric(dx)
  dy <- as.numeric(dy)
  if (length(dx) != length(dy) || any(!is.finite(c(dx, dy)))) {
    stop("expected matching, finite increment vectors")
  }
  w <- if (is.null(weights)) rep(1, length(dx)) else rep_len(as.numeric(weights), length(dx))
  if (any(w < 0) || any(!is.finite(w))) stop("weights must be nonnegative and finite")
  ex <- sum(w * dx^2)
  ey <- sum(w * dy^2)
  cr <- sum(w * dx * dy)
  sc <- power_mean(ex, ey, power)
  if (!isTRUE(sc > 0)) return(NA_real_)
  min(max(-cr / sc, -1), 1)
}

#' @rdname juxtaposed_value
score_edges <- function(X, graph, scale = 0L, ridge = 0, standardized = FALSE, power = 1,
                        smoother = c("heat", "lazy")) {
  smoother <- match.arg(smoother)
  if (ncol(as.matrix(X)) != 2L) stop("score_edges requires exactly two features")
  if (!is.finite(ridge) || ridge < 0) stop("ridge must be finite and nonnegative")
  if (smoother == "lazy") {
    p <- .prep(X, graph, 0L, standardized)
    P <- lazy_walk(p$graph)
    for (k in seq_len(as.integer(scale))) p$Z <- as.matrix(P %*% p$Z)
  } else {
    p <- .prep(X, graph, scale, standardized)
  }
  up <- .upper_edges(p$graph)
  i <- up$i
  j <- up$j
  w <- up$x
  n <- nrow(p$Z)
  dx <- p$Z[j, 1] - p$Z[i, 1]
  dy <- p$Z[j, 2] - p$Z[i, 2]
  B <- Matrix::sparseMatrix(i = c(i, j), j = rep(seq_along(i), 2L), x = 1,
                            dims = c(n, length(i)))
  pool <- function(v) {
    node <- as.vector(B %*% v)
    node[i] + node[j]
  }
  cross <- -2 * pool(w * dx * dy)
  ea <- pool(w * dx^2)
  eb <- pool(w * dy^2)
  mass <- pool(w)
  ref <- if (length(w)) sum(w * (dx^2 + dy^2)) / sum(w) else 0
  base <- power_mean(ea, eb, power)
  val <- rep(NA_real_, length(cross))
  pos <- base > 0
  val[pos] <- 0.5 * cross[pos] / (base[pos] + 0.5 * ridge * mass[pos] * ref)
  data.frame(from = i, to = j, j = pmin(pmax(val, -1), 1), signed_evidence = cross / mass,
             energy = (ea + eb) / mass)
}

#' @rdname juxtaposed_value
edges_to_nodes <- function(edges, n, field = "j") {
  v <- edges[[field]]
  ok <- is.finite(v)
  if (!any(ok)) return(rep(NA_real_, n))
  s <- rowsum(c(v[ok], v[ok]), c(edges$from[ok], edges$to[ok]))
  cnt <- rowsum(rep(1, 2 * sum(ok)), c(edges$from[ok], edges$to[ok]))
  out <- rep(NA_real_, n)
  idx <- as.integer(rownames(s))
  out[idx] <- s[, 1] / cnt[, 1]
  out
}
