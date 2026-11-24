# Spatial graphs, Laplacian and the smoothing operator shared by every statistic.

.as_dgc <- function(x) {
  methods::as(methods::as(methods::as(x, "dMatrix"), "generalMatrix"), "CsparseMatrix")
}

.check_graph <- function(graph, n = NULL) {
  a <- .as_dgc(graph)
  if (nrow(a) != ncol(a)) stop("graph must be square")
  if (!is.null(n) && nrow(a) != n) {
    stop(sprintf("graph has %d nodes but the data have %d observations", nrow(a), n))
  }
  if (length(a@x) && (any(a@x < 0) || any(!is.finite(a@x)))) {
    stop("graph weights must be finite and nonnegative")
  }
  if (length(a@x) && max(abs(a - Matrix::t(a))) > 1e-12) stop("graph must be symmetric")
  if (any(diag(a) != 0)) stop("graph must have a zero diagonal")
  Matrix::drop0(a)
}

#' @rdname spatial_graph
spatial_graph <- function(coords, neighbors = 6L, radius = NULL) {
  coords <- as.matrix(coords)
  storage.mode(coords) <- "double"
  n <- nrow(coords)
  if (n < 2L || any(!is.finite(coords))) {
    stop("coords must be a finite matrix with at least two rows")
  }
  neighbors <- as.integer(neighbors)
  if (is.na(neighbors) || neighbors < 1L) stop("neighbors must be a positive integer")
  k <- min(neighbors, n - 1L)
  if (requireNamespace("RANN", quietly = TRUE)) {
    nn <- RANN::nn2(coords, k = k + 1L)
    idx <- nn$nn.idx
    dst <- nn$nn.dists
  } else {
    d <- as.matrix(stats::dist(coords))
    idx <- matrix(0L, n, k + 1L)
    dst <- matrix(0, n, k + 1L)
    for (i in seq_len(n)) {
      o <- order(d[i, ])[seq_len(k + 1L)]
      idx[i, ] <- o
      dst[i, ] <- d[i, o]
    }
  }
  rows <- matrix(rep(seq_len(n), times = ncol(idx)), nrow = n)
  use <- rows != idx
  if (!is.null(radius)) {
    if (!is.finite(radius) || radius <= 0) stop("radius must be positive")
    use <- use & (dst <= radius)
  }
  a <- Matrix::sparseMatrix(i = rows[use], j = idx[use], x = 1, dims = c(n, n))
  a <- a + Matrix::t(a)
  a <- .as_dgc(a)
  a@x[] <- 1
  diag(a) <- 0
  Matrix::drop0(a)
}

#' @rdname spatial_graph
visium_graph <- function(array_row, array_col) {
  r <- as.integer(array_row)
  cc <- as.integer(array_col)
  n <- length(r)
  if (length(cc) != n) stop("array_row and array_col must have equal length")
  key <- paste(r, cc, sep = "_")
  if (anyDuplicated(key)) stop("Visium array coordinates must be unique")
  ii <- integer(0)
  jj <- integer(0)
  for (off in list(c(0L, 2L), c(1L, -1L), c(1L, 1L))) {
    m <- match(paste(r + off[1], cc + off[2], sep = "_"), key)
    ok <- !is.na(m)
    ii <- c(ii, which(ok))
    jj <- c(jj, m[ok])
  }
  a <- Matrix::sparseMatrix(i = c(ii, jj), j = c(jj, ii), x = 1, dims = c(n, n))
  a <- .as_dgc(a)
  if (length(a@x)) a@x[] <- 1
  a
}

.laplacian_kind <- function(laplacian) {
  match.arg(laplacian, c("combinatorial", "normalized"))
}

#' @rdname spatial_graph
graph_laplacian <- function(graph, laplacian = c("combinatorial", "normalized")) {
  laplacian <- .laplacian_kind(laplacian)
  a <- .check_graph(graph)
  deg <- Matrix::rowSums(a)
  if (laplacian == "normalized") {
    inv <- ifelse(deg > 0, 1 / sqrt(deg), 0)
    Dh <- Matrix::Diagonal(x = inv)
    L <- Matrix::Diagonal(x = as.numeric(deg > 0)) - Dh %*% a %*% Dh
    return(list(L = L, degree = deg))
  }
  list(L = Matrix::Diagonal(x = deg) - a, degree = deg)
}

#' @rdname spatial_graph
graph_smoother <- function(graph, laplacian = c("combinatorial", "normalized")) {
  laplacian <- .laplacian_kind(laplacian)
  lp <- graph_laplacian(graph, laplacian)
  n <- nrow(lp$L)
  if (laplacian == "normalized") return(Matrix::Diagonal(n) - lp$L / 2)
  dmax <- if (length(lp$degree)) max(lp$degree) else 0
  if (dmax == 0) return(Matrix::Diagonal(n))
  Matrix::Diagonal(n) - lp$L / (2 * dmax)
}

#' @rdname spatial_graph
lazy_walk <- function(graph) {
  a <- .check_graph(graph)
  deg <- Matrix::rowSums(a)
  inv <- ifelse(deg > 0, 1 / deg, 0)
  Matrix::Diagonal(x = ifelse(deg > 0, 0.5, 1)) + 0.5 * (Matrix::Diagonal(x = inv) %*% a)
}
