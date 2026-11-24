grid_data <- function() {
  set.seed(1)
  xy <- as.matrix(expand.grid(1:20, 1:20))
  g <- spatial_graph(xy, neighbors = 4)
  x <- sin(xy[, 1] / 3) + 0.3 * rnorm(400)
  y <- cos(xy[, 2] / 4) + 0.3 * rnorm(400)
  list(xy = xy, g = g, X = cbind(x = x, y = y, negx = -x, affx = 2 * x + 5, noise = rnorm(400)))
}

test_that("basic properties of J hold", {
  d <- grid_data()
  J <- juxtaposition_matrix(d$X, d$g, scale = 0)
  expect_equal(unname(diag(J)), rep(-1, 5))
  expect_equal(J["x", "negx"], 1)
  expect_equal(J["x", "affx"], -1)
  expect_equal(J, t(J))
  expect_true(max(abs(J), na.rm = TRUE) <= 1 + 1e-12)
})

test_that("J equals the edge-sum definition and juxtaposed_value", {
  d <- grid_data()
  Z <- standardize_cols(d$X[, 1:2])
  tr <- methods::as(Matrix::triu(d$g, k = 1), "TsparseMatrix")
  i <- tr@i + 1L
  j <- tr@j + 1L
  dx <- Z[j, 1] - Z[i, 1]
  dy <- Z[j, 2] - Z[i, 2]
  expected <- -2 * sum(dx * dy) / sum(dx^2 + dy^2)
  expect_equal(juxtaposition_score(d$X[, 1], d$X[, 2], d$g, scale = 0), expected)
  expect_equal(juxtaposed_value(dx, dy), expected)
})

test_that("regular-graph identity J = (I_xy - r) / (1 - mean I) holds", {
  nr <- 12
  nc <- 12
  idx <- matrix(seq_len(nr * nc), nr, nc, byrow = TRUE)
  ii <- integer(0)
  jj <- integer(0)
  for (r in 0:(nr - 1)) for (cc in 0:(nc - 1)) {
    offs <- list(c(0, 1), c(1, 0), c(1, if (r %% 2 == 0) -1 else 1))
    for (o in offs) {
      ii <- c(ii, idx[r + 1, cc + 1])
      jj <- c(jj, idx[(r + o[1]) %% nr + 1, (cc + o[2]) %% nc + 1])
    }
  }
  a <- Matrix::sparseMatrix(i = c(ii, jj), j = c(jj, ii), x = 1, dims = c(nr * nc, nr * nc))
  a <- methods::as(methods::as(methods::as(a, "dMatrix"), "generalMatrix"), "CsparseMatrix")
  a@x[] <- 1
  expect_true(all(Matrix::rowSums(a) == 6))
  set.seed(3)
  S <- graph_smoother(a)
  X <- matrix(rnorm(nr * nc * 3), ncol = 3)
  X <- as.matrix(S %*% (S %*% X)) + 0.2 * X
  J <- juxtaposition_matrix(X, a, scale = 0)
  I <- moran_bv_matrix(X, a)
  r <- pearson_matrix(X)
  for (p in 1:3) for (q in 1:3) {
    expect_equal(J[p, q], (I[p, q] - r[p, q]) / (1 - 0.5 * (I[p, p] + I[q, q])))
  }
})

test_that("spectral statistics equal the direct computation", {
  d <- grid_data()
  nl <- spectral_null(d$g)
  off <- row(diag(5)) != col(diag(5))
  for (s in c(0L, 1L, 3L)) {
    expect_equal(spectral_test_j(nl, d$X, scale = s)$stat[off],
                 juxtaposition_matrix(d$X, d$g, scale = s)[off], tolerance = 1e-8)
  }
  expect_equal(spectral_test_lee(nl, d$X)$stat[off], lee_l_matrix(d$X, d$g)[off], tolerance = 1e-8)
  expect_equal(spectral_test_moran(nl, d$X)$stat[off], moran_bv_matrix(d$X, d$g)[off],
               tolerance = 1e-8)
  expect_equal(spectral_test_pearson(nl, d$X)$stat[off], pearson_matrix(d$X)[off],
               tolerance = 1e-8)
})

test_that("permutation test, tables and zero-variance handling work", {
  d <- grid_data()
  pt <- permutation_test(d$X[, 1], d$X[, 2], d$g, statistic = "J", n_perm = 99L)
  expect_true(pt$p_value > 0 && pt$p_value <= 1)
  for (s in c("L", "I", "r")) {
    expect_true(permutation_test(d$X[, 1], d$X[, 2], d$g, statistic = s, n_perm = 99L)$p_value <= 1)
  }
  tab <- juxtaposition(d$X, d$g)
  expect_equal(nrow(tab), 10)
  expect_true(all(c("J", "z", "p", "q") %in% names(tab)))
  expect_false(is.unsorted(rev(tab$J), na.rm = TRUE))
  cmp <- compare_indices(d$X, d$g)
  expect_true(all(c("lee_L", "moran_I", "pearson_r", "neg_codispersion") %in% names(cmp)))
  X2 <- cbind(d$X[, 1:2], const = 1)
  J2 <- juxtaposition_matrix(X2, d$g)
  expect_true(is.na(J2[1, 3]) && is.finite(J2[1, 2]))
})
