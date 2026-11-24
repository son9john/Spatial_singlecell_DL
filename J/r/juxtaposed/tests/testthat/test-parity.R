# Cross-language parity: expected values were computed with the Python package juxtaposed 0.2.0.
fx <- function(name) as.matrix(utils::read.csv(test_path("fixtures", paste0(name, ".csv"))))

parity_inputs <- function() {
  co <- utils::read.csv(test_path("fixtures", "visium_coords.csv"))
  list(g = visium_graph(co$array_row, co$array_col), X = fx("X"))
}

cmp <- function(a, b, tol = 1e-9) {
  a <- unname(as.matrix(a))
  b <- unname(as.matrix(b))
  ok <- is.finite(a) & is.finite(b)
  expect_identical(is.finite(a), is.finite(b))
  expect_lt(max(abs(a[ok] - b[ok])), tol)
}

test_that("all-pairs indices match the Python implementation", {
  d <- parity_inputs()
  expect_equal(Matrix::nnzero(d$g) / 2, 492)
  cmp(juxtaposition_matrix(d$X, d$g, scale = 0), fx("J0"))
  cmp(juxtaposition_matrix(d$X, d$g, scale = 1), fx("J1"))
  cmp(juxtaposition_matrix(d$X, d$g, scale = 2), fx("J2"))
  cmp(codispersion_matrix(d$X, d$g, scale = 1), fx("codisp1"))
  cmp(gradient_coenergy_matrix(d$X, d$g, scale = 1), fx("coenergy1"))
  cmp(lee_l_matrix(d$X, d$g), fx("lee"))
  cmp(moran_bv_matrix(d$X, d$g), fx("moran"))
  cmp(pearson_matrix(d$X), fx("pearson"))
})

test_that("spectral-null z-scores and p-values match the Python implementation", {
  d <- parity_inputs()
  nl <- spectral_null(d$g)
  zj <- spectral_test_j(nl, d$X, scale = 1)
  off <- row(diag(5)) != col(diag(5))
  cmp(zj$z[off], fx("specJ1_z")[off], tol = 1e-6)
  cmp(zj$p[off], fx("specJ1_p")[off], tol = 1e-7)
  cmp(spectral_test_lee(nl, d$X)$z[off], fx("specLee_z")[off], tol = 1e-6)
  cmp(spectral_test_moran(nl, d$X)$z[off], fx("specMoran_z")[off], tol = 1e-6)
  cmp(spectral_test_pearson(nl, d$X)$z[off], fx("specPearson_z")[off], tol = 1e-6)
})

test_that("ranked table and local edges match the Python implementation", {
  d <- parity_inputs()
  tab <- juxtaposition(d$X, d$g)
  ref <- utils::read.csv(test_path("fixtures", "juxtaposition_table.csv"), stringsAsFactors = FALSE)
  key <- function(t) paste(pmin(t$feature_x, t$feature_y), pmax(t$feature_x, t$feature_y))
  m <- match(key(ref), key(tab))
  cmp(tab$J[m], ref$J)
  cmp(tab$q[m], ref$q, tol = 1e-7)
  e <- score_edges(d$X[, 1:2], d$g, scale = 1L, ridge = 1)
  er <- utils::read.csv(test_path("fixtures", "edges_ab_s1_r1.csv"))
  expect_equal(e$from, er$from)
  expect_equal(e$to, er$to)
  cmp(e$j, er$j)
})

test_that("kNN graphs match the Python implementation on tie-free coordinates", {
  xy <- as.matrix(utils::read.csv(test_path("fixtures", "random_coords.csv")))
  for (case in list(list(k = 4L, r = NULL, f = "knn4_edges"),
                    list(k = 6L, r = 1.5, f = "knn6_r15_edges"))) {
    g <- spatial_graph(xy, neighbors = case$k, radius = case$r)
    tr <- methods::as(Matrix::triu(g, k = 1), "TsparseMatrix")
    got <- data.frame(i = tr@i + 1L, j = tr@j + 1L)
    got <- got[order(got$i, got$j), ]
    ref <- utils::read.csv(test_path("fixtures", paste0(case$f, ".csv")))
    expect_equal(unname(as.matrix(got)), unname(as.matrix(ref)))
  }
})

test_that("0.3 options (power mean, normalized Laplacian, lazy local maps) match Python", {
  d <- parity_inputs()
  cmp(juxtaposition_matrix(d$X, d$g, scale = 1, power = 0.5), fx("J1_p05"))
  cmp(juxtaposition_matrix(d$X, d$g, scale = 1, laplacian = "normalized"), fx("Jnorm1"))
  nl <- spectral_null(d$g, laplacian = "normalized")
  off <- row(diag(5)) != col(diag(5))
  cmp(spectral_test_j(nl, d$X, scale = 1)$z[off], fx("specJnorm1_z")[off], tol = 1e-6)
  # z does not depend on the power mean
  cmp(spectral_test_j(nl, d$X, scale = 1, power = 0.25)$z[off], fx("specJnorm1_z")[off], tol = 1e-6)
  e <- score_edges(d$X[, 1:2], d$g, scale = 4L, ridge = 1, power = 0.5, smoother = "lazy")
  er <- utils::read.csv(test_path("fixtures", "edges_ab_lazy4_r1_p05.csv"))
  expect_equal(e$from, er$from)
  expect_equal(e$to, er$to)
  cmp(e$j, er$j)
  cmp(e$signed_evidence, er$signed_evidence)
  nodes <- edges_to_nodes(e, nrow(d$X))
  cmp(nodes, utils::read.csv(test_path("fixtures", "nodes_ab_lazy4_r1_p05.csv"))$node)
})
