test_that("regional scores equal the explicit edge-sum definition", {
  g <- spatial_graph(cbind(1:10, rep(0,10)), neighbors=2)
  X <- cbind(sin(1:10), cos(1:10), rep(1.23, 10))
  observed <- regional_juxtaposition_matrix(X, g, scales=c(1,2))
  ij <- which(as.matrix(g) > 0 & upper.tri(as.matrix(g)), arr.ind=TRUE)
  z <- standardize_cols(X)[,1:2]
  step <- graph_smoother(g)
  literal <- numeric(2)
  for (k in 1:2) {
    z <- as.matrix(step %*% z)
    f <- z[ij[,1],,drop=FALSE] + z[ij[,2],,drop=FALSE]
    literal[k] <- 2 * sum(f[,1] * f[,2]) / sum(f^2)
  }
  expect_equal(observed[1,2],mean(literal), tolerance=1e-12)
  expect_true(all(is.na(observed[,3])))
  expect_error(regional_juxtaposition_matrix(X,g,scales=c(1,1)), "distinct")
})
