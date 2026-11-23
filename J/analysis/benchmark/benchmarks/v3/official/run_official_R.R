# Calibration of the official R tests on the exported cross-donor null pairs.
#   Rscript benchmarks/v3/official/run_official_R.R results/v3/official_inputs
# MERINGUE::spatialCrossCorTest(n = 1000) and spdep::lee.mc(nsim = 999, alternative = "greater").
# A seeded random subset of pairs per unit is tested (default 250; second argument).
suppressPackageStartupMessages({library(Matrix)})
args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args)) args[1] else "results/v3/official_inputs"
max_pairs <- if (length(args) > 1) as.integer(args[2]) else 250L  # R permutation tests are slow
pval <- function(r) if (is.list(r)) as.numeric(r$p.value) else as.numeric(r)
units <- list.dirs(root, recursive = FALSE)
have_meringue <- requireNamespace("MERINGUE", quietly = TRUE)
have_spdep <- requireNamespace("spdep", quietly = TRUE)
out <- list()
for (u in units) {
  expr <- as.matrix(read.csv(file.path(u, "expression.csv"), check.names = FALSE))
  pairs <- read.csv(file.path(u, "pairs.csv"))
  set.seed(20261122)
  if (nrow(pairs) > max_pairs) pairs <- pairs[sort(sample(nrow(pairs), max_pairs)), ]
  g <- as(readMM(file.path(u, "graph.mtx")), "CsparseMatrix")
  ids <- as.character(seq_len(nrow(expr)))
  dimnames(g) <- list(ids, ids)
  rownames(expr) <- ids
  res <- data.frame(unit = basename(u), pair = pairs$pair)
  if (have_meringue) {
    res$meringue_perm <- vapply(seq_len(nrow(pairs)), function(k) {
      x <- expr[, pairs$gene_x[k]]; y <- expr[, pairs$gene_y[k]]
      names(x) <- names(y) <- ids
      pval(MERINGUE::spatialCrossCorTest(x, y, g, n = 1000, plot = FALSE))
    }, numeric(1))
  }
  if (have_spdep) {
    lw <- spdep::mat2listw(as.matrix(g), style = "W")
    res$lee_perm <- vapply(seq_len(nrow(pairs)), function(k) {
      spdep::lee.mc(expr[, pairs$gene_x[k]], expr[, pairs$gene_y[k]], lw, nsim = 999,
                    alternative = "greater")$p.value
    }, numeric(1))
  }
  out[[u]] <- res
  message(basename(u), " done")
}
res <- do.call(rbind, out)
write.csv(res, file.path(dirname(root), "official_R_pvalues.csv"), row.names = FALSE)
res$mode <- sub(".*_", "", res$unit)
print(aggregate(. ~ mode, data = res[, intersect(c("mode", "meringue_perm", "lee_perm"), names(res))],
                FUN = function(p) mean(p < 0.05)))
