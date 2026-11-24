# Jetwork tutorial helpers. Requires Matrix and the juxtaposed R package.

.jetwork_expression <- function(expression, genes = colnames(expression)) {
  measured <- colnames(expression)
  if (is.null(measured) || anyNA(measured) || any(!nzchar(measured)) ||
      anyDuplicated(measured) || length(genes) < 2L) {
    stop("Expression must have unique gene column names and at least two genes")
  }
  absent <- setdiff(genes, measured)
  if (length(absent)) stop("Expression is missing fitted genes: ", paste(absent, collapse = ", "))
  x <- as.matrix(expression[, genes, drop = FALSE])
  if (!is.numeric(x) || nrow(x) < 2L || any(!is.finite(x))) {
    stop("Expression must contain finite numeric values for at least two sites")
  }
  x
}

.jetwork_smoothed <- function(expression, graph) {
  if (nrow(graph) != nrow(expression)) {
    stop("Graph rows and expression rows must represent the same sites")
  }
  if (!is.null(rownames(graph)) && !is.null(rownames(expression)) &&
      !identical(rownames(graph), rownames(expression))) {
    stop("Graph row names and expression row names must match in order")
  }
  as.matrix(juxtaposed::graph_smoother(graph) %*% juxtaposed::standardize_cols(expression))
}

.jetwork_scaled <- function(score) {
  score[!is.finite(score)] <- 0
  diag(score) <- 0
  size <- sqrt(sum(score^2))
  if (size == 0) score else score / size
}

.jetwork_walk <- function(score, partners) {
  retained <- matrix(0, nrow(score), ncol(score))
  for (i in seq_len(nrow(score))) {
    positive <- which(score[i, ] > 0)
    chosen <- head(positive[order(-score[i, positive], positive)], partners)
    retained[i, chosen] <- score[i, chosen]
  }
  symmetric <- Matrix::Matrix(pmax(retained, t(retained)), sparse = TRUE)
  d <- Matrix::rowSums(symmetric)
  Matrix::Diagonal(x = ifelse(d > 0, 1 / d, 0)) %*% symmetric
}

# Training is a list of sections with expression (sites x genes), graph,
# and group (donor/animal ID). Gene columns of subsequent sections are aligned.
fit_jetwork <- function(training, partners = 12L) {
  if (!length(training)) stop("Supply at least one training section")
  if (length(partners) != 1L || !is.numeric(partners) || !is.finite(partners) ||
      partners < 1 || partners != floor(partners)) stop("partners must be a positive integer")
  groups <- vapply(training, function(s) {
    if (length(s$group) != 1L || is.na(s$group) || !nzchar(s$group)) {
      stop("Each training section needs a biological group ID")
    }
    as.character(s$group)
  }, character(1))
  genes <- colnames(training[[1L]]$expression)
  score <- lapply(training, function(s) {
    x <- .jetwork_expression(s$expression, genes)
    smoothed <- .jetwork_smoothed(x, s$graph)
    z <- juxtaposed::standardize_cols(smoothed)
    co <- crossprod(z) / nrow(z)
    op <- juxtaposed::juxtaposition_matrix(smoothed, s$graph, scale = 0L, standardized = TRUE)
    list(co = .jetwork_scaled(co), op = .jetwork_scaled(op))
  })
  avg <- function(key) {
    by_group <- lapply(split(seq_along(groups), groups), function(idx) {
      Reduce("+", lapply(score[idx], "[[", key)) / length(idx)
    })
    Reduce("+", by_group) / length(by_group)
  }
  co <- .jetwork_walk(avg("co"), partners)
  op <- .jetwork_walk(avg("op"), partners)
  mix <- Matrix::Diagonal(x = 1 - 0.25 * Matrix::rowSums(co)) + 0.25 * co
  list(genes = genes, q = mix %*% op %*% mix, groups = unique(groups))
}

# Score each undirected edge once. i and j index input expression rows.
score_jetwork_edges <- function(model, expression, graph) {
  x <- .jetwork_expression(expression, model$genes)
  u <- .jetwork_smoothed(x, graph)
  partner <- as.matrix(u %*% Matrix::t(model$q))
  e <- methods::as(Matrix::drop0(Matrix::triu(graph, k = 1)), "TsparseMatrix")
  order <- order(e@i, e@j)
  i <- e@i[order] + 1L
  j <- e@j[order] + 1L
  sites <- if (is.null(rownames(x))) as.character(seq_len(nrow(x))) else rownames(x)
  scores <- data.frame(i = i, j = j, site_i = sites[i], site_j = sites[j],
                       jetwork = numeric(length(i)), gradient = numeric(length(i)))
  for (start in seq.int(1L, length(i) + 1L, by = 8192L)) {
    if (start > length(i)) break
    block <- seq.int(start, min(start + 8191L, length(i)))
    dx <- u[i[block], , drop = FALSE] - u[j[block], , drop = FALSE]
    dp <- partner[i[block], , drop = FALSE] - partner[j[block], , drop = FALSE]
    scores$jetwork[block] <- -rowMeans(dx * dp)
    scores$gradient[block] <- rowMeans(dx^2)
  }
  scores
}

# Inspect gene partners, propagating anchor weights along the fitted network.
rank_jetwork_genes <- function(model, anchors) {
  anchors <- unique(anchors)
  if (!length(anchors) || any(!anchors %in% model$genes)) stop("Supply genes present in the fitted panel")
  keep <- !model$genes %in% anchors
  weights <- Matrix::colMeans(model$q[match(anchors, model$genes), , drop = FALSE])
  rank <- order(-weights[keep], model$genes[keep])
  data.frame(gene = model$genes[keep][rank], score = weights[keep][rank])
}

# Named held-out annotations are aligned by site ID. Unknown labels are omitted.
jetwork_edge_truth <- function(edges, annotation) {
  if (is.null(names(annotation)) || anyNA(names(annotation)) || anyDuplicated(names(annotation))) {
    stop("Provide annotations named with unique expression row IDs")
  }
  left <- as.character(annotation[edges$site_i])
  right <- as.character(annotation[edges$site_j])
  ifelse(!is.na(left) & !is.na(right) & nzchar(left) & nzchar(right), left != right, NA)
}

evaluate_jetwork_edges <- function(edges, annotation, score = "jetwork", top = 0.10) {
  if (length(top) != 1L || !is.numeric(top) || !is.finite(top) || top <= 0 || top > 1) {
    stop("top must be in (0, 1]")
  }
  if (length(score) != 1L || !score %in% names(edges) ||
      !is.numeric(edges[[score]]) || any(!is.finite(edges[[score]]))) stop("Supply a finite score column")
  truth <- jetwork_edge_truth(edges, annotation)
  ok <- !is.na(truth)
  value <- edges[[score]][ok]
  y <- truth[ok]
  n_positive <- sum(y)
  n_negative <- sum(!y)
  if (!n_positive || !n_negative) stop("Need both within-label and cross-label edges to evaluate")
  order <- order(-value, edges$i[ok], edges$j[ok])
  ends <- cumsum(rle(value[order])$lengths)
  hits <- cumsum(y[order])[ends]
  average_precision <- sum(diff(c(0, hits)) / n_positive * hits / ends)
  auroc <- (sum(rank(value)[y]) - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
  budget <- ceiling(top * length(y))
  matched <- sum(y[order[seq_len(budget)]])
  data.frame(score = score, edges = length(y), annotated_transitions = n_positive,
             prevalence = mean(y), average_precision = average_precision, auroc = auroc,
             selected = budget, matched = matched,
             jaccard = matched / (budget + n_positive - matched))
}
