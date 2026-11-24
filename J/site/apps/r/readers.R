# Readers for the juxtaposed app: every input format becomes one common structure.
#
# A dataset is a list with
#   counts   genes x spots dgCMatrix of raw counts, or NULL when only normalized values exist
#   data     genes x spots dgCMatrix of log-normalized values (filled by normalize_dataset())
#   coords   data.frame(x, y): plotting coordinates (image pixels when an image is present)
#   array    data.frame(row, col) of lattice indices (Visium, Visium HD, binned data), or NULL
#   lattice  "hex" (Visium), "square" (Visium HD, bins) or "none" (cells, beads)
#   image    numeric array (height x width x 3) of the tissue image, or NULL
#   meta     data.frame of annotations, one row per spot (clusters, cell types, ...)
#   platform short description; unit: coordinate unit; notes: character vector of messages

"%||%" <- function(a, b) if (is.null(a) || length(a) == 0) b else a

# ---- uploads ----------------------------------------------------------------------------------

# Named vector (names = file names, values = paths). Archives (.tar.gz, .tgz, .tar, .zip) are unpacked.
expand_uploads <- function(files) {
  out <- character(0)
  for (i in seq_len(nrow(files))) {
    nm <- files$name[i]
    p <- files$datapath[i]
    if (grepl("\\.(tar\\.gz|tgz|tar)$", nm, ignore.case = TRUE)) {
      d <- tempfile("untar_")
      dir.create(d)
      utils::untar(p, exdir = d, tar = "internal")
      f <- list.files(d, recursive = TRUE, full.names = TRUE)
      out <- c(out, stats::setNames(f, basename(f)))
    } else if (grepl("\\.zip$", nm, ignore.case = TRUE)) {
      d <- tempfile("unzip_")
      dir.create(d)
      utils::unzip(p, exdir = d)
      f <- list.files(d, recursive = TRUE, full.names = TRUE)
      out <- c(out, stats::setNames(f, basename(f)))
    } else {
      out <- c(out, stats::setNames(p, nm))
    }
  }
  out[!grepl("^\\._", names(out)) & !grepl("__MACOSX", out)]
}

pick <- function(files, patterns) {
  for (p in patterns) {
    hit <- grep(p, names(files), ignore.case = TRUE)
    if (length(hit)) return(unname(files[hit[1]]))
  }
  NULL
}

# ---- count matrices ---------------------------------------------------------------------------

as_dgc <- function(m) methods::as(methods::as(methods::as(m, "dMatrix"), "generalMatrix"), "CsparseMatrix")

# 10x HDF5 matrix converted in the browser (h5convert.js) into the .jx10x binary.
read_jx10x <- function(path, gene_expression_only = TRUE) {
  con <- file(path, "rb")
  on.exit(close(con))
  magic <- readBin(con, "raw", 8)
  if (!identical(rawToChar(magic[1:6]), "JX10X1")) stop("Not a converted 10x matrix")
  hl <- readBin(con, "integer", 1, size = 4, endian = "little")
  h <- jsonlite::fromJSON(rawToChar(readBin(con, "raw", hl)), simplifyVector = TRUE)
  pad <- (4 - ((8 + 4 + hl) %% 4)) %% 4
  if (pad) readBin(con, "raw", pad)
  indptr <- readBin(con, "integer", h$n_indptr, size = 4, endian = "little")
  indices <- readBin(con, "integer", h$nnz, size = 4, endian = "little")
  x <- if (h$data_type == "int") {
    as.numeric(readBin(con, "integer", h$nnz, size = 4, endian = "little"))
  } else {
    readBin(con, "double", h$nnz, size = 8, endian = "little")
  }
  m <- new("dgCMatrix", i = indices, p = indptr, x = x, Dim = as.integer(h$shape))
  if (!isTRUE(methods::validObject(m, test = TRUE))) {
    m <- Matrix::sparseMatrix(i = indices + 1L, p = indptr, x = x, dims = as.integer(h$shape))
  }
  types <- h$types %||% rep("Gene Expression", h$shape[1])
  keep <- if (gene_expression_only && any(types == "Gene Expression")) types == "Gene Expression" else rep(TRUE, length(types))
  m <- m[keep, , drop = FALSE]
  dimnames(m) <- list(make.unique(as.character(h$names[keep])), as.character(h$barcodes))
  m
}

read_mex <- function(files) {
  mtx <- pick(files, "^matrix\\.mtx(\\.gz)?$")
  bc <- pick(files, "^barcodes\\.tsv(\\.gz)?$")
  ft <- pick(files, c("^features\\.tsv(\\.gz)?$", "^genes\\.tsv(\\.gz)?$"))
  if (is.null(mtx) || is.null(bc) || is.null(ft)) return(NULL)
  m <- as_dgc(Matrix::readMM(mtx))
  barcodes <- readLines(bc)
  feat <- utils::read.delim(ft, header = FALSE, stringsAsFactors = FALSE, quote = "")
  genes <- if (ncol(feat) >= 2) feat[[2]] else feat[[1]]
  if (ncol(feat) >= 3 && any(feat[[3]] == "Gene Expression")) {
    keep <- feat[[3]] == "Gene Expression"
    m <- m[keep, , drop = FALSE]
    genes <- genes[keep]
  }
  dimnames(m) <- list(make.unique(genes), barcodes)
  m
}

read_counts <- function(files, prefer = c("filtered_feature_bc_matrix", "cell_feature_matrix")) {
  bin <- pick(files, c(paste0(prefer, ".*\\.jx10x$"), "\\.jx10x$"))
  if (!is.null(bin)) return(read_jx10x(bin))
  read_mex(files)
}

# ---- 10x Visium / Visium HD (Space Ranger) ----------------------------------------------------

read_positions <- function(path) {
  if (grepl("\\.parquet$", path, ignore.case = TRUE)) {
    pos <- as.data.frame(nanoparquet::read_parquet(path))
  } else {
    first <- readLines(path, n = 1)
    header <- grepl("barcode|in_tissue", first)
    pos <- utils::read.csv(path, header = header, stringsAsFactors = FALSE)
    if (!header) {
      names(pos)[1:6] <- c("barcode", "in_tissue", "array_row", "array_col",
                           "pxl_row_in_fullres", "pxl_col_in_fullres")
    }
  }
  if (!"barcode" %in% names(pos)) names(pos)[1] <- "barcode"
  pos
}

read_image <- function(path) {
  img <- if (grepl("\\.png$", path, ignore.case = TRUE)) png::readPNG(path) else jpeg::readJPEG(path)
  if (length(dim(img)) == 2) img <- array(img, c(dim(img), 3))
  img[, , 1:3, drop = FALSE]
}

read_spaceranger <- function(files) {
  counts <- read_counts(files)
  if (is.null(counts)) {
    stop("No count matrix found. Add filtered_feature_bc_matrix.h5, or matrix.mtx.gz with barcodes.tsv.gz and features.tsv.gz.")
  }
  pos_path <- pick(files, c("^tissue_positions\\.parquet$", "^tissue_positions\\.csv$", "^tissue_positions_list\\.csv$",
                            "tissue_positions"))
  if (is.null(pos_path)) {
    stop("No tissue_positions file found. Add the files of the spatial folder, or the spatial.tar.gz archive.")
  }
  pos <- read_positions(pos_path)
  if ("in_tissue" %in% names(pos)) pos <- pos[pos$in_tissue == 1, , drop = FALSE]
  pos <- pos[pos$barcode %in% colnames(counts), , drop = FALSE]
  if (!nrow(pos)) stop("The barcodes of the count matrix and of the tissue positions do not match.")
  counts <- counts[, pos$barcode, drop = FALSE]
  sf_path <- pick(files, "^scalefactors_json\\.json$")
  sf <- if (!is.null(sf_path)) jsonlite::fromJSON(sf_path) else list()
  img_path <- pick(files, c("^tissue_lowres_image\\.png$", "^tissue_hires_image\\.png$", "^detected_tissue_image\\.jpg$"))
  image <- NULL
  scale <- 1
  if (!is.null(img_path)) {
    s <- if (grepl("lowres", img_path)) sf$tissue_lowres_scalef else if (grepl("hires", img_path)) sf$tissue_hires_scalef else NULL
    if (!is.null(s)) {
      image <- read_image(img_path)
      scale <- s
    }
  }
  hd <- !is.null(sf$bin_size_um) || grepl("^s_[0-9]+um_", pos$barcode[1])
  bin_um <- sf$bin_size_um %||% suppressWarnings(as.numeric(sub("^s_0*([0-9]+)um_.*", "\\1", pos$barcode[1])))
  list(
    counts = counts, data = NULL,
    coords = data.frame(x = pos$pxl_col_in_fullres * scale, y = pos$pxl_row_in_fullres * scale),
    array = data.frame(row = as.integer(pos$array_row), col = as.integer(pos$array_col)),
    lattice = if (hd) "square" else "hex", image = image,
    meta = data.frame(row.names = pos$barcode),
    platform = if (hd) sprintf("Visium HD (%s µm bins)", format(bin_um)) else "Visium",
    unit = "pixels", notes = character(0)
  )
}

# ---- Xenium -----------------------------------------------------------------------------------

read_table_any <- function(path) {
  if (grepl("\\.parquet$", path, ignore.case = TRUE)) return(as.data.frame(nanoparquet::read_parquet(path)))
  if (requireNamespace("data.table", quietly = TRUE)) {
    return(as.data.frame(data.table::fread(path, data.table = FALSE, showProgress = FALSE)))
  }
  utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}

read_xenium <- function(files) {
  counts <- read_counts(files, prefer = "cell_feature_matrix")
  if (is.null(counts)) stop("No count matrix found. Add cell_feature_matrix.h5 (or the cell_feature_matrix MEX files).")
  cp <- pick(files, c("^cells\\.parquet$", "^cells\\.csv\\.gz$", "^cells\\.csv$"))
  if (is.null(cp)) stop("No cells.csv.gz or cells.parquet found (it holds the cell centroids).")
  ce <- read_table_any(cp)
  id <- as.character(ce$cell_id %||% ce[[1]])
  keep <- id %in% colnames(counts)
  if (!any(keep)) stop("The cell IDs of the count matrix and of the cells file do not match.")
  ce <- ce[keep, , drop = FALSE]
  counts <- counts[, id[keep], drop = FALSE]
  list(
    counts = counts, data = NULL,
    coords = data.frame(x = ce$x_centroid, y = ce$y_centroid),
    array = NULL, lattice = "none", image = NULL,
    meta = data.frame(row.names = id[keep]),
    platform = "Xenium (cells)", unit = "µm", notes = character(0)
  )
}

# ---- any platform: tables ---------------------------------------------------------------------

coord_patterns <- list(
  c("x", "y"), c("x_centroid", "y_centroid"), c("centerx_global_px", "centery_global_px"),
  c("center_x", "center_y"), c("imagecol", "imagerow"), c("pxl_col_in_fullres", "pxl_row_in_fullres"),
  c("xcoord", "ycoord"), c("x_location", "y_location"), c("spatial_x", "spatial_y"), c("coord_x", "coord_y")
)

find_coords <- function(df) {
  low <- tolower(names(df))
  for (p in coord_patterns) {
    if (all(p %in% low)) return(names(df)[match(p, low)])
  }
  NULL
}

non_gene <- function(nm) {
  grepl("^(negprb|neg_|negcontrol|systemcontrol|blank|unassigned|falsecode|deprecated)", nm, ignore.case = TRUE)
}

read_tables <- function(files) {
  tabs <- files[grepl("\\.(csv|tsv|txt)(\\.gz)?$", names(files), ignore.case = TRUE)]
  if (!length(tabs)) stop("No CSV/TSV tables found.")
  dfs <- lapply(tabs, read_table_any)
  has_xy <- vapply(dfs, function(d) !is.null(find_coords(d)), logical(1))
  if (!any(has_xy)) stop("No coordinate columns found (for example x and y, or x_centroid and y_centroid).")
  ci <- which(has_xy)[1]
  co <- dfs[[ci]]
  xy <- find_coords(co)
  if (length(dfs) == 1) {
    genes <- setdiff(names(co)[vapply(co, is.numeric, logical(1))], xy)
    genes <- genes[!non_gene(genes) & !tolower(genes) %in% c("fov", "cell_id", "cell", "in_tissue", "array_row", "array_col")]
    if (length(genes) < 2) stop("The table needs at least two numeric gene columns besides the coordinates.")
    m <- t(as.matrix(co[, genes]))
    ids <- if (is.character(co[[1]])) make.unique(co[[1]]) else paste0("spot", seq_len(nrow(co)))
    colnames(m) <- ids
    return(list(counts = NULL, raw = as_dgc(m), coords = data.frame(x = co[[xy[1]]], y = co[[xy[2]]]),
                array = NULL, lattice = "none", image = NULL, meta = data.frame(row.names = ids),
                platform = "Table", unit = "", notes = character(0)))
  }
  ex <- dfs[[setdiff(seq_along(dfs), ci)[1]]]
  key_of <- function(d) {
    low <- tolower(names(d))
    if (all(c("fov", "cell_id") %in% low)) return(paste(d[[match("fov", low)]], d[[match("cell_id", low)]], sep = "_"))
    as.character(d[[1]])
  }
  ck <- key_of(co)
  ek <- key_of(ex)
  if (mean(ek %in% ck) > 0.5) {
    drop <- tolower(names(ex)) %in% c("fov", "cell_id", "cell", "barcode") | seq_along(ex) == 1
    genes <- names(ex)[!drop & vapply(ex, is.numeric, logical(1)) & !non_gene(names(ex))]
    m <- t(as.matrix(ex[, genes]))
    colnames(m) <- ek
  } else if (mean(names(ex)[-1] %in% ck) > 0.5) {
    m <- as.matrix(ex[, -1])
    rownames(m) <- make.unique(as.character(ex[[1]]))
    m <- m[!non_gene(rownames(m)), , drop = FALSE]
  } else {
    stop("Could not match the expression table to the coordinate table by cell or spot IDs.")
  }
  shared <- intersect(colnames(m), ck)
  m <- m[, shared, drop = FALSE]
  co <- co[match(shared, ck), , drop = FALSE]
  list(counts = NULL, raw = as_dgc(m), coords = data.frame(x = co[[xy[1]]], y = co[[xy[2]]]),
       array = NULL, lattice = "none", image = NULL, meta = data.frame(row.names = shared),
       platform = "Tables", unit = "", notes = character(0))
}

# ---- Seurat objects ---------------------------------------------------------------------------

# SeuratObject is installed on first use only, so the app starts quickly for everyone else.
# (The package name is kept in a variable so that the app bundler does not preload it.)
seurat_object_ns <- function() {
  pkg <- "SeuratObject"
  if (!requireNamespace(pkg, quietly = TRUE)) {
    installer <- tryCatch(get("install", envir = asNamespace("webr")), error = function(e) NULL)
    if (is.null(installer)) stop("The SeuratObject package is not installed.")
    installer(pkg, quiet = TRUE)
  }
  suppressPackageStartupMessages(loadNamespace(pkg))
}

load_seurat_rds <- function(path) {
  seurat_object_ns()
  obj <- suppressWarnings(readRDS(path))
  if (!inherits(obj, "Seurat")) stop("This .rds file does not hold a Seurat object.")
  obj
}

seurat_images <- function(obj) {
  imgs <- names(obj@images) %||% character(0)
  if (!length(imgs) && "spatial" %in% names(obj@reductions)) imgs <- "(spatial reduction)"
  if (!length(imgs)) {
    if (!is.null(find_coords(obj@meta.data))) imgs <- "(coordinates in metadata)"
  }
  imgs
}

slot_or <- function(x, name, default = NULL) {
  if (isTRUE(methods::.hasSlot(x, name))) methods::slot(x, name) else default
}

seurat_layer <- function(obj, assay, layer) {
  empty <- function(m) is.null(m) || !length(dim(m)) || nrow(m) == 0 || ncol(m) == 0
  layer_data <- tryCatch(getExportedValue("SeuratObject", "LayerData"), error = function(e) NULL)
  out <- if (is.null(layer_data)) NULL else tryCatch(layer_data(obj, assay = assay, layer = layer), error = function(e) NULL)
  if (!empty(out)) return(out)
  a <- obj@assays[[assay]]
  if (isTRUE(methods::.hasSlot(a, "layers"))) {                       # Assay5 without dispatch
    lay <- a@layers[[layer]]
    if (empty(lay)) return(NULL)
    feats <- rownames(a@features)[a@features[, layer]]
    cells <- rownames(a@cells)[a@cells[, layer]]
    dimnames(lay) <- list(feats, cells)
    return(lay)
  }
  out <- slot_or(a, layer)                                            # Assay / SCTAssay slots
  if (empty(out)) return(NULL)
  out
}

seurat_extract <- function(obj, image = NULL, assay = NULL) {
  imgs <- seurat_images(obj)
  image <- image %||% imgs[1]
  if (!length(image) || is.na(image)) stop("This Seurat object has no spatial coordinates (no image, spatial reduction or x/y metadata).")
  img <- if (image %in% names(obj@images)) obj@images[[image]] else NULL
  cls <- if (is.null(img)) "" else class(img)[1]
  pic <- NULL
  arr <- NULL
  lattice <- "none"
  unit <- ""
  if (cls == "VisiumV1") {
    co <- img@coordinates
    sf <- img@scale.factors
    cells <- rownames(co)
    s <- sf$lowres %||% 1
    xy <- data.frame(x = co$imagecol * s, y = co$imagerow * s)
    arr <- data.frame(row = as.integer(co$row), col = as.integer(co$col))
    pic <- img@image
    lattice <- "hex"
    unit <- "pixels"
    platform <- "Visium (Seurat object, VisiumV1)"
  } else if (!is.null(img) && isTRUE(methods::.hasSlot(img, "boundaries"))) {
    b <- img@boundaries
    cen <- b$centroids %||% b[[1]]
    m <- cen@coords
    cells <- cen@cells
    if (cls == "VisiumV2") {
      sf <- img@scale.factors
      s <- sf$lowres %||% 1
      xy <- data.frame(x = m[, 2] * s, y = m[, 1] * s)
      pic <- slot_or(img, "image")
      lattice <- "detect"
      unit <- "pixels"
      platform <- "Visium (Seurat object, VisiumV2)"
    } else {
      xy <- data.frame(x = m[, 1], y = m[, 2])
      unit <- "µm"
      platform <- sprintf("Imaging-based (Seurat object, %s)", cls)
    }
  } else if (!is.null(img) && isTRUE(methods::.hasSlot(img, "coordinates"))) {
    co <- img@coordinates
    cells <- rownames(co)
    xn <- find_coords(co) %||% names(co)[1:2]
    xy <- data.frame(x = co[[xn[1]]], y = co[[xn[2]]])
    platform <- sprintf("Seurat object (%s)", cls)
  } else if (identical(image, "(spatial reduction)")) {
    e <- obj@reductions$spatial@cell.embeddings
    cells <- rownames(e)
    xy <- data.frame(x = e[, 1], y = e[, 2])
    platform <- "Seurat object (spatial reduction)"
  } else {
    md <- obj@meta.data
    xn <- find_coords(md)
    cells <- rownames(md)
    xy <- data.frame(x = md[[xn[1]]], y = md[[xn[2]]])
    platform <- "Seurat object (metadata coordinates)"
  }
  if (is.null(assay)) {
    # The active assay (for example SCT after SCTransform) when it covers this image, else the image's assay.
    act <- obj@active.assay
    ref_act <- seurat_layer(obj, act, "data") %||% seurat_layer(obj, act, "counts")
    assay <- if (!is.null(ref_act) && mean(cells %in% colnames(ref_act)) > 0.5) act else slot_or(img, "assay") %||% act
  }
  if (!assay %in% names(obj@assays)) assay <- obj@active.assay
  data <- seurat_layer(obj, assay, "data")
  counts <- seurat_layer(obj, assay, "counts")
  ref <- if (!is.null(data)) data else counts
  if (is.null(ref)) stop(sprintf("Assay %s has neither a counts nor a data layer.", assay))
  keep <- cells %in% colnames(ref)
  cells <- cells[keep]
  xy <- xy[keep, , drop = FALSE]
  if (!is.null(arr)) arr <- arr[keep, , drop = FALSE]
  if (!length(cells)) stop("The cells of the image and of the assay do not match.")
  if (!is.null(data)) data <- as_dgc(data[, cells, drop = FALSE])
  if (!is.null(counts)) counts <- as_dgc(counts[, cells, drop = FALSE])
  if (!is.null(data) && looks_like_counts(data)) {
    if (is.null(counts)) counts <- data
    data <- NULL
  }
  if (!is.null(pic) && length(dim(pic)) == 3 && dim(pic)[3] > 3) pic <- pic[, , 1:3, drop = FALSE]
  md <- obj@meta.data[cells, , drop = FALSE]
  # (no ncol(obj): it dispatches on the assay class, which may need the full Seurat package)
  if (length(obj@active.ident) == nrow(obj@meta.data)) md$Idents <- obj@active.ident[cells]
  list(counts = counts, data = data, coords = xy, array = arr, lattice = lattice, image = pic,
       meta = annotation_columns(md), platform = platform, unit = unit, assay = assay,
       notes = sprintf("Assay %s, %s layer used.", assay, if (!is.null(data)) "data" else "counts (log-normalized here)"))
}

# ---- annotations, normalization, lattice, binning ---------------------------------------------

annotation_columns <- function(md) {
  md <- md[, !grepl("^(nCount|nFeature|percent|nUMI|nGene|n_counts|n_genes|total_counts)", names(md)), drop = FALSE]
  if (!ncol(md)) return(md)
  ok <- vapply(md, function(v) {
    k <- length(unique(v[!is.na(v)]))
    (is.factor(v) || is.character(v) || is.logical(v) || (is.numeric(v) && all(v == round(v), na.rm = TRUE) && k <= 40)) &&
      k >= 2 && k <= 60
  }, logical(1))
  out <- md[, ok, drop = FALSE]
  out[] <- lapply(out, function(v) factor(as.character(v)))
  out
}

looks_like_counts <- function(m) {
  x <- if (methods::is(m, "dgCMatrix")) m@x else as.numeric(m)
  x <- utils::head(x, 2e5)
  length(x) > 0 && all(abs(x - round(x)) < 1e-8) && max(x) > 20
}

normalize_dataset <- function(d) {
  if (!is.null(d$data)) return(d)
  if (!is.null(d$counts)) {
    cs <- Matrix::colSums(d$counts)
    cs[cs == 0] <- 1
    m <- d$counts %*% Matrix::Diagonal(x = 1e4 / cs)
    m <- as_dgc(m)
    m@x <- log1p(m@x)
    dimnames(m) <- dimnames(d$counts)
    d$data <- m
    d$normalization <- "log1p(counts per 10,000)"
  } else {
    raw <- d$raw
    if (looks_like_counts(raw)) {
      d$counts <- raw
      d$raw <- NULL
      return(normalize_dataset(d))
    }
    d$data <- raw
    d$normalization <- "values as uploaded"
  }
  d
}

detect_lattice <- function(xy) {
  n <- nrow(xy)
  if (n < 20) return(list(type = "none", k = 6L))
  s <- if (n > 3000) sample.int(n, 3000) else seq_len(n)
  nn <- RANN::nn2(as.matrix(xy), as.matrix(xy[s, ]), k = min(9L, n))
  d1 <- stats::median(nn$nn.dists[, 2])
  cnt <- stats::median(rowSums(nn$nn.dists[, -1, drop = FALSE] <= 1.15 * d1))
  if (cnt >= 5.5) return(list(type = "hex", k = 6L, d = d1))
  if (cnt >= 3.5 && cnt <= 4.5) return(list(type = "square", k = 4L, d = d1))
  list(type = "none", k = 6L, d = d1)
}

square_graph <- function(r, cc) {
  key <- paste(r, cc, sep = "_")
  ii <- integer(0)
  jj <- integer(0)
  for (off in list(c(0L, 1L), c(1L, 0L))) {
    m <- match(paste(r + off[1], cc + off[2], sep = "_"), key)
    ok <- !is.na(m)
    ii <- c(ii, which(ok))
    jj <- c(jj, m[ok])
  }
  n <- length(r)
  a <- Matrix::sparseMatrix(i = c(ii, jj), j = c(jj, ii), x = 1, dims = c(n, n))
  a <- as_dgc(a)
  if (length(a@x)) a@x[] <- 1
  a
}

build_graph <- function(d, neighbors = NULL) {
  if (!is.null(d$array) && identical(d$lattice, "hex") && !anyDuplicated(paste(d$array$row, d$array$col))) {
    return(list(graph = visium_graph(d$array$row, d$array$col), how = "Visium hexagonal lattice (6 neighbours)"))
  }
  if (!is.null(d$array) && identical(d$lattice, "square") && !anyDuplicated(paste(d$array$row, d$array$col))) {
    return(list(graph = square_graph(d$array$row, d$array$col), how = "square lattice (4 neighbours)"))
  }
  xy <- as.matrix(d$coords[, c("x", "y")])
  lat <- detect_lattice(xy)
  k <- neighbors %||% lat$k
  if (lat$type %in% c("hex", "square")) {
    g <- spatial_graph(xy, neighbors = k, radius = 1.2 * lat$d)
    return(list(graph = g, how = sprintf("%s lattice detected from coordinates (%d neighbours)",
                                         if (lat$type == "hex") "hexagonal" else "square", k)))
  }
  list(graph = spatial_graph(xy, neighbors = k), how = sprintf("%d nearest neighbours", k))
}

majority <- function(v, f) {
  tab <- table(f, v)
  factor(colnames(tab)[max.col(tab, ties.method = "first")])
}

# Group neighbouring spots or cells into square bins of side size (coordinate units), or
# merge merge x merge lattice positions (Visium HD).
bin_dataset <- function(d, size = NULL, merge = NULL) {
  if (!is.null(merge) && !is.null(d$array)) {
    br <- d$array$row %/% merge
    bc <- d$array$col %/% merge
  } else {
    br <- floor((d$coords$y - min(d$coords$y)) / size)
    bc <- floor((d$coords$x - min(d$coords$x)) / size)
  }
  key <- factor(paste(br, bc, sep = "_"))
  A <- Matrix::sparseMatrix(i = seq_along(key), j = as.integer(key), x = 1,
                            dims = c(length(key), nlevels(key)))
  size_b <- Matrix::colSums(A)
  mean_of <- function(v) as.numeric(Matrix::crossprod(A, v)) / size_b
  out <- d
  if (!is.null(d$counts)) {
    out$counts <- as_dgc(d$counts %*% A)
    out$data <- NULL
  } else {
    src <- d$data %||% d$raw
    out$raw <- as_dgc((src %*% A) %*% Matrix::Diagonal(x = 1 / size_b))
    out$data <- NULL
  }
  ids <- paste0("bin_", levels(key))
  for (nm in c("counts", "raw")) if (!is.null(out[[nm]])) colnames(out[[nm]]) <- ids
  out$coords <- data.frame(x = mean_of(d$coords$x), y = mean_of(d$coords$y))
  lv <- do.call(rbind, strsplit(levels(key), "_"))
  out$array <- data.frame(row = as.integer(lv[, 1]), col = as.integer(lv[, 2]))
  out$lattice <- "square"
  md <- data.frame(row.names = ids)
  for (nm in names(d$meta)) md[[nm]] <- majority(as.character(d$meta[[nm]]), key)
  md$spots_in_bin <- NULL
  out$meta <- md
  out$n_original <- length(key)
  out$bin_label <- if (!is.null(merge)) {
    sprintf("%d x %d lattice positions per bin", merge, merge)
  } else {
    sprintf("square bins of %s %s", format(signif(size, 3)), d$unit)
  }
  out
}

# Bin size that brings the number of bins to about target.
auto_bin_size <- function(d, target) {
  r <- diff(range(d$coords$x)) * diff(range(d$coords$y))
  s <- sqrt(r / target)
  for (i in 1:6) {
    nb <- length(unique(paste(floor((d$coords$y - min(d$coords$y)) / s), floor((d$coords$x - min(d$coords$x)) / s))))
    if (nb <= target) break
    s <- s * sqrt(nb / target) * 1.02
  }
  s
}

auto_lattice_factor <- function(d, target) {
  f <- 1L
  repeat {
    nb <- length(unique(paste(d$array$row %/% f, d$array$col %/% f)))
    if (nb <= target || f > 64) return(f)
    f <- f + 1L
  }
}

# ---- gene panels ------------------------------------------------------------------------------

gene_pool <- function(data, min_frac = 0.05, max_genes = 3000) {
  det <- Matrix::rowMeans(data > 0)
  ok <- which(det >= min_frac)
  if (!length(ok)) ok <- order(det, decreasing = TRUE)[seq_len(min(nrow(data), 200))]
  m <- Matrix::rowMeans(data[ok, , drop = FALSE])
  v <- Matrix::rowMeans(data[ok, , drop = FALSE]^2) - m^2
  ok <- ok[order(v, decreasing = TRUE)][seq_len(min(length(ok), max_genes))]
  rownames(data)[ok]
}

morans_i <- function(data, graph, genes) {
  W <- graph
  rs <- Matrix::rowSums(W)
  rs[rs == 0] <- 1
  W <- Matrix::Diagonal(x = 1 / rs) %*% W
  out <- numeric(0)
  for (chunk in split(genes, ceiling(seq_along(genes) / 500))) {
    Z <- as.matrix(Matrix::t(data[chunk, , drop = FALSE]))
    Z <- sweep(Z, 2, colMeans(Z))
    num <- colSums(Z * as.matrix(W %*% Z))
    den <- colSums(Z^2)
    out <- c(out, stats::setNames(ifelse(den > 0, num / den, NA), chunk))
  }
  out
}

select_panel <- function(d, graph, method = c("svg", "hvg", "custom"), n = 100, custom = character(0)) {
  method <- match.arg(method)
  data <- d$data
  wanted <- character(0)
  if (length(custom)) {
    hit <- match(toupper(custom), toupper(rownames(data)))
    wanted <- rownames(data)[hit[!is.na(hit)]]
  }
  if (method == "custom") return(list(genes = unique(wanted), score = NULL, missing = custom[is.na(match(toupper(custom), toupper(rownames(data))))]))
  pool <- gene_pool(data)
  score <- if (method == "svg") morans_i(data, graph, pool) else {
    m <- Matrix::rowMeans(data[pool, , drop = FALSE])
    stats::setNames(Matrix::rowMeans(data[pool, , drop = FALSE]^2) - m^2, pool)
  }
  top <- names(sort(score, decreasing = TRUE))[seq_len(min(n, length(score)))]
  list(genes = unique(c(wanted, top)), score = score,
       missing = custom[is.na(match(toupper(custom), toupper(rownames(data))))])
}
