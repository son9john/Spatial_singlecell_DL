# juxtaposed: find where two expression programs meet in tissue.
# Shiny app for R, run in the browser with Shinylive (webR): data never leave this computer.
# Inputs: Seurat objects (.rds), 10x Space Ranger output (Visium, Visium HD), 10x Xenium output,
# or any platform as tables. The package code (juxtaposed/*.R) is copied next to this file at build time.

library(shiny)
library(bslib)
library(Matrix)
library(plotly)
library(DT)

options(shiny.maxRequestSize = 4000 * 1024^2)
for (f in list.files("juxtaposed", pattern = "\\.R$", full.names = TRUE)) source(f, local = FALSE)
source("readers.R", local = FALSE)

# ---- constants: default R (ggplot2) colours ---------------------------------------------------
COL_OPP <- "#F8766D"   # opposing change (interface), boundary J
COL_CON <- "#619CFF"   # concordant change
COL_REG <- "#00BA38"   # regional J
COL_NS <- "#BDBDBD"
DIV_SCALE <- list(list(0, "#3A3A98"), list(0.25, "#619CFF"), list(0.5, "#F7F7F7"), list(0.75, "#F8766D"), list(1, "#832424"))
EXPR_SCALE <- list(list(0, "#F2F2F2"), list(0.5, "#56B1F7"), list(1, "#132B43"))
gg_hue <- function(n) grDevices::hcl(h = seq(15, 375, length.out = n + 1)[seq_len(n)], c = 100, l = 65)
EXAMPLE_FALLBACK <- "https://son9john.github.io/Spatial_singlecell_DL/example/"
H5JS <- paste(readLines("h5convert.js", warn = FALSE), collapse = "\n")

# ---- helpers ----------------------------------------------------------------------------------

image_uri <- function(img, max_px = 900) {
  if (is.null(img)) return(NULL)
  h <- dim(img)[1]
  w <- dim(img)[2]
  f <- ceiling(max(h, w) / max_px)
  if (f > 1) img <- img[seq(1, h, f), seq(1, w, f), , drop = FALSE]
  img[img < 0] <- 0
  img[img > 1] <- 1
  # base64_enc() wraps lines; plotly would turn those line breaks into <br>, so remove them.
  paste0("data:image/png;base64,", gsub("[\r\n]", "", jsonlite::base64_enc(png::writePNG(img))))
}

spot_size <- function(d, height = 430) {
  xy <- as.matrix(d$coords[, c("x", "y")])
  if (nrow(xy) < 3) return(6)
  s <- if (nrow(xy) > 2000) sample.int(nrow(xy), 2000) else seq_len(nrow(xy))
  nn <- RANN::nn2(xy, xy[s, , drop = FALSE], k = 2)$nn.dists[, 2]
  span <- max(diff(range(xy[, 1])), diff(range(xy[, 2])))
  max(2.5, min(14, 0.95 * stats::median(nn) / span * height))
}

tissue_plot <- function(d, value, label, kind = c("continuous", "diverging", "categorical"),
                        hover = NULL, height = 430, title = NULL) {
  kind <- match.arg(kind)
  ms <- spot_size(d, height)
  df <- data.frame(x = d$coords$x, y = d$coords$y, v = value,
                   h = hover %||% rownames(d$meta) %||% seq_along(value))
  p <- plot_ly(height = height)
  if (kind == "categorical") {
    f <- factor(df$v)
    cols <- stats::setNames(gg_hue(nlevels(f)), levels(f))
    p <- add_markers(p, data = df, x = ~x, y = ~y, color = f, colors = cols,
                     marker = list(size = ms, opacity = 0.9, line = list(width = 0)),
                     text = ~paste0(h, "<br>", label, ": ", v), hoverinfo = "text")
  } else {
    ok <- is.finite(df$v)
    if (any(!ok)) {
      p <- add_markers(p, data = df[!ok, ], x = ~x, y = ~y, marker = list(size = ms, color = "#D9D9D9", opacity = 0.6),
                       hoverinfo = "skip", showlegend = FALSE)
    }
    lim <- max(abs(df$v[ok]), 1e-9)
    mk <- list(size = ms, color = df$v[ok], opacity = 0.92, line = list(width = 0),
               colorbar = list(title = list(text = label, side = "right"), thickness = 12, len = 0.8),
               colorscale = if (kind == "diverging") DIV_SCALE else EXPR_SCALE)
    if (kind == "diverging") mk <- c(mk, list(cmin = -lim, cmax = lim, cmid = 0))
    p <- add_markers(p, data = df[ok, ], x = ~x, y = ~y, marker = mk, showlegend = FALSE,
                     text = ~paste0(h, "<br>", label, ": ", signif(v, 3)), hoverinfo = "text")
  }
  lay <- list(
    xaxis = list(visible = FALSE, zeroline = FALSE),
    yaxis = list(visible = FALSE, autorange = "reversed", scaleanchor = "x", zeroline = FALSE),
    margin = list(l = 5, r = 5, t = if (is.null(title)) 5 else 35, b = 5),
    paper_bgcolor = "rgba(0,0,0,0)", plot_bgcolor = "rgba(0,0,0,0)",
    legend = list(itemsizing = "constant", font = list(size = 11))
  )
  if (!is.null(title)) lay$title <- list(text = title, font = list(size = 14), x = 0.02)
  if (!is.null(d$image_uri)) {
    lay$images <- list(list(source = d$image_uri, xref = "x", yref = "y", x = 0, y = 0,
                            sizex = dim(d$image)[2], sizey = dim(d$image)[1], sizing = "stretch",
                            xanchor = "left", yanchor = "top", layer = "below", opacity = 0.85))
    lay$xaxis$range <- c(0, dim(d$image)[2])
    lay$yaxis$range <- c(dim(d$image)[1], 0)
    lay$yaxis$autorange <- FALSE
  }
  p <- do.call(plotly::layout, c(list(p), lay))
  plotly::config(p, displaylogo = FALSE, modeBarButtonsToRemove = c("lasso2d", "select2d", "autoScale2d"))
}

parse_genes <- function(txt) {
  if (is.null(txt) || !nzchar(trimws(txt))) return(character(0))
  g <- unlist(strsplit(txt, "[[:space:],;]+"))
  unique(g[nzchar(g)])
}

fmt_n <- function(x) formatC(x, format = "d", big.mark = ",")

file_status <- function(files, kind) {
  has <- function(p) !is.null(pick(files, p))
  mtx <- has("\\.jx10x$") || (has("^matrix\\.mtx(\\.gz)?$") && has("^barcodes\\.tsv(\\.gz)?$") &&
                              has("^(features|genes)\\.tsv(\\.gz)?$"))
  if (kind == "visium") {
    list(
      list(ok = mtx, need = TRUE, text = "Count matrix: filtered_feature_bc_matrix.h5 (or matrix.mtx.gz + barcodes.tsv.gz + features.tsv.gz)"),
      list(ok = has("tissue_positions"), need = TRUE, text = "Spot positions: spatial/tissue_positions.csv, tissue_positions_list.csv or tissue_positions.parquet"),
      list(ok = has("scalefactors_json\\.json$"), need = FALSE, text = "Scale factors: spatial/scalefactors_json.json (to place spots on the image)"),
      list(ok = has("tissue_(lowres|hires)_image\\.png$"), need = FALSE, text = "Tissue image: spatial/tissue_lowres_image.png")
    )
  } else if (kind == "xenium") {
    list(
      list(ok = mtx, need = TRUE, text = "Count matrix: cell_feature_matrix.h5 (or the cell_feature_matrix MEX files)"),
      list(ok = has("^cells\\.(parquet|csv\\.gz|csv)$"), need = TRUE, text = "Cell centroids: cells.csv.gz or cells.parquet")
    )
  } else {
    list(list(ok = has("\\.(csv|tsv|txt)(\\.gz)?$"), need = TRUE,
              text = "A table with x and y columns and one column per gene, or an expression table plus a coordinate table"))
  }
}

methods_text <- function(r) {
  p <- r$params
  sprintf(paste0(
    "Spatial adjacency was defined as %s on %s %s%s. Expression was %s. ",
    "%d genes were analysed (%s). For each of the %s gene pairs, the juxtaposition index J ",
    "(juxtaposed R package v0.4.0) was computed after %d step(s) of graph smoothing, and two-sided p-values were ",
    "obtained from the exact spectral sign-flip null of the tissue graph, with Benjamini-Hochberg correction over all pairs. ",
    "Interfaces were mapped with edge-level J (lazy smoother, 4 steps, ridge 1). Co-localization was scored with regional J ",
    "over graph scales 1, 2, 4, 8 and 16."),
    p$graph, fmt_n(p$n), p$unit_label, if (!is.null(p$bin)) paste0(" (", p$bin, ")") else "",
    p$normalization, length(p$genes), p$panel_label, fmt_n(nrow(r$tab)), p$scale)
}

# ---- UI ---------------------------------------------------------------------------------------

css <- "
:root { --jx-accent: #2563EB; --jx-ink: #111827; --jx-muted: #6B7280; --jx-line: #E5E7EB; }
body { background: #F5F6F8; }
.navbar { background: #111827 !important; }
.navbar .navbar-brand, .navbar .nav-link { color: #F9FAFB !important; }
.navbar .nav-link.active { border-bottom: 2px solid #93C5FD; }
.card { border: 1px solid var(--jx-line); border-radius: 12px; box-shadow: 0 1px 2px rgba(17,24,39,.04); }
.card-header { background: #fff; font-weight: 600; border-bottom: 1px solid var(--jx-line); }
.jx-lede { color: var(--jx-muted); max-width: 900px; }
.jx-sources .shiny-options-group { display: grid; grid-template-columns: 1fr; gap: 10px; }
.jx-sources .radio { margin: 0; }
.jx-sources .radio label { display: block; width: 100%; border: 1px solid var(--jx-line); border-radius: 10px;
  padding: 12px 14px; background: #fff; cursor: pointer; }
.jx-sources .radio label:has(input:checked) { border-color: var(--jx-accent); background: #EFF6FF; box-shadow: 0 0 0 2px #BFDBFE; }
.jx-sources .radio input { margin-right: 8px; }
.jx-src-title { font-weight: 600; color: var(--jx-ink); }
.jx-src-desc { display: block; color: var(--jx-muted); font-size: .88rem; margin-left: 22px; }
.jx-drop { border: 2px dashed #CBD5E1; border-radius: 12px; padding: 18px; background: #fff; }
.jx-drop input[type=file] { width: 100%; }
.jx-status { color: var(--jx-muted); font-size: .9rem; margin-top: 6px; }
.jx-error { color: #B91C1C; }
.jx-check { list-style: none; padding-left: 0; margin: 10px 0 0; }
.jx-check li { padding: 3px 0; font-size: .92rem; }
.jx-ok { color: #047857; font-weight: 700; margin-right: 6px; }
.jx-miss { color: #B91C1C; font-weight: 700; margin-right: 6px; }
.jx-opt { color: #9CA3AF; font-weight: 700; margin-right: 6px; }
.jx-help-table td, .jx-help-table th { padding: 8px 10px; border-bottom: 1px solid var(--jx-line); vertical-align: top; }
.jx-pill { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: .8rem; font-weight: 600; }
.jx-pill-opp { background: #FDE2DF; color: #9F2A1E; }
.jx-pill-reg { background: #D7F5E1; color: #0B6B2E; }
.jx-small { font-size: .85rem; color: var(--jx-muted); }
.jx-methods { background: #fff; border: 1px solid var(--jx-line); border-radius: 10px; padding: 14px; font-size: .95rem; }
.bslib-value-box .value-box-title { font-size: .85rem; }
"

src_choice <- function(title, desc) {
  tags$span(tags$span(class = "jx-src-title", title), tags$span(class = "jx-src-desc", desc))
}

guide_table <- tags$table(class = "jx-help-table",
  tags$tr(tags$th("Your question"), tags$th("Use"), tags$th("How to read it")),
  tags$tr(tags$td("Do two expression programs meet at a tissue interface and change in opposite directions across it? (tumour–stroma, layer borders, niche edges)"),
          tags$td(tags$span(class = "jx-pill jx-pill-opp", "Boundary J")),
          tags$td("J > 0: opposing change across shared edges (an interface). J < 0: concordant change. J ≈ 0: no shared boundary, even if the genes are anticorrelated overall.")),
  tags$tr(tags$td("Where exactly is that interface?"), tags$td(tags$span(class = "jx-pill jx-pill-opp", "Interface map")),
          tags$td("Edge-level J of one pair, drawn on the tissue: red spots sit on the boundary between the two programs.")),
  tags$tr(tags$td("Are two genes expressed in the same neighbourhoods? (co-expression, co-localization)"),
          tags$td(tags$span(class = "jx-pill jx-pill-reg", "Regional J")),
          tags$td("> 0: shared regions; < 0: complementary regions. Boundary J is not a co-localization score."))
)

load_panel <- nav_panel(
  title = "1 · Load data", value = "load",
  layout_columns(
    col_widths = c(5, 7),
    card(
      card_header("Choose your data"),
      div(class = "jx-sources",
        radioButtons("source", NULL, selected = "example",
          choiceNames = list(
            src_choice("Example: mouse brain (10x Visium)", "One click. A real sagittal section on its H&E image."),
            src_choice("Seurat object (.rds)", "Seurat v4 or v5 with spatial data: Visium, Visium HD, Slide-seq, Xenium, CosMx, MERSCOPE."),
            src_choice("10x Visium or Visium HD (Space Ranger output)", "The files Load10X_Spatial() reads: the count matrix and the spatial folder."),
            src_choice("10x Xenium output", "cell_feature_matrix.h5 and cells.csv.gz (or cells.parquet)."),
            src_choice("Any platform: tables", "CosMx, MERSCOPE, Slide-seq, Stereo-seq, ...: an expression table and a coordinate table.")),
          choiceValues = list("example", "seurat", "visium", "xenium", "tables"))),
      uiOutput("source_help")
    ),
    card(
      card_header("Preview"),
      uiOutput("data_summary"),
      uiOutput("preview_controls"),
      plotlyOutput("preview_plot", height = "460px")
    )
  )
)

analyze_panel <- nav_panel(
  title = "2 · Analyze", value = "analyze",
  layout_sidebar(
    sidebar = sidebar(
      width = 340, open = "always",
      tags$h6("Genes"),
      radioButtons("panel_method", NULL, selected = "svg",
        choiceNames = list("Spatially variable genes (Moran's I)", "Highly variable genes", "Only my gene list"),
        choiceValues = list("svg", "hvg", "custom")),
      sliderInput("n_genes", "Number of genes", min = 20, max = 300, value = 100, step = 10),
      textAreaInput("custom_genes", "Always include these genes (optional)", rows = 3,
                    placeholder = "e.g. Epcam, Vim, Ptprc  (separate with spaces, commas or new lines)"),
      tags$h6("Spatial graph"),
      checkboxInput("k_auto", "Choose neighbours automatically", TRUE),
      conditionalPanel("!input.k_auto", numericInput("k", "Nearest neighbours", 6, min = 3, max = 20)),
      sliderInput("scale", "Smoothing steps for boundary J", min = 0, max = 4, value = 1, step = 1),
      tags$h6("Large datasets"),
      numericInput("max_nodes", "Largest number of spots for the exact test", 4000, min = 500, max = 15000, step = 500),
      tags$div(class = "jx-small", "Larger data are grouped into small square bins first. About 20 s for 2,700 spots in the browser; time grows with the cube of the number of spots."),
      uiOutput("annotation_choice"),
      actionButton("run", "Run analysis", class = "btn-primary btn-lg w-100 mt-2", icon = icon("play"))
    ),
    card(
      card_header("What will happen"),
      uiOutput("plan"),
      tags$hr(),
      guide_table
    )
  )
)

results_panel <- nav_panel(
  title = "3 · Interfaces", value = "results",
  uiOutput("result_boxes"),
  layout_columns(
    col_widths = c(5, 7),
    card(full_screen = TRUE, card_header("All gene pairs"), plotlyOutput("volcano", height = "430px"),
         tags$div(class = "jx-small", "Click a point or a table row to map that pair. Right: opposing change across shared edges.")),
    card(full_screen = TRUE, card_header("Pairs"),
         layout_columns(col_widths = c(7, 5),
           radioButtons("direction", NULL, inline = TRUE, selected = "opposing",
                        choices = c("Interfaces (J > 0)" = "opposing", "Concordant (J < 0)" = "concordant", "All" = "all")),
           numericInput("qmax", "Show q at most", 0.05, min = 0, max = 1, step = 0.01)),
         DTOutput("pair_table"))
  ),
  card(full_screen = TRUE, card_header(textOutput("map_title", inline = TRUE)),
       layout_columns(col_widths = c(4, 4, 4),
         plotlyOutput("map_x", height = "400px"), plotlyOutput("map_y", height = "400px"), plotlyOutput("map_j", height = "400px")),
       uiOutput("boundary_ui"))
)

coloc_panel <- nav_panel(
  title = "4 · Co-localization", value = "coloc",
  layout_columns(
    col_widths = c(7, 5),
    card(full_screen = TRUE, card_header("Regional J: shared neighbourhood expression"),
         plotlyOutput("regional_heatmap", height = "620px")),
    card(card_header("Most co-localized pairs"), DTOutput("regional_table"),
         tags$div(class = "jx-small", "Regional J averages an edge-sum concordance over graph scales 1, 2, 4, 8 and 16. It has no p-value."))
  )
)

export_panel <- nav_panel(
  title = "Export", value = "export",
  layout_columns(
    col_widths = c(5, 7),
    card(card_header("Download"),
         downloadButton("dl_pairs", "Boundary J, all pairs (CSV)", class = "w-100 mb-2"),
         downloadButton("dl_regional", "Regional J matrix (CSV)", class = "w-100 mb-2"),
         downloadButton("dl_map", "Interface map of the selected pair, per spot (CSV)", class = "w-100 mb-2"),
         downloadButton("dl_methods", "Methods paragraph (TXT)", class = "w-100")),
    card(card_header("Methods paragraph for your paper"), uiOutput("methods"))
  )
)

help_panel <- nav_panel(
  title = "Help", value = "help",
  card(card_header("Which score answers my question?"), guide_table),
  card(card_header("Input files"), tags$ul(
    tags$li(tags$b("Seurat object (.rds): "), "saved with saveRDS(). The app reads the spatial image (VisiumV1, VisiumV2, FOV, SlideSeq), the assay used by that image and its data layer (counts are log-normalized if no data layer exists). SCT assays work."),
    tags$li(tags$b("10x Visium / Visium HD: "), "from the Space Ranger outs folder select filtered_feature_bc_matrix.h5 and the files of the spatial folder (or the spatial.tar.gz that 10x distributes). For Visium HD use binned_outputs/square_016um (or square_008um for small regions)."),
    tags$li(tags$b("10x Xenium: "), "cell_feature_matrix.h5 and cells.csv.gz. Cells are grouped into small square bins so that the exact test stays fast."),
    tags$li(tags$b("Tables: "), "one CSV/TSV with x, y and one column per gene, or two tables (expression and coordinates) sharing cell IDs. CosMx (exprMat + metadata) and MERSCOPE (cell_by_gene + cell_metadata) files are recognized.")),
    tags$p(class = "jx-small", "Everything runs in this browser tab: files are not uploaded to any server. Very large objects (several GB) may exceed the browser's memory; use the juxtaposed R package directly for those (see the Seurat vignette).")),
  card(card_header("Citation"), tags$p("Song J, Kim J, Yoo J, Lee D, Kim K. The juxtaposition index J: calibrated detection of tissue interfaces in spatial transcriptomics. bioRxiv (2026)."),
       tags$p(class = "jx-small", "Example data: 10x Genomics, Mouse Brain Serial Section 1 (Sagittal-Anterior), Visium, CC BY 4.0."))
)

ui <- page_navbar(
  title = tags$span(tags$b("juxtaposed"), tags$span(style = "opacity:.7; font-weight:400; margin-left:8px;", "the juxtaposition index J")),
  id = "nav", fillable = FALSE, bg = "#111827",
  header = tags$head(tags$style(HTML(css)),
                     tags$script(type = "module", HTML(paste0(H5JS, "\nwirePicker('jx_picker', 'upload', 'jx_status');\n")))),
  load_panel, analyze_panel, results_panel, coloc_panel, export_panel, help_panel,
  nav_spacer(),
  nav_item(tags$a(href = "https://son9john.github.io/Spatial_singlecell_DL/vignette/seurat_juxtaposition.html",
                  target = "_blank", icon("book"), "Seurat vignette"))
)

# ---- server -----------------------------------------------------------------------------------

server <- function(input, output, session) {
  rv <- reactiveValues(files = NULL, obj = NULL, d = NULL, res = NULL, pair = NULL, msg = NULL)

  # The example lives next to the app on the website: <site>/example/ (the app is at <site>/r/).
  example_base <- reactive({
    cd <- session$clientData
    host <- cd$url_hostname %||% ""
    if (grepl("github\\.io$|^localhost$|^127\\.0\\.0\\.1$", host)) {
      port <- if (nzchar(cd$url_port %||% "")) paste0(":", cd$url_port) else ""
      paste0(cd$url_protocol, "//", host, port, sub("/r/.*$", "/", cd$url_pathname %||% "/"), "example/")
    } else {
      EXAMPLE_FALLBACK
    }
  })

  set_dataset <- function(d) {
    d <- normalize_dataset(d)
    d$image_uri <- image_uri(d$image)
    if (is.null(rownames(d$meta)) || nrow(d$meta) != ncol(d$data)) d$meta <- data.frame(row.names = colnames(d$data))
    d$total <- if (!is.null(d$counts)) Matrix::colSums(d$counts) else Matrix::colSums(d$data)
    rv$d <- d
    rv$res <- NULL
    rv$pair <- NULL
  }

  fail <- function(e) {
    msg <- conditionMessage(e)
    showNotification(msg, type = "error", duration = 12)
    rv$msg <- msg
  }

  # ---- loading --------------------------------------------------------------------------------
  output$source_help <- renderUI({
    s <- input$source
    if (s == "example") {
      return(tagList(tags$p(class = "jx-small mt-3", "2,695 spots, the 2,000 most variable genes, H&E image. Downloads about 6 MB."),
                     actionButton("load_example", "Load the example", class = "btn-primary", icon = icon("download"))))
    }
    if (s == "seurat") {
      return(tagList(
        fileInput("rds", "Seurat object (.rds)", accept = ".rds", width = "100%"),
        uiOutput("seurat_choices"),
        tags$p(class = "jx-small", "The first .rds needs about 30 s to set up SeuratObject in the browser.")))
    }
    tagList(
      div(class = "jx-drop mt-3",
          tags$label("for" = "jx_picker", class = "form-label fw-semibold", "Select files (you can select several, or add more in a second step)"),
          tags$input(id = "jx_picker", type = "file", multiple = NA, class = "form-control"),
          div(id = "jx_status", class = "jx-status", "HDF5 (.h5), CSV/TSV (.gz), Parquet, PNG, JSON and .tar.gz/.zip archives are accepted.")),
      div(style = "display:none", fileInput("upload", NULL, multiple = TRUE)),
      uiOutput("file_check"),
      div(class = "d-flex gap-2 mt-2",
          actionButton("load_files", "Load data", class = "btn-primary", icon = icon("check")),
          actionButton("clear_files", "Clear files", class = "btn-outline-secondary")))
  })

  observeEvent(input$source, {
    rv$files <- NULL
  })

  observeEvent(input$upload, {
    new <- tryCatch(expand_uploads(input$upload), error = function(e) { fail(e); NULL })
    if (is.null(new)) return()
    old <- rv$files
    if (length(old)) old <- old[!names(old) %in% names(new)]
    rv$files <- c(old, new)
  })

  observeEvent(input$clear_files, rv$files <- NULL)

  output$file_check <- renderUI({
    if (!input$source %in% c("visium", "xenium", "tables")) return(NULL)
    st <- file_status(rv$files %||% character(0), input$source)
    items <- lapply(st, function(s) {
      mark <- if (s$ok) tags$span(class = "jx-ok", "✓") else if (s$need) tags$span(class = "jx-miss", "✗") else tags$span(class = "jx-opt", "○")
      tags$li(mark, s$text, if (!s$need) tags$span(class = "jx-small", " (optional)"))
    })
    got <- names(rv$files)
    tagList(tags$ul(class = "jx-check", items),
            if (length(got)) tags$div(class = "jx-small", paste("Files:", paste(got, collapse = ", "))))
  })

  observeEvent(input$load_files, {
    f <- rv$files
    if (!length(f)) return(showNotification("Select the files first.", type = "warning"))
    withProgress(message = "Reading files", value = 0.3, {
      d <- tryCatch(switch(input$source,
                           visium = read_spaceranger(f), xenium = read_xenium(f), tables = read_tables(f)),
                    error = function(e) { fail(e); NULL })
      if (!is.null(d)) {
        setProgress(0.8, "Normalizing")
        set_dataset(d)
      }
    })
  })

  observeEvent(input$load_example, {
    base <- paste0(example_base(), "visium_mouse_brain/")
    files <- c("filtered_feature_bc_matrix.jx10x", "tissue_positions_list.csv", "scalefactors_json.json", "tissue_lowres_image.png")
    withProgress(message = "Downloading the example", value = 0.1, {
      dir <- tempfile("example_")
      dir.create(dir)
      ok <- tryCatch({
        for (i in seq_along(files)) {
          setProgress(0.1 + 0.6 * i / length(files), detail = files[i])
          utils::download.file(paste0(base, files[i]), file.path(dir, files[i]), mode = "wb", quiet = TRUE)
        }
        TRUE
      }, error = function(e) { fail(simpleError(paste("Could not download the example:", conditionMessage(e)))); FALSE })
      if (ok) {
        setProgress(0.8, "Reading")
        d <- tryCatch(read_spaceranger(stats::setNames(file.path(dir, files), files)), error = function(e) { fail(e); NULL })
        if (!is.null(d)) {
          d$platform <- "Visium, 10x mouse brain (example)"
          set_dataset(d)
        }
      }
    })
  })

  observeEvent(input$rds, {
    withProgress(message = "Reading the Seurat object", value = 0.2, {
      setProgress(0.2, detail = "Setting up SeuratObject")
      obj <- tryCatch(load_seurat_rds(input$rds$datapath), error = function(e) { fail(e); NULL })
      rv$obj <- obj
      if (!is.null(obj)) {
        imgs <- seurat_images(obj)
        if (!length(imgs)) {
          fail(simpleError("This Seurat object has no spatial coordinates (no image, spatial reduction or x/y metadata)."))
          return()
        }
        if (length(imgs) == 1 && length(obj@assays) == 1) extract_seurat(imgs, NULL)
      }
    })
  })

  extract_seurat <- function(image, assay) {
    d <- tryCatch(seurat_extract(rv$obj, image = image, assay = assay), error = function(e) { fail(e); NULL })
    if (!is.null(d)) set_dataset(d)
  }

  output$seurat_choices <- renderUI({
    obj <- rv$obj
    if (is.null(obj)) return(NULL)
    imgs <- seurat_images(obj)
    tagList(
      selectInput("seurat_image", "Image / field of view", imgs),
      selectInput("seurat_assay", "Assay", c("(the image's assay)" = "", names(obj@assays))),
      actionButton("seurat_use", "Use this object", class = "btn-primary", icon = icon("check")))
  })

  observeEvent(input$seurat_use, {
    withProgress(message = "Extracting expression and coordinates", value = 0.5,
                 extract_seurat(input$seurat_image, if (nzchar(input$seurat_assay %||% "")) input$seurat_assay else NULL))
  })

  output$data_summary <- renderUI({
    d <- rv$d
    if (is.null(d)) {
      return(tags$p(class = "jx-lede", "Load data on the left. The tissue appears here, on its image when one is available."))
    }
    tagList(
      layout_columns(col_widths = c(4, 4, 4),
        value_box("Spots or cells", fmt_n(ncol(d$data)), theme = "light"),
        value_box("Genes", fmt_n(nrow(d$data)), theme = "light"),
        value_box("Platform", tags$span(style = "font-size:1rem", d$platform), theme = "light")),
      tags$p(class = "jx-small", paste(c(d$notes, if (!is.null(d$normalization)) paste("Expression:", d$normalization)), collapse = " ")),
      actionButton("goto_analyze", "Continue to analysis", class = "btn-primary mb-2", icon = icon("arrow-right")))
  })

  observeEvent(input$goto_analyze, nav_select("nav", "analyze"))

  output$preview_controls <- renderUI({
    d <- rv$d
    if (is.null(d)) return(NULL)
    choices <- c("Total counts" = ".total", stats::setNames(names(d$meta), names(d$meta)))
    layout_columns(col_widths = c(6, 6),
      selectInput("preview_color", "Colour spots by", choices),
      selectizeInput("preview_gene", "…or by a gene", choices = NULL, options = list(placeholder = "type a gene")))
  })

  observe({
    d <- rv$d
    if (is.null(d)) return()
    updateSelectizeInput(session, "preview_gene", choices = c("", rownames(d$data)), server = TRUE, selected = "")
  })

  output$preview_plot <- renderPlotly({
    d <- rv$d
    req(d)
    g <- input$preview_gene %||% ""
    if (nzchar(g) && g %in% rownames(d$data)) {
      return(tissue_plot(d, as.numeric(d$data[g, ]), g))
    }
    col <- input$preview_color %||% ".total"
    if (col == ".total" || !col %in% names(d$meta)) {
      tissue_plot(d, as.numeric(d$total), "total")
    } else {
      tissue_plot(d, d$meta[[col]], col, kind = "categorical")
    }
  })

  # ---- analysis -------------------------------------------------------------------------------
  output$annotation_choice <- renderUI({
    d <- rv$d
    if (is.null(d) || !ncol(d$meta)) return(NULL)
    pref <- intersect(c("seurat_clusters", "Idents", "cell_type", "celltype", "annotation", "leiden", "louvain", "region"), names(d$meta))
    tagList(tags$h6("Annotation"),
            selectInput("annotation", "Which annotated regions does each interface separate?",
                        c("(none)" = "", names(d$meta)), selected = pref[1] %||% ""))
  })

  output$plan <- renderUI({
    d <- rv$d
    if (is.null(d)) return(tags$p(class = "jx-lede", "Load data first (step 1)."))
    n <- ncol(d$data)
    big <- n > (input$max_nodes %||% 4000)
    tags$ol(
      tags$li(if (big) sprintf("Group the %s spots into about %s square bins (sum of counts per bin).", fmt_n(n), fmt_n(input$max_nodes))
              else sprintf("Use all %s spots.", fmt_n(n))),
      tags$li("Build the spatial neighbour graph (hexagonal for Visium, square for bins, nearest neighbours otherwise)."),
      tags$li(switch(input$panel_method %||% "svg",
        svg = sprintf("Pick the %d most spatially variable genes (Moran's I) plus your genes.", input$n_genes),
        hvg = sprintf("Pick the %d most variable genes plus your genes.", input$n_genes),
        custom = "Use only your gene list.")),
      tags$li("Compute boundary J for every pair, with exact spectral p-values and Benjamini–Hochberg q-values."),
      tags$li("Compute regional J (co-localization) for the same genes."))
  })

  observeEvent(input$run, {
    d <- rv$d
    if (is.null(d)) return(showNotification("Load data first.", type = "warning"))
    custom <- parse_genes(input$custom_genes)
    if (input$panel_method == "custom" && length(custom) < 2) {
      return(showNotification("Enter at least two genes, or choose another gene selection.", type = "warning"))
    }
    t0 <- Sys.time()
    withProgress(message = "Running", value = 0, {
      res <- tryCatch({
        bin <- NULL
        n0 <- ncol(d$data)
        if (n0 > input$max_nodes) {
          setProgress(0.05, "Grouping spots into bins")
          d <- if (!is.null(d$array) && identical(d$lattice, "square")) {
            bin_dataset(d, merge = auto_lattice_factor(d, input$max_nodes))
          } else {
            bin_dataset(d, size = auto_bin_size(d, input$max_nodes))
          }
          bin <- d$bin_label
          d <- normalize_dataset(d)
          d$image_uri <- rv$d$image_uri
          d$total <- if (!is.null(d$counts)) Matrix::colSums(d$counts) else Matrix::colSums(d$data)
        }
        setProgress(0.15, "Building the spatial graph")
        g <- build_graph(d, neighbors = if (isTRUE(input$k_auto)) NULL else input$k)
        setProgress(0.25, "Selecting genes")
        panel <- select_panel(d, g$graph, input$panel_method, input$n_genes, custom)
        if (length(panel$missing)) showNotification(paste("Not found:", paste(panel$missing, collapse = ", ")), type = "warning")
        X <- as.matrix(Matrix::t(d$data[panel$genes, , drop = FALSE]))
        X <- X[, apply(X, 2, stats::sd) > 0, drop = FALSE]
        if (ncol(X) < 2) stop("Fewer than two genes vary across the tissue; choose other genes.")
        setProgress(0.35, sprintf("Exact spectral null for %s spots", fmt_n(nrow(X))))
        null <- spectral_null(g$graph)
        setProgress(0.75, sprintf("Boundary J for %s pairs", fmt_n(ncol(X) * (ncol(X) - 1) / 2)))
        tab <- juxtaposition(X, g$graph, null = null, scale = input$scale)
        tab$pair <- paste(tab$feature_x, tab$feature_y, sep = " – ")
        tab$call <- ifelse(tab$q < 0.05 & tab$J > 0, "interface (opposing)",
                           ifelse(tab$q < 0.05 & tab$J < 0, "concordant", "not significant"))
        setProgress(0.9, "Regional J")
        reg <- regional_juxtaposition_matrix(X, g$graph)
        dimnames(reg) <- list(colnames(X), colnames(X))
        list(d = d, graph = g$graph, X = X, tab = tab, regional = reg,
             params = list(graph = g$how, n = nrow(X), unit_label = if (n0 > nrow(X)) "bins" else "spots", bin = bin,
                           normalization = d$normalization %||% "as provided", genes = colnames(X),
                           panel_label = switch(input$panel_method, svg = "most spatially variable by Moran's I",
                                                hvg = "most variable", custom = "user-selected"),
                           scale = input$scale, seconds = as.numeric(difftime(Sys.time(), t0, units = "secs"))))
      }, error = function(e) { fail(e); NULL })
    })
    if (!is.null(res)) {
      rv$res <- res
      top <- res$tab[res$tab$J > 0, ]
      rv$pair <- if (nrow(top)) c(top$feature_x[1], top$feature_y[1]) else c(res$tab$feature_x[1], res$tab$feature_y[1])
      nav_select("nav", "results")
    }
  })

  # ---- results --------------------------------------------------------------------------------
  output$result_boxes <- renderUI({
    r <- rv$res
    if (is.null(r)) return(card(tags$p(class = "jx-lede", "Run the analysis (step 2) to see interfaces here.")))
    tab <- r$tab
    sig <- tab[tab$call == "interface (opposing)", ]
    layout_columns(col_widths = c(3, 3, 3, 3),
      value_box("Spots analyzed", fmt_n(r$params$n), r$params$graph, theme = "light"),
      value_box("Gene pairs tested", fmt_n(nrow(tab)), sprintf("%d genes, %.0f s", ncol(r$X), r$params$seconds), theme = "light"),
      value_box("Interfaces (q < 0.05)", fmt_n(nrow(sig)), "pairs with opposing change across shared edges",
                theme = value_box_theme(bg = "#FDE2DF", fg = "#7A1F14")),
      value_box("Strongest interface", if (nrow(sig)) sig$pair[1] else "none at q < 0.05",
                if (nrow(sig)) sprintf("J = %.2f, q = %.1e", sig$J[1], sig$q[1]) else "", theme = "light"))
  })

  shown <- reactive({
    r <- rv$res
    req(r)
    tab <- r$tab
    if (input$direction == "opposing") tab <- tab[tab$J > 0, ]
    if (input$direction == "concordant") {
      tab <- tab[tab$J < 0, ]
      tab <- tab[order(tab$J), ]
    }
    q <- input$qmax
    if (!is.null(q) && !is.na(q)) tab <- tab[tab$q <= q, ]
    tab
  })

  output$pair_table <- renderDT({
    tab <- shown()
    out <- data.frame(Pair = tab$pair, J = round(tab$J, 3), z = round(tab$z, 2),
                      q = signif(tab$q, 2), stringsAsFactors = FALSE)
    datatable(out, rownames = FALSE, selection = "single",
              options = list(pageLength = 10, dom = "ftip", order = list(), scrollX = TRUE))
  })

  observeEvent(input$pair_table_rows_selected, {
    tab <- shown()
    i <- input$pair_table_rows_selected
    if (length(i)) rv$pair <- c(tab$feature_x[i], tab$feature_y[i])
  })

  output$volcano <- renderPlotly({
    r <- rv$res
    req(r)
    tab <- r$tab
    tab$y <- -log10(pmax(tab$q, 1e-300))
    tab$key <- paste(tab$feature_x, tab$feature_y, sep = "|")
    cols <- c("interface (opposing)" = COL_OPP, "concordant" = COL_CON, "not significant" = COL_NS)
    p <- plot_ly(tab, x = ~J, y = ~y, color = ~call, colors = cols, type = "scattergl", mode = "markers",
                 marker = list(size = 6, opacity = 0.75), customdata = ~key, source = "volcano",
                 text = ~sprintf("%s<br>J = %.3f<br>q = %.2g", pair, J, q), hoverinfo = "text")
    p <- layout(p, xaxis = list(title = "boundary J", zeroline = TRUE, zerolinecolor = "#9CA3AF"),
                yaxis = list(title = "-log10 q"), legend = list(orientation = "h", y = -0.18),
                shapes = list(list(type = "line", xref = "paper", x0 = 0, x1 = 1, y0 = -log10(0.05), y1 = -log10(0.05),
                                   line = list(dash = "dot", color = "#9CA3AF"))),
                margin = list(l = 50, r = 10, t = 10, b = 40))
    p <- event_register(p, "plotly_click")
    config(p, displaylogo = FALSE)
  })

  observeEvent(suppressWarnings(event_data("plotly_click", source = "volcano")), {
    e <- suppressWarnings(event_data("plotly_click", source = "volcano"))
    if (!is.null(e$customdata)) rv$pair <- strsplit(e$customdata, "|", fixed = TRUE)[[1]]
  })

  edges <- reactive({
    r <- rv$res
    p <- rv$pair
    req(r, p)
    score_edges(r$X[, p, drop = FALSE], r$graph, scale = 4L, ridge = 1, smoother = "lazy")
  })

  output$map_title <- renderText({
    p <- rv$pair
    if (is.null(p)) return("Interface map")
    r <- rv$res
    row <- r$tab[(r$tab$feature_x == p[1] & r$tab$feature_y == p[2]) | (r$tab$feature_x == p[2] & r$tab$feature_y == p[1]), ]
    sprintf("Interface map: %s and %s  (J = %.3f, q = %.2g)", p[1], p[2], row$J[1], row$q[1])
  })

  output$map_x <- renderPlotly({
    r <- rv$res; p <- rv$pair; req(r, p)
    tissue_plot(r$d, r$X[, p[1]], p[1], title = p[1], height = 400)
  })
  output$map_y <- renderPlotly({
    r <- rv$res; p <- rv$pair; req(r, p)
    tissue_plot(r$d, r$X[, p[2]], p[2], title = p[2], height = 400)
  })
  output$map_j <- renderPlotly({
    r <- rv$res; req(r)
    v <- edges_to_nodes(edges(), nrow(r$X), field = "signed_evidence")
    tissue_plot(r$d, v, "interface", kind = "diverging", title = "Interface (signed evidence)", height = 400)
  })

  boundary_table <- reactive({
    r <- rv$res
    a <- input$annotation %||% ""
    if (is.null(r) || !nzchar(a) || !a %in% names(r$d$meta)) return(NULL)
    e <- edges()
    lab <- as.character(r$d$meta[[a]])
    A <- lab[e$from]
    B <- lab[e$to]
    cross <- !is.na(A) & !is.na(B) & A != B
    if (!any(cross)) return(NULL)
    key <- ifelse(A < B, paste(A, B, sep = " | "), paste(B, A, sep = " | "))[cross]
    s <- e$signed_evidence[cross]
    out <- stats::aggregate(s, list(boundary = key), function(v) c(n = length(v), mean = mean(v)))
    out <- data.frame(Boundary = out$boundary, Edges = out$x[, "n"], Evidence = round(out$x[, "mean"], 3))
    out <- out[out$Edges >= 5, ]
    out <- out[order(-out$Evidence), ]
    names(out)[3] <- "Mean signed evidence"
    out
  })

  output$boundary_ui <- renderUI({
    if (is.null(boundary_table())) return(NULL)
    tagList(tags$h6(class = "mt-3", sprintf("Which annotated regions does this interface separate? (%s)", input$annotation)),
            tags$p(class = "jx-small", "Edges between two different regions, ranked by the mean signed evidence of the selected pair."),
            DTOutput("boundary_table"))
  })
  output$boundary_table <- renderDT(datatable(boundary_table(), rownames = FALSE,
                                              options = list(pageLength = 5, dom = "tip")))

  # ---- co-localization ------------------------------------------------------------------------
  output$regional_heatmap <- renderPlotly({
    r <- rv$res
    req(r)
    R <- r$regional
    keep <- head(order(-apply(abs(R - diag(diag(R))), 1, max, na.rm = TRUE)), 60)
    R <- R[keep, keep]
    o <- stats::hclust(stats::as.dist(1 - R), method = "average")$order
    R <- R[o, o]
    p <- plot_ly(x = colnames(R), y = rownames(R), z = R, type = "heatmap", colorscale = DIV_SCALE, zmin = -1, zmax = 1,
                 colorbar = list(title = "regional J", thickness = 12),
                 hovertemplate = "%{y} – %{x}<br>regional J = %{z:.2f}<extra></extra>")
    p <- layout(p, xaxis = list(tickfont = list(size = 9), tickangle = -60), yaxis = list(tickfont = list(size = 9), autorange = "reversed"),
                margin = list(l = 80, r = 10, t = 10, b = 80))
    config(p, displaylogo = FALSE)
  })

  output$regional_table <- renderDT({
    r <- rv$res
    req(r)
    R <- r$regional
    iu <- which(upper.tri(R), arr.ind = TRUE)
    out <- data.frame(Pair = paste(rownames(R)[iu[, 1]], colnames(R)[iu[, 2]], sep = " – "),
                      RegionalJ = round(R[iu], 3))
    out <- out[order(-out$RegionalJ), ]
    names(out)[2] <- "Regional J"
    datatable(out, rownames = FALSE, options = list(pageLength = 12, dom = "ftip"))
  })

  # ---- export ---------------------------------------------------------------------------------
  output$methods <- renderUI({
    r <- rv$res
    if (is.null(r)) return(tags$p(class = "jx-lede", "Run the analysis first."))
    tags$div(class = "jx-methods", methods_text(r))
  })

  output$dl_pairs <- downloadHandler("boundary_J_all_pairs.csv", function(file) {
    r <- rv$res
    req(r)
    utils::write.csv(r$tab[, c("feature_x", "feature_y", "J", "z", "p", "q", "call")], file, row.names = FALSE)
  })
  output$dl_regional <- downloadHandler("regional_J_matrix.csv", function(file) {
    req(rv$res)
    utils::write.csv(rv$res$regional, file)
  })
  output$dl_map <- downloadHandler(function() paste0("interface_", paste(rv$pair, collapse = "_"), ".csv"), function(file) {
    r <- rv$res
    p <- rv$pair
    req(r, p)
    e <- edges()
    out <- data.frame(spot = rownames(r$X) %||% seq_len(nrow(r$X)), x = r$d$coords$x, y = r$d$coords$y,
                      r$X[, p, drop = FALSE], check.names = FALSE)
    out$signed_evidence <- edges_to_nodes(e, nrow(r$X), "signed_evidence")
    out$local_J <- edges_to_nodes(e, nrow(r$X), "j")
    utils::write.csv(out, file, row.names = FALSE)
  })
  output$dl_methods <- downloadHandler("juxtaposed_methods.txt", function(file) {
    req(rv$res)
    writeLines(methods_text(rv$res), file)
  })
}

shinyApp(ui, server)
