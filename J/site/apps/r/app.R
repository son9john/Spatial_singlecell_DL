# juxtaposed - Shiny app for R (runs in the browser through Shinylive/webR; data never leave the computer).
# Pipeline: data -> spatial graph -> boundary J for all pairs (calibrated) -> edge map of one pair -> regional J.
# The package's R files are copied into ./juxtaposed/ by the site workflow and sourced below.

library(shiny)
library(bslib)
library(ggplot2)
library(Matrix)
library(DT)

for (f in list.files("juxtaposed", pattern = "\\.R$", full.names = TRUE)) source(f, local = FALSE)

# Default R (ggplot2) palette
DIVERGING <- c("#3A3A98", "#619CFF", "#F7F7F7", "#F8766D", "#832424")
EXPRESSION <- c("#F2F2F2", "#56B1F7", "#132B43")

demo_tissue <- function(side = 30L, seed = 1L) {
  set.seed(seed)
  g <- expand.grid(col = seq_len(side) - 1L, row = seq_len(side) - 1L)
  x <- g$col + 0.5 * (g$row %% 2)
  y <- g$row * sqrt(3) / 2
  mid <- max(x) / 2
  soft <- 1 / (1 + exp(-(x - mid) / 0.6))
  a <- 3 * (1 - soft)
  island <- function(cx, cy) 3 * (sqrt((x - cx * max(x))^2 + (y - cy * max(y))^2) < 0.15 * max(x))
  noise <- function() rnorm(length(x), 0, 0.3)
  data.frame(x = x, y = y,
             GeneA = a + noise(), GeneB = 3 * soft + noise(), GeneC = a + noise(),
             GeneD = island(0.2, 0.25) + noise(), GeneE = island(0.8, 0.8) + noise(),
             GeneF = rnorm(length(x), 1.5, 1))
}

read_table <- function(path, name) {
  sep <- if (grepl("\\.(tsv|txt)$", tolower(name))) "\t" else ","
  df <- utils::read.csv(path, sep = sep, check.names = FALSE)
  names(df)[tolower(names(df)) %in% c("x", "y")] <- tolower(names(df)[tolower(names(df)) %in% c("x", "y")])
  if (!all(c("x", "y") %in% names(df))) stop("The file needs columns named x and y, then one column per gene.")
  genes <- setdiff(names(df)[vapply(df, is.numeric, TRUE)], c("x", "y"))
  if (length(genes) < 2L) stop("At least two numeric gene columns are needed.")
  stats::na.omit(df[, c("x", "y", genes)])
}

genes_of <- function(df) utils::head(setdiff(names(df), c("x", "y")), 200L)

spot_plot <- function(df, value, label, diverging = FALSE) {
  p <- ggplot(data.frame(x = df$x, y = df$y, v = value), aes(x, y, colour = v)) +
    geom_point(size = 1.6) + scale_y_reverse() + coord_equal() + theme_void(base_size = 13) +
    labs(colour = label)
  if (diverging) {
    lim <- max(abs(value), na.rm = TRUE)
    p + scale_colour_gradientn(colours = DIVERGING, limits = c(-lim, lim), na.value = "grey80")
  } else {
    p + scale_colour_gradientn(colours = EXPRESSION)
  }
}

guide <- markdown("
**Which score answers my question?**

| Question | Use | Reads |
|---|---|---|
| Do two programs meet at an interface and change in *opposite* directions across it (tumour–stroma, layer borders, niche edges)? | **Boundary J** | J > 0 opposing; J < 0 concordant; J ≈ 0 no shared boundary |
| Are two genes expressed in the *same* neighbourhoods (co-expression, co-localization)? | **Regional J** | > 0 shared region; < 0 complementary regions |
| *Where* is the interface? | **Edge map** (local J of one pair) | high values mark the edges where the pair changes in opposite directions |

Boundary J is not a co-expression score. For co-localization use regional J.

**Demo tissue (known answer).** GeneA and GeneB are complementary domains that touch (boundary J > 0).
GeneC shares GeneA's region (regional J > 0, boundary J < 0). GeneD and GeneE are separated islands:
anticorrelated, but J ≈ 0 because they never share an edge. GeneF is noise.
")

ui <- page_sidebar(
  title = "juxtaposed · the juxtaposition index J",
  sidebar = sidebar(
    width = 310,
    h5("1 · Data"),
    radioButtons("source", NULL, c("Demo tissue (synthetic)" = "demo", "Upload a table" = "upload")),
    conditionalPanel("input.source === 'upload'",
      fileInput("file", "CSV or TSV", accept = c(".csv", ".tsv", ".txt")),
      helpText("Columns: x, y (spot coordinates), then one column per gene, normalized (e.g. log1p CP10k).",
               "Up to 200 genes are scored.")),
    h5("2 · Graph"),
    sliderInput("neighbors", "Nearest neighbours", 4, 12, 6, step = 1),
    helpText("6 fits Visium's hexagonal grid."),
    h5("3 · Scale"),
    sliderInput("scale", "Smoothing steps for boundary J", 0, 4, 1, step = 1),
    helpText("1 is the default; raise it for noisy, sparse data."),
    actionButton("run", "Run analysis", class = "btn-primary"),
    downloadButton("dl", "Download pair table (CSV)")
  ),
  navset_card_tab(
    nav_panel("Start here", guide),
    nav_panel("Expression", selectInput("gene", "Gene", NULL), plotOutput("expr_plot", height = 520)),
    nav_panel("Boundary J (all pairs)",
      layout_columns(col_widths = c(8, 4),
        radioButtons("direction", "Show", inline = TRUE,
                     c("Opposing (J > 0)" = "opposing", "Concordant (J < 0)" = "concordant", "All pairs" = "all")),
        numericInput("qmax", "Max q (BH)", 1, min = 0, max = 1, step = 0.01)),
      DTOutput("pair_table"),
      helpText("z, p: exact spectral sign-flip null on this graph (no permutations).",
               "q: Benjamini–Hochberg over all unique pairs. Click a row to map it.")),
    nav_panel("Edge map",
      layout_columns(col_widths = c(4, 4, 4),
        selectInput("gx", "Gene x", NULL), selectInput("gy", "Gene y", NULL),
        selectInput("field", "Edge value", c("Signed evidence (localizes interfaces)" = "signed_evidence",
                                             "Local J (bounded, -1 to 1)" = "j"))),
      plotOutput("edge_plot", height = 520),
      helpText("Local J pooled over each edge's neighbourhood (lazy smoother, 4 steps, ridge 1),",
               "averaged onto spots. Red marks opposing change: the interface.")),
    nav_panel("Regional J (co-localization)",
      plotOutput("regional_plot", height = 560),
      helpText("Edge-sum concordance averaged over graph scales 1, 2, 4, 8, 16.",
               "For co-expression and co-localization, use this score rather than boundary J."))
  )
)

server <- function(input, output, session) {
  dat <- reactive({
    if (input$source == "demo") return(demo_tissue())
    req(input$file)
    read_table(input$file$datapath, input$file$name)
  })

  observe({
    g <- genes_of(dat())
    updateSelectInput(session, "gene", choices = g, selected = g[1])
    updateSelectInput(session, "gx", choices = g, selected = g[1])
    updateSelectInput(session, "gy", choices = g, selected = g[2])
  })

  graph <- eventReactive(input$run, spatial_graph(as.matrix(dat()[, c("x", "y")]), neighbors = input$neighbors))

  pairs <- eventReactive(input$run, {
    withProgress(message = "Boundary J with the spectral test…", {
      X <- as.matrix(dat()[, genes_of(dat())])
      tab <- juxtaposition(X, graph(), scale = input$scale)
      tab$direction <- ifelse(tab$J > 0, "opposing (interface)", "concordant")
      tab
    })
  })

  regional <- eventReactive(input$run, {
    g <- genes_of(dat())
    m <- regional_juxtaposition_matrix(as.matrix(dat()[, g]), graph())
    dimnames(m) <- list(g, g)
    m
  })

  shown <- reactive({
    tab <- pairs()
    if (input$direction == "opposing") tab <- tab[tab$J > 0, ]
    if (input$direction == "concordant") {
      tab <- tab[tab$J < 0, ]
      tab <- tab[order(tab$J), ]
    }
    if (!is.na(input$qmax)) tab <- tab[tab$q <= input$qmax, ]
    rownames(tab) <- NULL
    tab
  })

  output$pair_table <- renderDT(
    datatable(shown(), selection = "single", rownames = FALSE, options = list(pageLength = 15)) |>
      formatSignif(c("J", "z", "p", "q"), 3))

  observeEvent(input$pair_table_rows_selected, {
    r <- shown()[input$pair_table_rows_selected, ]
    updateSelectInput(session, "gx", selected = r$feature_x)
    updateSelectInput(session, "gy", selected = r$feature_y)
  })

  output$expr_plot <- renderPlot({
    req(input$gene %in% names(dat()))
    spot_plot(dat(), dat()[[input$gene]], input$gene)
  })

  output$edge_plot <- renderPlot({
    req(input$gx, input$gy, input$gx != input$gy)
    df <- dat()
    e <- score_edges(as.matrix(df[, c(input$gx, input$gy)]), graph(), scale = 4L, ridge = 1, smoother = "lazy")
    spot_plot(df, edges_to_nodes(e, nrow(df), input$field), paste(input$gx, "vs", input$gy), diverging = TRUE)
  })

  output$regional_plot <- renderPlot({
    m <- regional()
    long <- data.frame(x = factor(colnames(m)[col(m)], colnames(m)), y = factor(rownames(m)[row(m)], rev(rownames(m))),
                       v = as.vector(m))
    ggplot(long, aes(x, y, fill = v)) + geom_tile() + coord_equal() +
      scale_fill_gradientn(colours = DIVERGING, limits = c(-1, 1), name = "regional J") +
      labs(x = NULL, y = NULL) + theme_minimal(base_size = 12) +
      theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
  })

  output$dl <- downloadHandler("juxtaposition_pairs.csv", function(file) utils::write.csv(pairs(), file, row.names = FALSE))
}

shinyApp(ui, server)
