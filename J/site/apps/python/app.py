"""juxtaposed - Shiny for Python app (runs in the browser through Shinylive; nothing is uploaded to a server).

Pipeline: data -> spatial graph -> boundary J for all pairs (calibrated) -> edge map of one pair -> regional J.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from shiny import App, reactive, render, req, ui  # noqa: E402

import pipeline as P  # noqa: E402

DIV = LinearSegmentedColormap.from_list("jdiv", P.DIVERGING)
EXPR = LinearSegmentedColormap.from_list("jexpr", P.EXPRESSION)

GUIDE = ui.markdown(
    """
**Which score answers my question?**

| Question | Use | Reads |
|---|---|---|
| Do two programs meet at an interface and change in *opposite* directions across it (tumour–stroma, layer borders, niche edges)? | **Boundary J** | J > 0 opposing; J < 0 concordant; J ≈ 0 no shared boundary |
| Are two genes expressed in the *same* neighbourhoods (co-expression, co-localization)? | **Regional J** | > 0 shared region; < 0 complementary regions |
| *Where* is the interface? | **Edge map** (local J of one pair) | high values mark the edges where the pair changes in opposite directions |

Boundary J is not a co-expression score. For co-localization use regional J.
"""
)

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h5("1 · Data"),
        ui.input_radio_buttons("source", None, {"demo": "Demo tissue (synthetic)", "upload": "Upload a table"}),
        ui.panel_conditional(
            "input.source === 'upload'",
            ui.input_file("file", "CSV or TSV", accept=[".csv", ".tsv", ".txt"]),
            ui.help_text("Columns: x, y (spot coordinates), then one column per gene, normalized "
                         "(e.g. log1p CP10k). Up to 200 genes are scored."),
        ),
        ui.h5("2 · Graph"),
        ui.input_slider("neighbors", "Nearest neighbours", 4, 12, 6),
        ui.help_text("6 fits Visium's hexagonal grid."),
        ui.h5("3 · Scale"),
        ui.input_slider("scale", "Smoothing steps for boundary J", 0, 4, 1),
        ui.help_text("1 is the default; raise it for noisy, sparse data."),
        ui.input_action_button("run", "Run analysis", class_="btn-primary"),
        ui.download_button("dl", "Download pair table (CSV)"),
        width=310,
    ),
    ui.navset_card_tab(
        ui.nav_panel("Start here", GUIDE,
                     ui.p("Press Run analysis. The demo tissue has a known answer: GeneA and GeneB are complementary "
                          "domains that touch (boundary J > 0); GeneC shares GeneA's region (regional J > 0, "
                          "boundary J < 0); GeneD and GeneE are separated islands (anticorrelated, but J ≈ 0 "
                          "because they never share an edge); GeneF is noise.")),
        ui.nav_panel("Expression",
                     ui.input_select("gene", "Gene", choices=[]),
                     ui.output_plot("expr_plot", height="520px")),
        ui.nav_panel("Boundary J (all pairs)",
                     ui.layout_columns(
                         ui.input_radio_buttons("direction", "Show", {"opposing": "Opposing (J > 0)",
                                                                      "concordant": "Concordant (J < 0)",
                                                                      "all": "All pairs"}, inline=True),
                         ui.input_numeric("qmax", "Max q (BH)", 1.0, min=0, max=1, step=0.01),
                         col_widths=(8, 4)),
                     ui.output_data_frame("pair_table"),
                     ui.help_text("z, p: exact spectral sign-flip null on this graph (no permutations). "
                                  "q: Benjamini–Hochberg over all unique pairs. Click a row to map it.")),
        ui.nav_panel("Edge map",
                     ui.layout_columns(ui.input_select("gx", "Gene x", choices=[]),
                                       ui.input_select("gy", "Gene y", choices=[]),
                                       ui.input_select("field", "Edge value",
                                                       {"signed_evidence": "Signed evidence (localizes interfaces)",
                                                        "j": "Local J (bounded, -1 to 1)"}),
                                       col_widths=(4, 4, 4)),
                     ui.output_plot("edge_plot", height="520px"),
                     ui.help_text("Local J pooled over each edge's neighbourhood (lazy smoother, 4 steps, ridge 1), "
                                  "averaged onto spots. Red marks opposing change: the interface.")),
        ui.nav_panel("Regional J (co-localization)",
                     ui.output_plot("regional_plot", height="560px"),
                     ui.help_text("Edge-sum concordance averaged over graph scales 1, 2, 4, 8, 16. "
                                  "For co-expression and co-localization, use this score rather than boundary J.")),
    ),
    title="juxtaposed · the juxtaposition index J",
    fillable=False,
)


def _spot_axes(df, ax):
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def server(input, output, session):
    @reactive.calc
    def data():
        if input.source() == "demo":
            return P.demo_tissue()
        f = req(input.file())[0]
        with open(f["datapath"], "rb") as fh:
            return P.read_table(fh.read(), f["name"])

    @reactive.effect
    def _genes():
        g = P.genes_of(data())
        ui.update_select("gene", choices=g, selected=g[0])
        ui.update_select("gx", choices=g, selected=g[0])
        ui.update_select("gy", choices=g, selected=g[1])

    @reactive.calc
    @reactive.event(input.run)
    def graph():
        return P.build_graph(data(), input.neighbors())

    @reactive.calc
    @reactive.event(input.run)
    def pairs():
        with ui.Progress() as p:
            p.set(message="Boundary J with the spectral test…")
            return P.pairwise(data(), graph(), input.scale())

    @reactive.calc
    @reactive.event(input.run)
    def regional():
        return P.regional(data(), graph())

    @reactive.calc
    def shown():
        tab = pairs()
        if input.direction() == "opposing":
            tab = tab[tab["J"] > 0]
        elif input.direction() == "concordant":
            tab = tab[tab["J"] < 0].sort_values("J")
        q = input.qmax()
        if q is not None:
            tab = tab[tab["q"] <= q]
        return tab.reset_index(drop=True)

    @render.data_frame
    def pair_table():
        return render.DataGrid(shown().round(4), selection_mode="row", height="420px")

    @reactive.effect
    def _pick_row():
        rows = pair_table.cell_selection()["rows"]
        if rows:
            r = shown().iloc[rows[0]]
            ui.update_select("gx", selected=r["feature_x"])
            ui.update_select("gy", selected=r["feature_y"])

    @render.plot
    def expr_plot():
        df = data()
        g = req(input.gene())
        fig, ax = plt.subplots()
        sc = ax.scatter(df["x"], df["y"], c=df[g], cmap=EXPR, s=18, linewidths=0)
        fig.colorbar(sc, ax=ax, shrink=0.7, label=g)
        _spot_axes(df, ax)
        return fig

    @render.plot
    def edge_plot():
        df = data()
        gx, gy = req(input.gx()), req(input.gy())
        req(gx != gy)
        graph()  # needs Run
        val = P.edge_map(df, graph(), gx, gy, input.field())
        lim = np.nanmax(np.abs(val)) or 1.0
        fig, ax = plt.subplots()
        sc = ax.scatter(df["x"], df["y"], c=val, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), s=18, linewidths=0)
        fig.colorbar(sc, ax=ax, shrink=0.7, label=f"{gx} vs {gy}")
        _spot_axes(df, ax)
        return fig

    @render.plot
    def regional_plot():
        m = regional()
        fig, ax = plt.subplots()
        im = ax.imshow(m.to_numpy(), cmap=DIV, vmin=-1, vmax=1)
        n = len(m)
        ax.set_xticks(range(n), m.columns, rotation=90, fontsize=8 if n < 40 else 5)
        ax.set_yticks(range(n), m.index, fontsize=8 if n < 40 else 5)
        fig.colorbar(im, ax=ax, shrink=0.7, label="regional J")
        return fig

    @render.download(filename="juxtaposition_pairs.csv")
    def dl():
        yield pairs().to_csv(index=False)


app = App(app_ui, server)
