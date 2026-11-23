"""juxtaposed: find where two expression programs meet in tissue.

Shiny for Python app, run in the browser with Shinylive (Pyodide): data never leave this computer.
Inputs: AnnData (.h5ad), 10x Space Ranger output (Visium, Visium HD), 10x Xenium output, or tables.
"""
from __future__ import annotations

import base64
import io
import json
import time
import warnings

import numpy as np
import pandas as pd
from shiny import App, reactive, render, req, ui

import juxtaposed as jx
import readers as R

# ---- constants: default R (ggplot2) colours ---------------------------------------------------
COL_OPP, COL_CON, COL_REG, COL_NS = "#F8766D", "#619CFF", "#00BA38", "#BDBDBD"
DIV_SCALE = [[0, "#3A3A98"], [0.25, "#619CFF"], [0.5, "#F7F7F7"], [0.75, "#F8766D"], [1, "#832424"]]
EXPR_SCALE = [[0, "#F2F2F2"], [0.5, "#56B1F7"], [1, "#132B43"]]
N_DRAWS = 2000  # Monte Carlo sign-flip draws for pairs with few spectral modes
PLOTLY_JS = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"
EXAMPLE_FALLBACK = "https://son9john.github.io/Spatial_singlecell_DL/example/"
EXAMPLE_FILES = ["filtered_feature_bc_matrix.jx10x", "tissue_positions_list.csv", "scalefactors_json.json",
                 "tissue_lowres_image.png"]


def gg_hue(n: int) -> list[str]:
    """scales::hue_pal()(n): evenly spaced hues at chroma 100, luminance 65 (as in ggplot2)."""
    out = []
    for h in np.linspace(15, 375, n + 1)[:n]:
        out.append(_hcl_to_hex(h, 100, 65))
    return out


def _hcl_to_hex(h, c, l):  # noqa: E741 - CIE-LUV polar to sRGB, as grDevices::hcl()
    hr = np.deg2rad(h)
    L, U, V = l, c * np.cos(hr), c * np.sin(hr)
    xn, yn, zn = 0.95047, 1.0, 1.08883
    un, vn = 4 * xn / (xn + 15 * yn + 3 * zn), 9 * yn / (xn + 15 * yn + 3 * zn)
    Y = yn * (((L + 16) / 116) ** 3 if L > 8 else L / 903.3)
    u, v = U / (13 * L) + un, V / (13 * L) + vn
    X = 9 * Y * u / (4 * v)
    Z = -X / 3 - 5 * Y + 3 * Y / v
    rgb = np.array([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560],
                    [0.0556434, -0.2040259, 1.0572252]]) @ np.array([X, Y, Z])
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(np.clip(rgb, 0, None), 1 / 2.4) - 0.055)
    return "#" + "".join(f"{int(round(255 * min(max(x, 0), 1))):02X}" for x in rgb)


# ---- figures (plotly.js specs built by hand; no plotly Python package needed) -----------------

def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def image_uri(img, max_px: int = 900):
    if img is None:
        return None
    from PIL import Image

    h, w = img.shape[:2]
    f = int(np.ceil(max(h, w) / max_px))
    if f > 1:
        img = img[::f, ::f]
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img[..., :3])).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def spot_size(d, height=430):
    from scipy.spatial import cKDTree

    xy = d["coords"]
    if len(xy) < 3:
        return 6
    s = np.random.default_rng(0).choice(len(xy), min(2000, len(xy)), replace=False)
    nn = cKDTree(xy).query(xy[s], k=2)[0][:, 1]
    span = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])) or 1
    return float(max(2.5, min(14, 0.95 * np.median(nn) / span * height)))


def _finite(v):
    return np.where(np.isfinite(v), v, np.nan)


def tissue_fig(d, value, label, kind="continuous", title=None, height=430):
    xy = d["coords"]
    ms = spot_size(d, height)
    names = np.asarray(d["spots"], dtype=str)
    traces = []
    if kind == "categorical":
        cats = pd.Categorical(value)
        cols = gg_hue(len(cats.categories))
        for k, c in enumerate(cats.categories):
            m = np.asarray(cats == c)
            traces.append(dict(type="scattergl", mode="markers", name=str(c), x=xy[m, 0], y=xy[m, 1],
                               marker=dict(size=ms, color=cols[k], opacity=0.9),
                               text=[f"{n}<br>{label}: {c}" for n in names[m]], hoverinfo="text"))
    else:
        v = np.asarray(value, dtype=float)
        ok = np.isfinite(v)
        if (~ok).any():
            traces.append(dict(type="scattergl", mode="markers", x=xy[~ok, 0], y=xy[~ok, 1], showlegend=False,
                               marker=dict(size=ms, color="#D9D9D9", opacity=0.6), hoverinfo="skip"))
        lim = float(np.nanmax(np.abs(v[ok]))) if ok.any() else 1.0
        mk = dict(size=ms, color=v[ok], opacity=0.92, colorscale=DIV_SCALE if kind == "diverging" else EXPR_SCALE,
                  colorbar=dict(title=dict(text=label, side="right"), thickness=12, len=0.8))
        if kind == "diverging":
            mk.update(cmin=-lim, cmax=lim, cmid=0)
        traces.append(dict(type="scattergl", mode="markers", x=xy[ok, 0], y=xy[ok, 1], marker=mk, showlegend=False,
                           text=[f"{n}<br>{label}: {x:.3g}" for n, x in zip(names[ok], v[ok])], hoverinfo="text"))
    layout = dict(height=height, margin=dict(l=5, r=5, t=35 if title else 5, b=5),
                  xaxis=dict(visible=False, zeroline=False),
                  yaxis=dict(visible=False, autorange="reversed", scaleanchor="x", zeroline=False),
                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  legend=dict(itemsizing="constant", font=dict(size=11)))
    if title:
        layout["title"] = dict(text=title, font=dict(size=14), x=0.02)
    if d.get("image_uri"):
        h, w = d["image"].shape[:2]
        layout["images"] = [dict(source=d["image_uri"], xref="x", yref="y", x=0, y=0, sizex=w, sizey=h,
                                 sizing="stretch", xanchor="left", yanchor="top", layer="below", opacity=0.85)]
        layout["xaxis"]["range"] = [0, w]
        layout["yaxis"].update(range=[h, 0], autorange=False)
    return dict(data=traces, layout=layout)


def volcano_fig(tab: pd.DataFrame):
    cols = {"interface (opposing)": COL_OPP, "concordant": COL_CON, "not significant": COL_NS}
    y = -np.log10(np.clip(tab["q"].to_numpy(float), 1e-300, None))
    traces = []
    for call in ["not significant", "concordant", "interface (opposing)"]:
        m = (tab["call"] == call).to_numpy()
        if not m.any():
            continue
        sub = tab[m]
        traces.append(dict(type="scattergl", mode="markers", name=call, x=sub["J"], y=y[m],
                           marker=dict(size=6, color=cols[call], opacity=0.75),
                           customdata=(sub["feature_x"] + "|" + sub["feature_y"]).tolist(),
                           text=[f"{p}<br>J = {j:.3f}<br>q = {q:.2g}" for p, j, q in zip(sub["pair"], sub["J"], sub["q"])],
                           hoverinfo="text"))
    layout = dict(height=430, margin=dict(l=50, r=10, t=10, b=40), xaxis=dict(title="boundary J", zeroline=True),
                  yaxis=dict(title="-log10 q"), legend=dict(orientation="h", y=-0.18),
                  shapes=[dict(type="line", xref="paper", x0=0, x1=1, y0=-np.log10(0.05), y1=-np.log10(0.05),
                               line=dict(dash="dot", color="#9CA3AF"))])
    return dict(data=traces, layout=layout)


def heatmap_fig(mat: pd.DataFrame):
    return dict(data=[dict(type="heatmap", x=list(mat.columns), y=list(mat.index), z=mat.to_numpy(),
                           colorscale=DIV_SCALE, zmin=-1, zmax=1, colorbar=dict(title=dict(text="regional J"), thickness=12),
                           hovertemplate="%{y} – %{x}<br>regional J = %{z:.2f}<extra></extra>")],
                layout=dict(height=620, margin=dict(l=80, r=10, t=10, b=80),
                            xaxis=dict(tickfont=dict(size=9), tickangle=-60),
                            yaxis=dict(tickfont=dict(size=9), autorange="reversed")))


def plot_ui(fig: dict, div_id: str, click_input: str | None = None):
    fig = dict(fig)
    fig["config"] = dict(displaylogo=False, responsive=True, modeBarButtonsToRemove=["lasso2d", "select2d", "autoScale2d"])
    payload = json.dumps(fig, default=_json_default).replace("</", "<\\/")
    click = (f"el.on('plotly_click', function(e) {{ var p = e.points[0]; if (p && p.customdata) "
             f"Shiny.setInputValue('{click_input}', p.customdata, {{priority: 'event'}}); }});") if click_input else ""
    js = (f"(function() {{ var fig = {payload}; var tries = 0;"
          f" (function draw() {{ var el = document.getElementById('{div_id}');"
          f"   if (!window.Plotly || !el) {{ if (tries++ < 200) setTimeout(draw, 50); return; }}"
          f"   Plotly.react(el, fig.data, fig.layout, fig.config).then(function() {{ {click} }}); }})(); }})();")
    return ui.TagList(ui.div(id=div_id, style=f"min-height:{fig['layout'].get('height', 400)}px"), ui.tags.script(ui.HTML(js)))


# ---- text helpers -----------------------------------------------------------------------------

def parse_genes(txt: str | None) -> list[str]:
    import re

    if not txt or not txt.strip():
        return []
    return list(dict.fromkeys(g for g in re.split(r"[\s,;]+", txt) if g))


def fmt_n(x) -> str:
    return f"{int(x):,}"


def file_status(files: dict, kind: str):
    has = lambda p: R.pick(files, p) is not None  # noqa: E731
    mtx = has(r"\.(h5|jx10x)$") or (has(r"^matrix\.mtx(\.gz)?$") and has(r"^barcodes\.tsv(\.gz)?$") and
                                    has(r"^(features|genes)\.tsv(\.gz)?$"))
    if kind == "visium":
        return [(mtx, True, "Count matrix: filtered_feature_bc_matrix.h5 (or matrix.mtx.gz + barcodes.tsv.gz + features.tsv.gz)"),
                (has("tissue_positions"), True, "Spot positions: tissue_positions.csv, tissue_positions_list.csv or tissue_positions.parquet"),
                (has(r"^scalefactors_json\.json$"), False, "Scale factors: scalefactors_json.json (to place spots on the image)"),
                (has(r"^tissue_(lowres|hires)_image\.png$"), False, "Tissue image: tissue_lowres_image.png")]
    if kind == "xenium":
        return [(mtx, True, "Count matrix: cell_feature_matrix.h5 (or the cell_feature_matrix MEX files)"),
                (has(r"^cells\.(parquet|csv\.gz|csv)$"), True, "Cell centroids: cells.csv.gz or cells.parquet")]
    if kind == "h5ad":
        return [(has(r"\.h5ad$"), True, "AnnData file (.h5ad) with spatial coordinates in obsm['spatial']")]
    return [(has(r"\.(csv|tsv|txt)(\.gz)?$"), True,
             "A table with x and y columns and one column per gene, or an expression table plus a coordinate table")]


def methods_text(r) -> str:
    p = r["params"]
    bin_txt = f" ({p['bin']})" if p.get("bin") else ""
    return (f"Spatial adjacency was defined as {p['graph']} on {fmt_n(p['n'])} {p['unit_label']}{bin_txt}. "
            f"Expression was {p['normalization']}. {len(p['genes'])} genes were analysed ({p['panel_label']}). "
            f"For each of the {fmt_n(len(r['tab']))} gene pairs, the juxtaposition index J (juxtaposed Python package "
            f"v{jx.__version__}) was computed after {p['scale']} step(s) of graph smoothing, and two-sided p-values were "
            f"obtained from the spectral sign-flip null of the tissue graph ({p_value_text(p)}), with Benjamini-Hochberg "
            "correction over all pairs. Interfaces were mapped with edge-level J (lazy smoother, 4 steps, ridge 1). "
            "Co-localization was scored with regional J over graph scales 1, 2, 4, 8 and 16.")


def p_value_text(p) -> str:
    """How the p-values were obtained, for the Methods paragraph."""
    modes = (f"median {p['modes_median']:.0f} effective spectral modes per pair"
             if p.get("modes_median") is not None and np.isfinite(p["modes_median"]) else "")
    if p.get("p_method") == "monte_carlo":
        core = f"Monte Carlo p-values from {fmt_n(N_DRAWS)} sign-flip draws"
    elif p.get("p_method") == "mixed":
        core = (f"normal approximation, with Monte Carlo p-values from {fmt_n(N_DRAWS)} sign-flip draws for the "
                f"{fmt_n(p['n_sampled'])} pairs with fewer than 20 effective spectral modes")
    else:
        core = "normal approximation"
    return "; ".join(t for t in (core, modes) if t)


# ---- UI ---------------------------------------------------------------------------------------

CSS = """
:root { --jx-accent: #2563EB; --jx-ink: #111827; --jx-muted: #6B7280; --jx-line: #E5E7EB; }
body { background: #F5F6F8; }
.navbar { background: #111827 !important; }
.navbar .navbar-brand, .navbar .nav-link { color: #F9FAFB !important; }
.navbar .nav-link.active { border-bottom: 2px solid #93C5FD; }
.card { border: 1px solid var(--jx-line); border-radius: 12px; box-shadow: 0 1px 2px rgba(17,24,39,.04); }
.card-header { background: #fff; font-weight: 600; border-bottom: 1px solid var(--jx-line); }
.jx-lede { color: var(--jx-muted); max-width: 900px; }
.jx-sources .shiny-options-group { display: grid; grid-template-columns: 1fr; gap: 10px; }
.jx-sources .form-check, .jx-sources .radio { margin: 0; padding: 0; position: relative; }
.jx-sources .form-check label, .jx-sources .radio label { display: block; width: 100%; border: 1px solid var(--jx-line);
  border-radius: 10px; padding: 12px 14px 12px 38px; background: #fff; cursor: pointer; }
.jx-sources .form-check input, .jx-sources .radio input { position: absolute; left: 14px; top: 16px; margin: 0; z-index: 2; }
.jx-sources .form-check:has(input:checked) label, .jx-sources .radio:has(input:checked) label {
  border-color: var(--jx-accent); background: #EFF6FF; box-shadow: 0 0 0 2px #BFDBFE; }
.jx-src-title { font-weight: 600; color: var(--jx-ink); }
.jx-src-desc { display: block; color: var(--jx-muted); font-size: .88rem; }
.jx-drop { border: 2px dashed #CBD5E1; border-radius: 12px; padding: 14px 18px 4px; background: #fff; }
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
"""


def src_choice(title, desc):
    return ui.HTML(f'<span class="jx-src-title">{title}</span><span class="jx-src-desc">{desc}</span>')


GUIDE = ui.tags.table(
    {"class": "jx-help-table"},
    ui.tags.tr(ui.tags.th("Your question"), ui.tags.th("Use"), ui.tags.th("How to read it")),
    ui.tags.tr(ui.tags.td("Do two expression programs meet at a tissue interface and change in opposite directions across it? "
                          "(tumour–stroma, layer borders, niche edges)"),
               ui.tags.td(ui.span("Boundary J", class_="jx-pill jx-pill-opp")),
               ui.tags.td("J > 0: opposing change across shared edges (an interface). J < 0: concordant change. "
                          "J ≈ 0: no shared boundary, even if the genes are anticorrelated overall.")),
    ui.tags.tr(ui.tags.td("Where exactly is that interface?"), ui.tags.td(ui.span("Interface map", class_="jx-pill jx-pill-opp")),
               ui.tags.td("Edge-level J of one pair, drawn on the tissue: red spots sit on the boundary between the two programs.")),
    ui.tags.tr(ui.tags.td("Are two genes expressed in the same neighbourhoods? (co-expression, co-localization)"),
               ui.tags.td(ui.span("Regional J", class_="jx-pill jx-pill-reg")),
               ui.tags.td("> 0: shared regions; < 0: complementary regions. Boundary J is not a co-localization score.")),
)

SOURCES = {
    "example": src_choice("Example: mouse brain (10x Visium)", "One click. A real sagittal section on its H&amp;E image."),
    "h5ad": src_choice("AnnData (.h5ad)", "From Scanpy or Squidpy, with coordinates in obsm['spatial'] (and the H&amp;E image if stored)."),
    "visium": src_choice("10x Visium or Visium HD (Space Ranger output)", "The count matrix (.h5) and the files of the spatial folder."),
    "xenium": src_choice("10x Xenium output", "cell_feature_matrix.h5 and cells.csv.gz (or cells.parquet)."),
    "tables": src_choice("Any platform: tables", "CosMx, MERSCOPE, Slide-seq, Stereo-seq, ...: an expression table and a coordinate table."),
}

load_panel = ui.nav_panel(
    "1 · Load data",
    ui.layout_columns(
        ui.card(ui.card_header("Choose your data"),
                ui.div(ui.input_radio_buttons("source", None, SOURCES, selected="example"), class_="jx-sources"),
                ui.output_ui("source_help")),
        ui.card(ui.card_header("Preview"), ui.output_ui("data_summary"), ui.output_ui("preview_controls"),
                ui.output_ui("preview_plot")),
        col_widths=(5, 7)),
    value="load")

analyze_panel = ui.nav_panel(
    "2 · Analyze",
    ui.layout_sidebar(
        ui.sidebar(
            ui.h6("Genes"),
            ui.input_radio_buttons("panel_method", None, {"svg": "Spatially variable genes (Moran's I)",
                                                          "hvg": "Highly variable genes", "custom": "Only my gene list"},
                                   selected="svg"),
            ui.input_slider("n_genes", "Number of genes", 20, 300, 100, step=10),
            ui.input_text_area("custom_genes", "Always include these genes (optional)", rows=3,
                               placeholder="e.g. Epcam, Vim, Ptprc  (separate with spaces, commas or new lines)"),
            ui.h6("Spatial graph"),
            ui.input_checkbox("k_auto", "Choose neighbours automatically", True),
            ui.panel_conditional("!input.k_auto", ui.input_numeric("k", "Nearest neighbours", 6, min=3, max=20)),
            ui.input_slider("scale", "Smoothing steps for boundary J", 0, 4, 1),
            ui.h6("P-values"),
            ui.input_radio_buttons("p_method", None,
                                   {"auto": "Normal approximation; Monte Carlo for pairs with fewer than 20 spectral modes",
                                    "normal": "Normal approximation only (fastest)"}, selected="auto"),
            ui.div(f"Monte Carlo uses {N_DRAWS:,} sign-flip draws and runs only for pairs whose null rests on few "
                   "spectral modes, where the normal approximation is too conservative.", class_="jx-small"),
            ui.h6("Large datasets"),
            ui.input_numeric("max_nodes", "Largest number of spots for the spectral test", 4000, min=500, max=15000, step=500),
            ui.div("Larger data are grouped into small square bins first. About 15 s for 2,700 spots in the browser; "
                   "time grows with the cube of the number of spots.", class_="jx-small"),
            ui.output_ui("annotation_choice"),
            ui.input_action_button("run", "Run analysis", class_="btn-primary btn-lg w-100 mt-2"),
            width=340, open="always"),
        ui.card(ui.card_header("What will happen"), ui.output_ui("plan"), ui.hr(), GUIDE)),
    value="analyze")

results_panel = ui.nav_panel(
    "3 · Interfaces",
    ui.output_ui("result_boxes"),
    ui.layout_columns(
        ui.card(ui.card_header("All gene pairs"), ui.output_ui("volcano"),
                ui.div("Click a point or a table row to map that pair. Right: opposing change across shared edges.",
                       class_="jx-small"), full_screen=True),
        ui.card(ui.card_header("Pairs"),
                ui.layout_columns(
                    ui.input_radio_buttons("direction", None, {"opposing": "Interfaces (J > 0)",
                                                               "concordant": "Concordant (J < 0)", "all": "All"},
                                           selected="opposing", inline=True),
                    ui.input_numeric("qmax", "Show q at most", 0.05, min=0, max=1, step=0.01), col_widths=(7, 5)),
                ui.output_data_frame("pair_table"), full_screen=True),
        col_widths=(5, 7)),
    ui.card(ui.card_header(ui.output_text("map_title", inline=True)),
            ui.layout_columns(ui.output_ui("map_x"), ui.output_ui("map_y"), ui.output_ui("map_j"), col_widths=(4, 4, 4)),
            ui.output_ui("boundary_ui"), full_screen=True),
    value="results")

coloc_panel = ui.nav_panel(
    "4 · Co-localization",
    ui.layout_columns(
        ui.card(ui.card_header("Regional J: shared neighbourhood expression"), ui.output_ui("regional_heatmap"), full_screen=True),
        ui.card(ui.card_header("Most co-localized pairs"), ui.output_data_frame("regional_table"),
                ui.div("Regional J averages an edge-sum concordance over graph scales 1, 2, 4, 8 and 16. It has no p-value.",
                       class_="jx-small")),
        col_widths=(7, 5)),
    value="coloc")

export_panel = ui.nav_panel(
    "Export",
    ui.layout_columns(
        ui.card(ui.card_header("Download"),
                ui.download_button("dl_pairs", "Boundary J, all pairs (CSV)", class_="w-100 mb-2"),
                ui.download_button("dl_regional", "Regional J matrix (CSV)", class_="w-100 mb-2"),
                ui.download_button("dl_map", "Interface map of the selected pair, per spot (CSV)", class_="w-100 mb-2"),
                ui.download_button("dl_methods", "Methods paragraph (TXT)", class_="w-100")),
        ui.card(ui.card_header("Methods paragraph for your paper"), ui.output_ui("methods")),
        col_widths=(5, 7)),
    value="export")

help_panel = ui.nav_panel(
    "Help",
    ui.card(ui.card_header("Which score answers my question?"), GUIDE),
    ui.card(ui.card_header("Input files"), ui.tags.ul(
        ui.tags.li(ui.tags.b("AnnData (.h5ad): "), "X (or layers['counts'] / raw.X as counts), var names, obs annotations, "
                   "obsm['spatial'] and, if present, the image in uns['spatial']. Counts are log-normalized; normalized X is used as is."),
        ui.tags.li(ui.tags.b("10x Visium / Visium HD: "), "from the Space Ranger outs folder select filtered_feature_bc_matrix.h5 "
                   "and the files of the spatial folder (or the spatial.tar.gz that 10x distributes). For Visium HD use "
                   "binned_outputs/square_016um (or square_008um for small regions)."),
        ui.tags.li(ui.tags.b("10x Xenium: "), "cell_feature_matrix.h5 and cells.csv.gz. Cells are grouped into small square bins "
                   "so that the spectral test stays fast."),
        ui.tags.li(ui.tags.b("Tables: "), "one CSV/TSV with x, y and one column per gene, or two tables (expression and coordinates) "
                   "sharing cell IDs. CosMx (exprMat + metadata) and MERSCOPE (cell_by_gene + cell_metadata) files are recognized."),
        ui.tags.li(ui.tags.b("Seurat objects: "), "use the R app.")),
        ui.p("Everything runs in this browser tab: files are not uploaded to any server.", class_="jx-small")),
    ui.card(ui.card_header("Citation"),
            ui.p("Song J, Kim J, Yoo J, Lee D, Kim K. The juxtaposition index J maps where spatial expression domains "
                 "meet, with a spectral test for every gene pair. bioRxiv (2026)."),
            ui.p("Example data: 10x Genomics, Mouse Brain Serial Section 1 (Sagittal-Anterior), Visium, CC BY 4.0.",
                 class_="jx-small")),
    value="help")

app_ui = ui.page_navbar(
    load_panel, analyze_panel, results_panel, coloc_panel, export_panel, help_panel,
    ui.nav_spacer(),
    ui.nav_control(ui.a("Scanpy notebook", href="https://son9john.github.io/Spatial_singlecell_DL/vignette/scanpy_juxtaposition.html",
                        target="_blank")),
    title=ui.HTML('<b>juxtaposed</b><span style="opacity:.7;font-weight:400;margin-left:8px">the juxtaposition index J</span>'),
    id="nav", fillable=False, bg="#111827",
    header=ui.head_content(ui.tags.style(CSS), ui.tags.script(src=PLOTLY_JS)),
    window_title="juxtaposed",
)


# ---- server -----------------------------------------------------------------------------------

def server(input, output, session):
    files = reactive.value({})
    dataset = reactive.value(None)
    result = reactive.value(None)
    pair = reactive.value(None)

    def fail(e):
        ui.notification_show(str(e), type="error", duration=12)

    def set_dataset(d):
        d = R.normalize(d)
        d["image_uri"] = image_uri(d.get("image"))
        d["total"] = np.asarray((d["counts"] if d.get("counts") is not None else d["X"]).sum(axis=1)).ravel()
        dataset.set(d)
        result.set(None)
        pair.set(None)

    def example_base():
        cd = session.clientdata
        try:
            host = cd.url_hostname()
            if host.endswith("github.io") or host in ("localhost", "127.0.0.1"):
                port = f":{cd.url_port()}" if cd.url_port() else ""
                path = cd.url_pathname()
                base = path[: path.find("/python/") + 1] if "/python/" in path else "/"
                return f"{cd.url_protocol()}//{host}{port}{base}example/"
        except Exception:  # noqa: BLE001
            pass
        return EXAMPLE_FALLBACK

    # ---- loading ------------------------------------------------------------------------------
    @render.ui
    def source_help():
        s = input.source()
        if s == "example":
            return ui.TagList(ui.p("2,695 spots, the 2,000 most variable genes, H&E image. Downloads about 6 MB.",
                                   class_="jx-small mt-3"),
                              ui.input_action_button("load_example", "Load the example", class_="btn-primary"))
        accept = {"h5ad": [".h5ad"]}.get(s)
        return ui.TagList(
            ui.div(ui.input_file("upload", "Select files (several at once, or add more in a second step)", multiple=True,
                                 accept=accept, width="100%"), class_="jx-drop mt-3"),
            ui.output_ui("file_check"),
            ui.div(ui.input_action_button("load_files", "Load data", class_="btn-primary"),
                   ui.input_action_button("clear_files", "Clear files", class_="btn-outline-secondary"),
                   class_="d-flex gap-2 mt-2"))

    @reactive.effect
    @reactive.event(input.source)
    def _reset_files():
        files.set({})

    @reactive.effect
    @reactive.event(input.upload)
    def _add_files():
        try:
            new = R.expand_uploads(input.upload())
        except Exception as e:  # noqa: BLE001
            return fail(e)
        merged = {k: v for k, v in files.get().items() if k not in new}
        merged.update(new)
        files.set(merged)

    @reactive.effect
    @reactive.event(input.clear_files)
    def _clear():
        files.set({})

    @render.ui
    def file_check():
        st = file_status(files.get(), input.source())
        items = []
        for ok, need, text in st:
            mark = ui.span("✓", class_="jx-ok") if ok else (ui.span("✗", class_="jx-miss") if need else ui.span("○", class_="jx-opt"))
            items.append(ui.tags.li(mark, text, ui.span(" (optional)", class_="jx-small") if not need else None))
        got = list(files.get())
        return ui.TagList(ui.tags.ul(*items, class_="jx-check"),
                          ui.div("Files: " + ", ".join(got), class_="jx-small") if got else None)

    @reactive.effect
    @reactive.event(input.load_files)
    def _load_files():
        f = files.get()
        if not f:
            return ui.notification_show("Select the files first.", type="warning")
        with ui.Progress(min=0, max=1) as p:
            p.set(0.3, message="Reading files")
            try:
                s = input.source()
                if s == "h5ad":
                    d = R.read_h5ad(R.pick(f, r"\.h5ad$"))
                elif s == "visium":
                    d = R.read_spaceranger(f)
                elif s == "xenium":
                    d = R.read_xenium(f)
                else:
                    d = R.read_tables(f)
                p.set(0.8, message="Normalizing")
                set_dataset(d)
            except Exception as e:  # noqa: BLE001
                fail(e)

    @reactive.effect
    @reactive.event(input.load_example)
    async def _load_example():
        import os
        import tempfile

        from pyodide.http import pyfetch  # available in the browser (Shinylive)

        base = example_base() + "visium_mouse_brain/"
        d = tempfile.mkdtemp(prefix="example_")
        with ui.Progress(min=0, max=1) as p:
            try:
                for i, fn in enumerate(EXAMPLE_FILES):
                    p.set(0.1 + 0.6 * i / len(EXAMPLE_FILES), message="Downloading the example", detail=fn)
                    resp = await pyfetch(base + fn)
                    if not resp.ok:
                        raise RuntimeError(f"Could not download {fn} (HTTP {resp.status}).")
                    with open(os.path.join(d, fn), "wb") as fh:
                        fh.write(await resp.bytes())
                p.set(0.8, message="Reading")
                ds = R.read_spaceranger({fn: os.path.join(d, fn) for fn in EXAMPLE_FILES})
                ds["platform"] = "Visium, 10x mouse brain (example)"
                set_dataset(ds)
            except Exception as e:  # noqa: BLE001
                fail(e)

    @render.ui
    def data_summary():
        d = dataset.get()
        if d is None:
            return ui.p("Load data on the left. The tissue appears here, on its image when one is available.", class_="jx-lede")
        notes = " ".join(d.get("notes", []) + ([f"Expression: {d['normalization']}."] if d.get("normalization") else []))
        return ui.TagList(
            ui.layout_columns(ui.value_box("Spots or cells", fmt_n(d["X"].shape[0]), theme="light"),
                              ui.value_box("Genes", fmt_n(d["X"].shape[1]), theme="light"),
                              ui.value_box("Platform", ui.span(d["platform"], style="font-size:1rem"), theme="light"),
                              col_widths=(4, 4, 4)),
            ui.p(notes, class_="jx-small"),
            ui.input_action_button("goto_analyze", "Continue to analysis", class_="btn-primary mb-2"))

    @reactive.effect
    @reactive.event(input.goto_analyze)
    def _goto():
        ui.update_navset("nav", selected="analyze")

    @render.ui
    def preview_controls():
        d = dataset.get()
        if d is None:
            return None
        choices = {".total": "Total counts", **{c: c for c in d["meta"].columns}}
        return ui.layout_columns(ui.input_select("preview_color", "Colour spots by", choices),
                                 ui.input_selectize("preview_gene", "…or by a gene", choices=[], options={"placeholder": "type a gene"}),
                                 col_widths=(6, 6))

    @reactive.effect
    def _gene_choices():
        d = dataset.get()
        if d is not None:
            ui.update_selectize("preview_gene", choices=[""] + list(d["genes"]), selected="", server=True)

    @render.ui
    def preview_plot():
        d = dataset.get()
        req(d)
        try:
            g = input.preview_gene() or ""
        except Exception:  # noqa: BLE001
            g = ""
        if g and g in d["genes"]:
            j = d["genes"].index(g)
            return plot_ui(tissue_fig(d, d["X"][:, j].toarray().ravel(), g), "fig_preview")
        try:
            col = input.preview_color()
        except Exception:  # noqa: BLE001
            col = ".total"
        if col and col in d["meta"].columns:
            return plot_ui(tissue_fig(d, d["meta"][col], col, kind="categorical"), "fig_preview")
        return plot_ui(tissue_fig(d, d["total"], "total"), "fig_preview")

    # ---- analysis -----------------------------------------------------------------------------
    @render.ui
    def annotation_choice():
        d = dataset.get()
        if d is None or not len(d["meta"].columns):
            return None
        pref = [c for c in ("leiden", "louvain", "cell_type", "celltype", "annotation", "clusters", "region")
                if c in d["meta"].columns]
        return ui.TagList(ui.h6("Annotation"),
                          ui.input_select("annotation", "Which annotated regions does each interface separate?",
                                          {"": "(none)", **{c: c for c in d["meta"].columns}},
                                          selected=pref[0] if pref else ""))

    @render.ui
    def plan():
        d = dataset.get()
        if d is None:
            return ui.p("Load data first (step 1).", class_="jx-lede")
        n = d["X"].shape[0]
        mx = input.max_nodes() or 4000
        method = input.panel_method()
        step3 = {"svg": f"Pick the {input.n_genes()} most spatially variable genes (Moran's I) plus your genes.",
                 "hvg": f"Pick the {input.n_genes()} most variable genes plus your genes.",
                 "custom": "Use only your gene list."}[method]
        return ui.tags.ol(
            ui.tags.li(f"Group the {fmt_n(n)} spots into about {fmt_n(mx)} square bins (sum of counts per bin)." if n > mx
                       else f"Use all {fmt_n(n)} spots."),
            ui.tags.li("Build the spatial neighbour graph (hexagonal for Visium, square for bins, nearest neighbours otherwise)."),
            ui.tags.li(step3),
            ui.tags.li("Compute boundary J for every pair, with spectral sign-flip p-values and Benjamini–Hochberg q-values."),
            ui.tags.li("Compute regional J (co-localization) for the same genes."))

    @reactive.effect
    @reactive.event(input.run)
    def _run():
        d0 = dataset.get()
        if d0 is None:
            return ui.notification_show("Load data first.", type="warning")
        custom = parse_genes(input.custom_genes())
        if input.panel_method() == "custom" and len(custom) < 2:
            return ui.notification_show("Enter at least two genes, or choose another gene selection.", type="warning")
        t0 = time.time()
        with ui.Progress(min=0, max=1) as p:
            try:
                d = d0
                bin_label = None
                mx = int(input.max_nodes() or 4000)
                n0 = d["X"].shape[0]
                if n0 > mx:
                    p.set(0.05, message="Grouping spots into bins")
                    if d.get("array") is not None and d["lattice"] == "square":
                        d = R.bin_dataset(d, merge=R.auto_lattice_factor(d, mx))
                    else:
                        d = R.bin_dataset(d, size=R.auto_bin_size(d, mx))
                    bin_label = d["bin_label"]
                    d = R.normalize(d)
                    d["image_uri"] = d0.get("image_uri")
                p.set(0.15, message="Building the spatial graph")
                graph, how = R.build_graph(d, None if input.k_auto() else int(input.k()))
                p.set(0.25, message="Selecting genes")
                idx, missing = R.select_panel(d, graph, input.panel_method(), int(input.n_genes()), custom)
                if missing:
                    ui.notification_show("Not found: " + ", ".join(missing), type="warning")
                X = d["X"][:, idx].toarray()
                names = [d["genes"][i] for i in idx]
                ok = X.std(axis=0) > 0
                X, names = X[:, ok], [g for g, k in zip(names, ok) if k]
                if X.shape[1] < 2:
                    raise ValueError("Fewer than two genes vary across the tissue; choose other genes.")
                p.set(0.35, message=f"Spectral null for {fmt_n(X.shape[0])} spots")
                null = jx.SpectralNull(graph)
                p.set(0.75, message=f"Boundary J for {fmt_n(X.shape[1] * (X.shape[1] - 1) // 2)} pairs")
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    jres = jx.juxtaposition(X, graph, features=names, null=null, scale=int(input.scale()),
                                            method=input.p_method(), n_draws=N_DRAWS, seed=0)
                for w in caught:
                    if "spectral modes" in str(w.message):
                        ui.notification_show("Many pairs rest on few spectral modes; choose the Monte Carlo option "
                                             "for their p-values.", type="warning", duration=10)
                tab = jres.table(sort="J")
                p_method = jres.extra.get("p_method", "normal")
                modes = tab["modes"] if "modes" in tab else pd.Series(dtype=float)
                n_sampled = int(np.sum(modes < 20)) if p_method == "mixed" else 0
                tab["pair"] = tab["feature_x"] + " – " + tab["feature_y"]
                tab["call"] = np.where((tab["q"] < 0.05) & (tab["J"] > 0), "interface (opposing)",
                                       np.where((tab["q"] < 0.05) & (tab["J"] < 0), "concordant", "not significant"))
                p.set(0.9, message="Regional J")
                reg = pd.DataFrame(jx.regional_juxtaposition_matrix(X, graph), index=names, columns=names)
                res = dict(d=d, graph=graph, X=X, names=names, tab=tab, regional=reg,
                           params=dict(graph=how, n=X.shape[0], unit_label="bins" if n0 > X.shape[0] else "spots",
                                       bin=bin_label, normalization=d.get("normalization", "as provided"), genes=names,
                                       panel_label={"svg": "most spatially variable by Moran's I", "hvg": "most variable",
                                                    "custom": "user-selected"}[input.panel_method()],
                                       scale=int(input.scale()), seconds=time.time() - t0, p_method=p_method,
                                       modes_median=float(modes.median()) if len(modes) else None,
                                       n_sampled=n_sampled))
            except Exception as e:  # noqa: BLE001
                return fail(e)
        result.set(res)
        top = res["tab"][res["tab"]["J"] > 0]
        row = top.iloc[0] if len(top) else res["tab"].iloc[0]
        pair.set((row["feature_x"], row["feature_y"]))
        ui.update_navset("nav", selected="results")

    # ---- results ------------------------------------------------------------------------------
    @render.ui
    def result_boxes():
        r = result.get()
        if r is None:
            return ui.card(ui.p("Run the analysis (step 2) to see interfaces here.", class_="jx-lede"))
        tab = r["tab"]
        sig = tab[tab["call"] == "interface (opposing)"]
        return ui.layout_columns(
            ui.value_box("Spots analyzed", fmt_n(r["params"]["n"]), r["params"]["graph"], theme="light"),
            ui.value_box("Gene pairs tested", fmt_n(len(tab)), f"{len(r['names'])} genes, {r['params']['seconds']:.0f} s"
                         + (f"; median {r['params']['modes_median']:.0f} spectral modes"
                            if r["params"].get("modes_median") is not None else ""), theme="light"),
            ui.value_box("Interfaces (q < 0.05)", fmt_n(len(sig)), "pairs with opposing change across shared edges",
                         theme=ui.value_box_theme(bg="#FDE2DF", fg="#7A1F14")),
            ui.value_box("Strongest interface", sig["pair"].iloc[0] if len(sig) else "none at q < 0.05",
                         f"J = {sig['J'].iloc[0]:.2f}, q = {sig['q'].iloc[0]:.1e}" if len(sig) else "", theme="light"),
            col_widths=(3, 3, 3, 3))

    @reactive.calc
    def shown():
        r = result.get()
        req(r)
        tab = r["tab"]
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
        t = shown()
        out = pd.DataFrame({"Pair": t["pair"], "J": t["J"].round(3), "z": t["z"].round(2), "q": t["q"].map(lambda v: float(f"{v:.2g}"))})
        return render.DataGrid(out, selection_mode="row", height="380px", filters=False)

    @reactive.effect
    def _pick_row():
        try:
            rows = pair_table.cell_selection()["rows"]
        except Exception:  # noqa: BLE001
            return
        if rows:
            t = shown()
            i = rows[0]
            if i < len(t):
                pair.set((t["feature_x"].iloc[i], t["feature_y"].iloc[i]))

    @render.ui
    def volcano():
        r = result.get()
        req(r)
        return plot_ui(volcano_fig(r["tab"]), "fig_volcano", click_input="volcano_click")

    @reactive.effect
    @reactive.event(input.volcano_click)
    def _volcano_click():
        v = input.volcano_click()
        if v and "|" in v:
            pair.set(tuple(v.split("|", 1)))

    @reactive.calc
    def edges():
        r = result.get()
        p = pair.get()
        req(r, p)
        i, j = r["names"].index(p[0]), r["names"].index(p[1])
        return jx.score_edges(r["X"][:, [i, j]], r["graph"], scale=4, ridge=1.0, smoother_kind="lazy")

    @render.text
    def map_title():
        r, p = result.get(), pair.get()
        if r is None or p is None:
            return "Interface map"
        t = r["tab"]
        row = t[((t["feature_x"] == p[0]) & (t["feature_y"] == p[1])) | ((t["feature_x"] == p[1]) & (t["feature_y"] == p[0]))]
        return f"Interface map: {p[0]} and {p[1]}  (J = {row['J'].iloc[0]:.3f}, q = {row['q'].iloc[0]:.2g})"

    @render.ui
    def map_x():
        r, p = result.get(), pair.get()
        req(r, p)
        return plot_ui(tissue_fig(r["d"], r["X"][:, r["names"].index(p[0])], p[0], title=p[0], height=400), "fig_map_x")

    @render.ui
    def map_y():
        r, p = result.get(), pair.get()
        req(r, p)
        return plot_ui(tissue_fig(r["d"], r["X"][:, r["names"].index(p[1])], p[1], title=p[1], height=400), "fig_map_y")

    @render.ui
    def map_j():
        r = result.get()
        req(r)
        v = edges().to_nodes(r["X"].shape[0], field="signed_evidence")
        return plot_ui(tissue_fig(r["d"], v, "interface", kind="diverging", title="Interface (signed evidence)", height=400),
                       "fig_map_j")

    @reactive.calc
    def boundary_table():
        r = result.get()
        try:
            a = input.annotation() or ""
        except Exception:  # noqa: BLE001
            a = ""
        if r is None or not a or a not in r["d"]["meta"].columns:
            return None
        e = edges()
        lab = np.asarray(r["d"]["meta"][a].astype(str))
        A, B = lab[e.endpoints[:, 0]], lab[e.endpoints[:, 1]]
        cross = A != B
        if not cross.any():
            return None
        key = np.where(A < B, np.char.add(np.char.add(A, " | "), B), np.char.add(np.char.add(B, " | "), A))[cross]
        df = pd.DataFrame({"Boundary": key, "s": np.asarray(e.signed_evidence)[cross]})
        out = df.groupby("Boundary")["s"].agg(Edges="size", Evidence="mean").reset_index()
        out = out[out["Edges"] >= 5].sort_values("Evidence", ascending=False)
        out["Evidence"] = out["Evidence"].round(3)
        return out.rename(columns={"Evidence": "Mean signed evidence"})

    @render.ui
    def boundary_ui():
        if boundary_table() is None:
            return None
        return ui.TagList(ui.h6(f"Which annotated regions does this interface separate? ({input.annotation()})", class_="mt-3"),
                          ui.p("Edges between two different regions, ranked by the mean signed evidence of the selected pair.",
                               class_="jx-small"),
                          ui.output_data_frame("boundary_grid"))

    @render.data_frame
    def boundary_grid():
        t = boundary_table()
        req(t is not None)
        return render.DataGrid(t, height="260px")

    # ---- co-localization ------------------------------------------------------------------------
    @render.ui
    def regional_heatmap():
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        r = result.get()
        req(r)
        Rm = r["regional"]
        off = Rm.to_numpy() - np.diag(np.diag(Rm.to_numpy()))
        keep = np.argsort(-np.nanmax(np.abs(off), axis=1))[:60]
        sub = Rm.iloc[keep, keep]
        dist = np.clip(1 - sub.to_numpy(), 0, None)
        np.fill_diagonal(dist, 0)
        o = leaves_list(linkage(squareform(dist, checks=False), method="average"))
        return plot_ui(heatmap_fig(sub.iloc[o, o]), "fig_regional")

    @render.data_frame
    def regional_table():
        r = result.get()
        req(r)
        Rm = r["regional"]
        iu = np.triu_indices(len(Rm), k=1)
        out = pd.DataFrame({"Pair": [f"{Rm.index[i]} – {Rm.columns[j]}" for i, j in zip(*iu)],
                            "Regional J": np.round(Rm.to_numpy()[iu], 3)}).sort_values("Regional J", ascending=False)
        return render.DataGrid(out, height="560px")

    # ---- export -------------------------------------------------------------------------------
    @render.ui
    def methods():
        r = result.get()
        if r is None:
            return ui.p("Run the analysis first.", class_="jx-lede")
        return ui.div(methods_text(r), class_="jx-methods")

    @render.download(filename="boundary_J_all_pairs.csv")
    def dl_pairs():
        r = result.get()
        req(r)
        cols = [c for c in ("feature_x", "feature_y", "J", "z", "p", "q", "modes", "call") if c in r["tab"]]
        yield r["tab"][cols].to_csv(index=False)

    @render.download(filename="regional_J_matrix.csv")
    def dl_regional():
        r = result.get()
        req(r)
        yield r["regional"].to_csv()

    @render.download(filename=lambda: "interface_" + "_".join(pair.get() or ("pair",)) + ".csv")
    def dl_map():
        r, p = result.get(), pair.get()
        req(r, p)
        e = edges()
        out = pd.DataFrame({"spot": r["d"]["spots"], "x": r["d"]["coords"][:, 0], "y": r["d"]["coords"][:, 1],
                            p[0]: r["X"][:, r["names"].index(p[0])], p[1]: r["X"][:, r["names"].index(p[1])],
                            "signed_evidence": e.to_nodes(r["X"].shape[0], "signed_evidence"),
                            "local_J": e.to_nodes(r["X"].shape[0], "j")})
        yield out.to_csv(index=False)

    @render.download(filename="juxtaposed_methods.txt")
    def dl_methods():
        r = result.get()
        req(r)
        yield methods_text(r)


app = App(app_ui, server)
