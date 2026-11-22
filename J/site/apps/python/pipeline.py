"""Computation for the juxtaposed Shiny app (no Shiny imports, so it can be tested on its own)."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

import juxtaposed as jx

# Default R (ggplot2) palette
RED, GREEN, BLUE = "#F8766D", "#00BA38", "#619CFF"
DIVERGING = ["#3A3A98", "#619CFF", "#F7F7F7", "#F8766D", "#832424"]
EXPRESSION = ["#F2F2F2", "#56B1F7", "#132B43"]


def demo_tissue(side: int = 30, seed: int = 0) -> pd.DataFrame:
    """Synthetic hexagonal tissue with known structure.

    Columns: x, y, then genes.
      * GeneA / GeneB  - complementary domains that touch along a vertical interface (boundary J > 0)
      * GeneC          - co-localized with GeneA (shared region; regional J > 0, boundary J < 0)
      * GeneD / GeneE  - two separated islands (anticorrelated, but share no boundary; J near 0)
      * GeneF          - spatially random noise
    """
    rng = np.random.default_rng(seed)
    rows, cols = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    x = cols + 0.5 * (rows % 2)
    y = rows * np.sqrt(3) / 2
    x, y = x.ravel(), y.ravel()
    mid = x.max() / 2
    soft = 1 / (1 + np.exp(-(x - mid) / 0.6))
    a = 3 * (1 - soft)
    b = 3 * soft
    c = a.copy()
    d = 3 * (np.hypot(x - 0.2 * x.max(), y - 0.25 * y.max()) < 0.15 * x.max())
    e = 3 * (np.hypot(x - 0.8 * x.max(), y - 0.8 * y.max()) < 0.15 * x.max())
    noise = lambda: rng.normal(0, 0.3, x.size)  # noqa: E731
    return pd.DataFrame({
        "x": x, "y": y,
        "GeneA": a + noise(), "GeneB": b + noise(), "GeneC": c + noise(),
        "GeneD": d + noise(), "GeneE": e + noise(), "GeneF": rng.normal(1.5, 1.0, x.size),
    })


def read_table(raw: bytes, name: str) -> pd.DataFrame:
    """Read an uploaded CSV/TSV with columns x, y, then one column per gene."""
    sep = "\t" if name.lower().endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(io.BytesIO(raw), sep=sep)
    lower = {c.lower(): c for c in df.columns}
    for key in ("x", "y"):
        if key not in lower:
            raise ValueError("The file needs columns named x and y (spot coordinates), then one column per gene.")
    df = df.rename(columns={lower["x"]: "x", lower["y"]: "y"})
    genes = [c for c in df.columns if c not in ("x", "y") and pd.api.types.is_numeric_dtype(df[c])]
    if len(genes) < 2:
        raise ValueError("At least two numeric gene columns are needed.")
    return df[["x", "y", *genes]].dropna()


def genes_of(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("x", "y")]


def build_graph(df: pd.DataFrame, neighbors: int):
    return jx.spatial_graph(df[["x", "y"]].to_numpy(float), neighbors=int(neighbors))


def pairwise(df: pd.DataFrame, graph, scale: int, max_genes: int = 200) -> pd.DataFrame:
    """Boundary J for all pairs with the spectral sign-flip test (z, p, BH q)."""
    genes = genes_of(df)[:max_genes]
    res = jx.juxtaposition(df[genes].to_numpy(float), graph, features=genes, scale=int(scale), test=True)
    tab = res.table(sort="J")
    tab["direction"] = np.where(tab["J"] > 0, "opposing (interface)", "concordant")
    return tab


def regional(df: pd.DataFrame, graph, max_genes: int = 200) -> pd.DataFrame:
    """Regional J (shared neighbourhood expression), gene x gene."""
    genes = genes_of(df)[:max_genes]
    mat = jx.regional_juxtaposition_matrix(df[genes].to_numpy(float), graph)
    return pd.DataFrame(mat, index=genes, columns=genes)


def edge_map(df: pd.DataFrame, graph, gx: str, gy: str, field: str = "signed_evidence") -> np.ndarray:
    """Local J on every edge, averaged onto spots. Settings of the preprint's known-truth benchmark."""
    edges = jx.score_edges(df[[gx, gy]].to_numpy(float), graph, scale=4, ridge=1.0, smoother_kind="lazy")
    return edges.to_nodes(len(df), field=field)
