"""juxtaposed: the juxtaposition index J for spatial omics.

J asks whether two features change across the *same* tissue edges, and in which direction:

* J > 0 — opposing changes: complementary domains that touch (juxtaposition);
* J < 0 — concordant changes: the features rise and fall together across boundaries;
* J ≈ 0 — the features do not share boundaries (even if they are anticorrelated overall).

Quick start::

    import juxtaposed as jx
    graph = jx.spatial_graph(coords, neighbors=6)          # or jx.visium_graph(row, col)
    res = jx.juxtaposition(expr, graph, features=genes)    # J + spectral-null p/q for all pairs
    res.top(20)                                            # most juxtaposed pairs
"""

from ._api import JResult, compare, from_anndata, juxtaposition, score_pairs
from ._core import (
    codispersion_matrix, gradient_coenergy_matrix, juxtaposition_matrix, juxtaposition_score,
    lee_l_matrix, moran_bv_matrix, pearson_matrix, power_mean, standardize,
)
from ._graph import (
    laplacian, lazy_walk, normalized_laplacian, smoother, spatial_graph, visium_graph,
)
from ._regional import regional_juxtaposition_matrix
from ._inference import SpectralNull, bh_qvalues, permutation_test
from ._local import EdgeResult, juxtaposed_value, score_edges
from ._fast import available_backends

__version__ = "0.5.0"

__all__ = [
    "JResult", "SpectralNull", "EdgeResult", "available_backends", "bh_qvalues", "codispersion_matrix", "compare",
    "from_anndata", "gradient_coenergy_matrix", "juxtaposed_value", "juxtaposition",
    "juxtaposition_matrix", "juxtaposition_score", "laplacian", "lazy_walk", "lee_l_matrix",
    "moran_bv_matrix", "normalized_laplacian", "pearson_matrix", "permutation_test",
    "power_mean", "score_edges", "score_pairs", "smoother", "spatial_graph", "standardize",
    "visium_graph", "regional_juxtaposition_matrix",
]
