"""Signed bivariate analysis of spatial gradients."""

from ._api import EdgeResult, score_edges, score_pairs
from ._graph import spatial_graph as _spatial_graph
from ._seeds import (
    edge_weight, filter_seeds_by_cluster, first_elbow, node_j,
    panel_seed_map, seed_edge_scores, seed_global, seed_indices, seed_weights,
)
from ._seed_extensions import (
    MapFixedSpectralNull, bh_qvalues, build_heat_smoother, build_map_fixed_weights,
    build_weighted_laplacian, diffuse_heat, filter_clustered_seeds_2hop,
    first_elbow_curvature, regularized_node_j, seed_soft_j,
)
from ._value import juxtaposed_value

__version__ = "0.1.0rc1"
__all__ = [
    "EdgeResult", "juxtaposed_value", "score_edges", "score_pairs", "spatial_graph",
    "first_elbow", "seed_indices", "node_j", "seed_weights", "edge_weight",
    "seed_global", "seed_edge_scores", "filter_seeds_by_cluster", "panel_seed_map",
    "MapFixedSpectralNull", "bh_qvalues", "build_heat_smoother", "build_map_fixed_weights",
    "build_weighted_laplacian", "diffuse_heat", "filter_clustered_seeds_2hop",
    "first_elbow_curvature", "regularized_node_j", "seed_soft_j",
]


def spatial_graph(coordinates, neighbors=12, radius=None):
    """Build a symmetric binary nearest-neighbor graph as a SciPy CSR matrix.

    Coordinates are observations by spatial axes. An optional radius is in
    the same units as the coordinates, and defaults to no distance cutoff.
    """
    return _spatial_graph(coordinates, neighbors, radius)[0]
