"""Regression coverage for tiled all-pairs normalization and Metal scoring."""

import numpy as np
import pytest
from scipy import sparse

import juxtaposed as jx
from juxtaposed._core import graph_gram
from juxtaposed._graph import smooth


def test_cpu_tiles_match_dense_formula():
    rng = np.random.default_rng(8)
    graph = jx.spatial_graph(rng.random((23, 2)))
    values = rng.normal(size=(23, 271))
    values[:, 0] = .1
    z, valid = jx.standardize(values, return_mask=True)
    gram = graph_gram(smooth(z, graph, 2), graph)
    e = np.maximum(gram.diagonal(), 0)
    denom = e[:, None] + e
    out = np.divide(-2 * gram, denom, where=denom > 0, out=np.full_like(gram, np.nan))
    out[~valid, :] = out[:, ~valid] = np.nan
    result = jx.juxtaposition_matrix(values, graph, scale=2)
    np.testing.assert_allclose(result, np.clip(out, -1, 1), atol=1e-12, rtol=0)
    np.testing.assert_array_equal(result, result.T)


@pytest.mark.parametrize("p", [1, 16, 17])
def test_metal_matches_cpu_with_offsets_and_padding(p):
    if "mlx" not in jx.available_backends():
        pytest.skip("MLX or Metal unavailable")
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Apple Metal GPU unavailable")
    rng = np.random.default_rng(42)
    graph = jx.spatial_graph(rng.random((53, 2)))
    values = rng.normal(size=(53, p)) + 1e9
    if p > 1:
        values[:, 0] = .1
    for scale in [0, 1, 4]:
        expected = jx.juxtaposition_matrix(values, graph, scale=scale)
        actual = jx.juxtaposition_matrix(values, graph, scale=scale, backend="mlx")
        assert actual.dtype == np.float32
        np.testing.assert_allclose(expected, actual, atol=3e-6, rtol=0)
        np.testing.assert_array_equal(actual, actual.T)
    np.testing.assert_allclose(jx.lee_l_matrix(values, graph),
                               jx.lee_l_matrix(values, graph, backend="mlx"), atol=3e-6, rtol=0)
    no_edges = sparse.csr_matrix(graph.shape)
    assert np.isnan(jx.juxtaposition_matrix(values, no_edges, backend="mlx")).all()
    result = jx.juxtaposition(values, graph, test=False, backend="mlx")
    assert result.J.dtype == np.float32
