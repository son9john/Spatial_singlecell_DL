import numpy as np
import pytest
from scipy import sparse

import juxtaposed as jx


def fixture():
    xy = np.random.default_rng(22).normal(size=(33, 2))
    x = np.random.default_rng(10).normal(size=(33, 8))
    x[:, -1] = 1.23
    return x, jx.spatial_graph(xy)


def literal(x, g, scales):
    z = jx.standardize(x)
    tr = sparse.triu(g, 1).tocoo()
    f = jx.smoother(g)
    out = []
    for t in scales:
        a = (f ** t) @ z
        edges = (a[tr.row] + a[tr.col]) * np.sqrt(tr.data)[:, None]
        gram = edges.T @ edges
        den = np.add.outer(np.diag(gram), np.diag(gram))
        with np.errstate(invalid='ignore'):
            s = 2*gram/den
        s[-1, :] = np.nan
        s[:, -1] = np.nan
        out.append(s)
    return np.mean(out, axis=0)


def test_literal_edges_and_transforms():
    x, g = fixture()
    for scales in [(0,), (1, 2, 4, 8, 16)]:
        expected = literal(x, g, scales)
        got = jx.regional_juxtaposition_matrix(x, g, scales=scales)
        np.testing.assert_allclose(got, expected, atol=1e-12)
        np.testing.assert_allclose(jx.regional_juxtaposition_matrix(sparse.csr_matrix(x),2*g,scales=scales), got, atol=1e-12)
    for scales in ((), (-1,), (1, 1), (True,), (.5,)):
        with pytest.raises(ValueError):
            jx.regional_juxtaposition_matrix(x,g,scales=scales)
    assert np.isnan(jx.regional_juxtaposition_matrix(x,g*0)).all()


@pytest.mark.parametrize('backend',['auto','mlx','torch'])
def test_backend(backend):
    x, g = fixture()
    if backend == 'torch':
        torch=pytest.importorskip('torch')
        from juxtaposed._regional import _torch_region
        z, valid = jx.standardize(x,return_mask=True)
        actual = _torch_region(torch, torch.device('cpu'), z,g,(1,2,4,8,16),valid)
    else:
        if backend == 'mlx':
            if "mlx" not in jx.available_backends():
                pytest.skip("MLX or Metal unavailable")
            mx=pytest.importorskip('mlx.core')
            if not mx.metal.is_available():
                pytest.skip('Metal unavailable')
        actual = jx.regional_juxtaposition_matrix(x,g,backend=backend)
    np.testing.assert_allclose(actual,jx.regional_juxtaposition_matrix(x,g),atol=5e-6)
