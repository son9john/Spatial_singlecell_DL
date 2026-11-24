"""Portable fallback and optional accelerator dispatch."""

import builtins

import numpy as np
import pytest

import juxtaposed as jx
from juxtaposed import _backends, _torch


def sample():
    rng = np.random.default_rng(907)
    values = rng.normal(size=(23, 9))
    values[:, 0] = .1
    return values, jx.spatial_graph(rng.random((len(values), 2)))


@pytest.mark.parametrize(("system", "machine"), [
    ("Linux", "x86_64"), ("Windows", "AMD64"), ("Linux", "aarch64"), ("Darwin", "x86_64"),
])
def test_platform_cpu_fallback_without_optional_imports(monkeypatch, system, machine):
    monkeypatch.setattr(_backends.platform, "system", lambda: system)
    monkeypatch.setattr(_backends.platform, "machine", lambda: machine)
    original_import = builtins.__import__

    def import_without_accelerators(name, *args, **kwargs):
        if name == "_metal":
            pytest.fail("MLX module was imported on an unsupported platform")
        if name == "torch":
            raise ImportError("no torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_accelerators)
    values, graph = sample()
    expected = jx.juxtaposition_matrix(values, graph)
    np.testing.assert_array_equal(jx.juxtaposition_matrix(values, graph, backend="auto"), expected)
    np.testing.assert_array_equal(jx.lee_l_matrix(values, graph, backend="auto"),
                                  jx.lee_l_matrix(values, graph))
    with pytest.raises(RuntimeError, match="requires Apple Silicon"):
        jx.juxtaposition_matrix(values, graph, backend="mlx")
    with pytest.raises(RuntimeError, match="GPU-enabled PyTorch"):
        jx.juxtaposition_matrix(values, graph, backend="torch")


def test_auto_falls_back_on_accelerator_failure(monkeypatch):
    monkeypatch.setattr(_backends.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(_backends.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(_backends, "_metal_device_accessible", lambda: True)
    from juxtaposed import _metal

    def fail(*args, **kwargs):
        raise RuntimeError("simulated GPU failure")

    monkeypatch.setattr(_metal, "score_mlx", fail)

    def no_second_gpu():
        raise _backends.AcceleratorUnavailable("no second GPU")

    monkeypatch.setattr(_torch, "gpu", no_second_gpu)
    values, graph = sample()
    with pytest.warns(RuntimeWarning, match="simulated GPU failure"):
        actual = jx.juxtaposition(values, graph, backend="auto", test=False).J
    np.testing.assert_array_equal(actual, jx.juxtaposition_matrix(values, graph))
    with pytest.raises(RuntimeError, match="simulated GPU failure"):
        jx.juxtaposition_matrix(values, graph, backend="mlx")


def test_torch_mathematics_and_routing_on_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(_backends.platform, "system", lambda: "Linux")
    monkeypatch.setattr(_torch, "gpu", lambda: (torch, torch.device("cpu")))
    values, graph = sample()
    for backend in ("torch", "auto"):
        result = jx.juxtaposition_matrix(values, graph, scale=2, backend=backend)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, jx.juxtaposition_matrix(values, graph, scale=2),
                                   atol=3e-6, rtol=0)
        np.testing.assert_allclose(jx.lee_l_matrix(values, graph, backend=backend),
                                   jx.lee_l_matrix(values, graph), atol=3e-6, rtol=0)


def test_auto_keeps_spectral_inference_on_cpu():
    values, graph = sample()
    result = jx.juxtaposition(values, graph, backend="auto")
    assert result.J.dtype == np.float64
    assert result.q is not None


def test_auto_keeps_other_laplacians_and_powers_on_cpu():
    values, graph = sample()
    for kwargs in ({"power": 0}, {"laplacian": "normalized"}, {"dtype": np.float64}):
        expected = jx.juxtaposition_matrix(values, graph, **kwargs)
        actual = jx.juxtaposition_matrix(values, graph, backend="auto", **kwargs)
        np.testing.assert_array_equal(actual, expected)


def test_apple_process_without_metal_uses_cpu(monkeypatch):
    monkeypatch.setattr(_backends.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(_backends.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(_backends, "_metal_device_accessible", lambda: False)
    def no_gpu():
        raise _backends.AcceleratorUnavailable("no GPU")
    monkeypatch.setattr(_torch, "gpu", no_gpu)
    x, g = sample()
    np.testing.assert_allclose(jx.regional_juxtaposition_matrix(x, g, backend="auto"),
                               jx.regional_juxtaposition_matrix(x, g))
