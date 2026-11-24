"""Regional concordance of centered signals across spatial edges and diffusion scales."""
from __future__ import annotations

from numbers import Integral
import numpy as np
from scipy import sparse

from ._backends import accelerated
from ._core import standardize
from ._graph import check_graph, smoother


def _scales(steps):
    values = tuple(steps)
    if not values or any(isinstance(t, bool) or not isinstance(t, Integral) or t < 0 for t in values):
        raise ValueError("scales must contain distinct nonnegative integers")
    if len(set(values)) != len(values):
        raise ValueError("repeated scales would give unequal weights")
    return sorted(values)


def _numpy_region(z, a, steps, valid):
    step = smoother(a)
    degree = np.asarray(a.sum(axis=1)).ravel()
    q = sparse.diags(degree) + a
    total = np.zeros((z.shape[1], z.shape[1]), dtype=np.float64)
    previous = 0
    for t in steps:
        for _ in range(t - previous):
            z = step @ z
        gram = z.T @ (q @ z)
        energy = np.maximum(gram.diagonal(), 0)
        den = energy[:, None] + energy[None, :]
        total += np.divide(gram + gram.T, den, out=np.full_like(gram, np.nan), where=den > 0)
        previous = t
    total[~valid, :] = np.nan
    total[:, ~valid] = np.nan
    return np.clip(total / len(steps), -1, 1)


def _mlx_region(z, a, steps, valid):
    from ._metal import _apply, _inputs, _kernels
    mx, kernel, _ = _kernels()
    degree = np.asarray(a.sum(axis=1)).ravel()
    peak = degree.max()
    with mx.stream(mx.gpu):
        signal, ptr, cols, _ = _inputs(mx, z, a, True)
        weights = mx.array((a.data / peak).astype(np.float32))
        d = mx.array((degree / peak).astype(np.float32))[:, None]
        mask = mx.array(valid)
        total = mx.zeros((signal.shape[1], signal.shape[1]))
        previous = 0
        for t in steps:
            for _ in range(t - previous):
                signal = _apply(mx, kernel, signal, ptr, cols, weights, diffuse=True)
            lag = _apply(mx, kernel, signal, ptr, cols, weights, lag=True)
            gram = signal.T @ (d * signal + lag)
            e = mx.maximum(mx.diag(gram), 0.)
            den = e[:, None] + e[None, :]
            total = total + mx.where(den > 0, (gram + gram.T) / den, mx.nan)
            mx.eval(total, signal)
            previous = t
        result = mx.where(mask[:, None] & mask[None, :], mx.clip(total / len(steps), -1., 1.), mx.nan)
        mx.eval(result)
    return np.array(result, copy=True)


def _torch_region(torch, device, z, a, steps, valid):
    # Keep sparse graph multiplication on the host for compatibility across
    # PyTorch devices. The dense all-pairs products and reductions run on device.
    from ._torch import _tensor
    step = smoother(a)
    q = sparse.diags(np.asarray(a.sum(axis=1)).ravel()) + a
    with torch.inference_mode():
        total = torch.zeros((z.shape[1], z.shape[1]), dtype=torch.float32, device=device)
        mask = torch.as_tensor(valid, device=device)
        previous = 0
        for t in steps:
            for _ in range(t - previous):
                z = step @ z
            gram = _tensor(torch, z, device).T @ _tensor(torch, q @ z, device)
            e = gram.diagonal().clamp_min(0)
            den = e[:, None] + e[None, :]
            total.add_(((gram + gram.T) / den).masked_fill(den <= 0, float('nan')))
            previous = t
        result = (total / len(steps)).clamp(-1, 1)
        return result.masked_fill(~(mask[:, None] & mask[None, :]), float('nan')).cpu().numpy()


def regional_juxtaposition_matrix(X, graph, *, scales=(1, 2, 4, 8, 16), backend="cpu"):
    """Score shared regional expression of all gene pairs on a spatial graph.

    Rows are observations, columns are genes. The output is an equally weighted
    average of edge-sum concordances at the requested graph heat steps, in [-1, 1].
    Positive scores include shared high and shared low expression. Compare with
    juxtaposition_matrix for a pair's opposing transitions. Zero-variance features
    and zero-denominator comparisons return NaN. No p-value is assigned.
    CPU uses float64. 'auto' selects an available GPU with CPU fallback.
    """
    steps = _scales(scales)
    z, valid = standardize(X.toarray() if sparse.issparse(X) else X, return_mask=True)
    a = check_graph(graph, z.shape[0])
    if backend not in ("cpu", "auto", "mlx", "torch"):
        raise ValueError("backend must be 'cpu', 'auto', 'mlx' or 'torch'")
    if not a.nnz or not z.shape[1]:
        return np.full((z.shape[1], z.shape[1]), np.nan)
    gpu = accelerated(backend, lambda: _mlx_region(z, a, steps, valid),
                      lambda torch, device: _torch_region(torch, device, z, a, steps, valid))
    return gpu if gpu is not None else _numpy_region(z, a, steps, valid)
