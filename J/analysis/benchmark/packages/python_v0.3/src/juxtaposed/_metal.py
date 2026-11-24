"""Apple GPU kernels for dense all-pairs output on a sparse tissue graph."""

from functools import lru_cache

import numpy as np

from ._backends import AcceleratorUnavailable


@lru_cache(maxsize=1)
def _kernels():
    try:
        import mlx.core as mx
    except ImportError as error:
        raise ImportError("Install juxtaposed[mlx] on Apple silicon to use backend='mlx'") from error
    if not hasattr(getattr(mx, "fast", None), "metal_kernel"):
        raise AcceleratorUnavailable("The MLX installation lacks custom Metal kernel support")
    if not mx.metal.is_available():
        raise AcceleratorUnavailable("backend='mlx' requires an available Apple Metal GPU")

    graph = mx.fast.metal_kernel(
        name="juxtaposed_csr_graph",
        input_names=["signal", "indptr", "neighbor", "weight"],
        output_names=["out"],
        source="""
            uint index = thread_position_in_grid.x;
            uint features = signal_shape[1];
            uint row = index / features;
            uint gene = index % features;
            float center = signal[index];
            float total = 0.0f;
            for (int edge = indptr[row]; edge < indptr[row + 1]; ++edge) {
                float adjacent = signal[neighbor[edge] * features + gene];
                total = metal::fma(weight[edge], LAG ? adjacent : center - adjacent, total);
            }
            out[index] = DIFFUSE ? metal::fma(-0.5f, total, center) : total;
        """,
    )
    normalize = mx.fast.metal_kernel(
        name="juxtaposed_normalize_gram",
        input_names=["gram", "valid"],
        output_names=["out"],
        source="""
            const uint tile = 16;
            uint2 group = threadgroup_position_in_grid.xy;
            if (group.y > group.x)
                return;
            uint2 lane = thread_position_in_threadgroup.xy;
            uint genes = gram_shape[1];
            uint row = group.y * tile + lane.y;
            uint col = group.x * tile + lane.x;
            uint reverse_row = group.x * tile + lane.y;
            uint reverse_col = group.y * tile + lane.x;

            threadgroup float reverse_gram[16][17];
            threadgroup float reverse_score[16][17];
            reverse_gram[lane.y][lane.x] = reverse_row < genes && reverse_col < genes
                                        ? gram[reverse_row * genes + reverse_col] : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float value = NAN;
            if (row < genes && col < genes && valid[row] && valid[col]) {
                float energy = metal::max(0.0f, gram[row * genes + row])
                             + metal::max(0.0f, gram[col * genes + col]);
                float cross = gram[row * genes + col] + reverse_gram[lane.x][lane.y];
                if (energy > 0.0f)
                    value = metal::clamp(-cross / energy, -1.0f, 1.0f);
            }
            if (row < genes && col < genes)
                out[row * genes + col] = value;
            if (group.y < group.x) {
                reverse_score[lane.y][lane.x] = value;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (reverse_row < genes && reverse_col < genes)
                    out[reverse_row * genes + reverse_col] = reverse_score[lane.x][lane.y];
            }
        """,
    )
    return mx, graph, normalize


def _inputs(mx, z, operator, standardized):
    if max(z.size, z.shape[1] ** 2, operator.nnz) >= 2**31:
        raise AcceleratorUnavailable("MLX graph and result buffers must fit signed 32-bit indexing")
    if standardized:
        array = np.asarray(z, dtype=np.float32, order="C")
    else:
        # Subtract a float64 column offset before narrowing to preserve small
        # differences in expression values with a large common offset.
        array = np.empty(z.shape, dtype=np.float32)
        np.subtract(z, z[0], out=array, casting="unsafe")
    if not np.isfinite(array).all():
        raise AcceleratorUnavailable("MLX requires finite expression differences representable in float32")
    signal = mx.array(array)
    if standardized:
        valid = mx.ones(z.shape[1], dtype=mx.bool_)
    else:
        peak = mx.max(mx.abs(signal), axis=0)
        valid = peak > 0
        signal = signal / mx.where(valid, peak, 1.0)
        signal = signal - mx.mean(signal, axis=0)
        variance = mx.mean(mx.square(signal), axis=0)
        signal = signal * mx.rsqrt(mx.where(valid, variance, 1.0))
    pointers = mx.array(operator.indptr.astype(np.int32, copy=False))
    columns = mx.array(operator.indices.astype(np.int32, copy=False))
    return signal, pointers, columns, valid


def _apply(mx, kernel, signal, pointers, columns, weights, *, lag=False, diffuse=False):
    return kernel(
        inputs=[signal, pointers, columns, weights],
        template=[("LAG", lag), ("DIFFUSE", diffuse)],
        grid=(signal.size, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[signal.shape], output_dtypes=[mx.float32],
    )[0]


def score_mlx(z, a, scale, standardized):
    """Evaluate the same combinatorial J graph filter with float32 Metal operations."""
    if z.shape[1] == 0:
        return np.empty((0, 0), dtype=np.float32)
    mx, kernel, normalize = _kernels()
    max_degree = float(np.asarray(a.sum(axis=1)).max())
    if not np.isfinite(max_degree):
        raise AcceleratorUnavailable("MLX requires finite graph degrees")
    if max_degree == 0:
        return np.full((z.shape[1], z.shape[1]), np.nan, dtype=np.float32)

    with mx.stream(mx.gpu):
        signal, pointers, columns, valid = _inputs(mx, z, a, standardized)
        # Uniformly rescaling the Laplacian cancels from J and preserves its heat step.
        weights = mx.array(np.asarray(a.data / max_degree, dtype=np.float32))
        for _ in range(scale):
            signal = _apply(mx, kernel, signal, pointers, columns, weights, diffuse=True)
        lap_signal = _apply(mx, kernel, signal, pointers, columns, weights)
        gram = signal.T @ lap_signal
        padded = ((z.shape[1] + 15) // 16) * 16
        scores = normalize(
            inputs=[gram, valid], grid=(padded, padded, 1), threadgroup=(16, 16, 1),
            output_shapes=[gram.shape], output_dtypes=[mx.float32],
        )[0]
        mx.eval(scores)
    return np.array(scores, copy=True)


def lee_mlx(z, w, standardized):
    """Matched float32 Apple GPU baseline for global Lee's L."""
    if z.shape[1] == 0:
        return np.empty((0, 0), dtype=np.float32)
    mx, kernel, _ = _kernels()
    mass = float(np.sum(np.asarray(w.sum(axis=1)).ravel() ** 2))
    if mass == 0:
        return np.full((z.shape[1], z.shape[1]), np.nan, dtype=np.float32)
    with mx.stream(mx.gpu):
        signal, pointers, columns, valid = _inputs(mx, z, w, standardized)
        lag = _apply(mx, kernel, signal, pointers, columns,
                     mx.array(w.data.astype(np.float32, copy=False)), lag=True)
        scores = (lag.T @ lag) / mass
        scores = mx.where(valid[:, None] & valid[None, :], scores, mx.nan)
        mx.eval(scores)
    return np.array(scores, copy=True)
