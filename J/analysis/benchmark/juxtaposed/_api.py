"""Stable package entry points over the shared sparse graph kernels."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from ._graph import diffuse, graph_gram, graph_operators, local_moments
from ._value import energy_scale


@dataclass(frozen=True)
class EdgeResult:
    """Local J, original pair evidence, and mean joint energy for each edge."""

    endpoints: np.ndarray
    j: np.ndarray
    signed_evidence: np.ndarray
    energy: np.ndarray


def _inputs(values, graph, ridge, standardize):
    if sparse.issparse(values):
        values = values.toarray()
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("values must be an observation-by-feature matrix with at least two of each")
    if not np.isfinite(values).all():
        raise ValueError("values must be finite")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    graph = sparse.csr_matrix(graph, dtype=float)
    if graph.shape != (len(values), len(values)):
        raise ValueError("graph dimensions must match the observations")
    if np.any(graph.data < 0) or not np.isfinite(graph.data).all():
        raise ValueError("graph weights must be finite and nonnegative")
    if (graph != graph.T).nnz or graph.diagonal().any():
        raise ValueError("graph must be symmetric with a zero diagonal")
    if standardize:
        scale = values.std(axis=0)
        values = np.divide(values - values.mean(axis=0), scale, where=scale > 0,
                           out=np.zeros_like(values))
    return values, graph


def score_edges(values, graph, *, steps=0, ridge=0.0, standardize=True, power=1.0):
    """Return local J for a pair, pooling edges incident to each endpoint.

    Each focal edge is included at both endpoints. Nonnegative graph weights
    enter both the local sums and the whole-graph energy reference. Use
    `result.endpoints` to map scores to the original coordinate rows.
    """
    x, adjacency = _inputs(values, graph, ridge, standardize)
    if x.shape[1] != 2:
        raise ValueError("score_edges requires exactly two features")
    i, j = sparse.triu(adjacency, k=1, format="csr").nonzero()
    _, walk, _ = graph_operators(adjacency)
    x = diffuse(x, walk, steps)
    weights = np.asarray(adjacency[i, j]).ravel() if len(i) else np.empty(0)
    cross, a, b, mass, reference = local_moments(x, i, j, weights)
    base = energy_scale(a, b, power)
    value = np.divide(0.5 * cross, base + 0.5 * ridge * mass * reference, where=base > 0,
                      out=np.full_like(cross, np.nan))
    return EdgeResult(np.column_stack([i, j]), np.clip(value, -1, 1),
                      cross / mass, (a + b) / mass)


def score_pairs(values, graph, *, steps=0, ridge=0.0, standardize=True, power=1.0):
    """Return the dense all-pairs global J matrix on a symmetric spatial graph."""
    x, adjacency = _inputs(values, graph, ridge, standardize)
    _, walk, degree = graph_operators(adjacency)
    gram = graph_gram(diffuse(x, walk, steps), adjacency, degree)
    energy = np.maximum(np.diag(gram), 0)
    base = energy_scale(energy[:, None], energy, power)
    reference = 0.5 * (energy[:, None] + energy)
    value = np.divide(-gram, base + ridge * reference, where=base > 0,
                      out=np.full_like(gram, np.nan))
    return np.clip(value, -1, 1)
