"""Scalar J and weighted increment summaries."""

import numpy as np


def energy_scale(a, b, power):
    """Power mean of two nonnegative energies, computed without large powers."""
    if not np.isfinite(power) or power < 0:
        raise ValueError("power must be finite and nonnegative")
    if power == 1:
        return a * 0.5 + b * 0.5
    if power == 0:
        return np.sqrt(a) * np.sqrt(b)
    hi = np.maximum(a, b)
    lo = np.minimum(a, b)
    relative = np.divide(lo, hi, where=hi > 0, out=np.zeros_like(hi))
    with np.errstate(divide="ignore"):
        # expm1/log1p also stabilize the limit as the order approaches zero.
        factor = np.log1p(0.5 * np.expm1(power * np.log(relative))) / power
    return hi * np.exp(factor)


def juxtaposed_value(dx, dy, weights=None, axis=None, power=1.0):
    """Return signed J for already standardized feature increments.

    The default power of one uses the arithmetic mean of the two energies.
    Power zero uses the geometric mean and equals negative codispersion.
    NaN marks a zero denominator.
    """
    dx, dy = np.asarray(dx, dtype=float), np.asarray(dy, dtype=float)
    if dx.shape != dy.shape or not (np.isfinite(dx).all() and np.isfinite(dy).all()):
        raise ValueError("expected matching, finite increment arrays")
    if weights is None:
        weights = 1
    else:
        weights = np.broadcast_to(np.asarray(weights, dtype=float), dx.shape)
        if np.any(weights < 0) or not np.isfinite(weights).all():
            raise ValueError("weights must be nonnegative and finite")
    a = np.sum(weights * dx**2, axis=axis)
    b = np.sum(weights * dy**2, axis=axis)
    cross = np.sum(weights * dx * dy, axis=axis)
    scale = energy_scale(a, b, power)
    value = np.divide(-cross, scale, where=scale > 0, out=np.full_like(scale, np.nan))
    return np.clip(value, -1, 1)
