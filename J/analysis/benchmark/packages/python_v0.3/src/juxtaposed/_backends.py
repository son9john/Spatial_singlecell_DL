"""Runtime discovery for optional scoring accelerators."""

import platform
import warnings


class AcceleratorUnavailable(RuntimeError):
    """An optional backend cannot run on this machine or input."""


def _require_apple_silicon():
    if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        raise AcceleratorUnavailable("The MLX backend requires Apple Silicon macOS")


def accelerated(backend, run_mlx, run_torch):
    """Return an accelerator result or None for the portable CPU path."""
    if backend == "cpu":
        return None
    if backend not in ("auto", "mlx", "torch"):
        raise ValueError("backend must be 'cpu', 'auto', 'mlx' or 'torch'")

    candidates = ("mlx", "torch") if backend == "auto" else (backend,)
    for candidate in candidates:
        try:
            if candidate == "mlx":
                _require_apple_silicon()
                return run_mlx()
            from ._torch import gpu
            return run_torch(*gpu())
        except (AcceleratorUnavailable, ImportError, OSError):
            if backend != "auto":
                raise
        except RuntimeError as error:
            if backend != "auto":
                raise
            warnings.warn(f"{candidate} scoring failed: {error}. Trying a fallback.",
                          RuntimeWarning, stacklevel=3)
    return None
