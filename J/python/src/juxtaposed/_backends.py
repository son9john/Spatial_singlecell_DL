"""Runtime discovery for optional scoring accelerators."""

import ctypes
from functools import lru_cache
import platform
import warnings


class AcceleratorUnavailable(RuntimeError):
    """An optional backend cannot run on this machine or input."""


@lru_cache(maxsize=1)
def _metal_device_accessible():
    """MLX can abort at import when an Apple process has no Metal device."""
    try:
        metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
        get_device = metal.MTLCreateSystemDefaultDevice
        get_device.restype = ctypes.c_void_p
        get_device.argtypes = []
        device = get_device()
        if not device:
            return False
        release = ctypes.CDLL("/usr/lib/libobjc.A.dylib").objc_release
        release.argtypes = [ctypes.c_void_p]
        release.restype = None
        release(device)
        return True
    except (AttributeError, OSError):
        return False


def _require_apple_silicon():
    if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        raise AcceleratorUnavailable("The MLX backend requires Apple Silicon macOS")
    if not _metal_device_accessible():
        raise AcceleratorUnavailable("No Metal device is available to this process")


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
