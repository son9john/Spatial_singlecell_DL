"""Dense float32 GPU scoring through a user-installed PyTorch build."""

import numpy as np

from ._backends import AcceleratorUnavailable


def gpu():
    """Select an available CUDA/ROCm or Intel XPU device."""
    try:
        import torch
    except ImportError as error:
        raise AcceleratorUnavailable("Install a GPU-enabled PyTorch build to use backend='torch'") from error

    # PyTorch also exposes ROCm devices through torch.cuda.
    for name in ("cuda", "xpu"):
        backend = getattr(torch, name, None)
        if backend is not None and backend.is_available():
            return torch, torch.device(name)
    raise AcceleratorUnavailable("backend='torch' requires an available CUDA, ROCm or XPU device")


def _tensor(torch, values, device):
    with np.errstate(over="ignore"):
        values = np.asarray(values, dtype=np.float32, order="C")
    if not np.isfinite(values).all():
        raise AcceleratorUnavailable("GPU scoring requires finite inputs representable in float32")
    return torch.as_tensor(values, device=device)


def juxtaposition(torch, device, values, lap_values, valid):
    """Score a preprocessed graph signal and its Laplacian product."""
    with torch.inference_mode():
        x = _tensor(torch, values, device)
        lx = _tensor(torch, lap_values, device)
        gram = x.T @ lx
        energy = gram.diagonal().clamp_min(0)
        denom = energy[:, None] + energy[None, :]
        keep = torch.as_tensor(valid, device=device)
        scores = (-(gram + gram.T) / denom).clamp(-1, 1)
        scores.masked_fill_(~(keep[:, None] & keep[None, :] & (denom > 0)), float("nan"))
        return scores.cpu().numpy()


def lee(torch, device, lag, valid, mass):
    """Score Lee's L from the spatially lagged signal."""
    with torch.inference_mode():
        x = _tensor(torch, lag, device)
        keep = torch.as_tensor(valid, device=device)
        scores = x.T @ x / mass
        scores.masked_fill_(~(keep[:, None] & keep[None, :]), float("nan"))
        return scores.cpu().numpy()
