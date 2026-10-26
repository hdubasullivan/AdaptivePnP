"""Device selection shared by research scripts."""

from __future__ import annotations

import torch


def select_device() -> str:
    """The best available torch device: CUDA, then MPS, then CPU.

    Returns:
        ``'cuda'``, ``'mps'`` or ``'cpu'``.

    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
