"""The twelve-kernel blur benchmark distributed with KAIR (``kernels_12.mat``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

from adaptivepnp.paths import DATA_DIR

KERNELS_MAT_PATH = DATA_DIR / "kernels_12.mat"

KERNEL_TYPE_BY_INDEX: dict[int, str] = {
    **{i: "isotropic_gaussian" for i in range(4)},
    **{i: "anisotropic_gaussian" for i in range(4, 8)},
    **{i: "motion" for i in range(8, 12)},
}

_KERNEL_TYPE_PREFIX = {"isotropic_gaussian": "G", "anisotropic_gaussian": "A", "motion": "M"}


def kernel_type(kernel_idx: int) -> str:
    """The kernel family of a benchmark index.

    Args:
        kernel_idx: Kernel index, 0 to 11.

    Returns:
        ``'isotropic_gaussian'``, ``'anisotropic_gaussian'`` or ``'motion'``.

    """
    return KERNEL_TYPE_BY_INDEX[int(kernel_idx)]


def kernel_label(kernel_idx: int) -> str:
    """The paper-facing label of a benchmark index, such as ``'G4'``.

    Args:
        kernel_idx: Kernel index, 0 to 11.

    Returns:
        ``'G1'``-``'G4'``, ``'A1'``-``'A4'`` or ``'M1'``-``'M4'``.

    """
    kernel_idx = int(kernel_idx)
    return f"{_KERNEL_TYPE_PREFIX[kernel_type(kernel_idx)]}{kernel_idx % 4 + 1}"


def load_kernels(path: Path = KERNELS_MAT_PATH) -> list[torch.Tensor]:
    """Load the twelve 25 x 25 benchmark kernels: four isotropic Gaussian, four anisotropic Gaussian, four motion.

    Args:
        path: The ``kernels_12.mat`` file; defaults to ``data/kernels_12.mat``.

    Returns:
        Twelve float32 ``(25, 25)`` tensors, in benchmark order.

    Raises:
        FileNotFoundError: If the file is not there (see ``scripts/download_data.py``).

    """
    if not path.exists():
        raise FileNotFoundError(
            f"Kernel benchmark not found at {path}. Download it from "
            "https://raw.githubusercontent.com/cszn/KAIR/master/kernels/kernels_12.mat"
        )
    mat = sio.loadmat(str(path))
    return [torch.from_numpy(np.asarray(kernel, dtype=np.float32)) for kernel in mat["kernels"][0]]


__all__ = ["KERNEL_TYPE_BY_INDEX", "KERNELS_MAT_PATH", "kernel_label", "kernel_type", "load_kernels"]
