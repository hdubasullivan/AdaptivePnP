"""CBSD68 test images and the measurement synthesis used in the paper."""

from __future__ import annotations

from pathlib import Path

import torch
from datasets import load_from_disk
from torchvision.transforms.functional import pil_to_tensor

from adaptivepnp.operators import Convolution, SuperResolution
from adaptivepnp.paths import DATA_DIR


def load_cbsd68(index: int, data_dir: Path | None = None, return_rotation: bool = False):
    """CBSD68 image ``index`` as a ``(1, 3, H, W)`` float tensor in ``[0, 1]``.

    Landscape images are rotated to portrait, so every image is ``481 x 321``.

    Args:
        index: Image index, 0 to 67.
        data_dir: Directory holding ``CBSD68/``; defaults to ``data/`` or ``$ADAPTIVEPNP_DATA_DIR``.
        return_rotation: Also return whether the image was rotated, so a caller can undo it for display.

    Returns:
        The image, or ``(image, was_rotated)`` when ``return_rotation`` is set.

    Raises:
        FileNotFoundError: If the dataset is not there (see ``scripts/download_data.py``).

    """
    folder = (Path(data_dir) if data_dir else DATA_DIR) / "CBSD68"
    if not folder.exists():
        raise FileNotFoundError(f"CBSD68 not found at {folder}; run scripts/download_data.py.")
    tensor = pil_to_tensor(load_from_disk(str(folder))[int(index)]["png"].convert("RGB")).float().div_(255.0)
    was_rotated = tensor.shape[-1] > tensor.shape[-2]
    if was_rotated:
        tensor = torch.rot90(tensor, 1, dims=(-2, -1))
    image = tensor.unsqueeze(0)
    return (image, was_rotated) if return_rotation else image


def measure(x: torch.Tensor, operator, noise_level: float, seed: int = 0) -> torch.Tensor:
    """``y = A x + noise_level * w`` with ``w`` white Gaussian noise from a CPU generator seeded by ``seed``.

    Args:
        x: Ground-truth image(s).
        operator: Forward operator.
        noise_level: Noise standard deviation on the ``[0, 1]`` scale.
        seed: Seed of the noise generator.

    Returns:
        The measurement, on the CPU.

    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        clean = operator(x.to(operator.kernel.device if hasattr(operator, "kernel") else x.device)).cpu()
    return clean + noise_level * torch.randn(clean.shape, generator=generator)


def deblurring_problem(x: torch.Tensor, kernel: torch.Tensor, noise_level: float, seed: int = 0):
    """``(y, A)`` for circular blur by ``kernel`` plus white noise.

    Args:
        x: Ground-truth image ``(1, 3, H, W)``.
        kernel: Two-dimensional blur kernel.
        noise_level: Noise standard deviation.
        seed: Seed of the noise generator.

    Returns:
        The measurement and the :class:`adaptivepnp.operators.Convolution` that produced it.

    """
    operator = Convolution(kernel, input_shape=tuple(x.shape), input_dtype=x.dtype)
    return measure(x, operator, noise_level, seed), operator


def super_resolution_problem(x: torch.Tensor, kernel: torch.Tensor, factor: int, noise_level: float, seed: int = 0):
    """``(x, y, A)`` for blur, decimation by ``factor`` and white noise.

    Args:
        x: Ground-truth image ``(1, 3, H, W)``; cropped to a multiple of ``factor``.
        kernel: Two-dimensional blur kernel.
        factor: Integer decimation factor.
        noise_level: Noise standard deviation.
        seed: Seed of the noise generator.

    Returns:
        The cropped ground truth, the low-resolution measurement, and the
        :class:`adaptivepnp.operators.SuperResolution` that produced it.

    """
    h, w = x.shape[-2:]
    x = x[..., : h - h % factor, : w - w % factor]
    operator = SuperResolution(kernel, factor, input_shape=tuple(x.shape), input_dtype=x.dtype)
    return x, measure(x, operator, noise_level, seed), operator


__all__ = ["deblurring_problem", "load_cbsd68", "measure", "super_resolution_problem"]
