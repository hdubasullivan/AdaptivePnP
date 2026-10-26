"""Shared fixtures. Tests marked ``data`` need ``data/`` and ``model_zoo/`` and are skipped when they are missing."""

from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from adaptivepnp.operators import Convolution, SuperResolution  # noqa: E402
from adaptivepnp.paths import DATA_DIR, MODEL_ZOO  # noqa: E402

HAS_DATA = (
    (DATA_DIR / "kernels_12.mat").exists()
    and (DATA_DIR / "CBSD68").exists()
    and (MODEL_ZOO / "drunet_color.pth").exists()
)


def pytest_collection_modifyitems(items):
    skip = pytest.mark.skip(reason="needs data/ (kernels_12.mat, CBSD68) and model_zoo/drunet_color.pth")
    for item in items:
        if "data" in item.keywords and not HAS_DATA:
            item.add_marker(skip)


def gaussian_kernel(size: int = 7, sigma: float = 1.2) -> torch.Tensor:
    """A normalized ``size x size`` Gaussian blur kernel, float32."""
    ax = torch.arange(size, dtype=torch.float64) - size // 2
    g = torch.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma**2))
    return (g / g.sum()).float()


def make_operators(shape=(1, 3, 32, 32), dtype=torch.float32):
    """A circular blur and a 2x blur-then-decimate operator on ``shape``, sharing one Gaussian kernel."""
    k = gaussian_kernel().to(dtype)
    conv = Convolution(k, input_shape=shape, input_dtype=dtype)
    sr = SuperResolution(k, 2, input_shape=shape, input_dtype=dtype)
    return conv, sr


class Shrink(torch.nn.Module):
    """A stand-in denoiser: pull toward the local mean by an amount that grows with sigma."""

    def forward(self, x, sigma):
        blurred = torch.nn.functional.avg_pool2d(x, 3, stride=1, padding=1, count_include_pad=False)
        a = min(float(sigma) * 10, 0.9)
        return (1 - a) * x + a * blurred


@pytest.fixture
def kernel():
    return gaussian_kernel()


@pytest.fixture
def operators():
    """``(Convolution, SuperResolution)`` in float32 on a 32 x 32 grid."""
    return make_operators()


@pytest.fixture
def operators64():
    """``(Convolution, SuperResolution)`` in float64, for checks to round-off."""
    return make_operators(dtype=torch.float64)


@pytest.fixture
def shrink():
    return Shrink()
