"""End-to-end runs with the real DRUNet on one CBSD68 image; need ``data/`` and ``model_zoo/``."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from adaptivepnp.data import deblurring_problem, load_cbsd68, super_resolution_problem
from adaptivepnp.denoisers import load_drunet
from adaptivepnp.kernels import kernel_label, load_kernels
from adaptivepnp.metrics import psnr
from adaptivepnp.pnp import make_initial_guess, pnp_recon

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def denoiser():
    return load_drunet("cpu")


@pytest.fixture(scope="module")
def image():
    x, rotated = load_cbsd68(0, return_rotation=True)
    assert x.shape == (1, 3, 481, 321) and isinstance(rotated, bool)
    return x


def test_kernels_and_labels():
    kernels = load_kernels()
    assert len(kernels) == 12 and all(k.shape == (25, 25) for k in kernels)
    assert all(k.sum().item() == pytest.approx(1.0, abs=1e-4) for k in kernels)
    assert [kernel_label(i) for i in (0, 4, 8, 11)] == ["G1", "A1", "M1", "M4"]


@pytest.mark.parametrize("problem", ["deblurring", "super_resolution"])
def test_adaptive_pnp_improves_on_the_back_projection(denoiser, image, problem):
    kernel = load_kernels()[3]
    if problem == "deblurring":
        x, (y, A) = image, deblurring_problem(image, kernel, 0.01)
    else:
        x, y, A = super_resolution_problem(image, kernel, 2, 0.01)
    x0 = make_initial_guess(A, y, "adjoint").clone()
    _, recon, iterates, curves = pnp_recon(y, x0, A, denoiser, "pgm", K=2, record_iterates=True)
    assert np.isfinite(recon).all() and len(iterates) == 3
    assert psnr(torch.from_numpy(recon), x) > psnr(x0, x) + 1.0
    assert curves["sigma"][-1] == pytest.approx(0.01, rel=0.25)  # the noise level is recovered from the residual


def test_load_drunet_is_cached_per_device(denoiser):
    assert load_drunet("cpu") is denoiser
