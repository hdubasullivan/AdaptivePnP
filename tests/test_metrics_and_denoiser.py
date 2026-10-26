"""PSNR / SSIM with the border crop, and the DRUNet module at random initialization (no weights)."""

from __future__ import annotations

import pytest
import torch

from adaptivepnp.denoisers import DRUNet
from adaptivepnp.metrics import METRIC_BORDER_CROP, crop_border, psnr, ssim


def test_crop_border():
    x = torch.zeros(1, 3, 40, 50)
    assert crop_border(x).shape == (1, 3, 40 - 2 * METRIC_BORDER_CROP, 50 - 2 * METRIC_BORDER_CROP)
    assert crop_border(x, 0) is x


def test_psnr_and_ssim_known_values():
    reference = torch.rand(1, 3, 40, 50, generator=torch.Generator().manual_seed(0))
    assert psnr(reference + 0.1, reference) == pytest.approx(20.0, abs=1e-4)  # MSE 0.01 at data range 1
    assert psnr(reference, reference) == float("inf")
    assert ssim(reference, reference) == pytest.approx(1.0)
    noisy = reference + 0.1 * torch.randn(reference.shape, generator=torch.Generator().manual_seed(1))
    assert 0.0 < ssim(noisy, reference) < 1.0
    assert psnr(noisy[0], reference[0]) == pytest.approx(psnr(noisy, reference))  # (C, H, W) accepted too


def test_drunet_random_init_handles_odd_sizes_and_ranks():
    net = DRUNet(in_channels=3, out_channels=3, nc=(8, 8, 8, 8), nb=1)
    with torch.no_grad():
        assert net(torch.rand(1, 3, 37, 23), sigma=0.05).shape == (1, 3, 37, 23)
        assert net(torch.rand(3, 37, 23), sigma=0.05).shape == (3, 37, 23)
        assert net(torch.rand(37, 23).expand(3, -1, -1), sigma=torch.tensor(0.05)).shape == (3, 37, 23)
    with pytest.raises(FileNotFoundError, match="download_weights"):
        DRUNet(pretrained="/nonexistent/drunet_color.pth")
    with pytest.raises(ValueError, match="2D, 3D, or 4D"):
        net(torch.rand(1, 1, 3, 8, 8), sigma=0.05)
