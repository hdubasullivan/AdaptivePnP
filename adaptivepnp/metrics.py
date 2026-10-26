"""PSNR and SSIM metrics for PnP reconstructions, with border cropping."""

from __future__ import annotations

import torch
from skimage.metrics import structural_similarity

#: border pixels dropped from every edge before metrics
METRIC_BORDER_CROP = 10


def crop_border(image: torch.Tensor, border: int = METRIC_BORDER_CROP) -> torch.Tensor:
    """Drop ``border`` pixels from each spatial edge.

    Args:
        image: Tensor whose last two axes are spatial.
        border: Pixels to drop; ``0`` returns ``image`` unchanged.

    Returns:
        The cropped view.

    """
    if not border:
        return image
    return image[..., border:-border, border:-border]


def psnr(estimate: torch.Tensor, reference: torch.Tensor, border: int = METRIC_BORDER_CROP) -> float:
    """PSNR in dB of an estimate against the reference, data range 1, after the border crop.

    Args:
        estimate: ``(1, C, H, W)`` or ``(C, H, W)`` tensor.
        reference: Ground truth of the same shape.
        border: Pixels dropped from each spatial edge first.

    Returns:
        ``10 log10(1 / MSE)`` over all remaining elements.

    """
    a, b = crop_border(estimate, border).detach().cpu(), crop_border(reference, border).detach().cpu()
    mse = ((a - b) ** 2).mean()
    return float(10 * torch.log10(1.0 / mse))


def ssim(estimate: torch.Tensor, reference: torch.Tensor, border: int = METRIC_BORDER_CROP) -> float:
    """Mean SSIM (scikit-image, channel-last) of an estimate against the reference, data range 1, after the border crop.

    Args:
        estimate: ``(1, C, H, W)`` or ``(C, H, W)`` tensor.
        reference: Ground truth of the same shape.
        border: Pixels dropped from each spatial edge first.

    Returns:
        The mean structural similarity.

    """

    def _hwc(t: torch.Tensor):
        t = crop_border(t, border).detach().cpu()
        return t.reshape(-1, *t.shape[-2:])[-3:].permute(1, 2, 0).numpy()

    return float(structural_similarity(_hwc(reference), _hwc(estimate), data_range=1.0, channel_axis=-1))


__all__ = ["METRIC_BORDER_CROP", "crop_border", "psnr", "ssim"]
