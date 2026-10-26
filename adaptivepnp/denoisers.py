"""The DRUNet denoiser (Zhang et al. 2022) and its weight loading.

Architecture ported from the reference implementation in KAIR (https://github.com/cszn/KAIR, MIT; its notice is in ``THIRD_PARTY_LICENSES``). The pretrained
color weights ``drunet_color.pth`` are KAIR's release; the state-dict keys (``m_head``, ``m_down1`` ...,
``m_body``, ``m_up3`` ..., ``m_tail``) are preserved so that checkpoint loads directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptivepnp.paths import MODEL_ZOO

DRUNET_WEIGHTS_URL = "https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth"


class ResBlock(nn.Module):
    """Two 3x3 convolutions with a ReLU between them, added back to the input.

    Args:
        nc: Number of channels.

    """

    def __init__(self, nc: int):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(nc, nc, 3, 1, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(nc, nc, 3, 1, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block.

        Args:
            x: ``(B, nc, H, W)`` tensor.

        Returns:
            ``x + res(x)``.

        """
        return x + self.res(x)


def _down_block(nc_in: int, nc_out: int, nb: int) -> nn.Sequential:
    """``nb`` residual blocks at ``nc_in`` channels, then a strided-conv downsample to ``nc_out``.

    Args:
        nc_in: Channels in.
        nc_out: Channels out.
        nb: Number of residual blocks.

    Returns:
        The block as an ``nn.Sequential``.

    """
    layers: list[nn.Module] = [ResBlock(nc_in) for _ in range(nb)]
    layers.append(nn.Conv2d(nc_in, nc_out, 2, 2, 0, bias=False))
    return nn.Sequential(*layers)


def _up_block(nc_in: int, nc_out: int, nb: int) -> nn.Sequential:
    """A transposed-conv upsample from ``nc_in`` to ``nc_out``, then ``nb`` residual blocks.

    Args:
        nc_in: Channels in.
        nc_out: Channels out.
        nb: Number of residual blocks.

    Returns:
        The block as an ``nn.Sequential``.

    """
    layers: list[nn.Module] = [nn.ConvTranspose2d(nc_in, nc_out, 2, 2, 0, bias=False)]
    layers += [ResBlock(nc_out) for _ in range(nb)]
    return nn.Sequential(*layers)


def _normalize_rank(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Reshape ``(H,W)`` / ``(C,H,W)`` / ``(B,C,H,W)`` to ``(B,C,H,W)``.

    Args:
        x: Image tensor of rank 2, 3 or 4.

    Returns:
        The reshaped tensor and the original rank, for :func:`_restore_rank`.

    Raises:
        ValueError: For any other rank.

    """
    if x.ndim == 2:
        return x.unsqueeze(0).unsqueeze(0), 2
    if x.ndim == 3:
        return x.unsqueeze(0), 3
    if x.ndim == 4:
        return x, 4
    raise ValueError(f"Expected a 2D, 3D, or 4D image tensor, got shape {tuple(x.shape)}.")


def _restore_rank(x: torch.Tensor, ndim: int) -> torch.Tensor:
    """Undo :func:`_normalize_rank`.

    Args:
        x: ``(B,C,H,W)`` tensor.
        ndim: The original rank.

    Returns:
        ``x`` at the original rank.

    """
    if ndim == 2:
        return x.squeeze(0).squeeze(0)
    if ndim == 3:
        return x.squeeze(0)
    return x


class DRUNet(nn.Module):
    """DRUNet: a residual U-Net with a noise-level input channel, ``model(x, sigma)``.

    Four scales ``nc=(64,128,256,512)`` with ``nb=4`` residual blocks each, strided-conv downsampling,
    transposed-conv upsampling, additive skips, no biases. Inputs are padded to a multiple of 16.

    Args:
        in_channels: Image channels (3 for color).
        out_channels: Output channels.
        nc: Feature channels at each of the four scales.
        nb: Residual blocks per scale.
        pretrained: Path to a state dict (``model_zoo/drunet_color.pth``), an in-memory state dict, or
            ``None`` for random initialization.

    Raises:
        FileNotFoundError: If ``pretrained`` is a path that does not exist.
        ValueError: If the file does not hold a state dict.

    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        out_channels: int = 3,
        nc: tuple[int, int, int, int] = (64, 128, 256, 512),
        nb: int = 4,
        pretrained: str | Path | Mapping[str, torch.Tensor] | None = None,
    ):
        super().__init__()
        self._num_downs = len(nc) - 1
        self.m_head = nn.Conv2d(in_channels + 1, nc[0], 3, 1, 1, bias=False)
        for i in range(1, self._num_downs + 1):
            setattr(self, f"m_down{i}", _down_block(nc[i - 1], nc[i], nb))
        self.m_body = nn.Sequential(*[ResBlock(nc[-1]) for _ in range(nb)])
        for i in range(self._num_downs, 0, -1):
            setattr(self, f"m_up{i}", _up_block(nc[i], nc[i - 1], nb))
        self.m_tail = nn.Conv2d(nc[0], out_channels, 3, 1, 1, bias=False)

        if isinstance(pretrained, Mapping):
            self.load_state_dict(dict(pretrained), strict=True)
        elif pretrained is not None:
            path = Path(pretrained).expanduser()
            if not path.exists():
                raise FileNotFoundError(
                    f"DRUNet weights not found at {path}. Download them with scripts/download_weights.py "
                    f"(from {DRUNET_WEIGHTS_URL})."
                )
            state = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(state, Mapping):
                raise ValueError(f"{path} does not hold a state dict.")
            for key in ("model_state_dict", "state_dict", "params"):
                if key in state:
                    state = state[key]
                    break
            self.load_state_dict(dict(state), strict=True)
        self.eval()

    def _forward_unet(self, x0: torch.Tensor) -> torch.Tensor:
        """Run the backbone on an already-assembled head input.

        Args:
            x0: ``(B, in_channels + 1, H, W)`` tensor with ``H`` and ``W`` multiples of 16.

        Returns:
            ``(B, out_channels, H, W)`` tensor.

        """
        x = self.m_head(x0)
        skips = [x]
        for i in range(1, self._num_downs + 1):
            x = getattr(self, f"m_down{i}")(x)
            skips.append(x)
        x = self.m_body(x)
        for i in range(self._num_downs, 0, -1):
            x = getattr(self, f"m_up{i}")(x + skips[i])
        return self.m_tail(x + skips[0])

    def forward(self, x: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
        """Denoise ``x`` at noise level ``sigma``.

        Args:
            x: ``(H,W)``, ``(C,H,W)`` or ``(B,C,H,W)`` image tensor.
            sigma: Noise level, a float or a per-batch scalar tensor; broadcast as an extra input channel.

        Returns:
            The denoised tensor at ``x``'s rank.

        """
        x_norm, original_ndim = _normalize_rank(x)
        if isinstance(sigma, torch.Tensor):
            sigma_value = sigma.reshape(-1)[0] if sigma.numel() > 1 else sigma
        else:
            sigma_value = sigma
        noise_level_map = torch.full(
            (x_norm.size(0), 1, *x_norm.shape[2:]), float(sigma_value), device=x_norm.device, dtype=x_norm.dtype
        )
        x_in = torch.cat((x_norm, noise_level_map), dim=1)
        height, width = x_in.shape[2:]
        pad_h, pad_w = (-height) % 16, (-width) % 16
        if pad_h or pad_w:
            x_in = F.pad(x_in, (0, pad_w, 0, pad_h), mode="replicate")
        out = self._forward_unet(x_in)[..., :height, :width]
        return _restore_rank(out, original_ndim)


_DENOISER_CACHE: dict[tuple[str, str], DRUNet] = {}


def load_drunet(device: str, model_dir: str | Path | None = None) -> DRUNet:
    """Load (and cache per device) the pretrained color DRUNet from ``<model_dir>/drunet_color.pth``.

    Args:
        device: Device to load the network on.
        model_dir: Directory holding ``drunet_color.pth``; defaults to ``model_zoo/`` or
            ``$ADAPTIVEPNP_MODEL_DIR``.

    Returns:
        The network in eval mode on ``device``.

    Raises:
        FileNotFoundError: If the weights are not there (see ``scripts/download_weights.py``).

    """
    model_dir = Path(model_dir) if model_dir else MODEL_ZOO
    key = (str(model_dir), str(device))
    if key not in _DENOISER_CACHE:
        model = DRUNet(in_channels=3, out_channels=3, pretrained=model_dir / "drunet_color.pth")
        _DENOISER_CACHE[key] = model.to(device).eval()
    return _DENOISER_CACHE[key]


__all__ = ["DRUNET_WEIGHTS_URL", "DRUNet", "ResBlock", "load_drunet"]
