"""Linear forward operators: circular convolution, integer subsampling, and their composition."""

from __future__ import annotations

import math
from typing import Any

import torch

from adaptivepnp.estimator import RHO_SEARCH_BOUNDS, TAU_DEFAULT, estimate_sigma_gamma, rho_identifiable_window


def fft_power(x: torch.Tensor) -> torch.Tensor:
    """Squared-magnitude 2D FFT over the last two axes, divided by the number of pixels.

    This is ``|z~|²`` for the rotated residual ``z~ = F z / sqrt(M)``: the input of the likelihood.

    Args:
        x: Real tensor whose last two axes are spatial; leading axes are batch/channel copies.

    Returns:
        ``|fft2(x)|² / (H W)``, same shape as ``x``.

    """
    n = x.shape[-2] * x.shape[-1]
    return torch.fft.fft2(x, dim=(-2, -1)).abs() ** 2 / n


class LinearOperator:
    """A linear map on tensors whose last two axes are spatial; leading axes are batch/channel.

    Subclasses implement ``_forward`` and ``_adjoint``. ``input_shape``/``output_shape`` are checked on
    their trailing two dimensions only, so any batch of images with the declared spatial size is accepted.

    Args:
        input_shape: Shape of the domain; the last two entries are spatial.
        output_shape: Shape of the codomain; the last two entries are spatial.
        dtype: Dtype of the arrays the operator accepts and returns.

    """

    def __init__(self, input_shape: tuple[int, ...], output_shape: tuple[int, ...], dtype: torch.dtype):
        self.input_shape = tuple(int(s) for s in input_shape)
        self.output_shape = tuple(int(s) for s in output_shape)
        self.input_dtype = dtype
        self.output_dtype = dtype
        self._window_cache: dict[tuple[Any, ...], tuple[float, float]] = {}

    @staticmethod
    def _conforms(shape: tuple[int, ...], declared: tuple[int, ...]) -> bool:
        """Whether ``shape``'s trailing two dimensions match ``declared``'s."""
        return len(shape) >= 2 and tuple(shape[-2:]) == tuple(declared[-2:])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the operator to ``x``.

        Args:
            x: Array with the operator's spatial input shape in its last two axes.

        Returns:
            ``A x``.

        Raises:
            ValueError: If the spatial shape does not match ``input_shape``.

        """
        if not self._conforms(x.shape, self.input_shape):
            raise ValueError(f"{type(self).__name__}: input shape {tuple(x.shape)} does not match {self.input_shape}.")
        return self._forward(x)

    def adj(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the adjoint ``Aᴴ`` to ``y``.

        Args:
            y: Array with the operator's spatial output shape in its last two axes and its ``output_dtype``.

        Returns:
            ``A^* y``.

        Raises:
            ValueError: If the dtype or the spatial shape does not match the codomain.

        """
        if y.dtype != self.output_dtype:
            raise ValueError(f"Dtype error: expected {self.output_dtype}, got {y.dtype}.")
        if not self._conforms(y.shape, self.output_shape):
            raise ValueError(f"{type(self).__name__}: shape {tuple(y.shape)} does not match {self.output_shape}.")
        return self._adjoint(y)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward map, implemented by each subclass."""
        raise NotImplementedError

    def _adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Adjoint map, implemented by each subclass."""
        raise NotImplementedError

    def power_spectrum(self, like: torch.Tensor | None = None) -> torch.Tensor:
        """Eigenvalues of ``AA^T`` in the measurement-space FFT basis.

        Args:
            like: Template supplying the dtype and device (and, through its last two axes, the grid); its
                values are ignored. ``None`` uses zeros of ``output_shape`` on the CPU.

        Returns:
            Non-negative tensor of shape ``output_shape[-2:]``.

        """
        raise NotImplementedError

    def identifiable_window(
        self,
        tau: float = TAU_DEFAULT,
        *,
        n_copies: int | None = None,
        rho_bounds: tuple[float, float] = RHO_SEARCH_BOUNDS,
        sigma_known: bool = False,
        template: torch.Tensor | None = None,
    ) -> tuple[float, float]:
        """The identifiable window of this operator at tolerance ``tau``, computed once and cached.

        The window (:func:`adaptivepnp.estimator.rho_identifiable_window`) depends only on the spectrum of
        ``AA^T``, the number of measurements and ``tau``, not on the residual, so it is computed on
        the first call for each ``(n_copies, tau, rho_bounds, sigma_known)`` and reused at every later iteration.

        Args:
            tau: Tolerance on the normalized Cramér–Rao bounds.
            n_copies: Independent copies of the spectrum in a residual (the color channels); ``None`` takes
                them from ``output_shape``.
            rho_bounds: Range of ``rho`` scanned for the window's edges.
            sigma_known: Use the known-sigma bound on ``gamma^2`` alone (a floor, no ceiling).
            template: Tensor supplying the device and dtype of the spectrum; ``None`` uses the CPU.

        Returns:
            ``(rho_lo, rho_hi)``.

        Raises:
            ValueError: If the window is empty at ``tau`` (see :func:`adaptivepnp.estimator.rho_identifiable_window`).

        """
        if n_copies is None:
            n_copies = max(1, math.prod(int(s) for s in self.output_shape[:-2]))
        key = (int(n_copies), float(tau), (float(rho_bounds[0]), float(rho_bounds[1])), bool(sigma_known))
        window = self._window_cache.get(key)
        if window is None:
            spectrum = self.power_spectrum(template)
            window = rho_identifiable_window(
                spectrum.flatten(), n_copies * spectrum.numel(), tau=tau, rho_bounds=rho_bounds, sigma_known=sigma_known
            )
            self._window_cache[key] = window
        return window

    def estimate_sigma_gamma(self, residual: torch.Tensor, **kwargs: Any) -> tuple[float, float]:
        """IML estimate of ``(sigma, gamma)`` from a measurement-space residual.

        Rotates ``residual`` into the eigenbasis with :func:`fft_power` and hands it, with this operator's
        :meth:`power_spectrum` and its cached :meth:`identifiable_window`, to
        :func:`adaptivepnp.estimator.estimate_sigma_gamma`; the window is computed once per operator, not once
        per call.

        Args:
            residual: ``y - A v``, in measurement space.
            **kwargs: Forwarded to :func:`adaptivepnp.estimator.estimate_sigma_gamma` (``rho_floor``,
                ``rho_ceiling``, ``tau``, ``frozen_sigma``, ``rho_search_bounds``).

        Returns:
            ``(sigma_hat, gamma_hat)``.

        """
        spectrum = self.power_spectrum(residual)
        bounded = (
            kwargs.get("rho_floor", "cr_bound") == "cr_bound" or kwargs.get("rho_ceiling", "cr_bound") == "cr_bound"
        )
        if bounded and kwargs.get("window") is None:
            kwargs["window"] = self.identifiable_window(
                kwargs.get("tau", TAU_DEFAULT),
                n_copies=max(1, residual.numel() // spectrum.numel()),
                rho_bounds=kwargs.get("rho_search_bounds", RHO_SEARCH_BOUNDS),
                sigma_known=kwargs.get("frozen_sigma") is not None,
                template=residual,
            )
        return estimate_sigma_gamma(fft_power(residual), spectrum, **kwargs)


class Convolution(LinearOperator):
    r"""Same-shape two-dimensional circular convolution.

    Args:
        kernel: Two-dimensional convolution kernel; copied and cast to ``input_dtype``.
        input_shape: Shape of the images accepted; the last two entries are spatial.
        center: Kernel index representing zero displacement. Defaults to ``kernel.shape // 2``.
        input_dtype: Dtype of the images accepted and returned.

    Raises:
        ValueError: If the kernel is not two-dimensional or exceeds the spatial input shape.

    """

    def __init__(
        self,
        kernel: torch.Tensor,
        *,
        input_shape: tuple[int, ...],
        center: tuple[int, int] | None = None,
        input_dtype: torch.dtype = torch.float32,
    ) -> None:
        kernel = torch.as_tensor(kernel)
        if kernel.ndim != 2:
            raise ValueError(f"Kernel must be two-dimensional, got shape {tuple(kernel.shape)}.")
        if len(input_shape) < 2:
            raise ValueError(f"Convolution requires at least two input dimensions, got {input_shape}.")
        if kernel.shape[0] > input_shape[-2] or kernel.shape[1] > input_shape[-1]:
            raise ValueError(
                f"Kernel shape {tuple(kernel.shape)} cannot exceed spatial input shape {input_shape[-2:]}."
            )
        if center is None:
            center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
        self.center = (int(center[0]), int(center[1]))
        self._kernel = torch.as_tensor(kernel, dtype=input_dtype).detach().clone()
        self._frequency_response_cache: dict[tuple[Any, ...], torch.Tensor] = {}
        super().__init__(input_shape, input_shape, input_dtype)

    @property
    def kernel(self) -> torch.Tensor:
        """A copy of the convolution kernel."""
        return self._kernel.clone()

    def _frequency_response(self, like: torch.Tensor) -> torch.Tensor:
        """``H_k = fft2(h_pad)``, the kernel zero-padded to ``like``'s grid and centered, cached per grid and placement.

        Args:
            like: Template supplying the device, dtype and spatial size; its values are ignored.

        Returns:
            Complex tensor of shape ``like.shape[-2:]``.

        Raises:
            ValueError: If the kernel exceeds ``like``'s spatial size.

        """
        key = (like.device.type, like.device.index, like.dtype, tuple(like.shape[-2:]))
        cached = self._frequency_response_cache.get(key)
        if cached is not None:
            return cached
        height, width = (int(s) for s in like.shape[-2:])
        kernel = self._kernel.to(device=like.device, dtype=like.dtype)
        if kernel.shape[0] > height or kernel.shape[1] > width:
            raise ValueError(f"Kernel shape {tuple(kernel.shape)} cannot exceed spatial input shape {(height, width)}.")
        center_y, center_x = self.center
        embedded = torch.zeros((height, width), dtype=like.dtype, device=like.device)
        for row in range(kernel.shape[0]):
            for col in range(kernel.shape[1]):
                embedded[(row - center_y) % height, (col - center_x) % width] += kernel[row, col]
        response = torch.fft.fft2(embedded)
        self._frequency_response_cache[key] = response
        return response

    def _apply(self, x: torch.Tensor, *, adjoint: bool) -> torch.Tensor:
        """Multiply ``x`` by ``H_k`` (or its conjugate) in the Fourier domain.

        Args:
            x: Image(s) with the operator's dtype.
            adjoint: Use ``conj(H_k)`` instead of ``H_k``.

        Returns:
            The filtered image(s), real when ``x`` is real.

        Raises:
            ValueError: If ``x`` does not have the operator's dtype.

        """
        if x.dtype != self.input_dtype:
            raise ValueError(f"Dtype error: expected {self.input_dtype}, got {x.dtype}.")
        response = self._frequency_response(x)
        multiplier = response.conj() if adjoint else response
        result = torch.fft.ifft2(torch.fft.fft2(x, dim=(-2, -1)) * multiplier, dim=(-2, -1))
        if not x.is_complex():
            result = result.real
        return result.to(dtype=x.dtype)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Circular convolution with the kernel."""
        return self._apply(x, adjoint=False)

    def _adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Circular correlation with the kernel (convolution with the flipped kernel)."""
        return self._apply(y, adjoint=True)

    def frequency_response(self, like: torch.Tensor) -> torch.Tensor:
        """The complex eigenvalues ``H_k`` of ``A`` on ``like``'s grid, device and dtype.

        Args:
            like: Template supplying the device, dtype and spatial size; its values are ignored.

        Returns:
            Complex tensor of shape ``like.shape[-2:]``.

        """
        return self._frequency_response(like)

    def power_spectrum(self, like: torch.Tensor | None = None) -> torch.Tensor:
        """``|H_k|²``, the eigenvalues of ``AA^T``.

        Args:
            like: Template supplying the device, dtype and spatial size; its values are ignored. ``None`` uses
                zeros of ``output_shape`` on the CPU.

        Returns:
            Non-negative tensor of shape ``like.shape[-2:]``.

        """
        if like is None:
            like = torch.zeros(self.output_shape, dtype=self.output_dtype)
        return abs(self._frequency_response(like)) ** 2


class Subsampling(LinearOperator):
    """Integer-stride spatial subsampling ``x[..., ::f, ::f]``; the adjoint zero-inserts.

    Args:
        factor: Positive integer sampling factor, applied to both spatial axes.
        input_shape: Shape of the high-resolution input; its spatial sizes must be divisible by ``factor``.
        input_dtype: Dtype of the arrays accepted and returned.

    Raises:
        ValueError: If ``factor`` is not positive or does not divide the spatial input shape.

    """

    def __init__(self, factor: int, *, input_shape: tuple[int, ...], input_dtype: torch.dtype) -> None:
        factor = int(factor)
        if factor <= 0:
            raise ValueError(f"Sampling factor must be positive, got {factor}.")
        if any(size % factor for size in input_shape[-2:]):
            raise ValueError(f"Spatial input shape {input_shape[-2:]} must be divisible by the factor {factor}.")
        self.factor = (factor, factor)
        self._slices = (..., slice(0, None, factor), slice(0, None, factor))
        output_shape = (*input_shape[:-2], input_shape[-2] // factor, input_shape[-1] // factor)
        super().__init__(input_shape, output_shape, input_dtype)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Keep every ``factor``-th row and column.

        Args:
            x: High-resolution image(s) with the operator's dtype.

        Returns:
            The decimated image(s).

        Raises:
            ValueError: If ``x`` does not have the operator's dtype.

        """
        if x.dtype != self.input_dtype:
            raise ValueError(f"Dtype error: expected {self.input_dtype}, got {x.dtype}.")
        return x[self._slices]

    def _adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Zero-insert ``y`` onto the high-resolution grid."""
        spatial = tuple(stride * int(size) for size, stride in zip(y.shape[-2:], self.factor))
        out = torch.zeros((*y.shape[:-2], *spatial), dtype=y.dtype, device=y.device)
        out[self._slices] = y
        return out


class SuperResolution(LinearOperator):
    r"""Circular blur followed by subsampling, ``A = S H``; the adjoint is ``Hᴴ Sᴴ``.

    ``AA^T`` is diagonal in the low-resolution FFT basis with eigenvalues given by the aliased fold of
    ``|H_k|²`` (:meth:`power_spectrum`), which is what the ML estimate and the closed-form prox use.

    Args:
        kernel: Two-dimensional blur kernel.
        factor: Integer decimation factor, applied to both spatial axes.
        input_shape: Shape of the high-resolution images; its spatial sizes must be divisible by ``factor``.
        center: Kernel index representing zero displacement. Defaults to ``kernel.shape // 2``.
        input_dtype: Dtype of the arrays accepted and returned.

    """

    def __init__(
        self,
        kernel: torch.Tensor,
        factor: int,
        *,
        input_shape: tuple[int, ...],
        center: tuple[int, int] | None = None,
        input_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.convolution = Convolution(kernel, input_shape=input_shape, center=center, input_dtype=input_dtype)
        self.sampling = Subsampling(factor, input_shape=input_shape, input_dtype=input_dtype)
        super().__init__(input_shape, self.sampling.output_shape, input_dtype)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Blur, then decimate."""
        return self.sampling(self.convolution(x))

    def _adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Zero-insert, then correlate with the kernel."""
        return self.convolution.adj(self.sampling.adj(y))

    def power_spectrum(self, like: torch.Tensor | None = None) -> torch.Tensor:
        """Eigenvalues of ``AA^T`` on the low-resolution grid: the mean of ``|H_k|^2`` over each alias block.

        Args:
            like: Template supplying the device, dtype and low-resolution spatial size; its values are ignored.
                ``None`` uses zeros of ``output_shape`` on the CPU.

        Returns:
            Non-negative tensor of shape ``like.shape[-2:]``, i.e. ``(H / factor, W / factor)``.

        """
        if like is None:
            like = torch.zeros(self.output_shape, dtype=self.output_dtype)
        factor_y, factor_x = self.sampling.factor
        output_y, output_x = (int(s) for s in like.shape[-2:])
        template = torch.zeros((factor_y * output_y, factor_x * output_x), dtype=like.dtype, device=like.device)
        power = self.convolution.power_spectrum(template)
        return power.reshape(factor_y, output_y, factor_x, output_x).mean(dim=(0, 2))

    def estimate_sigma_gamma(self, residual: torch.Tensor, **kwargs: Any) -> tuple[float, float]:
        """IML estimate of ``(sigma, gamma)`` from a low-resolution residual.

        Args:
            residual: ``y - A v`` on the low-resolution grid, with the operator's dtype.
            **kwargs: Forwarded to :func:`adaptivepnp.estimator.estimate_sigma_gamma`.

        Returns:
            ``(sigma_hat, gamma_hat)``.

        Raises:
            ValueError: If ``residual`` does not have the operator's output shape or dtype.

        """
        if not self._conforms(residual.shape, self.output_shape):
            raise ValueError(f"Residual shape {tuple(residual.shape)} does not match output shape {self.output_shape}.")
        if residual.dtype != self.output_dtype:
            raise ValueError(f"Dtype error: expected {self.output_dtype}, got {residual.dtype}.")
        return super().estimate_sigma_gamma(residual, **kwargs)


__all__ = ["Convolution", "LinearOperator", "Subsampling", "SuperResolution", "fft_power"]
