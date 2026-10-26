"""The squared-L2 data-fidelity term and its exact proximal map ``F_{sigma,gamma}``"""

from __future__ import annotations

from typing import Any

import torch

from adaptivepnp.operators import Convolution, LinearOperator, SuperResolution


class SquaredL2:
    r"""The data-fitting term ``f(x) = scale * ||y - A x||^2 / sigma^2`` and its proximal map.

    The proximal map is the quadratic problem

        F_{sigma,gamma}(v) = argmin_x  scale * ||y - A x||^2 / sigma^2 + ||x - v||^2 / (2 gamma^2).

    Setting its gradient to zero and dividing by ``2 scale / sigma^2`` gives the normal equations

        (A^T A + lam I) x = A^T y + lam v =: b,        lam = sigma^2 / (2 scale gamma^2).

    Args:
        y: Measurement.
        A: Forward operator: a :class:`adaptivepnp.operators.Convolution` or
            :class:`adaptivepnp.operators.SuperResolution`.
        scale: Positive weight of the squared residual

    Raises:
        TypeError: If ``A`` is neither, since no closed-form proximal map is available.
        ValueError: If ``scale`` is not positive.

    """

    accepts_prox_sigma = True

    def __init__(self, y: torch.Tensor, A: LinearOperator, scale: float = 1.0):
        if not isinstance(A, (Convolution, SuperResolution)):
            raise TypeError(
                f"SquaredL2 has a closed-form prox for Convolution and SuperResolution, got {type(A).__name__}."
            )
        if not float(scale) > 0:
            raise ValueError(f"scale must be positive, got {scale!r}.")
        self.y = y
        self.A = A
        self.scale = float(scale)
        self.last_prox_info: dict[str, Any] | None = None

    def __call__(self, x: torch.Tensor) -> float:
        """Evaluate ``scale * ||y - A x||^2``.

        Args:
            x: Image(s) in the operator's domain.

        Returns:
            The loss as a float.

        """
        return float(self.scale * (abs(self.y - self.A(x)) ** 2).sum())

    def _lam(self, gamma: float, sigma: float) -> float:
        """The ridge ``lam = sigma^2 / (2 * scale * gamma^2)`` of the normal equations.

        Args:
            gamma: Proximal parameter.
            sigma: Noise level weighting the data term.

        Returns:
            ``lam`` as a float.

        Raises:
            ValueError: If the ``scale`` attribute has been set to a non-positive value.

        """
        if not self.scale > 0:
            raise ValueError(f"scale must be positive, got {self.scale!r}.")
        return sigma**2 / (2.0 * self.scale * gamma**2)

    def prox(self, v: torch.Tensor, gamma: float = 1.0, sigma: float | None = None) -> torch.Tensor:
        """The exact proximal map ``F_{sigma,gamma}(v)``.

        Args:
            v: Point at which to evaluate the prox.
            gamma: Proximal parameter (the PnP step size).
            sigma: Noise level weighting the data term; ``None`` means ``1``.

        Returns:
            ``F_{sigma,gamma}(v)``, same shape and dtype as ``v``.

        Raises:
            ValueError: If ``gamma`` or ``sigma`` is not positive.

        """
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma!r}.")
        if sigma is None:
            sigma = 1.0
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma!r}.")
        if isinstance(self.A, Convolution):
            return self._prox_circulant(v, gamma, sigma)
        return self._prox_super_resolution(v, gamma, sigma)

    def _prox_circulant(self, v: torch.Tensor, gamma: float, sigma: float) -> torch.Tensor:
        r"""Closed-form prox for a circular convolution ``A x = h * x``.

        The 2-D DFT ``F`` diagonalizes ``A``: ``F A F^{-1} = diag(H_k)`` with ``H_k = fft2(h)`` on ``v``'s grid,
        and hence ``F A^T A F^{-1} = diag(|H_k|^2)``. The normal equations

            (A^T A + lam I) x = A^T y + lam v,        lam = sigma^2 / (2 scale gamma^2),

        therefore decouple into one scalar equation per Fourier mode,

            (|H_k|^2 + lam) X_k = conj(H_k) Y_k + lam V_k.

        Args:
            v: Point at which to evaluate the prox.
            gamma: Proximal parameter.
            sigma: Noise level weighting the data term.

        Returns:
            The proximal map of ``v``.

        """
        lam = self._lam(gamma, sigma)  # sigma^2 / (2 scale gamma^2) = 1 / rho^2

        # 1. Frequency response of A on v's grid: H_k = fft2(h), cached by the operator per grid and dtype.
        H = self.A.frequency_response(v)

        # 2. Right-hand side b = A^T y + lam v in the Fourier domain: B_k = conj(H_k) Y_k + lam V_k.
        Y = torch.fft.fft2(self.y, dim=(-2, -1))
        V = torch.fft.fft2(v, dim=(-2, -1))
        B = H.conj() * Y + lam * V

        # 3. Solve the diagonal system (|H_k|^2 + lam) X_k = B_k and return to the spatial domain.
        X = B / (abs(H) ** 2 + lam)
        x = torch.fft.ifft2(X, dim=(-2, -1))
        if not v.is_complex():
            x = x.real

        self.last_prox_info = {"num_iter": 0, "rel_res": 0.0, "method": "circulant"}
        return x.to(dtype=v.dtype)

    def _prox_super_resolution(self, v: torch.Tensor, gamma: float, sigma: float) -> torch.Tensor:
        r"""Closed-form prox for blur-then-decimate ``A = S H`` (``H`` circular blur, ``S`` keeps every s-th pixel).

        ``A^T A`` is not diagonal in any FFT basis, but ``A A^T = S H H^T S^T`` is diagonal in the FFT basis
        of the low-resolution grid: ``F (A A^T) F^{-1} = diag(P_k)`` where ``P_k`` is the mean of ``|H_k|^2``
        over the ``s x s`` aliases of mode ``k`` (:meth:`SuperResolution.power_spectrum`; it equals the DFT of
        ``h~_0``, the decimated autocorrelation of ``h``, in Chan et al.). The Woodbury identity

            (A^T A + lam I)^{-1} = lam^{-1} [ I - A^T (A A^T + lam I)^{-1} A ]

        turns the normal equations ``(A^T A + lam I) x = b``, ``b = A^T y + lam v``, into eq. 26 of Chan et al.,

            x = lam^{-1} [ b - A^T (A A^T + lam I)^{-1} A b ],        lam = sigma^2 / (2 scale gamma^2),

        in which only a low-resolution diagonal system is inverted. Expanding ``b`` and collecting terms gives
        the equivalent residual form used here,

            x = v + A^T (A A^T + lam I)^{-1} (y - A v).

        Args:
            v: Point at which to evaluate the prox.
            gamma: Proximal parameter.
            sigma: Noise level weighting the data term.

        Returns:
            The proximal map of ``v``.

        """
        lam = self._lam(gamma, sigma)  # sigma^2 / (2 scale gamma^2) = 1 / rho^2

        # 1. Low-resolution residual r = y - A v, and the eigenvalues P_k of A A^T on that grid.
        residual = self.y - self.A(v)
        P = self.A.power_spectrum(residual)

        # 2. Solve (A A^T + lam I) w = r in the low-resolution Fourier domain: W_k = R_k / (P_k + lam).
        R = torch.fft.fft2(residual, dim=(-2, -1))
        w = torch.fft.ifft2(R / (P + lam), dim=(-2, -1))
        if not v.is_complex():
            w = w.real

        # 3. Lift the correction to the high-resolution grid: x = v + A^T w.
        x = v + self.A.adj(w.to(v.dtype))

        self.last_prox_info = {"num_iter": 0, "rel_res": 0.0, "method": "super_resolution"}
        return x


__all__ = ["SquaredL2"]
