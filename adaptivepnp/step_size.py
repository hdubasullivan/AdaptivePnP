"""Step-size policies: the ``(sigma, gamma)`` pair handed to the data-fidelity prox at each iteration."""

from __future__ import annotations

import time
from typing import NamedTuple

import torch

from adaptivepnp.estimator import RHO_SEARCH_BOUNDS, TAU_DEFAULT
from adaptivepnp.operators import LinearOperator


class StepSizeResult(NamedTuple):
    """One iteration's step size."""

    sigma: float
    """Noise level used by the data-fidelity prox (estimated or fixed)."""
    gamma: float
    """Gamma actually passed to the prox (post-EMA for :class:`AdaptiveStepSizePolicy`)."""
    estimate_time: float
    """Wall-clock seconds spent estimating; ``0.0`` if none."""


class StepSizePolicy:
    """Supplies ``(sigma, gamma)`` for the data-fidelity prox at each iteration.

    A policy is stateful: ``prev_gamma``/``prev_sigma`` hold the pair most recently returned and
    ``history`` logs every pair. Nothing in a policy's arithmetic reads ``history``; it is a log only.
    """

    anchor_dependent: bool = False
    _prev_gamma: float | None = None
    _prev_sigma: float | None = None
    history: dict[str, list[float]]

    @property
    def prev_gamma(self) -> float | None:
        """The gamma most recently returned, or ``None`` before the first call."""
        return self._prev_gamma

    @property
    def prev_sigma(self) -> float | None:
        """The sigma most recently returned, or ``None`` before the first call."""
        return self._prev_sigma

    def _record(self, sigma: float, gamma: float) -> None:
        """Latch this step's pair as the previous one and append it to the log."""
        self._prev_gamma = float(gamma)
        self._prev_sigma = float(sigma)
        self.history["gamma"].append(gamma)
        self.history["sigma"].append(sigma)

    def get_step_size(
        self, v: torch.Tensor, y: torch.Tensor, operator: LinearOperator, iteration: int
    ) -> StepSizeResult:
        """Return this step's ``(sigma, gamma)``.

        Args:
            v: Current image-space point the residual is formed against.
            y: Measurement.
            operator: Forward operator.
            iteration: Current iteration index (0 for the bootstrap call).

        Returns:
            A :class:`StepSizeResult` for this iteration.

        """
        raise NotImplementedError

    def bootstrap_prev_gamma(self, x: torch.Tensor, y: torch.Tensor, operator: LinearOperator) -> None:
        """Seed ``prev_gamma`` before the first real step.

        Args:
            x: Initial iterate.
            y: Measurement.
            operator: Forward operator.

        """
        return

    def reset(self) -> None:
        """Return the policy to its constructed state before a fresh run."""
        self._prev_gamma = None
        self._prev_sigma = None


class FixedStepSizePolicy(StepSizePolicy):
    """Constant ``(sigma, gamma)`` every step.

    Args:
        gamma: Fixed proximal parameter, must be positive.
        sigma: Fixed noise-level standard deviation.

    Raises:
        ValueError: If ``gamma`` is not positive.

    """

    def __init__(self, gamma: float, sigma: float):
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma!r}.")
        self.gamma = float(gamma)
        self.sigma = float(sigma)
        self.history = {"gamma": [], "sigma": []}
        self._record(self.sigma, self.gamma)

    def get_step_size(self, v, y, operator, iteration) -> StepSizeResult:
        """Return the fixed ``gamma`` and ``sigma``.

        Args:
            v: Ignored.
            y: Ignored.
            operator: Ignored.
            iteration: Ignored.

        Returns:
            A :class:`StepSizeResult` with the constant pair.

        """
        self._record(self.sigma, self.gamma)
        return StepSizeResult(sigma=self.sigma, gamma=self.gamma, estimate_time=0.0)

    def reset(self) -> None:
        """Re-seed the log with the constant pair; there is no other state to clear."""
        super().reset()
        self.history = {"gamma": [], "sigma": []}
        self._record(self.sigma, self.gamma)


class AdaptiveStepSizePolicy(StepSizePolicy):
    """Per-step maximum-likelihood estimate of ``(sigma, gamma)`` from the residual ``y - A v``, EMA-smoothed.

    Args:
        frozen_sigma: Hold sigma at this known value and estimate only gamma.
        freeze_estimated_sigma: Estimate sigma on the first call, then hold it fixed for the run.
        gamma_smoothing: EMA weight ``beta`` in ``[0, 1]`` on each iteration's estimate against the
            previous gamma: ``gamma_k = beta * gamma_hat_k + (1 - beta) * gamma_{k-1}``. ``1.0`` / ``None``
            is the raw estimate; ``0.0`` freezes gamma at its bootstrap value.
        rho_floor: ``'cr_bound'`` bounds the ML search below at the lower edge of the identifiable
            window; ``None`` leaves it unbounded.
        rho_ceiling: ``'cr_bound'`` bounds it above at the window's upper edge; ``None`` leaves it.
        tau: Tolerance on the normalized Cramér–Rao bounds that define the window.
        rho_search_bounds: Outer bounds of the search over ``rho = gamma / sigma``.

    Raises:
        ValueError: If ``tau`` is not positive, ``gamma_smoothing`` is outside ``[0, 1]``, or both
            ``frozen_sigma`` and ``freeze_estimated_sigma`` are set.

    """

    anchor_dependent = True

    def __init__(
        self,
        frozen_sigma: float | None = None,
        gamma_smoothing: float | None = None,
        freeze_estimated_sigma: bool = False,
        rho_floor: str | None = "cr_bound",
        rho_ceiling: str | None = "cr_bound",
        tau: float = TAU_DEFAULT,
        rho_search_bounds: tuple[float, float] = RHO_SEARCH_BOUNDS,
    ):
        if not tau > 0.0:
            raise ValueError(f"tau must be positive, got {tau!r}.")
        if gamma_smoothing is not None and not (0.0 <= gamma_smoothing <= 1.0):
            raise ValueError(f"gamma_smoothing must be in [0, 1] or None, got {gamma_smoothing!r}.")
        if freeze_estimated_sigma and frozen_sigma is not None:
            raise ValueError("freeze_estimated_sigma and frozen_sigma are mutually exclusive.")
        self.frozen_sigma = frozen_sigma
        self.freeze_estimated_sigma = freeze_estimated_sigma
        self.rho_floor = rho_floor
        self.rho_ceiling = rho_ceiling
        self.tau = tau
        self.rho_search_bounds = rho_search_bounds
        self._initial_frozen_sigma = frozen_sigma
        self.gamma_smoothing = gamma_smoothing
        self.history = {"gamma": [], "sigma": []}

    def _estimate(self, residual: torch.Tensor, operator: LinearOperator) -> tuple[float, float]:
        """Ask the operator for the ML ``(sigma_hat, gamma_hat)`` of ``residual`` under this policy's bounds.

        Args:
            residual: ``y - A v`` in measurement space.
            operator: Forward operator supplying the spectrum.

        Returns:
            ``(sigma_hat, gamma_hat)`` as floats.

        """
        sigma_hat, gamma_hat = operator.estimate_sigma_gamma(
            residual,
            rho_floor=self.rho_floor,
            rho_ceiling=self.rho_ceiling,
            tau=self.tau,
            frozen_sigma=self.frozen_sigma,
            rho_search_bounds=self.rho_search_bounds,
        )
        return float(sigma_hat), float(gamma_hat)

    def get_step_size(self, v, y, operator, iteration) -> StepSizeResult:
        """Estimate ``(sigma_hat, gamma_hat)`` from the residual ``y - A v``, then EMA-smooth gamma.

        Args:
            v: Current image-space point.
            y: Measurement.
            operator: Forward operator.
            iteration: Current iteration index; smoothing is skipped at iteration 0.

        Returns:
            A :class:`StepSizeResult` with the sigma used (estimated or frozen) and the smoothed gamma.

        """
        residual = y - operator(v)
        t0 = time.perf_counter()
        sigma_hat, gamma_hat = self._estimate(residual, operator)
        t1 = time.perf_counter()

        if self.freeze_estimated_sigma and self.frozen_sigma is None:
            self.frozen_sigma = sigma_hat

        if self.gamma_smoothing is not None and iteration > 0 and self.prev_gamma is not None:
            beta = self.gamma_smoothing
            gamma = beta * gamma_hat + (1 - beta) * self.prev_gamma
        else:
            gamma = gamma_hat

        sigma = sigma_hat if self.frozen_sigma is None else self.frozen_sigma
        self._record(sigma, gamma)
        return StepSizeResult(sigma=sigma, gamma=gamma, estimate_time=t1 - t0)

    def bootstrap_prev_gamma(self, x, y, operator) -> None:
        """Seed ``prev_gamma``/``prev_sigma`` at iteration 0 so the first real step has a value to smooth against.

        Args:
            x: Initial iterate.
            y: Measurement.
            operator: Forward operator.

        """
        self.get_step_size(x, y, operator, iteration=0)

    def reset(self) -> None:
        """Reset the EMA memory, the log, and any sigma latched by ``freeze_estimated_sigma``."""
        super().reset()
        self.history = {"gamma": [], "sigma": []}
        self.frozen_sigma = self._initial_frozen_sigma


__all__ = ["AdaptiveStepSizePolicy", "FixedStepSizePolicy", "StepSizePolicy", "StepSizeResult"]
