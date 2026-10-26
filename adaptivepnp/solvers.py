"""Plug-and-play PGM and ADMM, driven by a step-size policy.

All run exactly ``max_iter`` iterations when ``tolerance=0``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from adaptivepnp.data_fidelity import SquaredL2
from adaptivepnp.step_size import StepSizePolicy


def _norm(x: torch.Tensor) -> torch.Tensor:
    """Euclidean norm of a tensor."""
    return torch.linalg.vector_norm(x)


class PnPPrior:
    r"""A denoiser used as a proximal map: ``prox_{lam g}(x) = D_sigma(x)`` with ``sigma = sqrt(lam)``.

    Args:
        denoiser: The denoiser, called as ``denoiser(x, sigma=...)`` on a ``(1, C, H, W)`` tensor. A
            ``torch.nn.Module`` is run on its own device and dtype and switched to eval mode for the call.

    """

    has_prox = True
    has_grad_step = True

    def __init__(self, denoiser: Callable[..., torch.Tensor]):
        self.denoiser = denoiser

    def _placement(self) -> tuple[torch.device | None, torch.dtype | None]:
        """Device and floating dtype of the denoiser's parameters, or ``(None, None)`` if it has none."""
        getter = getattr(self.denoiser, "parameters", None)
        reference = next(iter(getter()), None) if callable(getter) else None
        if reference is None:
            return None, None
        return reference.device, (reference.dtype if reference.is_floating_point() else None)

    def prox(self, v: torch.Tensor, lam: float) -> torch.Tensor:
        """Denoise ``v`` at noise level ``sqrt(lam)``.

        Args:
            v: Point at which to evaluate the prox.
            lam: Proximal parameter; the denoiser runs at ``sqrt(lam)``.

        Returns:
            The denoised image, on ``v``'s device and dtype.

        """
        sigma = float(np.sqrt(lam))
        device, dtype = self._placement()
        x = v
        if dtype is not None and x.dtype != dtype:
            x = x.to(dtype)
        if device is not None and x.device != device:
            x = x.to(device)
        module = self.denoiser if isinstance(self.denoiser, torch.nn.Module) else None
        was_training = module is not None and module.training
        if module is not None:
            module.eval()
        try:
            with torch.no_grad():
                denoised = self.denoiser(x, sigma=sigma)
        finally:
            if module is not None:
                module.train(was_training)
        return denoised.to(device=v.device, dtype=v.dtype)

    def grad_step(self, x: torch.Tensor, lam: float) -> torch.Tensor:
        """The plug-and-play gradient step is the denoiser itself.

        Args:
            x: Current iterate.
            lam: Step parameter; the denoiser runs at ``sqrt(lam)``.

        Returns:
            ``D_sqrt(lam)(x)``.

        """
        return self.prox(x, lam)


@dataclass
class IterativeResult:
    """Outcome of a solve.

    Attributes:
        x: The final iterate, as returned by the solver's ``minimizer()``.
        converged: Whether the tolerance stopped the solve before ``max_iter``.
        num_iter: Number of iterations run.
        reason: ``'tolerance'`` or ``'max_iter'``.
        history: One record per iteration (``iter``, ``time``, and for ADMM the primal and dual residuals).

    """

    x: torch.Tensor
    converged: bool
    num_iter: int
    reason: str
    history: list[dict[str, Any]]


class _PolicyDrivenSolver:
    """Shared machinery of PGM and ADMM: resolve ``(gamma, sigma)`` from a policy and run the iteration loop.

    A subclass sets ``data_fidelity``, ``prior`` and the iterates, calls :meth:`_init_step_size` and
    :meth:`_init_loop` in its constructor, and implements :meth:`step`, :meth:`exit_condition` and
    :meth:`minimizer`.
    """

    _refresh_after_denoiser = False

    def _init_step_size(self, step_size: float | StepSizePolicy, data_fidelity: SquaredL2, x0: torch.Tensor) -> None:
        """Validate the step size and bootstrap the policy on ``x0``.

        Args:
            step_size: A positive scalar gamma, or a :class:`adaptivepnp.step_size.StepSizePolicy`, which is
                reset and bootstrapped on ``x0`` here.
            data_fidelity: The data term, supplying ``y`` and ``A`` to the policy.
            x0: Initial iterate.

        Raises:
            ValueError: If a scalar step size is not positive.

        """
        if isinstance(step_size, StepSizePolicy):
            self.step_size = None
            self.step_size_policy = step_size
            step_size.reset()
            step_size.bootstrap_prev_gamma(x0, data_fidelity.y, data_fidelity.A)
        else:
            if not float(step_size) > 0:
                raise ValueError(f"step_size must be a positive scalar or a StepSizePolicy, got {step_size!r}.")
            self.step_size = float(step_size)
            self.step_size_policy = None

    def _resolve(self, anchor: torch.Tensor) -> tuple[float, float | None]:
        """Ask the policy for ``(gamma, sigma)`` at ``anchor``; a scalar step size returns ``(gamma, None)``.

        Args:
            anchor: The iterate the residual is formed against.

        Returns:
            ``(gamma, sigma)``.

        """
        if self.step_size_policy is None:
            return float(self.step_size), None
        result = self.step_size_policy.get_step_size(
            v=anchor, y=self.data_fidelity.y, operator=self.data_fidelity.A, iteration=self.current_iter
        )
        return result.gamma, result.sigma

    def _seed_carried_step_size(self) -> tuple[float, float | None] | None:
        """Read ``(gamma_0, sigma_0)`` off the policy's bootstrap rather than estimating a second time.

        Returns:
            ``(gamma, sigma)`` for an iterator that re-anchors after the denoiser, else ``None``.

        """
        if not self._refresh_after_denoiser:
            return None
        gamma = getattr(self.step_size_policy, "prev_gamma", None)
        if gamma is None:
            return None
        return float(gamma), getattr(self.step_size_policy, "prev_sigma", None)

    def _data_prox(self, z: torch.Tensor, gamma: float, sigma: float | None) -> torch.Tensor:
        """Data-fitting proximal map ``F_{sigma,gamma}(z)``, forwarding ``sigma`` to a prox that accepts one.

        Args:
            z: Point at which to evaluate the prox.
            gamma: Proximal parameter (the PnP step size).
            sigma: Noise level for the data-term weighting, or ``None``.

        Returns:
            The proximal map of ``z``.

        """
        kwargs: dict[str, Any] = {}
        if getattr(self.data_fidelity, "accepts_prox_sigma", False) and sigma is not None:
            kwargs["sigma"] = sigma
        out = self.data_fidelity.prox(z, gamma, **kwargs)
        self.data_fidelity_prox_info = getattr(self.data_fidelity, "last_prox_info", None)
        return out

    # -- the iteration loop ----------------------------------------------------------------------
    def _init_loop(self, max_iter: int) -> None:
        """Initialize the iteration counter, history and timer for ``max_iter`` iterations."""
        self.max_iter = max_iter
        self.current_iter = 0
        self.history: list[dict[str, Any]] = []
        self._t0 = 0.0
        self._elapsed = 0.0

    def record_statistics(self) -> None:
        """Append this iteration's record (``iter``, elapsed ``time``) to ``history``."""
        self.history.append({"iter": self.current_iter, "time": self._elapsed + (time.perf_counter() - self._t0)})

    def step(self) -> None:
        """Perform one iteration; implemented by each subclass."""
        raise NotImplementedError

    def exit_condition(self) -> bool:
        """Whether the stopping criterion is met; implemented by each subclass."""
        raise NotImplementedError

    def minimizer(self) -> torch.Tensor:
        """The current estimate of the solution; implemented by each subclass."""
        raise NotImplementedError

    def solve(self, callback: Callable[[Any], None] | None = None) -> IterativeResult:
        """Run up to ``max_iter`` iterations.

        Args:
            callback: Called as ``callback(self)`` after every iteration; its wall time is excluded from
                the recorded ``time``.

        Returns:
            An :class:`IterativeResult`.

        """
        converged = False
        reason = "max_iter"
        self._t0 = time.perf_counter()
        for _ in range(self.max_iter):
            self.current_iter += 1
            self.step()
            self.record_statistics()
            converged = self.exit_condition()
            if callback:
                self._elapsed += time.perf_counter() - self._t0
                callback(self)
                self._t0 = time.perf_counter()
            if converged:
                reason = "tolerance"
                break
        return IterativeResult(
            x=self.minimizer(),
            converged=converged,
            num_iter=self.current_iter,
            reason=reason,
            history=list(self.history),
        )


class ProximalGradient(_PolicyDrivenSolver):
    r"""Plug-and-play proximal gradient: ``x_{k+1} = prox_{gamma² f}( D_{gamma}(x_k) )``.

    Args:
        data_fidelity: Term with a proximal map (:class:`adaptivepnp.data_fidelity.SquaredL2`).
        prior: A :class:`PnPPrior`.
        x0: Initial iterate.
        step_size: Positive scalar gamma, or a :class:`adaptivepnp.step_size.StepSizePolicy`.
        reg_strength: Scales the prior's prox parameter: ``(reg_strength * gamma)²``.
        max_iter: Iteration budget.
        tolerance: Relative iterate change below which to stop (``0`` runs all ``max_iter``).

    """

    def __init__(
        self,
        data_fidelity: SquaredL2,
        prior: PnPPrior,
        x0: torch.Tensor,
        step_size: float | StepSizePolicy,
        reg_strength: float = 1.0,
        max_iter: int = 100,
        tolerance: float = 1e-5,
    ):
        self.data_fidelity = data_fidelity
        self.prior = prior
        self.reg_strength = reg_strength
        self._init_step_size(step_size, data_fidelity, x0)
        self.tolerance = tolerance
        self.prev_est: torch.Tensor | None = None
        self.x = x0
        self.v = x0
        self.data_fidelity_prox_info = None
        self._carried_step_size = self._seed_carried_step_size()
        self._init_loop(max_iter)

    @property
    def _refresh_after_denoiser(self) -> bool:  # type: ignore[override]
        """Whether the policy is re-anchored on the denoiser output before the data prox."""
        return bool(getattr(self.step_size_policy, "anchor_dependent", False))

    def step(self) -> None:
        """One iteration: denoise at the carried gamma, re-estimate on the output, then the data prox."""
        refresh = self._refresh_after_denoiser
        if refresh and self._carried_step_size is not None:
            gamma, sigma = self._carried_step_size
        else:
            gamma, sigma = self._resolve(self.v)
        self.prev_est = self.minimizer()

        step_size = gamma**2
        grad_step = self.prior.grad_step(self.x, step_size * self.reg_strength**2)
        self.v = grad_step
        if refresh:
            gamma, sigma = self._resolve(self.v)
            self._carried_step_size = (gamma, sigma)
            step_size = gamma**2
        self.x = self._data_prox(grad_step, gamma, sigma)
        del grad_step

    def exit_condition(self) -> bool:
        """Relative iterate change ``||x_k - x_{k-1}|| / ||x_k||`` below ``tolerance``."""
        if self.prev_est is None:
            return False
        current = self.minimizer()
        return bool(_norm(self.prev_est - current) / (_norm(current) + 1e-12) < self.tolerance)

    def minimizer(self) -> torch.Tensor:
        """The current iterate ``x``."""
        return self.x


class ADMM(_PolicyDrivenSolver):
    r"""Unscaled-dual ADMM for ``min_x f(x) + g(x)``, data prox first.

    The policy is anchored on ``v`` (the denoiser output) at the top of each iteration. The reported
    iterate is ``v`` (Algorithm 3).

    Args:
        data_fidelity: Term with a proximal map (:class:`adaptivepnp.data_fidelity.SquaredL2`).
        prior: A :class:`PnPPrior`.
        x0: Initial iterate for both ``x`` and ``v``.
        step_size: Positive scalar gamma, or a :class:`adaptivepnp.step_size.StepSizePolicy`.
        reg_strength: The ``c`` scaling the prior's prox parameter.
        max_iter: Iteration budget.
        tolerance: Primal and dual residual threshold below which to stop (``0`` runs all ``max_iter``).

    """

    def __init__(
        self,
        data_fidelity: SquaredL2,
        prior: PnPPrior,
        x0: torch.Tensor,
        step_size: float | StepSizePolicy,
        reg_strength: float = 1.0,
        max_iter: int = 100,
        tolerance: float = 1e-5,
    ):
        self.data_fidelity = data_fidelity
        self.prior = prior
        self.reg_strength = reg_strength
        self._init_step_size(step_size, data_fidelity, x0)
        self.tolerance = tolerance
        self.v = x0
        self.x = x0
        self.lagrange_mult = 0.0 * x0
        self.data_fidelity_prox_info = None
        self.primal_residual: float | None = None
        self.dual_residual: float | None = None
        self._init_loop(max_iter)

    def step(self) -> None:
        """One iteration: resolve the step size on ``v``, data prox, denoiser, dual update."""
        gamma, sigma = self._resolve(self.v)
        step_size = gamma**2
        prev_v = self.v
        self.x = self._data_prox(torch.add(self.v, self.lagrange_mult, alpha=-step_size), gamma, sigma)
        self.v = self.prior.prox(
            torch.add(self.x, self.lagrange_mult, alpha=step_size), step_size * self.reg_strength**2
        )
        self.dual_residual = float(_norm(self.v - prev_v)) / step_size
        del prev_v
        primal_gap = self.x - self.v
        self.primal_residual = float(_norm(primal_gap))
        self.lagrange_mult.add_(primal_gap, alpha=1.0 / step_size)
        del primal_gap

    def record_statistics(self) -> None:
        """Append this iteration's record, including the primal and dual residuals."""
        super().record_statistics()
        self.history[-1].update({"primal_residual": self.primal_residual, "dual_residual": self.dual_residual})

    def exit_condition(self) -> bool:
        """Both the primal and the dual residual are below ``tolerance``."""
        return bool(self.primal_residual < self.tolerance and self.dual_residual < self.tolerance)

    def minimizer(self) -> torch.Tensor:
        """The denoiser output ``v``."""
        return self.v


__all__ = [
    "ADMM",
    "IterativeResult",
    "PnPPrior",
    "ProximalGradient",
]
