"""The reconstruction entry point: plug-and-play PGM or ADMM with a fixed or adaptive step size."""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from typing import Any

import torch
from torch.nn import functional as F

from adaptivepnp.data_fidelity import SquaredL2
from adaptivepnp.estimator import TAU_DEFAULT
from adaptivepnp.operators import LinearOperator
from adaptivepnp.solvers import ADMM, PnPPrior, ProximalGradient
from adaptivepnp.step_size import AdaptiveStepSizePolicy, FixedStepSizePolicy, StepSizePolicy

_ALGORITHMS = {"pgm": ProximalGradient, "admm": ADMM}

#: How ``rho_window`` constrains the ML search over ``rho = gamma / sigma``.
WINDOW = {
    "cr_bound": dict(rho_floor="cr_bound", rho_ceiling="cr_bound"),  # the identifiable window (the method)
    "none": dict(rho_floor=None, rho_ceiling=None),  # unconstrained ML (ablation)
}


def make_initial_guess(operator: LinearOperator, measurement: torch.Tensor, mode: str = "adjoint") -> torch.Tensor:
    """An image-grid initial iterate.

    Args:
        operator: Forward operator.
        measurement: ``y``.
        mode: ``'adjoint'`` (``A^T y``, the default), ``'measurement'`` (``y`` itself; only when it lives on
            the image grid) or ``'bicubic'`` (bicubic upsample to the image grid).

    Returns:
        The initial iterate.

    Raises:
        ValueError: For any other ``mode``.

    """
    if mode == "measurement":
        return measurement
    if mode == "adjoint":
        return operator.adj(measurement)
    if mode == "bicubic":
        return F.interpolate(measurement, size=operator.input_shape[-2:], mode="bicubic", align_corners=False)
    raise ValueError(f"Unknown initial_guess {mode!r}; choose adjoint, measurement, or bicubic.")


class _StageTimer:
    """Milliseconds of wall time per iteration and per stage: the estimate, one DRUNet evaluation, the data prox.

    Args:
        device: The device the solve runs on, or ``None``.

    """

    STAGES = ("iteration", "ml_estimate", "denoise", "prox")

    def __init__(self, device: Any) -> None:
        kind = torch.device(device).type if device is not None else "cpu"
        self.on_cuda = kind == "cuda"
        self._sync = getattr(getattr(torch, "mps", None), "synchronize", None) if kind == "mps" else None
        self.samples: dict[str, list[float]] = {stage: [] for stage in self.STAGES}
        self._events: dict[str, list[tuple[Any, Any]]] = {stage: [] for stage in self.STAGES}
        self._marks: list[dict[str, int]] = []

    @contextmanager
    def stage(self, name: str):
        """Context manager timing one call of ``name``, in milliseconds."""
        if self.on_cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self._events[name].append((start, end))
        else:
            if self._sync is not None:
                self._sync()
            started = time.perf_counter()
            try:
                yield
            finally:
                if self._sync is not None:
                    self._sync()
                self.samples[name].append(1e3 * (time.perf_counter() - started))

    def wrap(self, owner: Any, method: str, stage: str) -> None:
        """Time ``owner.method`` in place, attributing it to ``stage``.

        Args:
            owner: Object whose method is wrapped.
            method: Name of the method.
            stage: One of ``STAGES``.

        """
        original = getattr(owner, method, None)
        if original is None:
            return

        def timed(*args: Any, **kwargs: Any):
            with self.stage(stage):
                return original(*args, **kwargs)

        try:
            setattr(owner, method, timed)
        except (AttributeError, TypeError):
            pass

    def _count(self, stage: str) -> int:
        """Number of calls of ``stage`` timed so far."""
        return len(self._events[stage]) if self.on_cuda else len(self.samples[stage])

    def mark(self) -> None:
        """Close the current iterate: every call timed since the previous mark belongs to it."""
        self._marks.append({stage: self._count(stage) for stage in self.STAGES})

    def curves(self) -> dict[str, list[float]]:
        """``ms_*`` curves, one entry per mark: the milliseconds of one call of that stage in the iterate, NaN if none.

        Every solver here makes one call of each stage per iteration; should an iterate make several, the
        entry is their mean, so it always reads as the cost of one call.

        Returns:
            ``{'ms_iteration': [...], 'ms_ml_estimate': [...], 'ms_denoise': [...], 'ms_prox': [...]}``.

        """
        if self.on_cuda:
            torch.cuda.synchronize()
            for name, pairs in self._events.items():
                self.samples[name] = [start.elapsed_time(end) for start, end in pairs]  # already in ms
        out = {}
        for stage in self.STAGES:
            samples, curve, start = self.samples[stage], [], 0
            for mark in self._marks:
                stop = mark[stage]
                curve.append(sum(samples[start:stop]) / (stop - start) if stop > start else math.nan)
                start = stop
            out[f"ms_{stage}"] = curve
        return out


def run_solver(
    solver_cls: type,
    data_fidelity: SquaredL2,
    prior: PnPPrior,
    policy: StepSizePolicy,
    x0: torch.Tensor,
    K: int,
    record_iterates: bool = False,
    **solver_kwargs: Any,
) -> tuple[dict[str, Any], Any, list[torch.Tensor] | None, dict[str, list[float]] | None]:
    """Run ``solver_cls`` for exactly ``K`` iterations from ``x0`` and package the outcome.

    This is the tail of :func:`pnp_recon`: the per-iteration
    timing, the solve under ``torch.no_grad``, the recorded iterates and the policy's logs.

    Args:
        solver_cls: A solver class of :mod:`adaptivepnp.solvers`.
        data_fidelity: The data term.
        prior: The plug-and-play prior.
        policy: The step-size policy driving the solver.
        x0: Initial iterate on the image grid.
        K: Iteration budget (always run in full).
        record_iterates: Keep every iterate (on CPU) and the per-iteration policy logs and timing curves.
        **solver_kwargs: Extra constructor arguments of ``solver_cls`` (for example ``reg_strength``).

    Returns:
        ``(final_history_row, reconstruction as a numpy array, iterates or None, curves or None)``. The
        curves hold every log of ``policy.history`` (``sigma`` and ``gamma``, plus any the policy adds)
        and the ``ms_*`` curves of :class:`_StageTimer`: the milliseconds of each iteration, of its
        parameter estimate, of its DRUNet evaluation and of its data prox.

    Raises:
        RuntimeError: If the solve recorded no iterations.

    """
    timer = _StageTimer(getattr(data_fidelity.y, "device", None)) if record_iterates else None
    if timer is not None:
        timer.wrap(policy, "get_step_size", "ml_estimate")
        timer.wrap(prior, "prox", "denoise")  # ``grad_step`` calls ``prox``; wrapping both would count it twice
        timer.wrap(data_fidelity, "prox", "prox")

    with torch.no_grad():
        solver = solver_cls(
            data_fidelity=data_fidelity,
            prior=prior,
            x0=x0,
            step_size=policy,
            max_iter=K,
            tolerance=0,
            **solver_kwargs,
        )
        if record_iterates:
            iterates: list[torch.Tensor] | None = [x0.detach().cpu()]
            timer.wrap(solver, "step", "iteration")  # the whole of every iteration from here on
            timer.mark()  # iterate 0: the policy's bootstrap estimate, if any

            def record_iterate(optimizer) -> None:
                iterates.append(optimizer.minimizer().detach().cpu())
                timer.mark()

            result = solver.solve(callback=record_iterate)
        else:
            iterates = None
            result = solver.solve()

    recon_volume = result.x.detach().cpu().numpy()
    if not result.history:
        raise RuntimeError("PnP history is empty.")
    final_metrics = dict(result.history[-1])

    stepsize_history = None
    if record_iterates:
        stepsize_history = {
            **{key: list(values) for key, values in policy.history.items()},
            **(timer.curves() if timer is not None else {}),
        }
    return final_metrics, recon_volume, iterates, stepsize_history


def pnp_recon(
    y: torch.Tensor,
    x0: torch.Tensor,
    A: LinearOperator,
    denoiser: Any,
    pnp_type: str,
    *,
    K: int = 10,
    gamma: float | None = None,
    sigma: float | None = None,
    beta: float | None = 1.0,
    freeze_estimated_sigma: bool = False,
    rho_window: str = "cr_bound",
    tau: float = TAU_DEFAULT,
    record_iterates: bool = False,
) -> tuple[dict[str, Any], Any, list[torch.Tensor] | None, dict[str, list[float]] | None]:
    """Run plug-and-play ``pnp_type`` (``'pgm'`` or ``'admm'``) for ``K`` iterations from ``x0``.

    The step size is fixed when ``gamma`` is given (then ``sigma`` must be the noise level) and
    adaptive otherwise. In the adaptive case, sigma reaches the policy one of three ways: ``sigma``
    given pins it to that known value (oracle); ``sigma=None`` re-estimates it every iteration jointly
    with gamma; ``freeze_estimated_sigma`` estimates it once at iteration 0 and holds it.

    Args:
        y: Measurement ``(1, C, h, w)``.
        x0: Initial iterate on the image grid ``(1, C, H, W)``.
        A: Forward operator (``Convolution`` or ``SuperResolution``).
        denoiser: The DRUNet, called as ``denoiser(x, sigma=...)``.
        pnp_type: ``'pgm'`` or ``'admm'``.
        K: Iteration budget (always run in full).
        gamma: Fixed data-fidelity parameter, or ``None`` to estimate it.
        sigma: Known noise level for the fixed policy / oracle adaptive policy, or ``None``.
        beta: EMA weight on the adaptive estimate (1 = raw estimate, 0 = frozen at iteration 0).
        freeze_estimated_sigma: Adaptive policy: estimate sigma once, then hold it.
        rho_window: ``'cr_bound'`` (search inside the identifiable window) or ``'none'``.
        tau: Tolerance on the normalized Cramér–Rao bounds defining the identifiable window (Theorem 1).
        record_iterates: Keep every iterate (on CPU) and the per-iteration curves: ``sigma``, ``gamma`` and the
            ``ms_*`` timings of :func:`run_solver`.

    Returns:
        ``(final_history_row, reconstruction as a numpy array, iterates or None, curves or None)``.

    Raises:
        ValueError: For an unknown ``pnp_type`` or ``rho_window``, a fixed ``gamma`` without ``sigma``,
            or ``freeze_estimated_sigma`` together with ``sigma``.
        RuntimeError: If the solve recorded no iterations.

    """
    if pnp_type not in _ALGORITHMS:
        raise ValueError(f"pnp_type must be one of {sorted(_ALGORITHMS)}, got {pnp_type!r}.")
    if rho_window not in WINDOW:
        raise ValueError(f"rho_window must be one of {sorted(WINDOW)}, got {rho_window!r}.")
    if freeze_estimated_sigma and sigma is not None:
        raise ValueError("freeze_estimated_sigma requires sigma=None: sigma is estimated, not supplied.")

    data_fidelity = SquaredL2(y=y, A=A)

    if gamma is not None:
        if sigma is None:
            raise ValueError("sigma must be provided when gamma is specified.")
        policy = FixedStepSizePolicy(gamma=gamma, sigma=sigma)
    else:
        policy = AdaptiveStepSizePolicy(
            gamma_smoothing=beta,
            frozen_sigma=sigma,
            freeze_estimated_sigma=freeze_estimated_sigma,
            tau=tau,
            **WINDOW[rho_window],
        )

    return run_solver(_ALGORITHMS[pnp_type], data_fidelity, PnPPrior(denoiser), policy, x0, K, record_iterates)


__all__ = ["WINDOW", "make_initial_guess", "pnp_recon", "run_solver"]
