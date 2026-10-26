"""IML estimation of ``(sigma, gamma)`` from a measurement residual (Algorithm 1 of the paper)."""

from __future__ import annotations

import warnings

import numpy as np
import torch
from scipy.optimize import brentq, minimize_scalar

Array = torch.Tensor | np.ndarray

#: Outer bounds of every search over ``rho = gamma / sigma``.
RHO_SEARCH_BOUNDS: tuple[float, float] = (1e-6, 1e6)
#: Default tolerance on the normalized Cramér–Rao bounds that define the identifiable window.
TAU_DEFAULT = 0.05


def _as_backend(value: float, reference: Array) -> Array:
    """Convert ``value`` to a 0-d array/tensor matching ``reference``'s backend, dtype, and device.

    Args:
        value: Scalar value to convert.
        reference: Tensor or array to match backend, dtype, and device.

    Returns:
        0-d tensor/array matching ``reference``'s backend, dtype, and device.

    """
    if isinstance(reference, torch.Tensor):
        return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    return np.asarray(value, dtype=reference.dtype)


def _numel(x: Array) -> int:
    """Number of elements of a tensor or array.

    Args:
        x: Tensor or array.

    Returns:
        Number of elements.

    """
    return x.numel() if isinstance(x, torch.Tensor) else x.size


def _log(x: Array) -> Array:
    """Elementwise natural logarithm in ``x``'s backend."""
    return torch.log(x) if isinstance(x, torch.Tensor) else np.log(x)


def _clamp_min0(x: Array) -> Array:
    """Clip ``x`` below at zero in ``x``'s backend.

    Args:
        x: Tensor or array.

    Returns:
        Clipped array.

    """
    return x.clamp_min(0.0) if isinstance(x, torch.Tensor) else np.clip(x, 0.0, None)


def profiled_loglikelihood(rho: float, uTz2: Array, eigvals: Array, sigma: float | None = None) -> Array:
    """Log-likelihood of the residual at ``rho = gamma / sigma``, up to an additive constant.

    With ``sigma=None`` sigma is profiled out at its own MLE given rho, i.e. ``-L(rho) / 2`` with ``L`` the
    profiled negative log-likelihood of the paper. With ``sigma`` given, the exact log-likelihood at that
    fixed sigma is returned instead (the ``frozen_sigma`` case).

    Args:
        rho: Scalar ratio ``gamma / sigma``.
        uTz2: Squared residual coefficients in the diagonalizing basis (:func:`adaptivepnp.operators.fft_power`).
            Leading dimensions beyond ``eigvals.shape`` are independent copies sharing the spectrum
            (the RGB channels).
        eigvals: Eigenvalues of ``AA^T`` in that basis.
        sigma: Fixed noise level, or ``None`` to profile it out.

    Returns:
        A 0-d tensor/array in ``uTz2``'s backend.

    """
    rho = _as_backend(rho, uTz2)
    M = _numel(uTz2)  # modes of z~ over every channel
    n_copies = M // _numel(eigvals)  # channels sharing one spectrum

    # C_rho = rho^2 Lambda + I, the diagonal for one channel.
    C_rho = rho**2 * eigvals + 1.0
    # z~^T C_rho^{-1} z~ and log det C_rho, both over all M modes.
    quad = (uTz2 / C_rho).sum()
    logdet = n_copies * _log(C_rho).sum()

    if sigma is None:
        # Profiled: L(rho; z~) = M log(z~^T C_rho^{-1} z~) + log det C_rho; return -L / 2.
        return -0.5 * (M * _log(quad) + logdet)
    # Fixed sigma: -log p = (M / 2) log sigma^2 + (1 / 2) log det C_rho + z~^T C_rho^{-1} z~ / (2 sigma^2).
    sigma2 = _as_backend(float(sigma), uTz2) ** 2
    return -0.5 * (M * _log(sigma2) + logdet + quad / sigma2)


def _to_numpy_1d(x: Array) -> np.ndarray:
    """Flatten a torch or numpy array to a float64 numpy vector on the CPU.

    Args:
        x: Tensor or array.

    Returns:
        1-d float64 numpy array on the CPU.

    """
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).ravel()


def normalized_crb(rho: float, eigvals: Array, n_modes: int, sigma_known: bool = False) -> tuple[float, float]:
    """Diagonal of the normalized Cramér–Rao bound ``V(theta)`` at the ratio ``rho``.

    Args:
        rho: Ratio ``gamma / sigma`` at which to evaluate the bounds.
        eigvals: Eigenvalues of ``AA^T``.
        n_modes: Total number of real Gaussian modes, ``n_copies * numel(eigvals)``.
        sigma_known: Use the known-sigma bound on ``gamma^2`` alone.

    Returns:
        ``(V_sigma, V_gamma)``; both ``+inf`` when the spectrum is flat.

    """
    lam = _to_numpy_1d(eigvals)
    t = rho**2 * lam
    q = t / (1.0 + t)
    x = 1.0 / (1.0 + t)
    if sigma_known:
        mq2 = float(np.mean(q**2))
        return 0.0, (np.inf if mq2 <= 0.0 else 2.0 / (n_modes * mq2))
    mq, mx = float(q.mean()), float(x.mean())

    v = float(q.var()) if mq <= mx else float(x.var())
    if v <= 0.0:
        return np.inf, np.inf
    base = 2.0 / n_modes
    return base * (1.0 + mq**2 / v), base * (1.0 + mx**2 / v)


def tau_star(
    eigvals: Array,
    n_modes: int,
    rho_bounds: tuple[float, float] = RHO_SEARCH_BOUNDS,
    grid_size: int = 400,
    sigma_known: bool = False,
) -> float:
    """Smallest tolerance for which the identifiable window is non-empty.

    Args:
        eigvals: Eigenvalues of ``AA^T``.
        n_modes: Total number of real Gaussian modes.
        rho_bounds: Range of rho scanned.
        grid_size: Number of log-spaced scan points.
        sigma_known: Use the known-sigma bound on ``gamma^2`` alone.

    Returns:
        ``tau_star``.

    """
    lam = _to_numpy_1d(eigvals)
    grid = np.geomspace(rho_bounds[0], rho_bounds[1], grid_size)
    worst = np.fromiter(
        (max(normalized_crb(r, lam, n_modes, sigma_known)) for r in grid), dtype=np.float64, count=grid_size
    )
    return float(np.sqrt(worst.min()))


def rho_identifiable_window(
    eigvals: Array,
    n_modes: int,
    tau: float = TAU_DEFAULT,
    rho_bounds: tuple[float, float] = RHO_SEARCH_BOUNDS,
    grid_size: int = 400,
    sigma_known: bool = False,
) -> tuple[float, float]:
    """The identifiable window ``W_tau = {rho : V_sigma(rho) <= tau² and V_gamma(rho) <= tau²}``.

    Args:
        eigvals: Eigenvalues of ``AA^T`` (the squared singular values of ``A``).
        n_modes: Total number of real Gaussian modes, ``n_copies * numel(eigvals)``.
        tau: Tolerance on the normalized Cramér–Rao bounds of ``sigma^2`` and ``gamma^2``.
        rho_bounds: Range of rho to scan.
        grid_size: Number of log-spaced grid points used to bracket the edges.
        sigma_known: Use the known-sigma bound on ``gamma^2`` alone.

    Returns:
        ``(rho_lo, rho_hi)``: the window clipped to ``rho_bounds``.

    Raises:
        ValueError: If no rho in ``rho_bounds`` meets the tolerance; the message reports ``tau_star``.

    """
    lam = _to_numpy_1d(eigvals)
    if n_modes < 1:
        raise ValueError(f"n_modes must be positive, got {n_modes}.")
    lo, hi = rho_bounds
    if not 0.0 < lo < hi:
        raise ValueError(f"rho_bounds must satisfy 0 < lo < hi, got {rho_bounds}.")
    if not tau > 0.0:
        raise ValueError(f"tau must be positive, got {tau}.")
    tau2 = tau**2
    big = 1e300  # finite stand-in for +inf so the root finder never sees non-finite values

    def v_sigma(r: float) -> float:
        return min(normalized_crb(r, lam, n_modes, sigma_known)[0], big)

    def v_gamma(r: float) -> float:
        return min(normalized_crb(r, lam, n_modes, sigma_known)[1], big)

    grid = np.geomspace(lo, hi, grid_size)
    vs = np.fromiter((v_sigma(r) for r in grid), dtype=np.float64, count=grid_size)
    vg = np.fromiter((v_gamma(r) for r in grid), dtype=np.float64, count=grid_size)
    ok_gamma = np.flatnonzero(vg <= tau2)  # a suffix of the grid
    ok_sigma = np.flatnonzero(vs <= tau2)  # a prefix of the grid

    def _empty(reason: str) -> ValueError:
        kind = "known-sigma" if sigma_known else "joint (sigma^2, gamma^2)"
        smallest = float(np.sqrt(np.max(np.stack([vs, vg]), axis=0).min()))
        return ValueError(
            f"Identifiable window is empty for tau={tau} on rho_bounds={rho_bounds} ({kind} normalized CR "
            f"bound; {reason}). The smallest tolerance with a non-empty window on this scan is "
            f"tau_star = {smallest:.4g}. Raise tau or widen rho_bounds."
        )

    if ok_gamma.size == 0:
        raise _empty("the gamma^2-constraint fails for every rho scanned")
    if ok_sigma.size == 0:
        raise _empty("the sigma^2-constraint fails for every rho scanned")
    i0, i1 = int(ok_gamma[0]), int(ok_sigma[-1])
    if i0 > i1 + 1:
        raise _empty("the two constraints hold on disjoint rho ranges")
    rho_lo = grid[i0] if i0 == 0 else brentq(lambda r: v_gamma(r) - tau2, grid[i0 - 1], grid[i0])
    rho_hi = grid[i1] if i1 == grid_size - 1 else brentq(lambda r: v_sigma(r) - tau2, grid[i1], grid[i1 + 1])
    if not rho_lo <= rho_hi:
        raise _empty("the two constraints hold on disjoint rho ranges")
    return float(rho_lo), float(rho_hi)


def estimate_sigma_gamma(
    uTz2: Array,
    eigvals: Array,
    rho_floor: str | None = "cr_bound",
    rho_ceiling: str | None = "cr_bound",
    tau: float = TAU_DEFAULT,
    frozen_sigma: float | None = None,
    rho_search_bounds: tuple[float, float] = RHO_SEARCH_BOUNDS,
    window: tuple[float, float] | None = None,
    verbose: bool = False,
) -> tuple[float, float]:
    """Estimate ``(sigma, gamma)`` by maximizing :func:`profiled_loglikelihood` over ``rho = gamma / sigma``.

    Args:
        uTz2: Squared residual coefficients in the basis diagonalizing ``AA^T``
            (:func:`adaptivepnp.operators.fft_power`). Leading dimensions beyond ``eigvals.shape`` are
            independent copies sharing the spectrum.
        eigvals: Eigenvalues of ``AA^T`` in that basis.
        rho_floor: ``'cr_bound'`` clamps the search below at the lower edge of the identifiable window (the
            ``gamma^2`` constraint); ``None`` leaves the search bound.
        rho_ceiling: ``'cr_bound'`` clamps the search above at the window's upper edge (the ``sigma^2``
            constraint; only exists when sigma is estimated jointly); ``None`` leaves it.
        tau: Tolerance on the normalized Cramér–Rao bounds of ``sigma^2``-hat and ``gamma^2``-hat.
        frozen_sigma: Hold sigma at this value instead of profiling it out; the window then uses the
            known-sigma bound.
        rho_search_bounds: Outer bounds of the search; the window only ever tightens them.
        window: The identifiable window ``(rho_lo, rho_hi)`` of this spectrum, as returned by
            :func:`rho_identifiable_window` for the same ``tau``, ``rho_search_bounds`` and known-sigma case;
            when given, the window is not recomputed. It depends only on the operator and ``tau``, so
            :meth:`adaptivepnp.operators.LinearOperator.identifiable_window` computes it once per operator and
            passes it here at every iteration. ``None`` computes it from ``eigvals``.
        verbose: Warn when the estimate lands on an edge of the search interval (bound-limited).

    Returns:
        ``(sigma_hat, gamma_hat)``.

    Raises:
        ValueError: If ``uTz2`` is not whole copies of the spectrum, a bound is not ``'cr_bound'`` or
            ``None``, or the resolved search interval is empty.

    """
    n_spatial = _numel(eigvals)
    n_total = _numel(uTz2)
    if n_total % n_spatial != 0:
        raise ValueError(
            f"numel(uTz2) = {n_total} is not a multiple of numel(eigvals) = {n_spatial}; "
            "uTz2 must consist of whole copies of the spectrum."
        )
    n_modes = n_total

    search_lo, search_hi = (float(b) for b in rho_search_bounds)
    if not 0.0 < search_lo < search_hi:
        raise ValueError(f"rho_search_bounds must satisfy 0 < lo < hi, got {rho_search_bounds}.")
    for name, spec in (("rho_floor", rho_floor), ("rho_ceiling", rho_ceiling)):
        if spec not in (None, "cr_bound"):
            raise ValueError(f"{name} must be 'cr_bound' or None, got {spec!r}.")

    sigma_known = frozen_sigma is not None
    if rho_floor == "cr_bound" or rho_ceiling == "cr_bound":
        if window is None:
            window = rho_identifiable_window(
                eigvals, n_modes, tau=tau, rho_bounds=(search_lo, search_hi), sigma_known=sigma_known
            )
        if rho_floor == "cr_bound":
            search_lo = max(search_lo, window[0])
        # With sigma known there is no sigma^2-constraint, hence no upper edge.
        if rho_ceiling == "cr_bound" and not sigma_known:
            search_hi = min(search_hi, window[1])
    if not search_lo < search_hi:
        raise ValueError(
            f"Resolved rho interval is empty: floor {search_lo:.4g} >= ceiling {search_hi:.4g}. "
            "Widen rho_search_bounds, raise tau, or relax rho_floor / rho_ceiling."
        )

    def neg_ll(r: float) -> float:
        return -float(profiled_loglikelihood(r, uTz2, eigvals, sigma=frozen_sigma).item())

    res = minimize_scalar(neg_ll, bounds=(search_lo, search_hi), method="bounded")
    rho_hat = float(res.x)

    rel_tol = 1e-3
    if rho_hat <= search_lo * (1 + rel_tol) or rho_hat >= search_hi * (1 - rel_tol):
        if verbose:
            warnings.warn(
                f"rho_hat = {rho_hat:.4g} lies on the edge of the search interval "
                f"[{search_lo:.4g}, {search_hi:.4g}]; the estimate is bound-limited."
            )

    with torch.no_grad():
        if sigma_known:
            sigma_hat = _as_backend(float(frozen_sigma), uTz2)
        else:
            denom = rho_hat**2 * eigvals + 1.0
            sigma2_hat = (uTz2 / denom).sum() / n_modes
            sigma_hat = _clamp_min0(sigma2_hat) ** 0.5
        gamma_hat = sigma_hat * rho_hat
    sigma_out = float(sigma_hat.item()) if hasattr(sigma_hat, "item") else float(sigma_hat)
    gamma_out = float(gamma_hat.item()) if hasattr(gamma_hat, "item") else float(gamma_hat)
    return sigma_out, gamma_out


def identifiable_window_of(operator, tau: float, sigma_known: bool = False) -> tuple[float, float]:
    """The window :func:`rho_identifiable_window` clamps the search to, for ``operator``'s own spectrum.

    The operator computes it once and caches it
    (:meth:`adaptivepnp.operators.LinearOperator.identifiable_window`).

    Args:
        operator: A :class:`adaptivepnp.operators.LinearOperator` with a ``power_spectrum``.
        tau: Tolerance on the normalized Cramér–Rao bounds.
        sigma_known: Use the known-sigma bound on ``gamma^2`` alone.

    Returns:
        ``(rho_lo, rho_hi)``.

    """
    return operator.identifiable_window(tau, sigma_known=sigma_known)


__all__ = [
    "RHO_SEARCH_BOUNDS",
    "TAU_DEFAULT",
    "estimate_sigma_gamma",
    "identifiable_window_of",
    "normalized_crb",
    "profiled_loglikelihood",
    "rho_identifiable_window",
    "tau_star",
]
