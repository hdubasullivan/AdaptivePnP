"""The profiled likelihood, the IML estimate and the identifiable window (no data, no weights)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import gaussian_kernel

from adaptivepnp.estimator import (
    RHO_SEARCH_BOUNDS,
    estimate_sigma_gamma,
    identifiable_window_of,
    profiled_loglikelihood,
    rho_identifiable_window,
    tau_star,
)
from adaptivepnp.operators import Convolution, SuperResolution, fft_power


def _residual(op, gamma, sigma, seed):
    """A draw of the model behind the estimator: ``z = gamma A e + sigma w`` with white ``e`` and ``w``."""
    g = torch.Generator().manual_seed(seed)
    e = torch.randn(op.input_shape, generator=g, dtype=torch.float64)
    w = torch.randn(op.output_shape, generator=g, dtype=torch.float64)
    return gamma * op(e) + sigma * w


@pytest.mark.parametrize("which", [0, 1])
def test_estimator_recovers_sigma_and_gamma_from_the_generative_model(operators64, which):
    op = operators64[which]
    gamma, sigma = 0.08, 0.02
    residual = _residual(op, gamma, sigma, seed=2)
    uTz2, eig = fft_power(residual), op.power_spectrum(residual)
    s_hat, g_hat = estimate_sigma_gamma(uTz2, eig, rho_floor=None, rho_ceiling=None)
    assert s_hat == pytest.approx(sigma, rel=0.1) and g_hat == pytest.approx(gamma, rel=0.15)
    s_known, g_known = estimate_sigma_gamma(uTz2, eig, frozen_sigma=sigma, rho_floor=None, rho_ceiling=None)
    assert s_known == sigma and g_known == pytest.approx(gamma, rel=0.15)


def test_profiled_loglikelihood_peaks_near_the_true_ratio(operators64):
    op = operators64[0]
    gamma, sigma = 0.1, 0.03
    residual = _residual(op, gamma, sigma, seed=3)
    uTz2, eig = fft_power(residual), op.power_spectrum(residual)
    rhos = np.geomspace(0.1, 100, 200)
    ll = [float(profiled_loglikelihood(r, uTz2, eig)) for r in rhos]
    assert rhos[int(np.argmax(ll))] == pytest.approx(gamma / sigma, rel=0.2)


def test_identifiable_window_widens_with_tau_and_is_floor_only_when_sigma_is_known(kernel):
    shape = (1, 3, 64, 64)  # enough modes for tau = 0.1 to be attainable on the 32 x 32 low-resolution grid
    for op in (Convolution(kernel, input_shape=shape), SuperResolution(kernel, 2, input_shape=shape)):
        edges = [identifiable_window_of(op, tau) for tau in (0.1, 0.2, 0.5, 1.0)]
        los, his = [e[0] for e in edges], [e[1] for e in edges]
        assert los == sorted(los, reverse=True) and len(set(los)) == 4
        assert his == sorted(his) and all(h <= RHO_SEARCH_BOUNDS[1] for h in his)
        lo_known, hi_known = identifiable_window_of(op, 0.1, sigma_known=True)
        assert hi_known == pytest.approx(RHO_SEARCH_BOUNDS[1]) and lo_known > 0  # known sigma: a floor, no ceiling


def test_tau_star_is_the_smallest_tolerance_with_a_window(operators64):
    op = operators64[0]
    eig, n_modes = op.power_spectrum().flatten(), 3 * op.output_shape[-2] * op.output_shape[-1]
    t = tau_star(eig, n_modes)
    lo, hi = rho_identifiable_window(eig, n_modes, tau=1.5 * t)
    assert 0 < lo < hi
    with pytest.raises(ValueError, match="tau_star"):
        rho_identifiable_window(eig, n_modes, tau=0.5 * t)


def test_flat_spectrum_is_unidentifiable():
    with pytest.raises(ValueError, match="window is empty"):
        rho_identifiable_window(np.ones(100), 300, tau=0.1)
    lo, hi = rho_identifiable_window(np.ones(100), 300, tau=0.1, sigma_known=True)
    assert 0 < lo < hi


def test_pure_noise_pins_the_estimate_to_the_window_floor(operators64):
    op = operators64[0]
    residual = 0.01 * torch.randn(op.output_shape, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    lo, _ = identifiable_window_of(op, 0.1)
    with pytest.warns(UserWarning, match="bound-limited"):
        s, g = estimate_sigma_gamma(fft_power(residual), op.power_spectrum(residual), tau=0.1, verbose=True)
    assert g / s == pytest.approx(lo, rel=1e-3)


def test_operator_computes_the_window_once_and_reuses_it(monkeypatch):
    """The window depends only on the operator and tau, so the operator caches it across residuals."""
    import adaptivepnp.operators as operators_module

    op = SuperResolution(gaussian_kernel().double(), 2, input_shape=(1, 3, 64, 64), input_dtype=torch.float64)
    calls = []
    original = operators_module.rho_identifiable_window

    def counting(*args, **kwargs):
        calls.append(kwargs.get("sigma_known", False))
        return original(*args, **kwargs)

    monkeypatch.setattr(operators_module, "rho_identifiable_window", counting)
    residuals = [_residual(op, 0.05, 0.02, seed) for seed in (5, 6, 7)]
    cached = [op.estimate_sigma_gamma(r, tau=0.1) for r in residuals]
    assert calls == [False]  # one window for three estimates
    direct = [estimate_sigma_gamma(fft_power(r), op.power_spectrum(r), tau=0.1) for r in residuals]
    assert cached == direct  # the same window, hence the same estimates to the bit
    op.estimate_sigma_gamma(residuals[0], tau=0.1, frozen_sigma=0.02)
    assert calls == [False, True]  # the known-sigma window is its own cache entry
    op.estimate_sigma_gamma(residuals[0], tau=0.1, rho_floor=None, rho_ceiling=None)
    assert calls == [False, True]  # an unconstrained search needs no window


def test_estimator_validates_its_inputs(operators64):
    op = operators64[0]
    residual = _residual(op, 0.05, 0.02, seed=8)
    uTz2, eig = fft_power(residual), op.power_spectrum(residual)
    with pytest.raises(ValueError, match="whole copies"):
        estimate_sigma_gamma(uTz2[..., :10, :], eig)
    with pytest.raises(ValueError, match="cr_bound"):
        estimate_sigma_gamma(uTz2, eig, rho_floor="op_norm")
    with pytest.raises(ValueError, match="rho_search_bounds"):
        estimate_sigma_gamma(uTz2, eig, rho_search_bounds=(1.0, 0.1))
