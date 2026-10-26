"""The solvers, the step-size policies and the ``pnp_recon`` entry point, with a stand-in denoiser."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from adaptivepnp.data_fidelity import SquaredL2
from adaptivepnp.pnp import make_initial_guess, pnp_recon
from adaptivepnp.solvers import ADMM, PnPPrior, ProximalGradient
from adaptivepnp.step_size import AdaptiveStepSizePolicy, FixedStepSizePolicy


def _measurement(op, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(op.input_shape, generator=g)
    return x, op(x) + 0.02 * torch.randn(op.output_shape, generator=g)


@pytest.mark.parametrize("algorithm", ["pgm", "admm"])
@pytest.mark.parametrize("adaptive", [True, False])
def test_pnp_recon_runs_all_k_iterations_and_records_curves(operators, shrink, algorithm, adaptive):
    op = operators[0]
    _, y = _measurement(op, seed=7)
    kw = dict(beta=0.5) if adaptive else dict(gamma=0.03, sigma=0.02, beta=None)
    final, recon, iterates, curves = pnp_recon(
        y, make_initial_guess(op, y).clone(), op, shrink, algorithm, K=4, record_iterates=True, tau=0.5, **kw
    )
    assert recon.shape == tuple(op.input_shape) and np.isfinite(recon).all()
    assert len(iterates) == 5 and final["iter"] == 4
    assert len(curves["sigma"]) == len(curves["gamma"]) == 5
    # Iterate 0 ran no iteration, denoiser or data prox; it holds the policy's bootstrap estimate only when adaptive.
    stages = ("iteration", "ml_estimate", "denoise", "prox")
    assert all(math.isnan(curves[f"ms_{s}"][0]) for s in ("iteration", "denoise", "prox"))
    assert math.isnan(curves["ms_ml_estimate"][0]) != adaptive
    assert all(len(curves[f"ms_{s}"]) == 5 for s in stages)
    assert all(math.isfinite(t) for s in stages for t in curves[f"ms_{s}"][1:])
    # An iteration contains its estimate, its one denoiser call and its data prox, so its time bounds their sum.
    assert all(
        curves["ms_iteration"][k] >= curves["ms_ml_estimate"][k] + curves["ms_denoise"][k] + curves["ms_prox"][k]
        for k in range(1, 5)
    )
    if adaptive:
        assert all(s > 0 for s in curves["sigma"]) and len(set(curves["gamma"])) > 1
    else:
        assert set(curves["gamma"]) == {0.03} and set(curves["sigma"]) == {0.02}


def test_freeze_estimated_sigma_holds_the_first_estimate(operators, shrink):
    op = operators[0]
    _, y = _measurement(op, seed=8)
    x0 = make_initial_guess(op, y).clone()
    _, _, _, curves = pnp_recon(
        y, x0, op, shrink, "pgm", K=3, record_iterates=True, freeze_estimated_sigma=True, tau=0.5, rho_window="none"
    )
    assert len(set(curves["sigma"])) == 1
    with pytest.raises(ValueError, match="freeze_estimated_sigma"):
        pnp_recon(y, x0, op, shrink, "pgm", K=1, freeze_estimated_sigma=True, sigma=0.02)
    with pytest.raises(ValueError, match="rho_window"):
        pnp_recon(y, x0, op, shrink, "pgm", K=1, rho_window="op_norm")
    with pytest.raises(ValueError, match="pnp_type"):
        pnp_recon(y, x0, op, shrink, "fista", K=1)
    with pytest.raises(ValueError, match="sigma must be provided"):
        pnp_recon(y, x0, op, shrink, "pgm", K=1, gamma=0.03)


def test_policies_validate_their_arguments():
    with pytest.raises(ValueError):
        FixedStepSizePolicy(gamma=0.0, sigma=0.1)
    with pytest.raises(ValueError):
        AdaptiveStepSizePolicy(gamma_smoothing=1.5)
    with pytest.raises(ValueError):
        AdaptiveStepSizePolicy(frozen_sigma=0.1, freeze_estimated_sigma=True)
    with pytest.raises(ValueError):
        AdaptiveStepSizePolicy(tau=0.0)


def test_fixed_policy_returns_its_pair_and_logs_every_call():
    policy = FixedStepSizePolicy(gamma=0.03, sigma=0.02)
    for k in (1, 2):
        result = policy.get_step_size(None, None, None, k)
        assert (result.sigma, result.gamma, result.estimate_time) == (0.02, 0.03, 0.0)
    assert policy.history["gamma"] == [0.03] * 3 and policy.prev_gamma == 0.03  # the construction entry plus two
    policy.reset()
    assert policy.history["gamma"] == [0.03]


def test_solvers_accept_a_scalar_step_size(operators, shrink):
    op = operators[0]
    _, y = _measurement(op, seed=9)
    f = SquaredL2(y, op)
    for cls in (ProximalGradient, ADMM):
        result = cls(f, PnPPrior(shrink), y.clone(), step_size=0.05, max_iter=2, tolerance=0).solve()
        assert result.num_iter == 2 and result.reason == "max_iter" and torch.isfinite(result.x).all()
        assert [h["iter"] for h in result.history] == [1, 2]
    with pytest.raises(ValueError, match="positive"):
        ProximalGradient(f, PnPPrior(shrink), y.clone(), step_size=0.0)


def test_initial_guesses(operators):
    conv, sr = operators
    _, y = _measurement(sr, seed=10)
    torch.testing.assert_close(make_initial_guess(sr, y, "adjoint"), sr.adj(y))
    assert make_initial_guess(conv, y, "measurement") is y
    assert make_initial_guess(sr, y, "bicubic").shape == tuple(sr.input_shape)
    with pytest.raises(ValueError, match="initial_guess"):
        make_initial_guess(sr, y, "zeros")
