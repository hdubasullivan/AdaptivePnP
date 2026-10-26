"""The closed-form proximal map of the squared-L2 data term."""

from __future__ import annotations

import pytest
import torch

from adaptivepnp.data_fidelity import SquaredL2
from adaptivepnp.operators import Subsampling


def _problem(op, seed=6):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(op.input_shape, generator=g, dtype=torch.float64)
    y = op(x) + 0.02 * torch.randn(op.output_shape, generator=g, dtype=torch.float64)
    v = x + 0.1 * torch.randn(op.input_shape, generator=g, dtype=torch.float64)
    return x, y, v


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("scale", [0.5, 1.0])
def test_prox_satisfies_the_normal_equations(operators64, which, scale):
    """``(A^T A + lam I) x = A^T y + lam v`` with ``lam = sigma^2 / (2 scale gamma^2)``, for both operators."""
    op = operators64[which]
    x, y, v = _problem(op)
    f = SquaredL2(y, op, scale=scale)
    gamma, sigma = 0.05, 0.02
    out = f.prox(v, gamma=gamma, sigma=sigma)
    lam = sigma**2 / (2 * scale * gamma**2)
    torch.testing.assert_close(op.adj(op(out)) + lam * out, op.adj(y) + lam * v, rtol=1e-8, atol=1e-8)
    assert f.last_prox_info["num_iter"] == 0
    assert f(x) == pytest.approx(scale * float(((y - op(x)) ** 2).sum()))


@pytest.mark.parametrize("which", [0, 1])
def test_prox_tends_to_the_identity_for_a_tiny_step(operators64, which):
    op = operators64[which]
    _, y, v = _problem(op)
    out = SquaredL2(y, op).prox(v, gamma=1e-6, sigma=0.02)
    torch.testing.assert_close(out, v, rtol=0, atol=1e-6)


def test_validation(operators64):
    conv, _ = operators64
    _, y, v = _problem(conv)
    f = SquaredL2(y, conv)
    with pytest.raises(ValueError, match="scale"):
        SquaredL2(y, conv, scale=0.0)
    with pytest.raises(ValueError, match="gamma"):
        f.prox(v, gamma=0.0)
    with pytest.raises(ValueError, match="sigma"):
        f.prox(v, gamma=1.0, sigma=0.0)
    with pytest.raises(TypeError, match="closed-form"):
        SquaredL2(y, Subsampling(2, input_shape=conv.input_shape, input_dtype=torch.float64))
