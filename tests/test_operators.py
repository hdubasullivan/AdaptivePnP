"""The forward operators: adjointness, the FFT diagonalization of ``A A^T``, and input validation."""

from __future__ import annotations

import pytest
import torch

from adaptivepnp.operators import SuperResolution, fft_power


@pytest.mark.parametrize("which", [0, 1])
def test_adjointness(operators64, which):
    """``<A x, y> = <x, A^T y>`` for the blur and for blur-then-decimate."""
    op = operators64[which]
    g = torch.Generator().manual_seed(0)
    x = torch.randn(op.input_shape, generator=g, dtype=torch.float64)
    y = torch.randn(op.output_shape, generator=g, dtype=torch.float64)
    lhs = torch.vdot(op(x).flatten(), y.flatten()).item()
    rhs = torch.vdot(x.flatten(), op.adj(y).flatten()).item()
    assert lhs == pytest.approx(rhs, rel=1e-10)


@pytest.mark.parametrize("which", [0, 1])
def test_power_spectrum_diagonalizes_a_a_adjoint(operators64, which):
    """``A A^T`` acts as multiplication by ``power_spectrum`` in the measurement-space FFT basis."""
    op = operators64[which]
    y = torch.randn(op.output_shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    lhs = torch.fft.fft2(op(op.adj(y)))
    rhs = op.power_spectrum(y) * torch.fft.fft2(y)
    torch.testing.assert_close(lhs, rhs, rtol=1e-9, atol=1e-9)


def test_super_resolution_spectrum_is_the_alias_average(operators):
    conv, sr = operators
    full = conv.power_spectrum(torch.zeros(1, 3, 32, 32))
    folded = sr.power_spectrum(torch.zeros(1, 3, 16, 16))
    assert folded.shape == (16, 16)
    assert folded.sum().item() == pytest.approx(full.sum().item() / 4.0, rel=1e-5)


def test_fft_power_is_normalized_like_the_residual():
    """``fft_power`` returns ``|F z|^2 / (H W)``, so it sums to ``||z||^2`` (Parseval)."""
    z = torch.randn(1, 3, 16, 24, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    assert fft_power(z).sum().item() == pytest.approx((z**2).sum().item(), rel=1e-10)


def test_kernel_is_returned_as_a_copy(operators):
    conv, _ = operators
    conv.kernel.zero_()
    assert conv.kernel.sum().item() == pytest.approx(1.0, rel=1e-6)


def test_operators_reject_wrong_dtype_and_shape(operators, kernel):
    conv, _ = operators
    with pytest.raises(ValueError, match="Dtype"):
        conv(torch.zeros(1, 3, 32, 32, dtype=torch.float64))
    with pytest.raises(ValueError, match="shape"):
        conv(torch.zeros(1, 3, 16, 16))
    with pytest.raises(ValueError, match="divisible"):
        SuperResolution(kernel, 3, input_shape=(1, 3, 32, 32))
