"""Unit tests for the complex-LSQ phase identifier."""
from __future__ import annotations

import numpy as np
import pytest

from phase_margin.identification import fit_phase_response, fit_directional_spectrum


# ---------------------------------------------------------------------------
# Generate clean sinusoid -> recover (A, theta) exactly
# ---------------------------------------------------------------------------
# Exclude omega = pi (Nyquist) because sin(pi*k) = 0 for integer k makes the
# (cos, -sin) basis rank-deficient there: only the cosine component is
# identifiable.  Phase is non-unique at exactly Nyquist by design.
@pytest.mark.parametrize("omega", [0.3, 0.7, 1.2, np.pi / 2, np.pi - 0.05])
@pytest.mark.parametrize("theta_true", [-2.0, -0.5, 0.0, 0.7, 2.4])
def test_clean_sinusoid_recovery(omega, theta_true):
    eps = 0.1
    A_true = 0.8
    N = 64
    k = np.arange(N)
    r = eps * A_true * np.cos(omega * k + theta_true)

    fit = fit_phase_response(r, omega=omega, epsilon=eps)
    # Phase wraps; compare modulo 2*pi
    err = np.angle(np.exp(1j * (fit.theta - theta_true)))
    assert abs(err) < 1e-9, f"theta error {err}"
    assert abs(fit.amplitude - A_true) < 1e-9
    assert fit.residual < 1e-12


# ---------------------------------------------------------------------------
# Robustness to noise
# ---------------------------------------------------------------------------
def test_phase_recovery_under_noise():
    rng = np.random.default_rng(0)
    omega = 0.7
    eps = 0.1
    A_true = 0.6
    theta_true = 1.1
    N = 256
    k = np.arange(N)
    noise = 0.02 * rng.standard_normal(N)
    r = eps * A_true * np.cos(omega * k + theta_true) + noise

    fit = fit_phase_response(r, omega=omega, epsilon=eps)
    err = np.angle(np.exp(1j * (fit.theta - theta_true)))
    assert abs(err) < 0.05  # tens of milliradians under 2% noise
    assert fit.amplitude == pytest.approx(A_true, rel=0.1)
    assert 0.0 <= fit.residual <= 0.3


def test_directional_spectrum_shape():
    eps = 0.1
    # Stay strictly inside (0, pi) to avoid Nyquist degeneracy
    omegas = np.linspace(0.1, np.pi - 0.05, 6)
    A_true = 0.5
    theta_true = 0.3
    N = 64
    k = np.arange(N)
    grid = np.array([
        eps * A_true * np.cos(om * k + theta_true) for om in omegas
    ])
    spec = fit_directional_spectrum(grid, omegas=omegas, epsilon=eps, direction_name="v")
    assert spec.name == "v"
    assert spec.omegas.shape == omegas.shape
    assert np.allclose(spec.amplitudes, A_true, atol=1e-9)
    # Phase recovered modulo 2 pi in every bin
    for fit in spec.fits:
        err = np.angle(np.exp(1j * (fit.theta - theta_true)))
        assert abs(err) < 1e-9
