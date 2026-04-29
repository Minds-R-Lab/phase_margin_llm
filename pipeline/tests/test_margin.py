"""Unit tests for the phase-margin formula and regime classifier."""
from __future__ import annotations

import numpy as np
import pytest

from phase_margin.margin import directional_margin, compute_phase_margin, classify_regime
from phase_margin.types import DirectionalSpectrum, PhaseFit, Regime


# ---------------------------------------------------------------------------
# Helpers to build a DirectionalSpectrum with a constant phase across grid
# ---------------------------------------------------------------------------
def make_const_spectrum(name: str, theta: float, omegas=(0.5, 1.0, 1.5)):
    fits = [
        PhaseFit(omega=om, theta=theta, amplitude=1.0, residual=0.0,
                 n_samples=32, n_seeds=4)
        for om in omegas
    ]
    return DirectionalSpectrum(name=name, fits=fits)


# ---------------------------------------------------------------------------
# Eq. (6) reductions
# ---------------------------------------------------------------------------
def test_zero_phase_is_strongly_contractive():
    """An agent with phase identically 0 (and identity env) should give margin = pi."""
    a = make_const_spectrum("v", 0.0)
    margin = directional_margin(a, env_spec=None)
    assert margin == pytest.approx(np.pi, abs=1e-6)


def test_phase_at_minus_pi_over_two_gives_pi_over_two_margin():
    a = make_const_spectrum("v", -np.pi / 2)
    margin = directional_margin(a, env_spec=None)
    assert margin == pytest.approx(np.pi / 2, abs=1e-6)


def test_phase_at_minus_pi_gives_zero_margin():
    """Closed-loop pole on the unit circle -> margin 0 (oscillatory)."""
    a = make_const_spectrum("v", -np.pi)
    margin = directional_margin(a, env_spec=None)
    assert abs(margin) < 1e-6


def test_phase_beyond_minus_pi_gives_negative_margin():
    """Phase below -pi (sector edge crossed) -> negative margin."""
    a = make_const_spectrum("v", -np.pi - 0.3)
    margin = directional_margin(a, env_spec=None)
    assert margin < 0


# ---------------------------------------------------------------------------
# Regime classifier
# ---------------------------------------------------------------------------
def test_classify_contractive():
    margin, per = compute_phase_margin(
        agent_spectra={"v0": make_const_spectrum("v0", -0.3)},
    )
    regime = classify_regime(margin, per, margin_buffer=0.05)
    assert regime == Regime.CONTRACTIVE


def test_classify_oscillatory():
    margin, per = compute_phase_margin(
        agent_spectra={"v0": make_const_spectrum("v0", -np.pi + 0.02)},
    )
    regime = classify_regime(margin, per, margin_buffer=0.05)
    assert regime == Regime.OSCILLATORY


def test_classify_exploratory_when_strongly_negative_in_one_direction():
    margin, per = compute_phase_margin(
        agent_spectra={
            "v0": make_const_spectrum("v0", -0.2),
            "v1": make_const_spectrum("v1", -np.pi - 0.5),
        },
    )
    regime = classify_regime(margin, per, margin_buffer=0.05)
    assert regime == Regime.EXPLORATORY


def test_classify_unknown_when_empty():
    regime = classify_regime(float("nan"), {})
    assert regime == Regime.UNKNOWN
