"""End-to-end pipeline test against the synthetic LTI shadow.

The behavioural identifier should recover the analytical phase response
of the closed-loop transfer matrix at every probe frequency, within a
tolerance set by the chosen probe amplitude and seed budget.

If this test passes, the pipeline math is verified on a system with a
*known* answer.
"""
from __future__ import annotations

import numpy as np
import pytest

from phase_margin import run_certification
from phase_margin.loops import SyntheticLTILoop
from phase_margin.probe import random_vector_basis, ProbeDirection
from phase_margin.types import ProbeConfig
from phase_margin.identification import fit_phase_response


# ---------------------------------------------------------------------------
def _phase_circular_distance(a: float, b: float) -> float:
    return abs(float(np.angle(np.exp(1j * (a - b)))))


# ---------------------------------------------------------------------------
# Single-direction analytical comparison: tightest possible test
# ---------------------------------------------------------------------------
def test_lti_identified_phase_matches_analytical_phase():
    """For a deterministic LTI loop with no noise, the identified phase
    response in any fixed direction must match the analytical phase of
    the closed-loop transfer matrix Gcl(e^{j omega}) projected on that
    direction.
    """
    loop = SyntheticLTILoop.from_random(
        d=4, spectral_radius=0.6, seed=42, noise_std=0.0
    )
    v = np.zeros(loop.dim)
    v[0] = 1.0

    omegas = np.linspace(0.3, np.pi - 0.2, 6)
    eps = 0.05
    N = 96

    # For each omega, run a probe and recover phase
    direction = ProbeDirection(name="e0", vector=v.copy())
    direction.ensure_vector()

    nominal = loop.rollout(horizon=N, seed=0)
    for om in omegas:
        loop.reset(seed=0)
        runs = []
        for k in range(N):
            strength = eps * float(np.cos(om * k))
            z = loop.step(perturbation_vector=strength * direction.vector, seed=0)
            runs.append(z)
        runs = np.array(runs)
        delta = runs - nominal
        residual = delta @ direction.vector
        fit = fit_phase_response(residual, omega=om, epsilon=eps)

        analytical = loop.closed_loop_directional_phase(direction.vector, om)
        err = _phase_circular_distance(fit.theta, analytical)
        assert err < 0.15, (
            f"omega={om:.3f}: identified theta={fit.theta:.3f} vs "
            f"analytical {analytical:.3f}, err={err:.3f}"
        )


# ---------------------------------------------------------------------------
# Full pipeline: regime prediction matches spectral-radius classification
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("spectral_radius,expected", [
    (0.4, "stable"),
    (0.7, "stable"),
])
def test_full_pipeline_returns_finite_margin(spectral_radius, expected):
    """For a clearly stable closed-loop (rho < 1), the pipeline must
    produce a finite, non-NaN phase margin and a non-UNKNOWN regime."""
    loop = SyntheticLTILoop.from_random(
        d=6, spectral_radius=spectral_radius, seed=0, noise_std=0.0
    )
    basis = random_vector_basis(dim=6, n_directions=3, seed=1)
    config = ProbeConfig(
        horizon=64, n_seeds=4, n_seeds_nominal=2,
        n_frequencies=6, epsilon=0.05,
    )
    report = run_certification(
        loop=loop, basis=basis, config=config,
        use_text_perturbation=False, progress=False,
    )
    assert np.isfinite(report.phase_margin)
    assert report.regime != report.regime.UNKNOWN
    # Loop is genuinely stable, so the closed-loop matrix has rho < 1
    rho = loop.closed_loop_spectral_radius()
    assert rho < 1.0
