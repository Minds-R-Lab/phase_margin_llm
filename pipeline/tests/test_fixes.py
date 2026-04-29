"""Regression tests for the four fixes derived from the 18-cell run.

Each test is named after the failure mode it locks down:

  Fix B - informativeness flag stops sector-default zero margins from
          being mis-classified as oscillatory.
  Fix C - outer probe seed actually changes the inner probe rollouts.
  Fix D - residual cap default of 0.5 is the public default.
  Fix A - pca_text_basis recovers the dominant direction of an
          anisotropic synthetic trajectory.
"""
from __future__ import annotations

import numpy as np
import pytest

from phase_margin.types import (
    DirectionalSpectrum, PhaseFit, ProbeConfig, Regime,
)
from phase_margin.margin import compute_phase_margin, classify_regime


# ---------- helpers ----------------------------------------------------------
def make_spectrum(name, theta, residual, omegas=(0.5, 1.0, 1.5)):
    fits = [
        PhaseFit(omega=om, theta=theta, amplitude=1.0,
                 residual=residual, n_samples=32, n_seeds=4)
        for om in omegas
    ]
    return DirectionalSpectrum(name=name, fits=fits)


# ---------- Fix B ------------------------------------------------------------
def test_fixB_informativeness_flag_default_uninformative():
    spec = make_spectrum("v0", theta=0.0, residual=0.99)  # all rejected
    assert spec.is_informative(residual_cap=0.5) is False
    assert spec.is_informative(residual_cap=0.999) is True


def test_fixB_uninformative_basis_yields_unknown_regime():
    """All directions have residuals above 0.5 -> compute_phase_margin
    must return NaN, classify_regime must return UNKNOWN."""
    bad = {f"v{i}": make_spectrum(f"v{i}", 0.0, residual=0.99) for i in range(3)}
    margin, per_dir = compute_phase_margin(bad, residual_cap=0.5)
    assert np.isnan(margin), f"expected NaN, got {margin}"
    regime = classify_regime(margin, per_dir)
    assert regime == Regime.UNKNOWN, (
        f"uninformative spectrum was misclassified as {regime}, "
        "exactly the bug we are fixing"
    )


def test_fixB_partial_informativeness_uses_only_good_directions():
    """If some directions are informative and some are not, the margin
    should be the worst over the INFORMATIVE ones, not over all."""
    specs = {
        "good": make_spectrum("good", theta=-0.1, residual=0.0),
        "bad":  make_spectrum("bad",  theta= 0.0, residual=0.99),
    }
    margin, per_dir = compute_phase_margin(specs, residual_cap=0.5)
    expected = np.pi - 0.1   # directional_margin of good = min(pi-(-0.1), -0.1+pi) = pi - 0.1
    assert margin == pytest.approx(expected, abs=1e-9)
    regime = classify_regime(margin, per_dir)
    assert regime == Regime.CONTRACTIVE


# ---------- Fix C ------------------------------------------------------------
def test_fixC_outer_probe_seed_changes_inner_seeds():
    """Two probe_seed_base values must produce different empirical phase
    spectra on a stochastic synthetic shadow (otherwise the outer seed
    never reaches the inner LLM rollouts)."""
    from phase_margin import run_certification
    from phase_margin.loops import SyntheticLTILoop
    from phase_margin.probe import random_vector_basis
    loop = SyntheticLTILoop.from_random(d=4, spectral_radius=0.6,
                                        seed=0, noise_std=0.01)
    basis = random_vector_basis(dim=4, n_directions=2, seed=0)
    cfg = ProbeConfig(horizon=24, n_seeds=2, n_seeds_nominal=2,
                      n_frequencies=4, epsilon=0.05, residual_cap=0.5)

    r0 = run_certification(loop, basis, cfg, use_text_perturbation=False,
                           progress=False, probe_seed_base=0)
    r1 = run_certification(loop, basis, cfg, use_text_perturbation=False,
                           progress=False, probe_seed_base=1)

    # The empirical thetas must differ between seeds (stochastic noise
    # inside the loop changes the response).
    same = []
    for name in r0.agent_spectra:
        a = r0.agent_spectra[name].thetas
        b = r1.agent_spectra[name].thetas
        same.append(np.allclose(a, b))
    assert not all(same), (
        "probe_seed_base did NOT change the empirical phase spectra; "
        "outer seed is not propagating to inner probe rollouts."
    )


# ---------- Fix D ------------------------------------------------------------
def test_fixD_default_residual_cap_is_strict():
    """The shipped default residual_cap on ProbeConfig must be tight
    (<= 0.5).  A loose default of 0.9 caused the first 18-cell run to
    accept noise as fits."""
    cfg = ProbeConfig()
    assert cfg.residual_cap <= 0.5, (
        f"ProbeConfig.residual_cap = {cfg.residual_cap}; tighten it: "
        "loose values let noise fits through and produce meaningless margins."
    )


# ---------- Fix A ------------------------------------------------------------
def test_fixA_pca_text_basis_recovers_dominant_direction():
    """Build an anisotropic 'trajectory' whose variance is concentrated on
    a known unit vector; pca_text_basis must return a top direction
    aligned with it."""
    from phase_margin.probe import pca_text_basis

    class FakeEmb:
        dim = 8
        def embed(self, texts):
            # Map each text to a random-but-deterministic vector
            rng = np.random.default_rng(42)
            return rng.standard_normal((len(texts), self.dim))

    rng = np.random.default_rng(0)
    d = 8
    e = np.zeros(d); e[2] = 1.0
    n = 30
    # 95 % of variance along e, 5 % iso noise
    traj = rng.standard_normal((n, d)) * 0.05
    traj += np.outer(rng.standard_normal(n), e)

    basis = pca_text_basis(
        trajectory=traj, embedder=FakeEmb(), n_directions=2, seed=0,
    )
    assert len(basis) >= 1
    # First PCA direction must align with e (up to sign)
    cos = abs(float(np.dot(basis[0].vector, e)))
    assert cos > 0.95, f"top PCA direction misaligned: cos={cos:.3f}"
