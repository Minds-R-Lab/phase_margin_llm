"""Behavioural phase identification (Eq. (5) of the paper).

Given a sequence of residuals r_k = <Delta y_k, b(v)> obtained by
injecting a sinusoidal probe at the agent input and measuring the
seed-mean output projection on direction b(v), the identifier fits

    r_k ~= epsilon * A * cos(omega * k + theta)

by **exact least-squares** on the (cos, -sin) regression basis.  This
recovers (A, theta) with no DFT-leakage error at any probe frequency,
so the identification math can be unit-tested against analytical LTI
ground truth to floating-point precision.
"""
from __future__ import annotations

import numpy as np

from .types import PhaseFit, DirectionalSpectrum


# ---------------------------------------------------------------------------
# Single-frequency fit: closed-form LSQ
# ---------------------------------------------------------------------------
def fit_phase_response(
    residuals: np.ndarray,
    omega: float,
    epsilon: float,
    n_seeds: int = 1,
) -> PhaseFit:
    """Fit a sinusoid r_k = epsilon * A * cos(omega*k + theta) to residuals.

    Parameters
    ----------
    residuals : (N,) real array
        Per-iteration projection of the seed-averaged response on the
        probing direction.
    omega : float
        Probe frequency in radians / iteration, in (0, pi).
    epsilon : float
        Probe amplitude.
    n_seeds : int, default 1
        Number of seeds averaged into ``residuals`` (bookkeeping only).

    Returns
    -------
    PhaseFit
        ``theta`` in (-pi, pi], ``amplitude`` >= 0, ``residual`` in [0, 1].

    Notes
    -----
    Decomposing  cos(omega*k + theta) = cos(theta) cos(omega*k)
                                       - sin(theta) sin(omega*k),
    we fit  r_k ~= a_c * cos(omega*k) + a_s * (-sin(omega*k))  by exact
    least-squares.  Then  A = sqrt(a_c**2 + a_s**2) / epsilon  and
    theta = atan2(a_s, a_c).  This is exact (up to float precision)
    for any omega -- there is no DFT leakage.
    """
    r = np.asarray(residuals, dtype=float)
    if r.ndim != 1:
        raise ValueError("residuals must be 1-D")
    N = int(r.size)
    if N < 2:
        raise ValueError("need at least 2 samples")

    k = np.arange(N, dtype=float)
    cos_b = np.cos(omega * k)
    sin_b = -np.sin(omega * k)
    X = np.column_stack([cos_b, sin_b])
    coeffs, _, _, _ = np.linalg.lstsq(X, r, rcond=None)
    a_c = float(coeffs[0])
    a_s = float(coeffs[1])

    A_amp = float(np.hypot(a_c, a_s) / max(float(epsilon), 1e-12))
    theta = float(np.arctan2(a_s, a_c))

    fit_curve = a_c * cos_b + a_s * sin_b
    energy_total = float(np.sum(r * r) + 1e-18)
    energy_resid = float(np.sum((r - fit_curve) ** 2))
    residual = float(min(1.0, max(0.0, energy_resid / energy_total)))

    return PhaseFit(
        omega=float(omega),
        theta=theta,
        amplitude=A_amp,
        residual=residual,
        n_samples=N,
        n_seeds=int(n_seeds),
    )


# ---------------------------------------------------------------------------
# Multi-frequency fit (one direction)
# ---------------------------------------------------------------------------
def fit_directional_spectrum(
    residual_grid: np.ndarray,
    omegas: np.ndarray,
    epsilon: float,
    direction_name: str = "v",
    n_seeds: int = 1,
) -> DirectionalSpectrum:
    """Fit the phase response on a frequency grid for ONE probing direction.

    Parameters
    ----------
    residual_grid : (n_omega, N) array
        Row j is the residual sequence at frequency ``omegas[j]``.
    omegas : (n_omega,) array
        Probe frequencies in rad/iter.
    epsilon : float
        Probe amplitude.
    direction_name : str, default "v"
        Tag carried into the result.
    n_seeds : int, default 1
        Bookkeeping; number of seeds averaged into each row.
    """
    grid = np.asarray(residual_grid)
    if grid.ndim != 2:
        raise ValueError("residual_grid must be 2-D (n_omega, N)")
    om = np.asarray(omegas, dtype=float)
    if grid.shape[0] != om.size:
        raise ValueError("residual_grid rows must match omegas length")
    fits = [
        fit_phase_response(
            residuals=grid[j],
            omega=float(om[j]),
            epsilon=float(epsilon),
            n_seeds=int(n_seeds),
        )
        for j in range(om.size)
    ]
    return DirectionalSpectrum(name=direction_name, fits=fits)


# ---------------------------------------------------------------------------
# Convenience: phase from an analytical LTI G(z) at given omegas
# (used by tests as a ground-truth baseline)
# ---------------------------------------------------------------------------
def lti_phase_at(G_z, omegas: np.ndarray) -> np.ndarray:
    """Evaluate arg(G(e^{j*omega})) for callable G_z over an omega grid.

    ``G_z`` should accept a complex z and return a complex scalar (or
    a complex matrix; in that case we take a scalar via tr(G(z))/d).
    """
    om = np.asarray(omegas, dtype=float)
    out = np.zeros(om.size, dtype=float)
    for j, w in enumerate(om):
        z = complex(np.cos(w), np.sin(w))
        v = G_z(z)
        if np.ndim(v) == 0:
            out[j] = float(np.angle(v))
        else:
            arr = np.asarray(v, dtype=complex)
            scalar = complex(np.trace(arr) / max(arr.shape[0], 1))
            out[j] = float(np.angle(scalar))
    return out
