"""Empirical regime detection from observed long trajectories.

Given a long rollout of a closed loop, classify the observed dynamics
as contractive, oscillatory, or exploratory.  This is the *ground
truth* against which the phase-margin predictor is evaluated.

Heuristic rules (deliberately simple, calibratable per experiment):

- **Contractive.** The trajectory's variance falls below a threshold by
  the second half: var(z[N//2:]) <= variance_floor.  Equivalently, the
  trajectory settles to a stable point (or near one).
- **Oscillatory.** Autocorrelation of the trajectory at some lag p in
  [2, max_period] exceeds a threshold AND the variance does not decay
  to zero.  A 2-period limit cycle (the Wang et al. attractor)
  manifests as high autocorrelation at lag 2.
- **Exploratory.** The trajectory's distance from the origin grows
  faster than a threshold rate, or the running variance keeps
  increasing through the second half.

These rules are conservative; in ambiguous cases ``Regime.UNKNOWN`` is
returned.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import Regime


@dataclass
class GroundTruthResult:
    regime: Regime
    final_variance: float
    period_score: float  # peak autocorr at integer lag in [2, max_period]
    period_lag: int      # lag achieving period_score
    growth_rate: float   # slope of log||z_k||


def detect_regime(
    trajectory: np.ndarray,
    *,
    variance_floor: float = 1e-3,
    period_min: int = 2,
    period_max: int = 8,
    autocorr_threshold: float = 0.85,
    growth_threshold: float = 0.05,
) -> GroundTruthResult:
    """Classify a single trajectory ``trajectory`` of shape (T, d).

    ``T`` should be at least 2 * period_max + 4 for the autocorrelation
    estimate to be meaningful.
    """
    Z = np.asarray(trajectory, dtype=float)
    T, d = Z.shape

    # Centre by the long-run mean (the candidate fixed point)
    centroid = Z.mean(axis=0)
    Zc = Z - centroid

    # Variance in the second half of the trajectory
    second = Zc[T // 2 :]
    final_variance = float(np.mean(np.sum(second * second, axis=1)))

    # Per-step norm growth
    norms = np.linalg.norm(Zc, axis=1) + 1e-12
    log_norms = np.log(norms)
    if T >= 4:
        # Slope via least-squares against time index
        ts = np.arange(T, dtype=float)
        slope = float(np.polyfit(ts, log_norms, 1)[0])
    else:
        slope = 0.0
    growth_rate = slope

    # Autocorrelation across lags
    period_score = -np.inf
    period_lag = 0
    if T >= 2 * period_max + 4:
        max_lag = min(period_max, T - 2)
        for lag in range(period_min, max_lag + 1):
            num = np.sum(Zc[lag:] * Zc[:-lag])
            den = np.sqrt(
                np.sum(Zc[lag:] ** 2) * np.sum(Zc[:-lag] ** 2) + 1e-12
            )
            corr = float(num / den) if den > 0 else 0.0
            if corr > period_score:
                period_score, period_lag = corr, lag

    # ---------- decision tree ---------------------------------------------
    # Exploratory: positive growth rate
    if growth_rate > growth_threshold:
        regime = Regime.EXPLORATORY
    # Contractive: variance has collapsed
    elif final_variance <= variance_floor:
        regime = Regime.CONTRACTIVE
    # Oscillatory: high autocorr AND variance has not collapsed
    elif period_score >= autocorr_threshold and final_variance > variance_floor:
        regime = Regime.OSCILLATORY
    # Otherwise: ambiguous, default contractive if variance modest
    elif final_variance < 10 * variance_floor:
        regime = Regime.CONTRACTIVE
    else:
        regime = Regime.UNKNOWN

    return GroundTruthResult(
        regime=regime,
        final_variance=final_variance,
        period_score=float(period_score) if np.isfinite(period_score) else 0.0,
        period_lag=int(period_lag),
        growth_rate=float(growth_rate),
    )


# ---------------------------------------------------------------------------
# Soft-cycle detector for natural-language loops
# ---------------------------------------------------------------------------
def detect_period_by_within_across(
    trajectory: np.ndarray,
    period: int = 2,
    purity_threshold: float = 1.15,
):
    """Return (regime, purity) for a period-`period` attractor cycle.

    `purity` = (mean cross-class distance) / (mean within-class distance).
    A clean p-cycle in semantic space gives purity > 1.  Empirically a
    threshold of ~1.15 separates Wang et al.-style soft 2-cycles from
    contractive paraphrasing on frontier LLMs.
    """
    Z = np.asarray(trajectory, dtype=float)
    classes = [Z[i::period] for i in range(period)]
    within = []
    for C in classes:
        if len(C) < 2:
            continue
        D = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=-1)
        within.append(D[np.triu_indices(len(C), k=1)].mean())
    if not within:
        return Regime.UNKNOWN, 0.0
    cross = []
    for i in range(period):
        for j in range(i + 1, period):
            D = np.linalg.norm(classes[i][:, None, :] - classes[j][None, :, :], axis=-1)
            cross.append(D.mean())
    if not cross:
        return Regime.UNKNOWN, 0.0
    purity = float(np.mean(cross) / max(np.mean(within), 1e-12))
    if purity > purity_threshold:
        return Regime.OSCILLATORY, purity
    if purity < 1.0 / purity_threshold:
        return Regime.CONTRACTIVE, purity
    return Regime.UNKNOWN, purity
