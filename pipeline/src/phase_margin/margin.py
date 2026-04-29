"""Phase-margin computation and regime classification.

Implements:
- Eq. (6) of the paper:  Phi_margin(v) = min(pi - (beta_v + delta_v),
                                              (alpha_v + gamma_v) + pi)
- Corollary 6.2:  contractive / oscillatory / exploratory predicate.

Both functions take ``DirectionalSpectrum`` objects and return scalars,
with no LLM-specific assumptions, so they can be tested directly against
the synthetic LTI shadow.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from .types import DirectionalSpectrum, MarginReport, ProbeConfig, Regime


# ---------------------------------------------------------------------------
# Per-direction margin
# ---------------------------------------------------------------------------
def directional_margin(
    agent_spec: DirectionalSpectrum,
    env_spec: DirectionalSpectrum | None,
    residual_cap: float = 0.5,
) -> float:
    """Eq. (6): phase margin in one probing direction.

    Parameters
    ----------
    agent_spec, env_spec : DirectionalSpectrum
        The agent's and the environment's spectra in the same direction.
        ``env_spec`` may be ``None`` (treated as the identity), modelling
        an autonomous closed loop where the environment is the LLM
        feeding back to itself.
    """
    alpha, beta = agent_spec.sector(residual_cap)
    if env_spec is None:
        gamma, delta = 0.0, 0.0
    else:
        gamma, delta = env_spec.sector(residual_cap)
    upper = math.pi - (beta + delta)
    lower = (alpha + gamma) + math.pi
    return float(min(upper, lower))


# ---------------------------------------------------------------------------
# Aggregate global margin and regime classifier
# ---------------------------------------------------------------------------
def compute_phase_margin(
    agent_spectra: Mapping[str, DirectionalSpectrum],
    env_spectra: Mapping[str, DirectionalSpectrum] | None = None,
    residual_cap: float = 0.5,
) -> tuple[float, dict[str, float]]:
    """Aggregate Phi_margin across the probing basis (worst direction).

    Margins are computed only over INFORMATIVE directional spectra (those
    in which at least one probe frequency had a fit residual below cap).
    Non-informative directions are excluded from the min so that a
    direction whose probe never landed on a sinusoid does not dominate
    the headline margin with a default-zero artefact.
    """
    if not agent_spectra:
        return float("nan"), {}
    env_spectra = env_spectra or {}
    per_dir: dict[str, float] = {}
    informative: dict[str, bool] = {}
    for name, a_spec in agent_spectra.items():
        e_spec = env_spectra.get(name)
        per_dir[name] = directional_margin(a_spec, e_spec, residual_cap=residual_cap)
        informative[name] = a_spec.is_informative(residual_cap)
    info_only = {k: v for k, v in per_dir.items() if informative[k]}
    if not info_only:
        # No direction produced a usable phase fit.
        return float("nan"), per_dir
    margin = float(min(info_only.values()))
    return margin, per_dir


def classify_regime(
    phase_margin: float,
    per_direction_margin: Mapping[str, float],
    margin_buffer: float = 0.05,
) -> Regime:
    """Corollary 6.2 regime classifier, hardened against fit-failure defaults.

    A NaN aggregate margin is the signal that ``compute_phase_margin``
    refused to publish a number (no informative direction).  In that case
    we return UNKNOWN -- the certificate has no information, and an
    accidental margin of 0 from a sector default must NOT be silently
    promoted to OSCILLATORY.

    Otherwise:
        margin > +buffer    -> CONTRACTIVE
        margin in (-buffer, +buffer)  -> OSCILLATORY (limit-cycle warning)
        margin < -buffer or any direction strongly negative -> EXPLORATORY
    """
    if math.isnan(phase_margin):
        return Regime.UNKNOWN
    if not per_direction_margin:
        return Regime.UNKNOWN
    worst_dir_margin = min(per_direction_margin.values())
    if worst_dir_margin < -2 * margin_buffer:
        return Regime.EXPLORATORY
    if phase_margin > margin_buffer:
        return Regime.CONTRACTIVE
    if phase_margin < -margin_buffer:
        return Regime.EXPLORATORY
    return Regime.OSCILLATORY


# ---------------------------------------------------------------------------
# Build a MarginReport from spectra (used by the pipeline)
# ---------------------------------------------------------------------------
def build_report(
    agent_spectra: Mapping[str, DirectionalSpectrum],
    env_spectra: Mapping[str, DirectionalSpectrum] | None,
    config: ProbeConfig,
    notes: str = "",
) -> MarginReport:
    margin, per_dir = compute_phase_margin(
        agent_spectra=agent_spectra,
        env_spectra=env_spectra,
        residual_cap=config.residual_cap,
    )
    regime = classify_regime(
        phase_margin=margin,
        per_direction_margin=per_dir,
        margin_buffer=config.margin_buffer,
    )
    return MarginReport(
        phase_margin=margin,
        regime=regime,
        per_direction_margin=per_dir,
        agent_spectra=dict(agent_spectra),
        env_spectra=dict(env_spectra or {}),
        config=config,
        notes=notes,
    )
