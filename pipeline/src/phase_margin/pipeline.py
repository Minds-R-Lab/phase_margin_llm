"""End-to-end orchestration.

``run_certification`` implements Algorithm 1 of the paper for a given
loop and probe basis, and returns a ``MarginReport``.  It is the single
entry point used by the CLI and the validation notebooks.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
from tqdm.auto import tqdm

from .identification import fit_directional_spectrum
from .loops import AgentLoop
from .margin import build_report
from .probe import ProbeDirection
from .types import DirectionalSpectrum, MarginReport, ProbeConfig, Regime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _embed_step(loop: AgentLoop, ptext: str, pvec, seed: Optional[int]) -> np.ndarray:
    return loop.step(perturbation_text=ptext, perturbation_vector=pvec, seed=seed)


def _nominal_trajectory(loop: AgentLoop, config: ProbeConfig, seed_base: int = 1000) -> np.ndarray:
    """Average M_0 nominal rollouts of length N."""
    runs = np.zeros((config.n_seeds_nominal, config.horizon, loop.dim), dtype=float)
    for s in range(config.n_seeds_nominal):
        loop.reset(seed=seed_base + s)
        for k in range(config.horizon):
            runs[s, k] = _embed_step(loop, "", None, seed=seed_base + s)
    return runs.mean(axis=0)  # (N, d)


def _probe_trajectory(
    loop: AgentLoop,
    direction: ProbeDirection,
    omega: float,
    config: ProbeConfig,
    *,
    use_text: bool,
    seed_base: int = 0,
) -> np.ndarray:
    """Average M perturbed rollouts of length N at frequency omega."""
    v = direction.ensure_vector(loop.embedder if not direction.vector is not None else None)
    runs = np.zeros((config.n_seeds, config.horizon, loop.dim), dtype=float)
    for s in range(config.n_seeds):
        loop.reset(seed=seed_base + s)
        for k in range(config.horizon):
            strength = config.epsilon * float(np.cos(omega * k))
            ptext = direction.text_for_strength(strength) if use_text else ""
            pvec = (strength * v) if not use_text else None
            runs[s, k] = _embed_step(loop, ptext, pvec, seed=seed_base + s)
    return runs.mean(axis=0)  # (N, d)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_certification(
    loop: AgentLoop,
    basis: Sequence[ProbeDirection],
    config: ProbeConfig | None = None,
    *,
    use_text_perturbation: bool = False,
    progress: bool = True,
    probe_seed_base: int = 0,
) -> MarginReport:
    """Run Algorithm 1 of the paper.

    Parameters
    ----------
    loop : AgentLoop
        The closed loop under test.
    basis : sequence of ProbeDirection
        The probing basis.  For text loops, supply directions with
        ``pos_modifier``/``neg_modifier`` so that ``ensure_vector(emb)``
        can compute the embedding-space representation; for vector
        loops, supply directions with ``vector``.
    config : ProbeConfig, optional
        Probe protocol parameters.
    use_text_perturbation : bool
        If True, the loop is fed the textual modifier; if False, the
        loop is fed the vector ``strength * direction.vector``.  The
        default (False) is appropriate for the synthetic LTI loop;
        text loops should pass True.
    """
    config = config or ProbeConfig()
    omegas = config.grid()

    # Resolve direction vectors (lazily) once
    for d in basis:
        d.ensure_vector(loop.embedder)

    # ----- nominal trajectory --------------------------------------------
    z_star = _nominal_trajectory(loop, config)  # (N, d)

    # ----- probe each direction × frequency ------------------------------
    agent_spectra: dict[str, DirectionalSpectrum] = {}
    iter_dirs = tqdm(basis, desc="directions", leave=False) if progress else basis
    for d in iter_dirs:
        residual_grid = np.zeros((omegas.size, config.horizon), dtype=float)
        iter_om = tqdm(
            list(enumerate(omegas)),
            desc=f"  {d.name}",
            leave=False,
        ) if progress else list(enumerate(omegas))
        for j, omega in iter_om:
            z_bar = _probe_trajectory(
                loop=loop,
                direction=d,
                omega=float(omega),
                config=config,
                use_text=use_text_perturbation,
                seed_base=int((hash((d.name, j)) & 0xFFFF) ^ (int(probe_seed_base) * 1009)),
            )
            delta_z = z_bar - z_star  # (N, d)
            v = d.vector
            residual_grid[j] = delta_z @ v  # (N,)
        spectrum = fit_directional_spectrum(
            residual_grid=residual_grid,
            omegas=omegas,
            epsilon=config.epsilon,
            direction_name=d.name,
            n_seeds=config.n_seeds,
        )
        agent_spectra[d.name] = spectrum

    # No environment block here: the "loop" is autonomous (closed by
    # construction).  The Theorem 6.1 environment ε contributes phase
    # only if there is a separate identifiable environment block; for
    # the autonomous loops in this pipeline we set it to identity.
    return build_report(
        agent_spectra=agent_spectra,
        env_spectra=None,
        config=config,
        notes=f"loop={type(loop).__name__}, dim={loop.dim}",
    )
