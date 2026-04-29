"""phase_margin — black-box phase-margin certification of LLM agentic loops.

Public API:
    run_certification : end-to-end orchestrator
    PhaseMargin       : margin and regime classifier
    PhaseFit          : single-frequency identification result
    Regime            : enum of {CONTRACTIVE, OSCILLATORY, EXPLORATORY, UNKNOWN}
    SyntheticLTILoop  : ground-truth LTI shadow for validation
"""
from .types import (
    Regime,
    PhaseFit,
    DirectionalSpectrum,
    MarginReport,
    ProbeConfig,
)
from .identification import fit_phase_response, fit_directional_spectrum
from .margin import compute_phase_margin, classify_regime
from .pipeline import run_certification

# Loop / LLM imports are kept lazy so that the core package does not
# require sentence-transformers / anthropic / ollama just to import.

__all__ = [
    "Regime",
    "PhaseFit",
    "DirectionalSpectrum",
    "MarginReport",
    "ProbeConfig",
    "fit_phase_response",
    "fit_directional_spectrum",
    "compute_phase_margin",
    "classify_regime",
    "run_certification",
]

__version__ = "0.1.0"
