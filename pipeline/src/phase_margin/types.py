"""Dataclass types and enums."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import numpy as np


class Regime(str, Enum):
    CONTRACTIVE = "contractive"
    OSCILLATORY = "oscillatory"
    EXPLORATORY = "exploratory"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PhaseFit:
    omega: float
    theta: float
    amplitude: float
    residual: float
    n_samples: int
    n_seeds: int


@dataclass
class DirectionalSpectrum:
    name: str
    fits: list = field(default_factory=list)

    @property
    def omegas(self):
        return np.array([f.omega for f in self.fits])

    @property
    def thetas(self):
        return np.array([f.theta for f in self.fits])

    @property
    def amplitudes(self):
        return np.array([f.amplitude for f in self.fits])

    @property
    def residuals(self):
        return np.array([f.residual for f in self.fits])

    def sector(self, residual_cap: float = 0.5):
        if not self.fits:
            return (-np.pi, np.pi)
        keep = self.residuals <= residual_cap
        if not keep.any():
            return (-np.pi, np.pi)
        good = self.thetas[keep]
        return float(good.min()), float(good.max())

    def is_informative(self, residual_cap: float = 0.5) -> bool:
        if not self.fits:
            return False
        return bool((self.residuals <= residual_cap).any())


@dataclass
class ProbeConfig:
    horizon: int = 32
    n_seeds: int = 8
    n_seeds_nominal: int = 4
    n_frequencies: int = 8
    omega_min: float = 0.0
    omega_max: float = np.pi
    epsilon: float = 0.1
    residual_cap: float = 0.5
    margin_buffer: float = 0.05

    def grid(self):
        k = np.arange(1, self.n_frequencies + 1)
        return self.omega_min + (self.omega_max - self.omega_min) * k / self.n_frequencies


@dataclass
class MarginReport:
    phase_margin: float
    regime: Regime
    per_direction_margin: dict
    agent_spectra: dict
    env_spectra: dict
    config: ProbeConfig
    notes: str = ""

    def summary(self) -> str:
        lines = [
            f"phase margin = {self.phase_margin:+.3f} rad",
            f"predicted regime: {self.regime.value}",
            "per-direction margins:",
        ]
        for name, m in self.per_direction_margin.items():
            lines.append(f"  {name:24s}  Phi_v = {m:+.3f}")
        if self.notes:
            lines.append(f"notes: {self.notes}")
        return "\n".join(lines)
