"""Synthetic / mock LLM clients.

These are not LLMs at all; they are controllable dynamical systems used
to validate the pipeline math.  In particular ``SyntheticLTIClient``
implements

    x_{k+1} = A x_k + B u_k + xi_k
    y_k    = C x_k

where ``u_k`` is a vector parsed from the user message ("comma separated
floats"), the response is the comma-separated floats of ``y_k``, and the
internal state is hidden.  Because this system is LTI we can compute
the analytical phase response of G(z) = C (zI - A)^{-1} B at any
frequency, which the tests use as ground truth for the identifier.

``NonlinearShadowClient`` adds a bounded saturating nonlinearity to make
the test more realistic without losing the ability to compute a local
Jacobian-based ground truth.

``EchoClient`` is a trivial debugging client.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .base import LLMClient, Message


def _parse_vector(text: str, d: int) -> np.ndarray:
    parts = [p for p in text.replace("\n", ",").split(",") if p.strip()]
    vec = np.zeros(d, dtype=float)
    for i, p in enumerate(parts[:d]):
        try:
            vec[i] = float(p.strip())
        except ValueError:
            vec[i] = 0.0
    return vec


def _format_vector(v: np.ndarray) -> str:
    return ", ".join(f"{x:.6g}" for x in np.asarray(v).ravel())


# ---------------------------------------------------------------------------
# Trivial echo client
# ---------------------------------------------------------------------------
class EchoClient(LLMClient):
    """Returns the last user message verbatim (sanity check)."""

    @property
    def name(self) -> str:
        return "mock:echo"

    def chat(
        self,
        messages: Sequence[Message],
        *,
        seed: int | None = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> str:
        for m in reversed(messages):
            if m.role == "user":
                return m.content
        return ""


# ---------------------------------------------------------------------------
# Synthetic LTI shadow
# ---------------------------------------------------------------------------
@dataclass
class SyntheticLTIClient(LLMClient):
    """Linear time-invariant 'agent' for ground-truth validation.

    Reads a vector u_k from the last user message (comma-separated
    floats), updates internal state, and writes y_k as a string.

    The transfer matrix is G(z) = C (zI - A)^{-1} B.  Tests use the
    analytical phase response of G(e^{j omega}) as ground truth for the
    behavioural identifier.
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    noise_std: float = 0.0
    state: np.ndarray = field(init=False)
    _rng: np.random.Generator = field(init=False, default_factory=lambda: np.random.default_rng(0))

    def __post_init__(self):
        self.A = np.asarray(self.A, dtype=float)
        self.B = np.asarray(self.B, dtype=float)
        self.C = np.asarray(self.C, dtype=float)
        self.state = np.zeros(self.A.shape[0], dtype=float)

    # ----- helpers -----------------------------------------------------
    @classmethod
    def from_random(
        cls,
        d: int = 8,
        spectral_radius: float = 0.7,
        seed: int = 0,
        noise_std: float = 0.0,
    ) -> "SyntheticLTIClient":
        """Build an LTI shadow whose CLOSED-LOOP matrix A + B C has the
        requested spectral radius.  We pick B = C = I and set
        A = M - I where M is a random matrix scaled to spectral radius
        ``spectral_radius``.  This makes the closed loop stable iff
        ``spectral_radius < 1``.
        """
        rng = np.random.default_rng(seed)
        M = rng.standard_normal((d, d)) / math.sqrt(d)
        eig = np.max(np.abs(np.linalg.eigvals(M)))
        if eig > 0:
            M *= spectral_radius / eig
        I_d = np.eye(d)
        A = M - I_d            # so A + B C = M with target spectral radius
        B = I_d
        C = I_d
        obj = cls(A=A, B=B, C=C, noise_std=noise_std)
        obj._rng = np.random.default_rng(seed + 1)
        return obj

    def reset(self, seed: int | None = None) -> None:
        self.state = np.zeros_like(self.state)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return f"mock:lti(d={self.A.shape[0]})"

    @property
    def dim(self) -> int:
        return int(self.A.shape[0])

    # ----- analytic phase response -------------------------------------
    def transfer(self, z: complex) -> np.ndarray:
        """Evaluate the transfer matrix G(z) at z."""
        I = np.eye(self.A.shape[0])
        return self.C @ np.linalg.solve(z * I - self.A, self.B)

    def transfer_at_omega(self, omega: float) -> np.ndarray:
        return self.transfer(np.exp(1j * float(omega)))

    def directional_phase(self, v: np.ndarray, omega: float) -> float:
        """arg ⟨v, G(e^{j omega}) v⟩ — the directional phase along v."""
        v = np.asarray(v, dtype=float).ravel()
        v = v / (np.linalg.norm(v) + 1e-12)
        Gz = self.transfer_at_omega(omega)
        return float(np.angle(v @ Gz @ v))

    # ----- LLMClient interface -----------------------------------------
    def chat(
        self,
        messages: Sequence[Message],
        *,
        seed: int | None = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> str:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        for m in reversed(messages):
            if m.role == "user":
                u = _parse_vector(m.content, self.A.shape[0])
                break
        else:
            u = np.zeros(self.A.shape[0])

        # State update
        noise = self.noise_std * self._rng.standard_normal(self.state.shape)
        self.state = self.A @ self.state + self.B @ u + noise
        y = self.C @ self.state
        return _format_vector(y)


# ---------------------------------------------------------------------------
# Nonlinear shadow with bounded saturation
# ---------------------------------------------------------------------------
@dataclass
class NonlinearShadowClient(SyntheticLTIClient):
    """LTI dynamics with a saturating output nonlinearity.

    y_k = tanh(C x_k * gain) / gain

    Around the origin this collapses to the LTI shadow; far from the
    origin it saturates, producing limit-cycle / multi-attractor
    behaviour that mirrors real LLM loops more faithfully.
    """

    gain: float = 1.5

    def chat(
        self,
        messages: Sequence[Message],
        *,
        seed: int | None = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> str:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        for m in reversed(messages):
            if m.role == "user":
                u = _parse_vector(m.content, self.A.shape[0])
                break
        else:
            u = np.zeros(self.A.shape[0])
        noise = self.noise_std * self._rng.standard_normal(self.state.shape)
        self.state = self.A @ self.state + self.B @ u + noise
        y_lin = self.C @ self.state
        y = np.tanh(y_lin * self.gain) / max(self.gain, 1e-9)
        return _format_vector(y)

    @property
    def name(self) -> str:
        return f"mock:nonlinear-shadow(d={self.A.shape[0]},gain={self.gain})"
