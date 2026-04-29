"""Closed-loop topologies under test.

An ``AgentLoop`` is an autonomous discrete-time system in some embedding
space R^d.  At each iteration the loop:
  1. forms the agent's input (possibly perturbed) from its internal state,
  2. invokes the LLM client,
  3. embeds the response, and
  4. updates its state.

Two concrete loops are provided:

- ``SyntheticLTILoop`` — wraps a ``SyntheticLTIClient``; vector-valued
  perturbation; identity embedder; closed-form ground truth via the
  spectral radius of the closed-loop matrix.
- ``ParaphraseLoop`` — successive paraphrasing with a real or mock LLM
  and a sentence-transformers embedder.  Mirrors the experimental setup
  of Wang et al. (2025), in which a 2-period limit cycle is the
  documented attractor.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np

from .embedder import Embedder, IdentityEmbedder
from .llm.base import LLMClient, Message
from .llm.mock import SyntheticLTIClient, _format_vector


# ---------------------------------------------------------------------------
# Abstract loop
# ---------------------------------------------------------------------------
class AgentLoop(ABC):
    """Closed loop u_k = f(z_{k-1}); z_k = agent(u_k + perturbation)."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimension."""

    @property
    @abstractmethod
    def embedder(self) -> Embedder:
        """The embedder used for observations."""

    @abstractmethod
    def reset(self, seed: int | None = None) -> None: ...

    @abstractmethod
    def step(
        self,
        perturbation_text: str = "",
        perturbation_vector: Optional[np.ndarray] = None,
        *,
        seed: int | None = None,
    ) -> np.ndarray:
        """Advance one closed-loop step; return embedded observation, shape (d,)."""

    # ----- convenience: rollout ------------------------------------------
    def rollout(
        self,
        horizon: int,
        seed: int | None = None,
        perturb_per_k: Sequence[tuple[str, Optional[np.ndarray]]] | None = None,
    ) -> np.ndarray:
        """Run ``horizon`` steps; return (horizon, d) trajectory of embeddings."""
        self.reset(seed=seed)
        out = np.zeros((horizon, self.dim), dtype=float)
        for k in range(horizon):
            ptext, pvec = ("", None) if perturb_per_k is None else perturb_per_k[k]
            out[k] = self.step(
                perturbation_text=ptext,
                perturbation_vector=pvec,
                seed=seed,
            )
        return out


# ---------------------------------------------------------------------------
# Synthetic LTI loop (validation ground truth)
# ---------------------------------------------------------------------------
class SyntheticLTILoop(AgentLoop):
    """Closed loop with the synthetic LTI client feeding back to itself.

    The LLM client implements x_{k+1} = A x_k + B u_k; here we close the
    loop with u_k = y_{k-1} + perturbation_vector_k.  This makes the
    closed-loop dynamics

        x_{k+1} = (A + B C) x_k + B p_k,                   (no perturb)
        x_{k+1} = (A + B C) x_k + B (perturb_vector_k),    (with probe)

    so the closed-loop transfer matrix is Gcl(z) = C (zI - (A+BC))^{-1} B.
    Tests use the analytical phase of this Gcl(z) as ground truth for the
    behavioural identifier.
    """

    def __init__(self, client: SyntheticLTIClient):
        self._client = client
        self._embedder = IdentityEmbedder(dim=client.dim)
        self._last_y = np.zeros(client.dim, dtype=float)

    @classmethod
    def from_random(
        cls,
        d: int = 8,
        spectral_radius: float = 0.7,
        seed: int = 0,
        noise_std: float = 0.0,
    ) -> "SyntheticLTILoop":
        client = SyntheticLTIClient.from_random(
            d=d, spectral_radius=spectral_radius, seed=seed, noise_std=noise_std
        )
        return cls(client)

    @property
    def dim(self) -> int:
        return int(self._client.dim)

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    @property
    def client(self) -> SyntheticLTIClient:
        return self._client

    def reset(self, seed: int | None = None) -> None:
        self._client.reset(seed=seed)
        self._last_y = np.zeros(self.dim, dtype=float)

    def step(
        self,
        perturbation_text: str = "",
        perturbation_vector: Optional[np.ndarray] = None,
        *,
        seed: int | None = None,
    ) -> np.ndarray:
        u = self._last_y.copy()
        if perturbation_vector is not None:
            u = u + np.asarray(perturbation_vector, dtype=float).ravel()
        msg = Message(role="user", content=_format_vector(u))
        text = self._client.chat([msg], seed=seed)
        # IdentityEmbedder: the response IS a vector encoded as text.
        z = self._embedder.embed([text])[0]
        self._last_y = z
        return z

    # ----- analytical helpers used by tests ------------------------------
    def closed_loop_matrix(self) -> np.ndarray:
        """A_cl = A + B C (closed-loop dynamics matrix)."""
        return self._client.A + self._client.B @ self._client.C

    def closed_loop_transfer_at_omega(self, omega: float) -> np.ndarray:
        """Closed-loop transfer matrix at e^{j omega}.

        The synthetic LTI client uses the convention
            x_{k+1} = A x_k + B u_k,    y_k = C x_{k+1}
        so the open-loop input-to-output transfer is z C (zI-A)^{-1} B
        rather than the standard C (zI-A)^{-1} B.  The closed-loop
        formula inherits the same factor of z, so we include it here
        to make the analytical phase match the empirical identification.
        """
        Acl = self.closed_loop_matrix()
        z = np.exp(1j * float(omega))
        I = np.eye(self.dim)
        return z * (self._client.C @ np.linalg.solve(z * I - Acl, self._client.B))

    def closed_loop_directional_phase(self, v: np.ndarray, omega: float) -> float:
        v = np.asarray(v, dtype=float).ravel()
        v = v / (np.linalg.norm(v) + 1e-12)
        Gz = self.closed_loop_transfer_at_omega(omega)
        return float(np.angle(v @ Gz @ v))

    def closed_loop_spectral_radius(self) -> float:
        return float(np.max(np.abs(np.linalg.eigvals(self.closed_loop_matrix()))))


# ---------------------------------------------------------------------------
# Paraphrase loop (Wang et al. 2025 setup)
# ---------------------------------------------------------------------------
PARAPHRASE_SYSTEM = (
    "You are an expert paraphraser.  Rephrase the user's sentence in your "
    "own words, preserving the meaning.  Reply with the rephrased "
    "sentence ONLY, no commentary."
)


class ParaphraseLoop(AgentLoop):
    """Successive paraphrasing.

    Wang et al. (2025) demonstrate that this loop converges to a 2-period
    limit cycle on multiple frontier LLMs.  We use it as the canonical
    'oscillatory' regime for predictor validation.
    """

    def __init__(
        self,
        llm: LLMClient,
        embedder: Embedder,
        initial_text: str = "Deep learning models trained on large corpora exhibit emergent capabilities.",
        max_tokens: int = 96,
        temperature: float = 0.7,
    ):
        self._llm = llm
        self._embedder = embedder
        self._initial = initial_text
        self._state = initial_text
        self._max_tokens = max_tokens
        self._temperature = temperature

    @property
    def dim(self) -> int:
        return int(self._embedder.dim)

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    def reset(self, seed: int | None = None) -> None:
        self._state = self._initial

    def step(
        self,
        perturbation_text: str = "",
        perturbation_vector: Optional[np.ndarray] = None,
        *,
        seed: int | None = None,
    ) -> np.ndarray:
        instruction = (
            "Rephrase the following sentence in your own words. "
            "Keep the meaning identical."
        )
        if perturbation_text:
            instruction += " Style hint: " + perturbation_text + "."
        prompt = f"{instruction}\n\nSentence: {self._state}"
        msg = [
            Message(role="system", content=PARAPHRASE_SYSTEM),
            Message(role="user", content=prompt),
        ]
        reply = self._llm.chat(
            msg,
            seed=seed,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        self._state = (reply or "").strip() or self._state
        z = self._embedder.embed([self._state])[0]
        return z

    @property
    def state(self) -> str:
        return self._state
