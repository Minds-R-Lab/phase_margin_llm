"""Abstract LLM client interface.

All concrete clients implement ``chat(messages, *, seed) -> str``.  The
seed is used (where the backend supports it) to make rollouts
reproducible; for stochastic backends without a seed parameter we hash
the seed into the temperature schedule or prefix the prompt with a
deterministic marker.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Message:
    """OpenAI-style message."""
    role: str   # one of {"system", "user", "assistant"}
    content: str


class LLMClient(ABC):
    """Minimal LLM interface used by all loops."""

    @abstractmethod
    def chat(
        self,
        messages: Sequence[Message],
        *,
        seed: int | None = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> str:
        """Return the LLM's reply string for the given conversation."""
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...
