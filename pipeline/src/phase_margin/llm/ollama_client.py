"""Local Ollama adapter (HTTP).

Talks to a running Ollama daemon at ``OLLAMA_HOST`` (default
``http://localhost:11434``).  Uses the bundled ``ollama`` Python SDK if
available; otherwise falls back to ``urllib`` so the dependency is
optional.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Sequence

from .base import LLMClient, Message


class OllamaClient(LLMClient):
    def __init__(
        self,
        model: str = "llama3.2:3b",
        host: str | None = None,
        timeout: float = 120.0,
    ):
        self._model = model
        self._host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self._timeout = timeout

    @property
    def name(self) -> str:
        return f"ollama:{self._model}"

    def chat(
        self,
        messages: Sequence[Message],
        *,
        seed: int | None = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> str:
        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
                **({"seed": int(seed)} if seed is not None else {}),
            },
        }
        req = urllib.request.Request(
            url=f"{self._host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            data = json.loads(resp.read().decode())
        return data.get("message", {}).get("content", "")
