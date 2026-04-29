"""Anthropic Claude adapter.

Lazy-imports the ``anthropic`` SDK so the package can be installed and
tested on a machine without it.

Set ``ANTHROPIC_API_KEY`` before use.  The client respects an optional
on-disk JSON cache so repeated identical calls do not re-bill.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

from .base import LLMClient, Message


class AnthropicClient(LLMClient):
    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        cache_dir: str | os.PathLike | None = None,
    ):
        try:
            import anthropic  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "anthropic SDK is not installed; `pip install anthropic` "
                "or use a different backend"
            ) from e
        self._anthropic = anthropic
        self._client = anthropic.Anthropic()
        self._model = model
        self._cache = Path(cache_dir) if cache_dir else None
        if self._cache:
            self._cache.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return f"anthropic:{self._model}"

    # ----- caching ----------------------------------------------------
    def _cache_key(self, messages, temperature, max_tokens, seed) -> str:
        payload = json.dumps(
            {
                "model": self._model,
                "msgs": [(m.role, m.content) for m in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_get(self, key: str) -> str | None:
        if not self._cache:
            return None
        f = self._cache / f"{key}.json"
        if f.exists():
            return json.loads(f.read_text())["response"]
        return None

    def _cache_put(self, key: str, response: str) -> None:
        if not self._cache:
            return
        (self._cache / f"{key}.json").write_text(
            json.dumps({"response": response}, ensure_ascii=False)
        )

    # ----- main -------------------------------------------------------
    def chat(
        self,
        messages: Sequence[Message],
        *,
        seed: int | None = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> str:
        key = self._cache_key(messages, temperature, max_tokens, seed)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        # Anthropic API: separate system + user/assistant rolls
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        chat_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]
        # The SDK seeds via the prompt prefix, since the public API does
        # not currently take a numeric seed for Claude.
        if seed is not None:
            chat_messages = [{"role": "system", "content": ""}] if False else chat_messages
            # Append a deterministic-marker comment to the last user msg.
            for i in range(len(chat_messages) - 1, -1, -1):
                if chat_messages[i]["role"] == "user":
                    chat_messages[i] = {
                        **chat_messages[i],
                        "content": chat_messages[i]["content"]
                        + f"\n\n[probe:{seed}]",
                    }
                    break

        resp = self._client.messages.create(
            model=self._model,
            system=system or None,
            messages=chat_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = "".join(getattr(b, "text", "") for b in resp.content)
        self._cache_put(key, text)
        return text
