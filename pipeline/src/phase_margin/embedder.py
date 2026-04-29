"""Sentence-embedding adapters.

The pipeline only ever needs an ``embed(texts) -> ndarray`` callable.
We provide:

- ``SentenceTransformersEmbedder`` — local, free, default.  Uses the
  ``all-MiniLM-L6-v2`` model by default (384-d).
- ``IdentityEmbedder`` — for the synthetic LTI experiments where the
  "text" is already a vector.

Both implement the ``Embedder`` protocol so loops are agnostic of which
backend is in use.
"""
from __future__ import annotations

from typing import Iterable, Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    """Maps a list of texts to a 2-D array of shape (n_texts, d)."""

    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class IdentityEmbedder:
    """Pass-through embedder for vectors-as-text-of-floats experiments.

    Used by the synthetic LTI loop where the "state" already lives in
    R^d and we encode it as a string of comma-separated floats.
    """

    def __init__(self, dim: int):
        self._dim = int(dim)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=float)
        for i, t in enumerate(texts):
            try:
                vals = [float(x) for x in str(t).split(",") if x.strip()]
            except ValueError:
                vals = []
            n = min(len(vals), self._dim)
            out[i, :n] = vals[:n]
        return out


class SentenceTransformersEmbedder:
    """Default semantic embedder via the sentence-transformers library.

    Lazy-imports the dependency so the rest of the package does not
    require the heavyweight torch/transformers stack to be present.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:  # pragma: no cover - exercised only without dep
            raise ImportError(
                "sentence-transformers is not installed; either `pip install "
                "sentence-transformers` or pass an alternative Embedder."
            ) from e

        self._model = SentenceTransformer(model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())
        self.model_name = model_name

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=float)
        emb = self._model.encode(
            list(texts), convert_to_numpy=True, normalize_embeddings=False
        )
        return np.asarray(emb, dtype=float)
