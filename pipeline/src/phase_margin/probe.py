"""Probe-direction definitions and basis builders.

A ``ProbeDirection`` carries (i) a unit vector in the embedding space
along which the response will be measured, and (ii) a way to translate
a scalar perturbation strength ``s in [-eps, +eps]`` into either a
real-valued perturbation vector (for vector-valued loops like the LTI
shadow) or a textual modifier appended to the agent's prompt (for
text-valued loops like paraphrasing).

The pipeline calls ``direction.vector`` and either
``direction.text_for_strength(s)`` or ``s * direction.vector`` per
iteration; the loop's ``step`` consumes whichever it needs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------
@dataclass
class ProbeDirection:
    name: str
    vector: Optional[np.ndarray] = None        # unit vector in R^d (lazy)
    pos_modifier: Optional[str] = None         # text describing the +1 end
    neg_modifier: Optional[str] = None         # text describing the -1 end
    template: Optional[Callable[[float], str]] = None  # custom modifier-text generator
    _embedded: bool = field(default=False, repr=False)

    # ----- text injection --------------------------------------------------
    def text_for_strength(self, s: float) -> str:
        """Return a textual modifier of intensity |s| toward sign(s).

        If a custom ``template`` is provided it is used.  Otherwise we
        build a generic intensity-scaled modifier from ``pos_modifier``
        and ``neg_modifier``.
        """
        if self.template is not None:
            return self.template(float(s))
        if self.pos_modifier is None and self.neg_modifier is None:
            return ""

        a = float(abs(s))
        if a < 0.05:
            return ""
        qualifier = (
            "slightly" if a < 0.25 else
            "moderately" if a < 0.6 else
            "strongly"
        )
        target = self.pos_modifier if s > 0 else (self.neg_modifier or "")
        if not target:
            return ""
        return f"({qualifier} {target})"

    # ----- vector embedding -----------------------------------------------
    def ensure_vector(self, embedder=None, dim: int | None = None) -> np.ndarray:
        """Lazily set ``self.vector`` from the +/- modifier embeddings.

        If ``self.vector`` is already a vector, this is a no-op.
        Otherwise we embed both extremes via ``embedder`` and take the
        normalised difference.
        """
        if self.vector is not None and not self._embedded:
            v = np.asarray(self.vector, dtype=float).ravel()
            n = np.linalg.norm(v)
            self.vector = v / n if n > 0 else v
            self._embedded = True
            return self.vector

        if self.vector is not None:
            return self.vector

        if embedder is None:
            raise ValueError(
                f"Direction '{self.name}' has no vector and no embedder "
                f"was supplied."
            )
        if not (self.pos_modifier and self.neg_modifier):
            raise ValueError(
                f"Direction '{self.name}' needs pos_modifier and "
                f"neg_modifier to be embedded."
            )
        embs = embedder.embed([self.pos_modifier, self.neg_modifier])
        diff = embs[0] - embs[1]
        n = np.linalg.norm(diff)
        self.vector = diff / n if n > 0 else diff
        self._embedded = True
        return self.vector


# ---------------------------------------------------------------------------
# Basis builders
# ---------------------------------------------------------------------------
def random_vector_basis(dim: int, n_directions: int = 4, seed: int = 0) -> list[ProbeDirection]:
    """Random unit-vector basis for vector-valued loops (LTI shadow).

    Used in the synthetic-LTI validation: the directions span R^d
    so the certificate is non-trivial across all loop modes.
    """
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n_directions, dim))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True) + 1e-12
    return [
        ProbeDirection(name=f"v{i}", vector=raw[i])
        for i in range(n_directions)
    ]


def text_basis_paraphrase() -> list[ProbeDirection]:
    """Default semantic basis for the paraphrase loop.

    Four orthogonal-ish axes that empirically pull paraphrase output
    along distinct directions: verbosity, formality, abstraction, and
    sentiment.  Each axis is realised as a (positive, negative)
    text-modifier pair; the embedding-space direction is computed
    lazily by the embedder.
    """
    return [
        ProbeDirection(
            name="verbosity",
            pos_modifier="be more verbose and detailed",
            neg_modifier="be more concise and minimal",
        ),
        ProbeDirection(
            name="formality",
            pos_modifier="use formal academic register",
            neg_modifier="use informal conversational register",
        ),
        ProbeDirection(
            name="abstraction",
            pos_modifier="speak abstractly and theoretically",
            neg_modifier="speak concretely with examples",
        ),
        ProbeDirection(
            name="sentiment",
            pos_modifier="frame positively and optimistically",
            neg_modifier="frame critically and skeptically",
        ),
    ]


def task_directions(*pairs: tuple[str, str, str]) -> list[ProbeDirection]:
    """Build a list of directions from (name, positive, negative) tuples."""
    return [
        ProbeDirection(name=n, pos_modifier=p, neg_modifier=q)
        for (n, p, q) in pairs
    ]


# ---------------------------------------------------------------------------
# Default anchor library for the data-driven (PCA) text basis
# ---------------------------------------------------------------------------
DEFAULT_PROBE_ANCHORS: list[str] = [
    "be more verbose and detailed",
    "be more concise and minimal",
    "use formal academic register",
    "use informal conversational register",
    "speak abstractly and theoretically",
    "speak concretely with examples",
    "frame positively and optimistically",
    "frame critically and skeptically",
    "use rich elaborate synonyms",
    "use plain simple wording",
    "use complex sentence structure with subordinate clauses",
    "use direct simple sentences",
    "use technical jargon",
    "use everyday language",
    "be poetic and metaphorical",
    "be literal and matter-of-fact",
    "emphasize causes and reasons",
    "emphasize consequences and outcomes",
    "use first-person perspective",
    "use third-person perspective",
]


def pca_text_basis(
    loop=None,
    *,
    embedder=None,
    trajectory: "np.ndarray | None" = None,
    n_pca_steps: int = 24,
    n_directions: int = 4,
    anchor_texts: "Sequence[str] | None" = None,
    seed: int = 0,
) -> list[ProbeDirection]:
    """Data-driven probing basis built from the loop's own observed dynamics.

    1. Run an unperturbed nominal rollout for ``n_pca_steps`` (or accept a
       pre-computed ``trajectory`` of shape ``(n, d)``).
    2. SVD the centred trajectory.
    3. Pick the top-`n_directions` principal vectors v_1, ..., v_p.
    4. For each v_i, choose the anchor text whose embedding maximally
       projects on +v_i (the "positive end" of that semantic direction)
       and the anchor that maximally projects on -v_i.  These become the
       textual perturbation modifiers.
    5. Return ProbeDirection objects whose ``vector`` is already set to
       v_i in the embedding space.

    The result satisfies §5.3 of the paper (empirical PCA basis): the
    probing axes are *the directions the loop actually moves on*, not
    the four hand-picked semantic axes that may not span the LLM's
    natural attractors.
    """
    if anchor_texts is None:
        anchor_texts = DEFAULT_PROBE_ANCHORS

    # Resolve embedder + trajectory
    if trajectory is None:
        if loop is None:
            raise ValueError("either `loop` or `trajectory` must be provided")
        embedder = embedder or loop.embedder
        loop.reset(seed=seed)
        Z = np.array([loop.step(seed=seed) for _ in range(int(n_pca_steps))])
    else:
        Z = np.asarray(trajectory, dtype=float)
        if embedder is None:
            raise ValueError("`embedder` is required when `trajectory` is supplied")
    if Z.ndim != 2 or Z.shape[0] < 3:
        raise ValueError("trajectory must be (n, d) with n >= 3")

    # PCA via SVD on the centred trajectory
    Zc = Z - Z.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    n_dir = int(min(n_directions, Vt.shape[0]))
    top = Vt[:n_dir]                       # (n_dir, d)

    # Embed all anchor texts once and unit-normalise
    anchor_embs = np.asarray(embedder.embed(list(anchor_texts)), dtype=float)
    anchor_unit = anchor_embs / (
        np.linalg.norm(anchor_embs, axis=1, keepdims=True) + 1e-12
    )

    directions: list[ProbeDirection] = []
    used: set[int] = set()
    for i, v in enumerate(top):
        v_unit = v / (np.linalg.norm(v) + 1e-12)
        proj = anchor_unit @ v_unit         # (n_anchors,)
        order = np.argsort(proj)            # ascending
        neg_idx = next((int(k) for k in order if int(k) not in used),
                       int(order[0]))
        used.add(neg_idx)
        pos_idx = next((int(k) for k in order[::-1] if int(k) not in used),
                       int(order[-1]))
        used.add(pos_idx)
        if pos_idx == neg_idx:
            continue
        d = ProbeDirection(
            name=f"pca{i}",
            vector=v_unit.astype(float).copy(),
            pos_modifier=str(anchor_texts[pos_idx]),
            neg_modifier=str(anchor_texts[neg_idx]),
        )
        d._embedded = True   # vector is already in the embedding space
        directions.append(d)
    return directions


def hybrid_basis_paraphrase(
    loop=None,
    *,
    embedder=None,
    n_pca_steps: int = 24,
    n_pca_directions: int = 3,
    seed: int = 0,
):
    """Hand-picked text basis plus top-N PCA directions of the loop's own dynamics."""
    base = text_basis_paraphrase()
    pca = pca_text_basis(
        loop=loop, embedder=embedder,
        n_pca_steps=n_pca_steps, n_directions=n_pca_directions,
        seed=seed,
    )
    return base + pca
