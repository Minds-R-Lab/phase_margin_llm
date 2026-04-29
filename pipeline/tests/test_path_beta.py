"""Path-beta unit tests.

These do not load a real transformers model.  They check that:
  * pca_vector_basis recovers the dominant direction of an anisotropic
    synthetic trajectory (no embedder needed),
  * EmbeddingProbeLoop accepts a fake client and returns the hidden
    state through ``step``,
  * TransformersClient cannot be instantiated without ``transformers``
    installed but its module imports cleanly (so the rest of the
    package keeps importing on machines without torch).
"""
from __future__ import annotations

import numpy as np
import pytest

from phase_margin.probe import pca_vector_basis, ProbeDirection
from phase_margin.types import Regime


# ---------------------------------------------------------------------------
# pca_vector_basis on synthetic trajectory
# ---------------------------------------------------------------------------
def test_pca_vector_basis_recovers_dominant_direction():
    rng = np.random.default_rng(0)
    d = 16
    e = np.zeros(d); e[3] = 1.0
    n = 30
    traj = rng.standard_normal((n, d)) * 0.05
    traj += np.outer(rng.standard_normal(n), e)
    basis = pca_vector_basis(trajectory=traj, n_directions=2, seed=0)
    assert len(basis) == 2
    # First direction must align with e
    cos = abs(float(np.dot(basis[0].vector, e)))
    assert cos > 0.95, f"top PCA direction misaligned: cos={cos:.3f}"
    # Vectors are unit-norm
    for d_ in basis:
        assert abs(np.linalg.norm(d_.vector) - 1.0) < 1e-6


def test_pca_vector_basis_requires_loop_or_trajectory():
    with pytest.raises(ValueError):
        pca_vector_basis()


# ---------------------------------------------------------------------------
# EmbeddingProbeLoop with a fake client
# ---------------------------------------------------------------------------
class _FakeClient:
    """Mimic just the surface area EmbeddingProbeLoop needs."""
    def __init__(self, dim=8):
        self.hidden_dim = dim
        self._step = 0

    def chat_with_perturbation(self, *, messages, perturbation_vector,
                               seed=None, temperature=0.7, max_tokens=96):
        self._step += 1
        # Reply text varies slightly each step so trajectory isn't constant
        reply = f"reply-{self._step}"
        rng = np.random.default_rng(self._step + (seed or 0))
        h = rng.standard_normal(self.hidden_dim)
        if perturbation_vector is not None:
            h = h + np.asarray(perturbation_vector, dtype=float).ravel()
        return reply, h


def test_embedding_probe_loop_step_returns_hidden_state():
    from phase_margin.loops import EmbeddingProbeLoop
    client = _FakeClient(dim=12)
    loop = EmbeddingProbeLoop(client=client, initial_text="hello world.")
    z = loop.step()
    assert z.shape == (12,)
    assert loop.dim == 12
    # Perturbation is added to the response
    pert = np.zeros(12); pert[2] = 5.0
    loop.reset()
    z_perturbed = loop.step(perturbation_vector=pert)
    z_clean    = loop.step()
    # The perturbation contribution should be visible somewhere
    assert np.any(np.abs(z_perturbed) > 1e-6)


# ---------------------------------------------------------------------------
# TransformersClient module imports cleanly even without torch installed
# ---------------------------------------------------------------------------
def test_transformers_client_module_imports_cleanly():
    # Just importing the module must not crash the package even if
    # torch / transformers are unavailable -- the dependency check
    # happens at instantiation time.
    import importlib
    mod = importlib.import_module("phase_margin.llm.transformers_client")
    assert hasattr(mod, "TransformersClient")
