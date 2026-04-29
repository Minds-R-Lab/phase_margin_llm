"""Command-line interface for the phase-margin pipeline.

Examples
--------
phase-margin synthetic --d 8 --horizon 32 --frequencies 8
phase-margin paraphrase --backend mock --steps 20 --topic "Deep learning"
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import numpy as np

from . import run_certification
from .loops import SyntheticLTILoop, ParaphraseLoop
from .probe import random_vector_basis, text_basis_paraphrase
from .types import ProbeConfig
from .ground_truth import detect_regime


# ---------------------------------------------------------------------------
def cmd_synthetic(args) -> int:
    loop = SyntheticLTILoop.from_random(
        d=args.d,
        spectral_radius=args.spectral_radius,
        seed=args.seed,
        noise_std=args.noise_std,
    )
    basis = random_vector_basis(dim=args.d, n_directions=args.basis, seed=args.seed)
    config = ProbeConfig(
        horizon=args.horizon,
        n_seeds=args.seeds,
        n_seeds_nominal=max(2, args.seeds // 2),
        n_frequencies=args.frequencies,
        epsilon=args.epsilon,
    )
    report = run_certification(
        loop=loop, basis=basis, config=config, use_text_perturbation=False, progress=args.progress
    )
    print(report.summary())
    print(f"\nClosed-loop spectral radius (ground truth): "
          f"{loop.closed_loop_spectral_radius():.3f}")

    # Long ground-truth trajectory
    traj = loop.rollout(horizon=max(args.horizon, 200), seed=args.seed)
    gt = detect_regime(traj)
    print(f"Ground-truth regime from trajectory: {gt.regime.value}")
    print(f"  final_var={gt.final_variance:.4g}  "
          f"period_score={gt.period_score:.3f}  growth={gt.growth_rate:+.3f}")
    return 0


def cmd_paraphrase(args) -> int:
    # Lazy imports
    from .embedder import SentenceTransformersEmbedder
    if args.backend == "mock":
        from .llm.mock import EchoClient
        client = EchoClient()
    elif args.backend == "anthropic":
        from .llm.anthropic_client import AnthropicClient
        client = AnthropicClient(model=args.model, cache_dir=args.cache)
    elif args.backend == "ollama":
        from .llm.ollama_client import OllamaClient
        client = OllamaClient(model=args.model)
    else:
        raise ValueError(args.backend)

    embedder = SentenceTransformersEmbedder()
    loop = ParaphraseLoop(
        llm=client,
        embedder=embedder,
        initial_text=args.topic,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    basis = text_basis_paraphrase()
    config = ProbeConfig(
        horizon=args.horizon,
        n_seeds=args.seeds,
        n_seeds_nominal=max(2, args.seeds // 2),
        n_frequencies=args.frequencies,
        epsilon=args.epsilon,
    )
    report = run_certification(
        loop=loop, basis=basis, config=config, use_text_perturbation=True, progress=args.progress
    )
    print(report.summary())
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="phase-margin",
        description="Black-box phase-margin certification of LLM agentic loops.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    syn = sub.add_parser("synthetic", help="Synthetic LTI shadow validation")
    syn.add_argument("--d", type=int, default=8)
    syn.add_argument("--spectral-radius", type=float, default=0.7)
    syn.add_argument("--noise-std", type=float, default=0.0)
    syn.add_argument("--horizon", type=int, default=32)
    syn.add_argument("--seeds", type=int, default=8)
    syn.add_argument("--frequencies", type=int, default=8)
    syn.add_argument("--basis", type=int, default=4)
    syn.add_argument("--epsilon", type=float, default=0.1)
    syn.add_argument("--seed", type=int, default=0)
    syn.add_argument("--progress", action="store_true")
    syn.set_defaults(func=cmd_synthetic)

    pp = sub.add_parser("paraphrase", help="Paraphrase loop diagnosis")
    pp.add_argument("--backend", choices=["mock", "anthropic", "ollama"], default="mock")
    pp.add_argument("--model", default="claude-haiku-4-5-20251001")
    pp.add_argument("--topic", default="Large language models exhibit emergent capabilities.")
    pp.add_argument("--horizon", type=int, default=16)
    pp.add_argument("--seeds", type=int, default=4)
    pp.add_argument("--frequencies", type=int, default=4)
    pp.add_argument("--epsilon", type=float, default=0.6)
    pp.add_argument("--max-tokens", type=int, default=96)
    pp.add_argument("--temperature", type=float, default=0.7)
    pp.add_argument("--cache", default="data/cache/anthropic")
    pp.add_argument("--progress", action="store_true")
    pp.set_defaults(func=cmd_paraphrase)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
