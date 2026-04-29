#!/usr/bin/env python3
"""
run_imp_experiment.py
=====================

Test the *Internal Model Principle* (Francis-Wonham, 1976) applied to
LLM agents.

The IMP says: to track / reject a class of signals in steady state, the
controller must contain a model of those signals.  Translated to LLMs:
an agent that is **explicitly instructed to maintain a hypothesis about
its environment** should track non-stationary targets better than an
agent that is not, *especially* on structured (periodic, drifting)
targets where the hypothesis is informative.

Experiment
----------
Sequence-prediction game.  At step k a hidden generator f(k) produces
the integer target y_k.  The agent predicts y_k_hat, then sees y_k and
the history grows by one row.  We track |y_k_hat - y_k| over K steps
and report the *asymptotic* tracking error (mean over the last
``asym_window`` steps).

Five generators, two conditions, multiple seeds.  Paired t-test on
log-error.

Generators
----------
  constant           y_k = 7
  linear             y_k = 3 + 2*k
  period_2           y_k cycles [3, 8]
  period_3           y_k cycles [2, 6, 9]
  period_4           y_k cycles [5, 10, 5, 0]

Conditions
----------
  reactive           system prompt: predict the next value, no instruction
                     to model the generator.
  internal_model     system prompt: first STATE A HYPOTHESIS about the
                     generating function, update the hypothesis with each
                     new datum, then use it to predict the next value.

Backends
--------
  mock-smart         deterministic agent that, in `internal_model` mode,
                     attempts to detect the period; in `reactive` mode
                     just echoes the most recent observation.  IMP holds
                     for this mock by construction.
  mock-memoryless    deterministic agent that ignores the IM instruction
                     and always echoes the most recent observation.
                     IMP does NOT hold here.
  ollama             local Qwen / Llama via http://localhost:11434
  anthropic          Claude API (requires ANTHROPIC_API_KEY)

Usage
-----
    python experiments/run_imp_experiment.py --backend mock-smart       --tag mock_smart
    python experiments/run_imp_experiment.py --backend mock-memoryless  --tag mock_memo
    python experiments/run_imp_experiment.py --backend ollama --model qwen2.5:7b --tag qwen7b
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "src"))

from phase_margin.llm.base import LLMClient, Message


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
GENERATORS: dict[str, "callable"] = {
    "constant": lambda k: 7,
    "linear":   lambda k: 3 + 2 * k,
    "period_2": lambda k: [3, 8][k % 2],
    "period_3": lambda k: [2, 6, 9][k % 3],
    "period_4": lambda k: [5, 10, 5, 0][k % 4],
}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_REACTIVE = (
    "You are a sequence-prediction agent.  At each step you will be shown "
    "all observations so far in the form `k=0: y=...; k=1: y=...; ...` "
    "and asked to predict the next value y_k.  Reply with EXACTLY one "
    "integer on a line by itself.  No commentary, no labels, no quotes."
)

SYSTEM_INTERNAL_MODEL = (
    "You are a sequence-prediction agent operating under the Internal "
    "Model Principle.  At each step you will be shown all observations "
    "so far and asked to predict the next value y_k.  You MUST first "
    "state a one-line HYPOTHESIS about the function f(k) that generated "
    "the observed values, then on a NEW LINE write your single integer "
    "prediction for y_k.\n\n"
    "Reply format (exactly two lines):\n"
    "    HYPOTHESIS: <your one-line hypothesis>\n"
    "    <integer prediction>"
)


def build_user_prompt(history: list[tuple[int, int]], k_next: int) -> str:
    if not history:
        return f"No observations yet. Predict y_{k_next}."
    obs = "; ".join(f"k={k}: y={y}" for k, y in history)
    return f"Observations so far: {obs}\n\nPredict y_{k_next}."


def parse_prediction(text: str) -> int | None:
    """Extract the LAST integer the agent emitted."""
    if text is None:
        return None
    matches = re.findall(r"-?\d+", text)
    if not matches:
        return None
    return int(matches[-1])


# ---------------------------------------------------------------------------
# Mock agents (used for sandbox-side framework verification only)
# ---------------------------------------------------------------------------
class MockSmartClient(LLMClient):
    """Deterministic agent.  Behaves correctly *if and only if* given the
    internal-model system prompt.  In reactive mode just echoes the most
    recent observation.

    Used to confirm the framework discriminates IMP-following behaviour
    from non-IMP behaviour.  NOT a substitute for a real LLM.
    """

    @property
    def name(self) -> str:
        return "mock:smart"

    def chat(self, messages, *, seed=None, temperature=0.7, max_tokens=256) -> str:
        sys_msg = next((m.content for m in messages if m.role == "system"), "")
        user = next((m.content for m in messages if m.role == "user"), "")
        history = self._parse_history(user)
        next_k = self._parse_next_k(user)
        if next_k is None or not history:
            return "0"
        if "Internal Model Principle" in sys_msg:
            # Try to detect a period and use it
            ys = [y for _, y in history]
            for period in (1, 2, 3, 4):
                if len(ys) >= 2 * period and all(
                    ys[-i - 1] == ys[-i - 1 - period] for i in range(period)
                ):
                    pred = ys[next_k % period - period] if next_k >= period else ys[next_k % period]
                    pred = ys[(next_k - period) if (next_k - period) >= 0 else next_k] \
                           if next_k >= period else ys[next_k] if next_k < len(ys) else ys[-1]
                    # Cleaner: project the cycle forward
                    pred = ys[(len(ys) - period) + (next_k - len(ys)) % period]
                    return f"HYPOTHESIS: period-{period} cycle\n{pred}"
            # Try linear fit
            if len(ys) >= 3:
                ks = np.array([k for k, _ in history], dtype=float)
                xs = np.array(ys, dtype=float)
                slope = float(np.polyfit(ks, xs, 1)[0])
                intercept = float(np.polyfit(ks, xs, 1)[1])
                pred = int(round(slope * next_k + intercept))
                return f"HYPOTHESIS: linear y=a+b*k\n{pred}"
            # Fallback: constant = mean
            return f"HYPOTHESIS: constant\n{int(round(np.mean(ys)))}"
        # Reactive mode: just echo last value
        return str(history[-1][1])

    @staticmethod
    def _parse_history(user: str) -> list[tuple[int, int]]:
        out = []
        for m in re.finditer(r"k=(-?\d+):\s*y=(-?\d+)", user):
            out.append((int(m.group(1)), int(m.group(2))))
        return out

    @staticmethod
    def _parse_next_k(user: str) -> int | None:
        m = re.search(r"y_(\d+)", user)
        return int(m.group(1)) if m else None


class MockMemorylessClient(MockSmartClient):
    """Always echoes the most recent observation, regardless of system prompt.
    Models an agent that ignores any instruction to maintain an internal
    model.  IMP should *not* hold here; if it does, the framework is
    measuring something other than what we think.
    """

    @property
    def name(self) -> str:
        return "mock:memoryless"

    def chat(self, messages, *, seed=None, temperature=0.7, max_tokens=256) -> str:
        user = next((m.content for m in messages if m.role == "user"), "")
        history = self._parse_history(user)
        if not history:
            return "0"
        return str(history[-1][1])


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------
@dataclass
class CellResult:
    backend: str
    pattern: str
    condition: str
    seed: int
    K: int
    asym_window: int
    targets: list[int]
    predictions: list[int | None]
    raw_replies: list[str]
    asym_error: float
    elapsed_s: float


def run_cell(
    *,
    client: LLMClient,
    pattern: str,
    condition: str,
    K: int,
    asym_window: int,
    seed: int,
    temperature: float,
) -> CellResult:
    gen = GENERATORS[pattern]
    sys_prompt = SYSTEM_INTERNAL_MODEL if condition == "internal_model" else SYSTEM_REACTIVE

    history: list[tuple[int, int]] = []
    targets: list[int] = []
    preds: list[int | None] = []
    replies: list[str] = []

    t0 = time.time()
    for k in range(K):
        msgs = [
            Message(role="system", content=sys_prompt),
            Message(role="user",   content=build_user_prompt(history, k)),
        ]
        reply = client.chat(msgs, seed=seed + k, temperature=temperature, max_tokens=128)
        replies.append(reply or "")
        pred = parse_prediction(reply)
        preds.append(pred)
        true_y = int(gen(k))
        targets.append(true_y)
        history.append((k, true_y))

    # Asymptotic error over last `asym_window` steps
    last = preds[-asym_window:]
    last_true = targets[-asym_window:]
    abs_err = [abs(p - t) for p, t in zip(last, last_true) if p is not None]
    asym = float(np.mean(abs_err)) if abs_err else float("inf")

    return CellResult(
        backend=client.name, pattern=pattern, condition=condition, seed=seed,
        K=K, asym_window=asym_window,
        targets=targets, predictions=preds, raw_replies=replies,
        asym_error=asym, elapsed_s=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Aggregation, statistics, output
# ---------------------------------------------------------------------------
def paired_ttest_log(
    rows: list[CellResult],
    eps: float = 0.5,
) -> dict[str, Any]:
    """Paired t-test on log(asym_error_reactive + eps) - log(asym_error_im + eps).
    Negative t-stat => IM helps (IM error is smaller).
    """
    pairs = {}
    for r in rows:
        key = (r.backend, r.pattern, r.seed)
        pairs.setdefault(key, {})[r.condition] = r.asym_error

    paired = [
        (vals["reactive"], vals["internal_model"])
        for vals in pairs.values()
        if "reactive" in vals and "internal_model" in vals
    ]
    if len(paired) < 2:
        return {"n_pairs": len(paired)}

    a = np.array([np.log(p[0] + eps) for p in paired])
    b = np.array([np.log(p[1] + eps) for p in paired])
    diff = b - a   # negative => IM helps
    from scipy.stats import ttest_rel
    t, p = ttest_rel(a, b)   # scipy: positive t means a > b => reactive > IM => IM helps
    return {
        "n_pairs": int(len(paired)),
        "mean_log_reactive": float(a.mean()),
        "mean_log_internal": float(b.mean()),
        "mean_diff_log": float(diff.mean()),
        "t_stat": float(t),
        "p_value_two_sided": float(p),
        "imp_helps_at_p_0_05": bool(p < 0.05 and a.mean() > b.mean()),
        "median_ratio_reactive_over_im": float(
            np.median([(p[0] + eps) / (p[1] + eps) for p in paired])
        ),
    }


def per_pattern_summary(rows: list[CellResult]) -> dict[str, dict[str, Any]]:
    summary = {}
    for pat in sorted({r.pattern for r in rows}):
        pat_rows = [r for r in rows if r.pattern == pat]
        for cond in ("reactive", "internal_model"):
            errs = [r.asym_error for r in pat_rows if r.condition == cond]
            if not errs:
                continue
            summary.setdefault(pat, {})[cond] = {
                "n": len(errs),
                "mean_asym_error": float(np.mean(errs)),
                "std_asym_error":  float(np.std(errs)),
                "errors":          [float(e) for e in errs],
            }
    return summary


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist()
                               if hasattr(o, 'tolist') else str(o)))


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("phase_margin.imp")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w"); fh.setFormatter(fmt); log.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout);       sh.setFormatter(fmt); log.addHandler(sh)
    return log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def get_client(backend: str, model: str | None) -> LLMClient:
    if backend == "mock-smart":
        return MockSmartClient()
    if backend == "mock-memoryless":
        return MockMemorylessClient()
    if backend == "ollama":
        from phase_margin.llm.ollama_client import OllamaClient
        return OllamaClient(model=model or "qwen2.5:7b")
    if backend == "anthropic":
        from phase_margin.llm.anthropic_client import AnthropicClient
        return AnthropicClient(model=model or "claude-haiku-4-5-20251001",
                               cache_dir="data/cache/anthropic")
    raise ValueError(f"unknown backend: {backend}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="mock-smart",
                   choices=["mock-smart", "mock-memoryless", "ollama", "anthropic"])
    p.add_argument("--model", default=None)
    p.add_argument("--patterns", nargs="+", default=list(GENERATORS),
                   choices=list(GENERATORS))
    p.add_argument("--conditions", nargs="+",
                   default=["reactive", "internal_model"],
                   choices=["reactive", "internal_model"])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=16,
                   help="K, total prediction steps per cell")
    p.add_argument("--asym-window", type=int, default=8,
                   help="number of trailing steps used for asymptotic error")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 = greedy decoding (recommended)")
    p.add_argument("--results-root", default="results")
    p.add_argument("--tag", default="")
    args = p.parse_args(argv)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_tag = f"_{args.tag}" if args.tag else ""
    out_dir = Path(args.results_root) / f"experiment_imp_{timestamp}{folder_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out_dir / "log.txt")

    log.info("=" * 76)
    log.info(f"Internal Model Principle for LLM Agents -- experiment harness")
    log.info(f"output dir: {out_dir}")
    log.info("=" * 76)
    log.info(f"backend  : {args.backend}    model: {args.model}")
    log.info(f"patterns : {', '.join(args.patterns)}")
    log.info(f"conditions: {', '.join(args.conditions)}")
    log.info(f"seeds    : {args.seeds}")
    log.info(f"K        : {args.steps}    asym_window: {args.asym_window}")
    log.info(f"temp     : {args.temperature}")

    client = get_client(args.backend, args.model)
    log.info(f"client   : {client.name}")

    save_json({
        "argv": list(sys.argv),
        "args": vars(args),
        "patterns": list(args.patterns),
        "conditions": list(args.conditions),
        "seeds": args.seeds,
        "K": args.steps,
        "asym_window": args.asym_window,
    }, out_dir / "manifest.json")

    rows: list[CellResult] = []
    for pattern in args.patterns:
        for condition in args.conditions:
            for seed in range(args.seeds):
                t0 = time.time()
                r = run_cell(
                    client=client, pattern=pattern, condition=condition,
                    K=args.steps, asym_window=args.asym_window, seed=seed,
                    temperature=args.temperature,
                )
                rows.append(r)
                log.info(
                    f"  {pattern:9s}  {condition:14s}  seed={seed}  "
                    f"asym_err={r.asym_error:6.2f}  ({time.time()-t0:.0f}s)"
                )

    # Save raw cells
    save_json([asdict(r) for r in rows], out_dir / "cells.json")

    summary = per_pattern_summary(rows)
    save_json(summary, out_dir / "per_pattern_summary.json")

    log.info("\nper-pattern asymptotic error (mean over seeds):")
    for pat, by_cond in summary.items():
        re_e = by_cond.get("reactive", {}).get("mean_asym_error", float("nan"))
        im_e = by_cond.get("internal_model", {}).get("mean_asym_error", float("nan"))
        delta = im_e - re_e if not (np.isnan(re_e) or np.isnan(im_e)) else float("nan")
        log.info(f"  {pat:9s}  reactive={re_e:6.2f}  internal_model={im_e:6.2f}  "
                 f"delta={delta:+.2f}  "
                 f"{'(IMP helps)' if delta < 0 else '(IMP hurts or no effect)'}")

    test = paired_ttest_log(rows)
    save_json(test, out_dir / "paired_ttest.json")
    log.info("\npaired t-test on log(asym_error):")
    log.info(json.dumps(test, indent=2))

    log.info("\n" + "=" * 76)
    log.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
