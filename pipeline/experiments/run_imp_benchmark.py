#!/usr/bin/env python3
"""
run_imp_benchmark.py
====================

Diverse-task benchmark for the Internal Model Principle (IMP) claim.

Background
----------
``run_imp_experiment.py`` showed that on a 5-generator sequence-prediction
game, prepending an "Internal Model" (IM) instruction *hurts* small open
LLMs (qwen2.5:7b/14b/32b) and the harm decays with scale.  A reviewer
correctly objected that 5 generators is too small a base to make a
generalisable claim.  This script broadens to a 30-task benchmark
spanning three difficulty bands:

  EASY (10):    pattern-completion targets where a no-reasoning agent
                tends to get the answer right by echoing recent values
                or by trivial inspection (constants, slow linear, simple
                low-period cycles).

  MEDIUM (10):  patterns that need explicit model identification but
                only a single-step computation once identified
                (quadratic, longer periods, piecewise, parity-modulated).

  HARD (10):    patterns where even with a correct hypothesis the next
                step requires non-trivial computation (fibonacci-like
                recurrences, polynomial in k, conditional rules, regime
                switches).

Conditions
----------
  reactive            no instruction to model the generator.
  internal_hypothesis "state a one-line HYPOTHESIS, then predict".
  internal_cot        "Let's think step by step. Then on the LAST line
                      print one integer."

Hypotheses (pre-registered)
---------------------------
  H1: For small models (<=14B), IM conditions HURT on EASY tasks.
  H2: For small models, IM conditions HELP on HARD tasks (CoT-literature
      replication: Wei et al. 2022; Kojima et al. 2022).
  H3: There is a significant 3-way interaction
      difficulty x condition x scale: the harm on EASY decays with
      scale; the help on HARD persists or grows.

Backends and usage
------------------
  python experiments/run_imp_benchmark.py --backend mock-smart --tag mock
  python experiments/run_imp_benchmark.py --backend ollama --model qwen2.5:7b   --tag qwen7b
  python experiments/run_imp_benchmark.py --backend ollama --model qwen2.5:14b  --tag qwen14b
  python experiments/run_imp_benchmark.py --backend ollama --model qwen2.5:32b  --tag qwen32b
  python experiments/run_imp_benchmark.py --backend ollama --model llama3.1:70b-instruct-q4_K_M --tag llama70b
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "src"))

from phase_margin.llm.base import LLMClient, Message


# ---------------------------------------------------------------------------
# Benchmark task table
# ---------------------------------------------------------------------------
@dataclass
class Task:
    name: str
    difficulty: str          # "easy" | "medium" | "hard"
    generator: Callable[[int], int]
    K: int = 14              # default trajectory length
    asym_window: int = 6     # tail used for asymptotic error
    description: str = ""    # human-readable note (not shown to LLM)


def _make_fib() -> Callable[[int], int]:
    """Memoized Fibonacci with f(0) = f(1) = 1."""
    cache: dict[int, int] = {0: 1, 1: 1}

    def fib(k: int) -> int:
        if k in cache:
            return cache[k]
        # build up sequentially so we never hit a missing predecessor
        for j in range(max(cache) + 1, k + 1):
            cache[j] = cache[j - 1] + cache[j - 2]
        return cache[k]
    return fib


_FIB = _make_fib()


# 30 tasks.  All generators are deterministic functions of k = 0, 1, 2, ...
BENCHMARK_TASKS: list[Task] = [
    # ---------- EASY (10) -------------------------------------------------
    Task("e01_constant_seven",  "easy", lambda k: 7,                    description="y = 7"),
    Task("e02_constant_zero",   "easy", lambda k: 0,                    description="y = 0"),
    Task("e03_linear_2k+3",     "easy", lambda k: 2*k + 3,              description="y = 2k+3"),
    Task("e04_linear_5k",       "easy", lambda k: 5*k,                  description="y = 5k"),
    Task("e05_linear_desc_3",   "easy", lambda k: 20 - 3*k,             description="y = 20-3k"),
    Task("e06_period2_3_8",     "easy", lambda k: [3, 8][k % 2],        description="cycle [3,8]"),
    Task("e07_period2_10_0",    "easy", lambda k: [10, 0][k % 2],       description="cycle [10,0]"),
    Task("e08_period3_2_6_9",   "easy", lambda k: [2, 6, 9][k % 3],     description="cycle [2,6,9]"),
    Task("e09_period4_5_10_5_0","easy", lambda k: [5, 10, 5, 0][k % 4], description="cycle [5,10,5,0]"),
    Task("e10_mod5",            "easy", lambda k: k % 5,                description="y = k mod 5"),

    # ---------- MEDIUM (10) -----------------------------------------------
    Task("m01_quadratic_k2",            "medium", lambda k: k * k,                       description="y = k^2"),
    Task("m02_quadratic_offset",        "medium", lambda k: k*k - 2*k + 1,                description="y = (k-1)^2"),
    Task("m03_period5_pi",              "medium", lambda k: [3, 1, 4, 1, 5][k % 5],       description="cycle [3,1,4,1,5]"),
    Task("m04_period6_e",               "medium", lambda k: [2, 7, 1, 8, 2, 8][k % 6],    description="cycle [2,7,1,8,2,8]"),
    Task("m05_piecewise_v",             "medium", lambda k: k if k < 5 else 10 - k,       description="y = k if k<5 else 10-k"),
    Task("m06_zigzag",                  "medium", lambda k: k * (1 if k % 2 == 0 else -1),description="y = k*(-1)^k"),
    Task("m07_step_floor3",             "medium", lambda k: k // 3,                       description="y = floor(k/3)"),
    Task("m08_triangle_wave",           "medium", lambda k: abs((k % 6) - 3),             description="triangle wave amp 3"),
    Task("m09_linear_plus_period",      "medium", lambda k: k + 3*(k % 2),                 description="y = k + 3*(k%2)"),
    Task("m10_quadratic_bounded",       "medium", lambda k: -k*k + 10*k,                   description="y = -k^2 + 10k"),

    # ---------- HARD (10) -------------------------------------------------
    Task("h01_fibonacci",         "hard", _FIB,
         description="Fibonacci with f(0)=f(1)=1"),
    Task("h02_geometric_2",       "hard", lambda k: 2 ** k if k <= 12 else 4096,           K=12, asym_window=4,
         description="y = 2^k (capped K=12)"),
    Task("h03_squares_minus_k",   "hard", lambda k: k*k - k,                                description="y = k(k-1)"),
    Task("h04_triangle_numbers",  "hard", lambda k: k*(k+1)//2,                             description="y = k(k+1)/2"),
    Task("h05_pi_digits",         "hard", lambda k: [3,1,4,1,5,9,2,6,5,3,5,8,9,7,9,3][k%16],description="cycle of 16 pi digits"),
    Task("h06_e_digits",          "hard", lambda k: [2,7,1,8,2,8,1,8,2,8,4,5,9,0,4,5][k%16],description="cycle of 16 e digits"),
    Task("h07_cubic_mod100",      "hard", lambda k: (k**3) % 100,                            description="y = k^3 mod 100"),
    Task("h08_conditional_rule",  "hard", lambda k: 2*k if k % 3 == 0 else k + 1,            description="y = 2k if k%3==0 else k+1"),
    Task("h09_regime_switch",     "hard", lambda k: ([1,4][k%2] if k < 7 else [5,2][k%2]),   description="period-2 with regime switch at k=7"),
    Task("h10_sum_two_periods",   "hard", lambda k: (k % 3) + (k % 5),                       description="y = (k%3) + (k%5)"),
]


# Quick sanity check at import time so a typo in the table fails loudly.
def _validate_tasks() -> None:
    seen = set()
    for t in BENCHMARK_TASKS:
        assert t.name not in seen, f"duplicate task name {t.name}"
        seen.add(t.name)
        assert t.difficulty in ("easy", "medium", "hard"), t.difficulty
        # generator must produce ints over k = 0..K-1 without raising
        for k in range(t.K):
            v = t.generator(k)
            assert isinstance(v, (int, np.integer)), f"{t.name} k={k} -> {v!r}"
    counts = {"easy": 0, "medium": 0, "hard": 0}
    for t in BENCHMARK_TASKS:
        counts[t.difficulty] += 1
    assert counts == {"easy": 10, "medium": 10, "hard": 10}, counts


_validate_tasks()


# ---------------------------------------------------------------------------
# Prompts (one reactive baseline + two IM phrasings)
# ---------------------------------------------------------------------------
SYSTEM_REACTIVE = (
    "You are a sequence-prediction agent.  At each step you will be shown "
    "all observations so far in the form `k=0: y=...; k=1: y=...; ...` "
    "and asked to predict the next value y_k.  Reply with EXACTLY one "
    "integer on a line by itself.  No commentary, no labels, no quotes."
)

SYSTEM_IM_HYPOTHESIS = (
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

SYSTEM_IM_COT = (
    "You are a sequence-prediction agent.  At each step you will be "
    "shown all observations so far and asked to predict the next value "
    "y_k.  Let's think step by step: briefly reason about what rule "
    "could be generating these values and why your candidate next value "
    "is consistent with it.  After your reasoning, write your single "
    "integer prediction for y_k as the LAST line of your reply, by "
    "itself, with no labels or punctuation.  The integer on the last "
    "line will be parsed automatically."
)

CONDITIONS: dict[str, str] = {
    "reactive":             SYSTEM_REACTIVE,
    "internal_hypothesis":  SYSTEM_IM_HYPOTHESIS,
    "internal_cot":         SYSTEM_IM_COT,
}


def build_user_prompt(history: list[tuple[int, int]], k_next: int) -> str:
    if not history:
        return f"No observations yet. Predict y_{k_next}."
    obs = "; ".join(f"k={k}: y={y}" for k, y in history)
    return f"Observations so far: {obs}\n\nPredict y_{k_next}."


def parse_prediction(text: str) -> int | None:
    """Extract the LAST integer the agent emitted."""
    if not text:
        return None
    # Prefer the last non-empty line that contains an integer
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    for ln in reversed(lines):
        m = re.findall(r"-?\d+", ln)
        if m:
            return int(m[-1])
    matches = re.findall(r"-?\d+", text)
    return int(matches[-1]) if matches else None


# ---------------------------------------------------------------------------
# Mock clients (sandbox smoke-test only)
# ---------------------------------------------------------------------------
class MockSmartClient(LLMClient):
    """Deterministic agent that, in IM modes, attempts a small bag of
    hypothesis fits (constant / linear / period 2..6).  In reactive mode
    just echoes the most recent observation.  Behaviour is the same for
    both IM phrasings (the mock doesn't model the prompt-style
    sensitivity that real LLMs show).  Used for framework verification
    only.
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
        is_im = ("Internal Model Principle" in sys_msg) or ("step by step" in sys_msg)
        if is_im:
            ys = [y for _, y in history]
            ks = [k for k, _ in history]
            # Try periods 2..6
            for period in range(2, 7):
                if len(ys) >= 2 * period and all(
                    ys[-i - 1] == ys[-i - 1 - period] for i in range(period)
                ):
                    pred = ys[len(ys) - period + (next_k - len(ys)) % period]
                    return f"HYPOTHESIS: period-{period} cycle\n{pred}"
            # Try linear
            if len(ys) >= 3:
                slope, intercept = np.polyfit(np.array(ks, dtype=float),
                                              np.array(ys, dtype=float), 1)
                pred = int(round(slope * next_k + intercept))
                return f"HYPOTHESIS: linear y=a+b*k\n{pred}"
            return f"HYPOTHESIS: constant\n{int(round(np.mean(ys)))}"
        return str(history[-1][1])

    @staticmethod
    def _parse_history(user: str) -> list[tuple[int, int]]:
        return [(int(m.group(1)), int(m.group(2)))
                for m in re.finditer(r"k=(-?\d+):\s*y=(-?\d+)", user)]

    @staticmethod
    def _parse_next_k(user: str) -> int | None:
        m = re.search(r"y_(\d+)", user)
        return int(m.group(1)) if m else None


class MockMemorylessClient(MockSmartClient):
    @property
    def name(self) -> str:
        return "mock:memoryless"

    def chat(self, messages, *, seed=None, temperature=0.7, max_tokens=256) -> str:
        user = next((m.content for m in messages if m.role == "user"), "")
        history = self._parse_history(user)
        return "0" if not history else str(history[-1][1])


# ---------------------------------------------------------------------------
# Experiment cell
# ---------------------------------------------------------------------------
@dataclass
class CellResult:
    backend: str
    task: str
    difficulty: str
    condition: str
    seed: int
    K: int
    asym_window: int
    targets: list[int]
    predictions: list[int | None]
    raw_replies: list[str]
    asym_error: float
    rmse_error: float
    elapsed_s: float


def run_cell(
    *,
    client: LLMClient,
    task: Task,
    condition: str,
    seed: int,
    temperature: float,
) -> CellResult:
    sys_prompt = CONDITIONS[condition]

    history: list[tuple[int, int]] = []
    targets: list[int] = []
    preds: list[int | None] = []
    replies: list[str] = []

    t0 = time.time()
    for k in range(task.K):
        msgs = [
            Message(role="system", content=sys_prompt),
            Message(role="user",   content=build_user_prompt(history, k)),
        ]
        # Larger max_tokens for CoT; modest for hypothesis / reactive
        max_tok = 256 if condition == "internal_cot" else 96
        reply = client.chat(msgs, seed=seed + k,
                            temperature=temperature, max_tokens=max_tok)
        replies.append(reply or "")
        preds.append(parse_prediction(reply))
        true_y = int(task.generator(k))
        targets.append(true_y)
        history.append((k, true_y))

    last = preds[-task.asym_window:]
    last_true = targets[-task.asym_window:]
    abs_err = [abs(p - t) for p, t in zip(last, last_true) if p is not None]
    asym = float(np.mean(abs_err)) if abs_err else float("inf")
    rmse = float(np.sqrt(np.mean([(p - t) ** 2 for p, t in zip(last, last_true)
                                  if p is not None]))) if abs_err else float("inf")
    return CellResult(
        backend=client.name, task=task.name, difficulty=task.difficulty,
        condition=condition, seed=seed,
        K=task.K, asym_window=task.asym_window,
        targets=targets, predictions=preds, raw_replies=replies,
        asym_error=asym, rmse_error=rmse, elapsed_s=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Aggregation, statistics
# ---------------------------------------------------------------------------
def paired_ttest_log(
    rows: list[CellResult],
    *,
    cond_a: str,
    cond_b: str,
    eps: float = 0.5,
) -> dict[str, Any]:
    """Paired t-test on log(asym_error).  Pairing is over (task, seed).

    Reports test statistic for log(a) - log(b);
    positive t => cond_b is on average lower error => cond_b helps.
    """
    pairs: dict[tuple[str, int], dict[str, float]] = {}
    for r in rows:
        key = (r.task, r.seed)
        pairs.setdefault(key, {})[r.condition] = r.asym_error
    paired = [
        (vals[cond_a], vals[cond_b])
        for vals in pairs.values()
        if cond_a in vals and cond_b in vals
    ]
    if len(paired) < 2:
        return {"n_pairs": len(paired)}

    a = np.array([np.log(p[0] + eps) for p in paired])
    b = np.array([np.log(p[1] + eps) for p in paired])
    diff = b - a
    from scipy.stats import ttest_rel
    t, p = ttest_rel(a, b)
    return {
        "cond_a": cond_a,
        "cond_b": cond_b,
        "n_pairs": int(len(paired)),
        "mean_log_a": float(a.mean()),
        "mean_log_b": float(b.mean()),
        "mean_diff_log_b_minus_a": float(diff.mean()),
        "t_stat": float(t),
        "p_value_two_sided": float(p),
        "b_helps_at_p_0_05": bool(p < 0.05 and a.mean() > b.mean()),
        "b_hurts_at_p_0_05": bool(p < 0.05 and a.mean() < b.mean()),
        "median_ratio_a_over_b": float(
            np.median([(p[0] + eps) / (p[1] + eps) for p in paired])
        ),
    }


def difficulty_summary(rows: list[CellResult]) -> dict[str, Any]:
    """Per-difficulty x condition mean/median asymptotic error."""
    out: dict[str, Any] = {}
    for diff in ("easy", "medium", "hard"):
        sub = [r for r in rows if r.difficulty == diff]
        if not sub:
            continue
        per_cond: dict[str, Any] = {}
        for cond in CONDITIONS:
            errs = [r.asym_error for r in sub if r.condition == cond]
            if not errs:
                continue
            per_cond[cond] = {
                "n": len(errs),
                "mean":   float(np.mean(errs)),
                "median": float(np.median(errs)),
                "std":    float(np.std(errs)),
            }
        out[diff] = per_cond
    return out


def per_task_summary(rows: list[CellResult]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tname in sorted({r.task for r in rows}):
        sub = [r for r in rows if r.task == tname]
        diff = sub[0].difficulty
        per_cond: dict[str, Any] = {}
        for cond in CONDITIONS:
            errs = [r.asym_error for r in sub if r.condition == cond]
            if not errs:
                continue
            per_cond[cond] = {
                "n": len(errs),
                "mean": float(np.mean(errs)),
                "errors": [float(e) for e in errs],
            }
        out[tname] = {"difficulty": diff, "by_condition": per_cond}
    return out


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        obj, indent=2,
        default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)
    ))


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("phase_margin.imp_bench")
    log.setLevel(logging.INFO); log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w"); fh.setFormatter(fmt); log.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout);       sh.setFormatter(fmt); log.addHandler(sh)
    return log


# ---------------------------------------------------------------------------
# Backends
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="mock-smart",
                   choices=["mock-smart", "mock-memoryless", "ollama", "anthropic"])
    p.add_argument("--model", default=None)
    p.add_argument("--difficulties", nargs="+",
                   default=["easy", "medium", "hard"],
                   choices=["easy", "medium", "hard"])
    p.add_argument("--conditions", nargs="+",
                   default=list(CONDITIONS),
                   choices=list(CONDITIONS))
    p.add_argument("--task-filter", default=None,
                   help="optional regex; only run tasks whose name matches")
    p.add_argument("--seeds", type=int, default=1,
                   help="number of seeds per (task, condition); default 1 for greedy decoding")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 = greedy decoding (recommended)")
    p.add_argument("--results-root", default="results")
    p.add_argument("--tag", default="")
    args = p.parse_args(argv)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_tag = f"_{args.tag}" if args.tag else ""
    out_dir = Path(args.results_root) / f"experiment_imp_bench_{timestamp}{folder_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out_dir / "log.txt")

    # Filter tasks
    tasks = [t for t in BENCHMARK_TASKS if t.difficulty in args.difficulties]
    if args.task_filter:
        rx = re.compile(args.task_filter)
        tasks = [t for t in tasks if rx.search(t.name)]
    if not tasks:
        log.error("no tasks matched the filters; exiting")
        return 1

    log.info("=" * 76)
    log.info("Internal Model Principle for LLM Agents -- 30-task benchmark")
    log.info(f"output dir : {out_dir}")
    log.info("=" * 76)
    log.info(f"backend    : {args.backend}    model: {args.model}")
    log.info(f"difficulties: {', '.join(args.difficulties)}")
    log.info(f"conditions : {', '.join(args.conditions)}")
    log.info(f"n_tasks    : {len(tasks)}    seeds: {args.seeds}    temp: {args.temperature}")

    client = get_client(args.backend, args.model)
    log.info(f"client     : {client.name}")

    save_json({
        "argv": list(sys.argv),
        "args": vars(args),
        "tasks": [{"name": t.name, "difficulty": t.difficulty,
                   "K": t.K, "asym_window": t.asym_window,
                   "description": t.description} for t in tasks],
        "conditions": list(args.conditions),
    }, out_dir / "manifest.json")

    rows: list[CellResult] = []
    n_total = len(tasks) * len(args.conditions) * args.seeds
    n_done = 0
    t_start = time.time()
    for task in tasks:
        for cond in args.conditions:
            for seed in range(args.seeds):
                t0 = time.time()
                r = run_cell(client=client, task=task, condition=cond,
                             seed=seed, temperature=args.temperature)
                rows.append(r)
                n_done += 1
                log.info(
                    f"[{n_done:3d}/{n_total}] {task.difficulty:6s} {task.name:24s} "
                    f"{cond:20s} seed={seed} asym={r.asym_error:6.2f} "
                    f"({time.time()-t0:.1f}s)"
                )

    save_json([asdict(r) for r in rows], out_dir / "cells.json")
    save_json(per_task_summary(rows), out_dir / "per_task_summary.json")

    diff_summary = difficulty_summary(rows)
    save_json(diff_summary, out_dir / "difficulty_summary.json")

    # Pre-registered statistical tests: for each difficulty band and each
    # IM phrasing, run a paired t-test against reactive baseline.
    tests: dict[str, Any] = {}
    for diff in args.difficulties:
        sub = [r for r in rows if r.difficulty == diff]
        for im_cond in [c for c in args.conditions if c != "reactive"]:
            tests[f"{diff}__reactive_vs_{im_cond}"] = paired_ttest_log(
                sub, cond_a="reactive", cond_b=im_cond
            )
    save_json(tests, out_dir / "paired_ttests.json")

    log.info("\nasymptotic error by difficulty x condition (mean):")
    for diff, per_cond in diff_summary.items():
        line = f"  {diff:6s} "
        for cond in args.conditions:
            v = per_cond.get(cond, {}).get("mean", float("nan"))
            line += f" {cond}={v:6.2f}"
        log.info(line)

    log.info("\npaired t-tests (reactive vs each IM phrasing):")
    for k, v in tests.items():
        if "n_pairs" not in v or v["n_pairs"] < 2:
            log.info(f"  {k:50s}  n_pairs={v.get('n_pairs', 0)}  (skipped)")
            continue
        log.info(
            f"  {k:50s}  n={v['n_pairs']:3d}  "
            f"diff_log={v['mean_diff_log_b_minus_a']:+.3f}  "
            f"t={v['t_stat']:+6.2f}  p={v['p_value_two_sided']:.4g}  "
            f"helps={v['b_helps_at_p_0_05']}  hurts={v['b_hurts_at_p_0_05']}"
        )

    log.info(f"\ntotal wall time: {time.time() - t_start:.1f}s")
    log.info("=" * 76)
    log.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
