#!/usr/bin/env python3
"""
run_imp_benchmark2.py
=====================

Statistical-strength upgrade to ``run_imp_benchmark.py``.

Two upgrades, both targeted at the standard NeurIPS-reviewer critique
of the v1 benchmark ("n=10 is small; single-seed greedy gives no
within-cell variance"):

  1.  PROGRAMMATIC TASK GENERATION.  Each difficulty band is now a
      pool of 100 tasks (configurable) drawn from 10 parametric
      generator families.  v1's 10 hand-picked tasks are subsumed
      as a special case.  Deterministic given ``--task-seed`` so any
      reviewer can regenerate the exact task set.

  2.  MULTI-SEED SAMPLED DECODING.  The default decoding is now
      ``temperature = 0.7`` with ``--n-seeds`` seeds per cell.  This
      is what enables within-cell variance estimation and pairing in
      the (task, seed) rather than just task dimension.

The output is written incrementally to ``cells.jsonl`` so an
overnight run that crashes at hour 8 can be resumed by re-invoking
the same command -- already-completed cells are skipped.

Statistical analyses included in the end-of-run summary:
  * paired t-test on log-error per (band, IM-condition) pairing over
    (task, seed) -- this is the v1 test, now with up to 500 paired
    observations per band rather than 10;
  * cluster-bootstrap 95% CI on the median log-error gap, resampling
    over tasks (not seeds) so the CI reflects the heaviest source of
    variance the reviewer asked about;
  * within-task variance summary so reviewers can see the variance
    decomposition.

The cross-model mixed-effects analysis (log_err ~ cond*diff*scale +
(1|task) + (1|seed)) is not folded in here -- it requires combining
results across model runs and is left to a separate analysis script
that consumes the ``cells.jsonl`` files.

Backends and usage
------------------
  python experiments/run_imp_benchmark2.py --backend mock-smart \\
        --tasks-per-band 30 --n-seeds 3 --tag mock_smoke

  python experiments/run_imp_benchmark2.py --backend ollama --model qwen2.5:7b \\
        --tasks-per-band 100 --n-seeds 5 --tag qwen7b_v2

  python experiments/run_imp_benchmark2.py --backend ollama --model qwen2.5:14b \\
        --tasks-per-band 60  --n-seeds 5 --tag qwen14b_v2

  python experiments/run_imp_benchmark2.py --backend ollama --model qwen2.5:32b \\
        --tasks-per-band 40  --n-seeds 3 --tag qwen32b_v2

  python experiments/run_imp_benchmark2.py --backend ollama \\
        --model llama3.1:70b-instruct-q4_K_M \\
        --tasks-per-band 20  --n-seeds 3 --tag llama70b_v2
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
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


# ===========================================================================
# 1.  PROGRAMMATIC TASK GENERATION
# ===========================================================================
# Each task is a deterministic integer-valued generator f: N -> Z, plus
# metadata (name, difficulty, K, asym_window, optional description).
#
# Family functions take a numpy Generator and return (description_str, fn).
# All targets are clamped via _safe_int(...) so prompts stay short.
# ---------------------------------------------------------------------------
INT_CAP = 50_000   # absolute-value cap on integer targets


def _safe(v) -> int:
    """Clamp to an int with bounded absolute value."""
    if v > INT_CAP:
        return INT_CAP
    if v < -INT_CAP:
        return -INT_CAP
    return int(v)


# ---------- EASY families -------------------------------------------------
def fam_easy_constant(rng: np.random.Generator):
    c = int(rng.integers(-20, 21))
    return f"y = {c}", (lambda k, c=c: _safe(c))


def fam_easy_linear_pos(rng):
    a = int(rng.integers(1, 6))                  # 1..5
    b = int(rng.integers(-10, 21))               # -10..20
    return f"y = {a}k + {b}", (lambda k, a=a, b=b: _safe(a*k + b))


def fam_easy_linear_neg(rng):
    a = int(rng.integers(1, 6))
    b = int(rng.integers(0, 31))
    return f"y = -{a}k + {b}", (lambda k, a=a, b=b: _safe(-a*k + b))


def fam_easy_period2(rng):
    a, b = int(rng.integers(-10, 16)), int(rng.integers(-10, 16))
    if a == b:
        b = a + 1
    return f"cycle [{a},{b}]", (lambda k, a=a, b=b: _safe([a, b][k % 2]))


def fam_easy_period3(rng):
    triple = sorted(set(int(x) for x in rng.choice(np.arange(-10, 16), size=4, replace=False)))[:3]
    while len(triple) < 3:
        triple.append(triple[-1] + 1)
    a, b, c = triple
    return f"cycle [{a},{b},{c}]", (lambda k, abc=(a, b, c): _safe(abc[k % 3]))


def fam_easy_period4_distinct(rng):
    quad = list(rng.choice(np.arange(-10, 16), size=4, replace=False))
    quad = [int(x) for x in quad]
    return ("cycle " + str(quad)), (lambda k, q=quad: _safe(q[k % 4]))


def fam_easy_period4_palindrome(rng):
    a = int(rng.integers(-10, 16)); b = int(rng.integers(-10, 16))
    c = int(rng.integers(-10, 16))
    while c in (a, b):
        c = int(rng.integers(-10, 16))
    return f"cycle [{a},{b},{a},{c}]", \
           (lambda k, a=a, b=b, c=c: _safe([a, b, a, c][k % 4]))


def fam_easy_mod_n(rng):
    n = int(rng.integers(3, 8))
    return f"y = k mod {n}", (lambda k, n=n: _safe(k % n))


def fam_easy_arith_step(rng):
    a = int(rng.choice(list(range(-7, 0)) + list(range(1, 8))))
    return f"y = {a}k", (lambda k, a=a: _safe(a*k))


def fam_easy_arith_offset(rng):
    a = int(rng.integers(1, 6))
    c = int(rng.choice(list(range(-15, 0)) + list(range(1, 16))))
    return f"y = {a}k + {c}", (lambda k, a=a, c=c: _safe(a*k + c))


# ---------- MEDIUM families -----------------------------------------------
def fam_med_quadratic(rng):
    return "y = k^2", (lambda k: _safe(k * k))


def fam_med_quadratic_offset(rng):
    a = int(rng.integers(0, 6))
    return f"y = (k - {a})^2", (lambda k, a=a: _safe((k - a) ** 2))


def fam_med_quadratic_neg(rng):
    a = int(rng.integers(4, 13))
    return f"y = -k^2 + {a}k", (lambda k, a=a: _safe(-k*k + a*k))


def fam_med_period5(rng):
    cyc = list(rng.choice(np.arange(-5, 11), size=5, replace=True))
    cyc = [int(x) for x in cyc]
    return ("cycle " + str(cyc)), (lambda k, c=cyc: _safe(c[k % 5]))


def fam_med_period6(rng):
    cyc = list(rng.choice(np.arange(-5, 11), size=6, replace=True))
    cyc = [int(x) for x in cyc]
    return ("cycle " + str(cyc)), (lambda k, c=cyc: _safe(c[k % 6]))


def fam_med_piecewise_v(rng):
    pivot = int(rng.integers(3, 7))
    height = int(rng.integers(5, 12))
    def fn(k, p=pivot, h=height):
        return _safe(k if k < p else max(0, h - k))
    return f"y = k if k<{pivot} else max(0,{height}-k)", fn


def fam_med_zigzag(rng):
    s = int(rng.integers(1, 4))
    return f"y = {s}*k * (-1)^k", (lambda k, s=s: _safe(s*k*((-1)**k)))


def fam_med_step_floor(rng):
    n = int(rng.integers(2, 6))
    return f"y = floor(k/{n})", (lambda k, n=n: _safe(k // n))


def fam_med_triangle_wave(rng):
    period = int(rng.choice([4, 6, 8]))
    amp = period // 2
    return f"triangle wave period={period} amp={amp}", \
           (lambda k, p=period, a=amp: _safe(abs((k % p) - a)))


def fam_med_linear_plus_period(rng):
    a = int(rng.integers(1, 4))
    b = int(rng.integers(2, 6))
    return f"y = {a}k + {b}*(k%2)", (lambda k, a=a, b=b: _safe(a*k + b*(k % 2)))


# ---------- HARD families -------------------------------------------------
def fam_hard_fibonacci(rng):
    a = int(rng.integers(1, 4))
    b = int(rng.integers(1, 4))
    cache = {0: a, 1: b}
    def fn(k, c=cache):
        if k in c:
            return _safe(c[k])
        for j in range(max(c) + 1, k + 1):
            c[j] = _safe(c[j-1] + c[j-2])
        return _safe(c[k])
    return f"f(0)={a},f(1)={b},f(k)=f(k-1)+f(k-2)", fn


def fam_hard_geometric_2(rng):
    c = int(rng.integers(1, 4))
    def fn(k, c=c):
        v = c * (2 ** min(k, 14))      # cap exponent so prompts don't explode
        return _safe(v)
    return f"y = {c}*2^k (capped)", fn


def fam_hard_polynomial_kkminus1(rng):
    return "y = k(k-1)", (lambda k: _safe(k * (k - 1)))


def fam_hard_triangle_numbers(rng):
    return "y = k(k+1)/2", (lambda k: _safe(k * (k + 1) // 2))


def fam_hard_digit_cycle(rng):
    """Cycle of length L drawn from a random 0-9 string."""
    L = int(rng.choice([10, 12, 14, 16]))
    digits = [int(d) for d in rng.integers(0, 10, size=L)]
    return ("digit cycle len " + str(L)), (lambda k, d=digits, L=L: _safe(d[k % L]))


def fam_hard_cubic_modn(rng):
    n = int(rng.integers(50, 151))
    return f"y = k^3 mod {n}", (lambda k, n=n: _safe((k ** 3) % n))


def fam_hard_conditional_rule(rng):
    m = int(rng.integers(2, 5))
    a = int(rng.integers(2, 5))
    b = int(rng.integers(0, 5))
    return f"y = {a}k if k%{m}==0 else k+{b}", \
           (lambda k, m=m, a=a, b=b: _safe(a*k if k % m == 0 else k + b))


def fam_hard_regime_switch(rng):
    pivot = int(rng.integers(5, 9))
    A = [int(rng.integers(-5, 11)), int(rng.integers(-5, 11))]
    B = [int(rng.integers(-5, 11)), int(rng.integers(-5, 11))]
    return f"cycle {A} for k<{pivot} then cycle {B}", \
           (lambda k, p=pivot, A=A, B=B: _safe((A if k < p else B)[k % 2]))


def fam_hard_sum_two_periods(rng):
    a = int(rng.choice([2, 3, 4]))
    b = int(rng.choice([3, 5, 7]))
    while b == a:
        b = int(rng.choice([3, 5, 7]))
    return f"y = (k%{a}) + (k%{b})", (lambda k, a=a, b=b: _safe((k % a) + (k % b)))


def fam_hard_polynomial_general(rng):
    a = int(rng.integers(-3, 4))
    b = int(rng.integers(-5, 6))
    c = int(rng.integers(-5, 6))
    return f"y = {a}k^2 + {b}k + {c}", \
           (lambda k, a=a, b=b, c=c: _safe(a*k*k + b*k + c))


# ---------- Family registry ----------------------------------------------
EASY_FAMILIES: list[Callable] = [
    fam_easy_constant, fam_easy_linear_pos, fam_easy_linear_neg,
    fam_easy_period2, fam_easy_period3, fam_easy_period4_distinct,
    fam_easy_period4_palindrome, fam_easy_mod_n, fam_easy_arith_step,
    fam_easy_arith_offset,
]

MEDIUM_FAMILIES: list[Callable] = [
    fam_med_quadratic, fam_med_quadratic_offset, fam_med_quadratic_neg,
    fam_med_period5, fam_med_period6, fam_med_piecewise_v, fam_med_zigzag,
    fam_med_step_floor, fam_med_triangle_wave, fam_med_linear_plus_period,
]

HARD_FAMILIES: list[Callable] = [
    fam_hard_fibonacci, fam_hard_geometric_2, fam_hard_polynomial_kkminus1,
    fam_hard_triangle_numbers, fam_hard_digit_cycle, fam_hard_cubic_modn,
    fam_hard_conditional_rule, fam_hard_regime_switch,
    fam_hard_sum_two_periods, fam_hard_polynomial_general,
]

BAND_FAMILIES = {"easy": EASY_FAMILIES, "medium": MEDIUM_FAMILIES, "hard": HARD_FAMILIES}


# ---------- Task dataclass ------------------------------------------------
@dataclass
class Task:
    name: str
    difficulty: str
    family: str
    description: str
    K: int
    asym_window: int
    targets: list[int] = field(default_factory=list)


def build_task_set(
    n_per_band: int, task_seed: int, K: int, asym_window: int,
) -> list[Task]:
    """Sample ``n_per_band`` tasks per difficulty band deterministically."""
    rng_master = np.random.default_rng(task_seed)
    tasks: list[Task] = []
    for diff, families in BAND_FAMILIES.items():
        # round-robin: instances per family = ceil(n_per_band / n_families)
        n_fam = len(families)
        per_fam = math.ceil(n_per_band / n_fam)
        slot = 0
        for fi, fam in enumerate(families):
            for inst in range(per_fam):
                if slot >= n_per_band:
                    break
                instance_seed = int(rng_master.integers(2**31))
                rng = np.random.default_rng(instance_seed)
                desc, fn = fam(rng)
                # Precompute targets through K to keep the runtime path
                # generator-free (and to detect overflow at task-build time).
                try:
                    targets = [int(fn(k)) for k in range(K)]
                except Exception:                          # noqa: BLE001
                    continue
                fam_short = fam.__name__.replace("fam_", "")
                tname = f"{diff[0]}_{slot:03d}_{fam_short}"
                tasks.append(Task(
                    name=tname, difficulty=diff, family=fam_short,
                    description=desc, K=K, asym_window=asym_window,
                    targets=targets,
                ))
                slot += 1
            if slot >= n_per_band:
                break
    return tasks


# ===========================================================================
# 2.  PROMPTS  (verbatim from v1 -- do not change without re-running v1)
# ===========================================================================
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

CONDITIONS = {
    "reactive": SYSTEM_REACTIVE,
    "internal_hypothesis": SYSTEM_IM_HYPOTHESIS,
    "internal_cot": SYSTEM_IM_COT,
}


def build_user_prompt(history: list[tuple[int, int]], k_next: int) -> str:
    if not history:
        return f"No observations yet. Predict y_{k_next}."
    obs = "; ".join(f"k={k}: y={y}" for k, y in history)
    return f"Observations so far: {obs}\n\nPredict y_{k_next}."


def parse_prediction(text: str) -> int | None:
    if not text:
        return None
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    for ln in reversed(lines):
        m = re.findall(r"-?\d+", ln)
        if m:
            return int(m[-1])
    matches = re.findall(r"-?\d+", text)
    return int(matches[-1]) if matches else None


# ===========================================================================
# 3.  Mock backends (smoke testing only)
# ===========================================================================
class MockSmartClient(LLMClient):
    @property
    def name(self): return "mock:smart"

    def chat(self, messages, *, seed=None, temperature=0.7, max_tokens=256):
        sys_msg = next((m.content for m in messages if m.role == "system"), "")
        user = next((m.content for m in messages if m.role == "user"), "")
        history = [(int(a), int(b)) for a, b in re.findall(r"k=(-?\d+):\s*y=(-?\d+)", user)]
        if not history:
            return "0"
        next_k = int(re.search(r"y_(\d+)", user).group(1))
        is_im = ("Internal Model Principle" in sys_msg) or ("step by step" in sys_msg)
        if is_im:
            ys = [y for _, y in history]
            for period in range(2, 8):
                if len(ys) >= 2 * period and all(
                    ys[-i - 1] == ys[-i - 1 - period] for i in range(period)
                ):
                    pred = ys[len(ys) - period + (next_k - len(ys)) % period]
                    return f"HYPOTHESIS: period-{period}\n{pred}"
            if len(ys) >= 3:
                ks = np.array([k for k, _ in history], dtype=float)
                xs = np.array(ys, dtype=float)
                slope, intercept = np.polyfit(ks, xs, 1)
                pred = int(round(slope * next_k + intercept))
                return f"HYPOTHESIS: linear\n{pred}"
            return f"HYPOTHESIS: const\n{int(round(np.mean(ys)))}"
        return str(history[-1][1])


class MockMemorylessClient(MockSmartClient):
    @property
    def name(self): return "mock:memoryless"

    def chat(self, messages, *, seed=None, temperature=0.7, max_tokens=256):
        user = next((m.content for m in messages if m.role == "user"), "")
        history = [(int(a), int(b)) for a, b in re.findall(r"k=(-?\d+):\s*y=(-?\d+)", user)]
        return "0" if not history else str(history[-1][1])


# ===========================================================================
# 4.  Cell runner with resume support
# ===========================================================================
@dataclass
class CellResult:
    backend: str
    model: str
    task: str
    family: str
    difficulty: str
    condition: str
    seed: int
    K: int
    asym_window: int
    temperature: float
    targets: list[int]
    predictions: list[int | None]
    raw_replies: list[str]
    asym_error: float
    elapsed_s: float


def cell_key(model: str, task: str, condition: str, seed: int) -> str:
    return f"{model}::{task}::{condition}::{seed}"


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
    targets = list(task.targets)
    preds: list[int | None] = []
    replies: list[str] = []
    t0 = time.time()
    for k in range(task.K):
        msgs = [
            Message(role="system", content=sys_prompt),
            Message(role="user",   content=build_user_prompt(history, k)),
        ]
        max_tok = 256 if condition == "internal_cot" else 96
        reply = client.chat(msgs, seed=seed * 10_000 + k,
                            temperature=temperature, max_tokens=max_tok)
        replies.append(reply or "")
        preds.append(parse_prediction(reply))
        history.append((k, int(targets[k])))
    last = preds[-task.asym_window:]
    last_true = targets[-task.asym_window:]
    abs_err = [abs(p - t) for p, t in zip(last, last_true) if p is not None]
    asym = float(np.mean(abs_err)) if abs_err else float("inf")
    return CellResult(
        backend=client.name.split(":")[0], model=client.name,
        task=task.name, family=task.family, difficulty=task.difficulty,
        condition=condition, seed=seed,
        K=task.K, asym_window=task.asym_window, temperature=temperature,
        targets=targets, predictions=preds, raw_replies=replies,
        asym_error=asym, elapsed_s=time.time() - t0,
    )


# ===========================================================================
# 5.  Statistics:  paired t + cluster-bootstrap CI on (band, IM-cond) pair
# ===========================================================================
def _aggregate_per_pair(rows: list[CellResult], cond_a: str, cond_b: str
                        ) -> dict[tuple[str, int], tuple[float, float]]:
    """Map (task, seed) -> (err_a, err_b).  Cells lacking either are dropped."""
    by_key: dict[tuple[str, int], dict[str, float]] = {}
    for r in rows:
        by_key.setdefault((r.task, r.seed), {})[r.condition] = r.asym_error
    pairs = {}
    for k, v in by_key.items():
        if cond_a in v and cond_b in v:
            pairs[k] = (v[cond_a], v[cond_b])
    return pairs


def paired_t_log(rows: list[CellResult], *, cond_a: str, cond_b: str,
                 eps: float = 0.5) -> dict[str, Any]:
    """Paired t over (task, seed) on log(err+eps).  Reports a vs b: positive
    t means cond_b's log-error is larger -> cond_b hurts."""
    pairs = _aggregate_per_pair(rows, cond_a, cond_b)
    if len(pairs) < 2:
        return {"n_pairs": len(pairs)}
    a = np.array([np.log(va + eps) for va, _ in pairs.values()])
    b = np.array([np.log(vb + eps) for _, vb in pairs.values()])
    diff = b - a
    from scipy.stats import ttest_rel
    t, p = ttest_rel(a, b)
    return {
        "cond_a": cond_a, "cond_b": cond_b, "n_pairs": len(pairs),
        "mean_log_a": float(a.mean()), "mean_log_b": float(b.mean()),
        "mean_diff_log_b_minus_a": float(diff.mean()),
        "median_diff_log_b_minus_a": float(np.median(diff)),
        "t_stat": float(t), "p_value_two_sided": float(p),
        "b_helps_at_p_0_05": bool(p < 0.05 and a.mean() > b.mean()),
        "b_hurts_at_p_0_05": bool(p < 0.05 and a.mean() < b.mean()),
    }


def cluster_bootstrap_ci(rows: list[CellResult], *, cond_a: str, cond_b: str,
                         n_boot: int = 5000, eps: float = 0.5,
                         alpha: float = 0.05, rng_seed: int = 0,
                         ) -> dict[str, Any]:
    """Resample tasks (the cluster), keep all seeds within each task, compute
    median log-difference per resample, return percentile CI.  Resampling at
    the task level is what addresses the reviewer's "sensitive to which 10
    tasks?" concern."""
    pairs = _aggregate_per_pair(rows, cond_a, cond_b)
    if len(pairs) < 2:
        return {"n_pairs": len(pairs)}
    by_task: dict[str, list[tuple[float, float]]] = {}
    for (task, _seed), (va, vb) in pairs.items():
        by_task.setdefault(task, []).append((va, vb))
    tasks = list(by_task.keys())
    rng = np.random.default_rng(rng_seed)
    medians = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(tasks), size=len(tasks))
        diffs = []
        for j in idx:
            for va, vb in by_task[tasks[j]]:
                diffs.append(np.log(vb + eps) - np.log(va + eps))
        if diffs:
            medians.append(float(np.median(diffs)))
    medians = np.array(medians)
    lo = float(np.percentile(medians, 100 * (alpha / 2)))
    hi = float(np.percentile(medians, 100 * (1 - alpha / 2)))
    point = float(np.median(medians))
    return {
        "cond_a": cond_a, "cond_b": cond_b,
        "n_tasks": len(tasks), "n_boot": n_boot,
        "median_diff_log_point": point,
        "ci_lo": lo, "ci_hi": hi,
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "ci_direction": ("b_hurts" if lo > 0 else "b_helps" if hi < 0 else "inconclusive"),
    }


def variance_decomp(rows: list[CellResult]) -> dict[str, Any]:
    """Within-task vs across-task variance of log-error per (band, condition)."""
    out = {}
    for diff in ("easy", "medium", "hard"):
        for cond in CONDITIONS:
            sub = [r for r in rows if r.difficulty == diff and r.condition == cond]
            if len(sub) < 2:
                continue
            by_task: dict[str, list[float]] = {}
            for r in sub:
                by_task.setdefault(r.task, []).append(np.log(r.asym_error + 0.5))
            within = []
            task_means = []
            for ys in by_task.values():
                if len(ys) >= 2:
                    within.append(np.var(ys, ddof=1))
                task_means.append(np.mean(ys))
            out[f"{diff}__{cond}"] = {
                "n_tasks": len(by_task),
                "n_total_cells": len(sub),
                "mean_within_task_var": float(np.mean(within)) if within else None,
                "across_task_var": float(np.var(task_means, ddof=1)) if len(task_means) > 1 else None,
            }
    return out


# ===========================================================================
# 6.  IO + resume
# ===========================================================================
def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2,
                               default=lambda o: o.tolist()
                               if hasattr(o, 'tolist') else str(o)))


def load_done_keys(jsonl_path: Path) -> set[str]:
    """Return the cell_key()s already in jsonl (for resume)."""
    if not jsonl_path.exists():
        return set()
    done = set()
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add(cell_key(d["model"], d["task"], d["condition"], d["seed"]))
    return done


def load_cells(jsonl_path: Path) -> list[CellResult]:
    rows = []
    if not jsonl_path.exists():
        return rows
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(CellResult(**d))
    return rows


def append_cell(jsonl_path: Path, r: CellResult) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a") as f:
        f.write(json.dumps(asdict(r), default=str) + "\n")


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("phase_margin.imp_bench2")
    log.setLevel(logging.INFO); log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a"); fh.setFormatter(fmt); log.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout);       sh.setFormatter(fmt); log.addHandler(sh)
    return log


# ===========================================================================
# 7.  Backends
# ===========================================================================
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


# ===========================================================================
# 8.  Main
# ===========================================================================
def fmt_eta(sec: float) -> str:
    if sec < 90:
        return f"{sec:.0f}s"
    if sec < 5400:
        return f"{sec/60:.1f}m"
    return f"{sec/3600:.1f}h"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="mock-smart",
                   choices=["mock-smart", "mock-memoryless", "ollama", "anthropic"])
    p.add_argument("--model", default=None)
    p.add_argument("--tasks-per-band", type=int, default=100,
                   help="number of tasks per difficulty band (10 families * "
                        "ceil(n/10) instances)")
    p.add_argument("--difficulties", nargs="+",
                   default=["easy", "medium", "hard"],
                   choices=["easy", "medium", "hard"])
    p.add_argument("--conditions", nargs="+",
                   default=list(CONDITIONS),
                   choices=list(CONDITIONS))
    p.add_argument("--n-seeds", type=int, default=5,
                   help="number of LLM seeds per cell at temperature > 0")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="0.0 = greedy (collapses seeds); 0.7 = recommended")
    p.add_argument("--K", type=int, default=12, dest="K_steps",
                   help="trajectory length")
    p.add_argument("--asym-window", type=int, default=5)
    p.add_argument("--task-seed", type=int, default=2026,
                   help="deterministic task generation seed; same seed reproduces "
                        "the same task set across machines")
    p.add_argument("--results-root", default="results")
    p.add_argument("--tag", default="")
    p.add_argument("--results-dir", default=None,
                   help="resume into this exact dir if it exists; otherwise "
                        "a fresh timestamped dir is created")
    p.add_argument("--n-boot", type=int, default=5000)
    args = p.parse_args(argv)

    # Output dir (resume vs fresh)
    if args.results_dir:
        out_dir = Path(args.results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_tag = f"_{args.tag}" if args.tag else ""
        out_dir = Path(args.results_root) / f"experiment_imp_bench2_{timestamp}{folder_tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out_dir / "log.txt")
    cells_path = out_dir / "cells.jsonl"

    # Build the task set deterministically
    tasks_all = build_task_set(
        n_per_band=args.tasks_per_band,
        task_seed=args.task_seed,
        K=args.K_steps,
        asym_window=args.asym_window,
    )
    tasks = [t for t in tasks_all if t.difficulty in args.difficulties]

    # Manifest
    save_json({
        "argv": list(sys.argv),
        "args": vars(args),
        "n_tasks_total": len(tasks),
        "n_per_band": {
            d: sum(1 for t in tasks if t.difficulty == d)
            for d in ["easy", "medium", "hard"]
        },
        "tasks": [
            {"name": t.name, "difficulty": t.difficulty,
             "family": t.family, "description": t.description,
             "K": t.K, "asym_window": t.asym_window,
             "first_six_targets": t.targets[:6]}
            for t in tasks
        ],
    }, out_dir / "manifest.json")

    # Resume
    done = load_done_keys(cells_path)
    log.info("=" * 76)
    log.info("Internal Model Principle for LLM Agents -- benchmark v2 (multi-seed)")
    log.info(f"output dir : {out_dir}")
    log.info("=" * 76)
    log.info(f"backend     : {args.backend}    model: {args.model}")
    log.info(f"tasks/band  : {args.tasks_per_band}    total: {len(tasks)}")
    log.info(f"conditions  : {', '.join(args.conditions)}")
    log.info(f"n_seeds     : {args.n_seeds}    temperature: {args.temperature}")
    log.info(f"K, asym_w   : {args.K_steps}, {args.asym_window}")
    log.info(f"task_seed   : {args.task_seed}    (regenerable)")
    log.info(f"resume      : {len(done)} cells already done")

    client = get_client(args.backend, args.model)
    log.info(f"client      : {client.name}")

    n_total = len(tasks) * len(args.conditions) * args.n_seeds
    n_done = len(done)
    n_remaining = n_total - n_done
    log.info(f"cells total : {n_total}    remaining: {n_remaining}")

    rows: list[CellResult] = load_cells(cells_path)
    t_start = time.time()
    n_run_this_session = 0
    for task in tasks:
        for cond in args.conditions:
            for seed in range(args.n_seeds):
                key = cell_key(client.name, task.name, cond, seed)
                if key in done:
                    continue
                t0 = time.time()
                r = run_cell(client=client, task=task, condition=cond,
                             seed=seed, temperature=args.temperature)
                rows.append(r)
                done.add(key)
                append_cell(cells_path, r)
                n_run_this_session += 1
                # ETA
                elapsed = time.time() - t_start
                rate = n_run_this_session / max(elapsed, 1e-3)
                remaining_now = n_total - len(done)
                eta_sec = remaining_now / max(rate, 1e-9)
                log.info(
                    f"[{len(done):4d}/{n_total}] {task.difficulty:6s} "
                    f"{task.family:24s} {cond:20s} s={seed} "
                    f"asym={r.asym_error:7.2f} ({time.time()-t0:.1f}s) "
                    f"eta={fmt_eta(eta_sec)}"
                )

    # Final summary
    log.info("\nFinal summary:")
    summary: dict[str, Any] = {
        "n_cells": len(rows),
        "by_difficulty_condition": {},
        "paired_ttests": {},
        "bootstrap_ci": {},
        "variance_decomposition": variance_decomp(rows),
    }
    for diff in args.difficulties:
        for cond in args.conditions:
            sub = [r for r in rows if r.difficulty == diff and r.condition == cond]
            if not sub:
                continue
            errs = np.array([r.asym_error for r in sub])
            summary["by_difficulty_condition"][f"{diff}__{cond}"] = {
                "n_cells": len(sub),
                "mean":   float(np.mean(errs)),
                "median": float(np.median(errs)),
                "std":    float(np.std(errs)),
            }

    for diff in args.difficulties:
        sub = [r for r in rows if r.difficulty == diff]
        for im in [c for c in args.conditions if c != "reactive"]:
            tt = paired_t_log(sub, cond_a="reactive", cond_b=im)
            summary["paired_ttests"][f"{diff}__reactive_vs_{im}"] = tt
            ci = cluster_bootstrap_ci(sub, cond_a="reactive", cond_b=im,
                                      n_boot=args.n_boot, rng_seed=42)
            summary["bootstrap_ci"][f"{diff}__reactive_vs_{im}"] = ci

    save_json(summary, out_dir / "summary.json")

    log.info("\nasymptotic error by difficulty x condition (mean):")
    for diff in args.difficulties:
        line = f"  {diff:6s} "
        for cond in args.conditions:
            v = summary["by_difficulty_condition"].get(
                f"{diff}__{cond}", {}).get("mean", float("nan"))
            line += f" {cond:>22s}={v:6.2f}"
        log.info(line)

    log.info("\npaired t-tests (reactive vs each IM):")
    for k, v in summary["paired_ttests"].items():
        if v.get("n_pairs", 0) < 2:
            log.info(f"  {k:46s} n={v.get('n_pairs',0)}  (skipped)")
            continue
        log.info(
            f"  {k:46s} n={v['n_pairs']:4d} "
            f"diff={v['mean_diff_log_b_minus_a']:+.3f} "
            f"t={v['t_stat']:+6.2f} p={v['p_value_two_sided']:.3g} "
            f"hurts={v['b_hurts_at_p_0_05']}"
        )

    log.info("\ncluster-bootstrap 95% CI (resampling tasks):")
    for k, v in summary["bootstrap_ci"].items():
        if v.get("n_tasks", 0) < 2:
            log.info(f"  {k:46s} n_tasks={v.get('n_tasks',0)}  (skipped)")
            continue
        log.info(
            f"  {k:46s} n_tasks={v['n_tasks']:4d} "
            f"med={v['median_diff_log_point']:+.3f} "
            f"CI=[{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]  "
            f"{v['ci_direction']}"
        )

    log.info(f"\nwall time this session: {fmt_eta(time.time() - t_start)}")
    log.info("=" * 76)
    log.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
