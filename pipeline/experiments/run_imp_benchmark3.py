#!/usr/bin/env python3
"""
run_imp_benchmark3.py
=====================

Mechanism-test extension to ``run_imp_benchmark2.py``.

Benchmark v3 adds four new prompting arms designed to isolate the
mechanism behind the IM-Hypothesis harm documented in v1 and v2.
The new arms are:

  1. ``im_hyp_sc5``     -- IM-Hypothesis with SELF-CONSISTENCY over
                            N=5 sampled hypotheses per step.
                            Predicts via majority vote over the
                            5 parsed integers.  Direct test of the
                            tokenized-commitment mechanism: if
                            committing to a single wrong hypothesis
                            is the cause, aggregating over multiple
                            sampled hypotheses should restore
                            reactive-level performance.

  2. ``im_oracle``      -- The TRUE generator rule is given to the
                            agent in the system prompt; the agent
                            is told to apply it.  Isolates the
                            "wrong rule" component from the
                            "articulation-per-se" component.  If
                            im_oracle <= reactive, articulation of
                            a CORRECT rule is harmless and the
                            entire harm is attributable to wrong
                            rules.  If im_oracle > reactive,
                            articulation per se is part of the
                            problem.

  3. ``im_hyp_first``   -- Reverses the order: integer FIRST, then
                            HYPOTHESIS line.  Direct test of
                            commitment direction: under the
                            tokenized-commitment account, the
                            integer being generated before any rule
                            is articulated should make this arm
                            indistinguishable from reactive.

  4. ``im_hyp_eqbudget``-- Same as ``im_hypothesis`` (v2) but with
                            ``max_tokens = 256`` to match
                            ``im_cot``.  Rules out the budget
                            confound flagged as Limitation 5.

The same task pool from v2 is reused (deterministic given
``--task-seed=2026``) so the new arms pair directly task-for-task,
seed-for-seed with the existing v2 reactive / im_hypothesis /
im_cot data.

Usage
-----
  python experiments/run_imp_benchmark3.py --backend mock-smart \\
        --tasks-per-band 30 --n-seeds 3 --tag mock_v3

  # Recommended overnight on H100 (parallel-friendly with --skip-conditions
  # to split arms across runs):
  python experiments/run_imp_benchmark3.py --backend ollama \\
        --model qwen2.5:7b \\
        --tasks-per-band 30 --n-seeds 3 \\
        --conditions im_hyp_sc5 im_oracle im_hyp_first im_hyp_eqbudget \\
        --tag qwen7b_v3

The reactive / im_hypothesis / im_cot conditions are NOT re-run by
default (they exist in v2).  Pass ``--include-baselines`` to also
re-run them on the v3 task pool for direct comparison.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "src"))

from phase_margin.llm.base import LLMClient, Message

# Reuse v2 task generation and base prompts so v2 and v3 are paired
import run_imp_benchmark2 as v2


# ===========================================================================
# 1.  Prompts -- new arms (v2 prompts reused via aliases)
# ===========================================================================
SYSTEM_REACTIVE = v2.SYSTEM_REACTIVE
SYSTEM_IM_HYPOTHESIS = v2.SYSTEM_IM_HYPOTHESIS
SYSTEM_IM_COT = v2.SYSTEM_IM_COT


SYSTEM_IM_HYP_FIRST = (
    "You are a sequence-prediction agent.  At each step you will "
    "be shown all observations so far and asked to predict the "
    "next value y_k.  Write your single integer prediction FIRST "
    "on its own line, then on a NEW LINE state a one-line "
    "hypothesis about the function f(k) that generated the "
    "observed values.\n\n"
    "Reply format (exactly two lines):\n"
    "    <integer prediction>\n"
    "    HYPOTHESIS: <your one-line hypothesis>"
)


def system_im_oracle(task_description: str) -> str:
    return (
        "You are a sequence-prediction agent.  The function "
        "generating the observed values is exactly:\n\n"
        f"    f(k) = {task_description}\n\n"
        "At each step you will be shown all observations so far "
        "and asked to predict the next value y_k.  Apply the rule "
        "above and reply with EXACTLY one integer on a line by "
        "itself.  No commentary, no labels, no quotes."
    )


CONDITIONS = {
    # Names match v2 (internal_*) so cells.jsonl from v2 and v3 can be
    # joined task-for-task seed-for-seed without remapping.
    "reactive":             SYSTEM_REACTIVE,            # v2 baseline
    "internal_hypothesis":  SYSTEM_IM_HYPOTHESIS,       # v2 main arm
    "internal_cot":         SYSTEM_IM_COT,              # v2 CoT arm
    "im_hyp_sc5":           SYSTEM_IM_HYPOTHESIS,       # v3 NEW (uses base hyp prompt)
    "im_oracle":            "<dynamic>",                 # v3 NEW (per-task)
    "im_hyp_first":         SYSTEM_IM_HYP_FIRST,        # v3 NEW
    "im_hyp_eqbudget":      SYSTEM_IM_HYPOTHESIS,       # v3 NEW (same prompt, more tokens)
}

NEW_V3_CONDITIONS = ["im_hyp_sc5", "im_oracle", "im_hyp_first", "im_hyp_eqbudget"]
BASELINE_CONDITIONS = ["reactive", "internal_hypothesis", "internal_cot"]


# ===========================================================================
# 2.  Cell runner with the new arm semantics
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
    sc_samples: list[list[str]]   # only populated for sc5; otherwise empty per-step lists
    sc_votes:   list[list[int]]   # parsed integers per sample, per step
    asym_error: float
    elapsed_s: float


def _parse_first_line_int(text: str) -> int | None:
    """Parse the integer on the FIRST non-empty line of ``text``."""
    if not text:
        return None
    for ln in text.strip().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        m = re.findall(r"-?\d+", ln)
        if m:
            return int(m[0])
    return None


def _parse_last_line_int(text: str) -> int | None:
    """v2's behaviour: integer from the LAST non-empty line."""
    return v2.parse_prediction(text)


def _majority(ints: list[int | None]) -> int | None:
    """Majority vote over a list of (possibly None) integers.  Ties broken
    by the median of the non-None values."""
    keep = [x for x in ints if x is not None]
    if not keep:
        return None
    cnt = Counter(keep)
    top, top_n = cnt.most_common(1)[0]
    runners_up = [v for v, n in cnt.most_common() if n == top_n]
    if len(runners_up) == 1:
        return int(top)
    return int(round(np.median(runners_up)))


def run_cell_v3(
    *,
    client: LLMClient,
    task: v2.Task,
    condition: str,
    seed: int,
    temperature: float,
    n_sc_samples: int = 5,
) -> CellResult:
    """Single cell, v3 condition semantics."""
    history: list[tuple[int, int]] = []
    targets = list(task.targets)
    preds: list[int | None] = []
    replies: list[str] = []
    sc_samples_per_step: list[list[str]] = []
    sc_votes_per_step:   list[list[int]] = []

    # Resolve per-condition system prompt and decoding settings
    if condition == "im_oracle":
        sys_prompt = system_im_oracle(task.description)
    else:
        sys_prompt = CONDITIONS[condition]
    if condition in ("internal_cot", "im_hyp_eqbudget"):
        max_tok = 256
    else:
        max_tok = 96

    t0 = time.time()
    for k in range(task.K):
        msgs = [
            Message(role="system", content=sys_prompt),
            Message(role="user", content=v2.build_user_prompt(history, k)),
        ]
        if condition == "im_hyp_sc5":
            # Sample N independent hypotheses, each at temperature > 0,
            # then majority-vote the parsed integer.
            samples_text: list[str] = []
            samples_int:  list[int | None] = []
            for s in range(n_sc_samples):
                reply = client.chat(
                    msgs,
                    seed=seed * 100_000 + k * 100 + s,
                    temperature=max(temperature, 0.5),  # SC needs sampling
                    max_tokens=max_tok,
                )
                samples_text.append(reply or "")
                samples_int.append(_parse_last_line_int(reply))
            sc_samples_per_step.append(samples_text)
            sc_votes_per_step.append([x for x in samples_int if x is not None])
            replies.append(samples_text[0])  # representative reply
            preds.append(_majority(samples_int))
        else:
            reply = client.chat(
                msgs,
                seed=seed * 10_000 + k,
                temperature=temperature,
                max_tokens=max_tok,
            )
            replies.append(reply or "")
            sc_samples_per_step.append([])
            sc_votes_per_step.append([])
            # Parsing direction: im_hyp_first wants the FIRST integer line.
            if condition == "im_hyp_first":
                preds.append(_parse_first_line_int(reply))
            else:
                preds.append(_parse_last_line_int(reply))
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
        sc_samples=sc_samples_per_step, sc_votes=sc_votes_per_step,
        asym_error=asym, elapsed_s=time.time() - t0,
    )


# ===========================================================================
# 3.  IO + resume
# ===========================================================================
def cell_key(model: str, task: str, condition: str, seed: int) -> str:
    return f"{model}::{task}::{condition}::{seed}"


def load_done_keys(jsonl_path: Path) -> set[str]:
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


def append_cell(jsonl_path: Path, r: CellResult) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a") as f:
        f.write(json.dumps(asdict(r), default=str) + "\n")


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("phase_margin.imp_bench3")
    log.setLevel(logging.INFO); log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a"); fh.setFormatter(fmt); log.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout);       sh.setFormatter(fmt); log.addHandler(sh)
    return log


def fmt_eta(sec: float) -> str:
    if sec < 90: return f"{sec:.0f}s"
    if sec < 5400: return f"{sec/60:.1f}m"
    return f"{sec/3600:.1f}h"


def get_client(backend: str, model: str | None) -> LLMClient:
    if backend == "mock-smart":
        return v2.MockSmartClient()
    if backend == "mock-memoryless":
        return v2.MockMemorylessClient()
    if backend == "ollama":
        from phase_margin.llm.ollama_client import OllamaClient
        return OllamaClient(model=model or "qwen2.5:7b")
    if backend == "anthropic":
        from phase_margin.llm.anthropic_client import AnthropicClient
        return AnthropicClient(model=model or "claude-haiku-4-5-20251001",
                               cache_dir="data/cache/anthropic")
    raise ValueError(f"unknown backend: {backend}")


# ===========================================================================
# 4.  Main
# ===========================================================================
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="mock-smart",
                   choices=["mock-smart", "mock-memoryless", "ollama", "anthropic"])
    p.add_argument("--model", default=None)
    p.add_argument("--tasks-per-band", type=int, default=30)
    p.add_argument("--difficulties", nargs="+",
                   default=["easy", "medium", "hard"],
                   choices=["easy", "medium", "hard"])
    p.add_argument("--conditions", nargs="+",
                   default=NEW_V3_CONDITIONS,
                   choices=BASELINE_CONDITIONS + NEW_V3_CONDITIONS,
                   help="conditions to run; default = the four NEW v3 arms")
    p.add_argument("--include-baselines", action="store_true",
                   help="also run reactive / im_hypothesis / im_cot on the "
                        "v3 task pool (useful for self-contained v3 tables)")
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--n-sc-samples", type=int, default=5,
                   help="hypotheses per step in im_hyp_sc5")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--K", type=int, default=12, dest="K_steps")
    p.add_argument("--asym-window", type=int, default=5)
    p.add_argument("--task-seed", type=int, default=2026,
                   help="MUST match v2 (default 2026) for paired comparison")
    p.add_argument("--results-root", default="results")
    p.add_argument("--tag", default="")
    p.add_argument("--results-dir", default=None,
                   help="resume into this exact dir if it exists")
    args = p.parse_args(argv)

    if args.include_baselines:
        conds = list(dict.fromkeys(BASELINE_CONDITIONS + list(args.conditions)))
    else:
        conds = list(args.conditions)

    # Build task set deterministically from v2's generator
    tasks_all = v2.build_task_set(
        n_per_band=args.tasks_per_band,
        task_seed=args.task_seed,
        K=args.K_steps,
        asym_window=args.asym_window,
    )
    tasks = [t for t in tasks_all if t.difficulty in args.difficulties]

    # Output directory
    if args.results_dir:
        out_dir = Path(args.results_dir); out_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_tag = f"_{args.tag}" if args.tag else ""
        out_dir = Path(args.results_root) / f"experiment_imp_bench3_{timestamp}{folder_tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
    cells_path = out_dir / "cells.jsonl"
    log = setup_logging(out_dir / "log.txt")

    v2.save_json({
        "argv": list(sys.argv),
        "args": vars(args),
        "conditions_actually_run": conds,
        "n_tasks_total": len(tasks),
        "tasks": [{"name": t.name, "difficulty": t.difficulty,
                   "family": t.family, "description": t.description,
                   "K": t.K, "asym_window": t.asym_window,
                   "first_six_targets": t.targets[:6]} for t in tasks],
    }, out_dir / "manifest.json")

    done = load_done_keys(cells_path)
    log.info("=" * 76)
    log.info("Internal Model Principle for LLM Agents -- benchmark v3 (mechanism arms)")
    log.info(f"output dir   : {out_dir}")
    log.info("=" * 76)
    log.info(f"backend      : {args.backend}    model: {args.model}")
    log.info(f"tasks/band   : {args.tasks_per_band}    total: {len(tasks)}")
    log.info(f"conditions   : {conds}")
    log.info(f"n_seeds      : {args.n_seeds}    sc_samples: {args.n_sc_samples}    T: {args.temperature}")
    log.info(f"K, asym_w    : {args.K_steps}, {args.asym_window}")
    log.info(f"task_seed    : {args.task_seed}")
    log.info(f"resume       : {len(done)} cells already done")

    client = get_client(args.backend, args.model)
    log.info(f"client       : {client.name}")

    n_total = len(tasks) * len(conds) * args.n_seeds
    log.info(f"cells total  : {n_total}    remaining: {n_total - len(done)}")

    rows: list[CellResult] = []
    t_start = time.time(); n_run = 0
    for task in tasks:
        for cond in conds:
            for seed in range(args.n_seeds):
                key = cell_key(client.name, task.name, cond, seed)
                if key in done: continue
                t0 = time.time()
                r = run_cell_v3(
                    client=client, task=task, condition=cond, seed=seed,
                    temperature=args.temperature,
                    n_sc_samples=args.n_sc_samples,
                )
                rows.append(r); done.add(key); append_cell(cells_path, r); n_run += 1
                rate = n_run / max(time.time() - t_start, 1e-3)
                eta = (n_total - len(done)) / max(rate, 1e-9)
                log.info(
                    f"[{len(done):4d}/{n_total}] {task.difficulty:6s} "
                    f"{task.family:24s} {cond:18s} s={seed} "
                    f"asym={r.asym_error:7.2f} ({time.time()-t0:.1f}s) "
                    f"eta={fmt_eta(eta)}"
                )

    # Final summary -- per-condition mean per band, plus paired tests
    # against reactive (where reactive cells exist in this run)
    rows = []
    if cells_path.exists():
        with cells_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: d = json.loads(line)
                except: continue
                rows.append(CellResult(**d))
    log.info("\nFinal summary:")
    summary: dict[str, Any] = {
        "n_cells": len(rows),
        "by_difficulty_condition": {},
        "paired_ttests_vs_reactive": {},
    }
    for diff in args.difficulties:
        for cond in conds:
            sub = [r for r in rows if r.difficulty == diff and r.condition == cond]
            if not sub: continue
            errs = np.array([r.asym_error for r in sub])
            summary["by_difficulty_condition"][f"{diff}__{cond}"] = {
                "n": len(sub), "mean": float(np.mean(errs)),
                "median": float(np.median(errs)), "std": float(np.std(errs)),
            }

    # paired-t against reactive only if we have reactive cells in THIS run
    have_reactive = any(r.condition == "reactive" for r in rows)
    if have_reactive:
        for diff in args.difficulties:
            sub = [r for r in rows if r.difficulty == diff]
            for cond in [c for c in conds if c != "reactive"]:
                # Reuse v2's paired test
                from run_imp_benchmark2 import paired_t_log, cluster_bootstrap_ci
                tt = paired_t_log(sub, cond_a="reactive", cond_b=cond)
                summary["paired_ttests_vs_reactive"][f"{diff}__reactive_vs_{cond}"] = tt

    v2.save_json(summary, out_dir / "summary.json")

    log.info("\nasymptotic error by difficulty x condition (mean):")
    for diff in args.difficulties:
        line = f"  {diff:6s} "
        for cond in conds:
            v = summary["by_difficulty_condition"].get(
                f"{diff}__{cond}", {}).get("mean", float("nan"))
            line += f" {cond}={v:6.2f}"
        log.info(line)

    if have_reactive:
        log.info("\npaired t-tests (reactive vs each non-reactive arm):")
        for k, v in summary["paired_ttests_vs_reactive"].items():
            if v.get("n_pairs", 0) < 2:
                log.info(f"  {k}  n={v.get('n_pairs',0)}  (skipped)"); continue
            log.info(
                f"  {k:50s} n={v['n_pairs']:4d} "
                f"diff={v['mean_diff_log_b_minus_a']:+.3f} "
                f"t={v['t_stat']:+6.2f} p={v['p_value_two_sided']:.3g} "
                f"hurts={v['b_hurts_at_p_0_05']}"
            )

    log.info(f"\nwall time this session: {fmt_eta(time.time() - t_start)}")
    log.info("=" * 76)
    log.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
