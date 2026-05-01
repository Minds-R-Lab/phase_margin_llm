#!/usr/bin/env python3
"""
run_imp_code_benchmark.py
=========================

Code-debugging extension of the Apply-Don't-Articulate study.

The experiment mirrors run_imp_benchmark3.py but in the code-debugging
paradigm.  An LLM is shown a buggy Python function and asked to emit
a corrected version under one of seven prompting conditions.  The
corrected function is graded by hidden test cases.  We pair conditions
by (task, seed) and report pass-rate gaps to the reactive baseline.

Conditions
----------
  reactive            "fix the function, output only the def"
  internal_diagnosis  "first state the bug as # BUG: ..., then the def"
  internal_cot        "let's think step by step, then the def"
  im_diag_sc5         sample N=5 diagnoses+fixes, pick the modal fix
                      by canonical AST hash (deployment-realistic SC)
  im_oracle           true bug description provided in system prompt
  im_diag_first       def first, then # BUG comment after
  im_diag_eqbudget    same as internal_diagnosis with max_tokens=512

Usage
-----
  python experiments/run_imp_code_benchmark.py --backend mock-fix \\
        --n-seeds 2 --tag mock_smoke

  python experiments/run_imp_code_benchmark.py --backend ollama \\
        --model qwen2.5-coder:7b \\
        --conditions im_diag_sc5 im_oracle im_diag_first im_diag_eqbudget \\
        --include-baselines --n-seeds 3 --n-sc-samples 5 \\
        --tag qwen7b_code_v1
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "src"))

from phase_margin.llm.base import LLMClient, Message
from phase_margin.coding.bug_templates import build_task_set, CodeTask
from phase_margin.coding.grader import grade, GradeResult, extract_function_code


# ===========================================================================
# 1.  Prompts
# ===========================================================================
SYSTEM_REACTIVE = (
    "You are a Python debugging assistant.  The user will show you a "
    "buggy Python function.  Output the corrected function as a single "
    "Python def block.  No commentary, no explanation, no markdown "
    "fences.  Just the def block."
)

SYSTEM_INTERNAL_DIAGNOSIS = (
    "You are a Python debugging assistant operating under the Internal "
    "Model Principle.  The user will show you a buggy Python function. "
    "You MUST first state the bug in a single comment line beginning "
    "with `# BUG:`.  Then on the next lines, output the corrected "
    "function as a Python def block.\n\n"
    "Reply format (exactly this):\n"
    "    # BUG: <one-line diagnosis>\n"
    "    def corrected_function(...):\n"
    "        ...\n"
)

SYSTEM_INTERNAL_COT = (
    "You are a Python debugging assistant.  The user will show you a "
    "buggy Python function.  Let's think step by step about what's "
    "wrong.  After your reasoning, output the corrected function as a "
    "Python def block at the END of your reply.  The final def block "
    "in your reply will be parsed automatically."
)

SYSTEM_IM_DIAG_FIRST = (
    "You are a Python debugging assistant.  The user will show you a "
    "buggy Python function.  Output the corrected function FIRST as a "
    "Python def block.  Then on the line BELOW the function, write a "
    "single comment line beginning with `# BUG:` stating what the bug "
    "was.\n\n"
    "Reply format (exactly this):\n"
    "    def corrected_function(...):\n"
    "        ...\n"
    "    # BUG: <one-line diagnosis>\n"
)


def system_im_oracle(bug_description: str) -> str:
    return (
        "You are a Python debugging assistant.  The user will show "
        "you a buggy Python function.  The bug is: "
        f"{bug_description}.  Output the corrected function as a "
        "single Python def block.  No commentary, no explanation, no "
        "markdown fences."
    )


CONDITIONS = {
    "reactive":            ("static", SYSTEM_REACTIVE,           256),
    "internal_diagnosis":  ("static", SYSTEM_INTERNAL_DIAGNOSIS, 256),
    "internal_cot":        ("static", SYSTEM_INTERNAL_COT,       512),
    "im_diag_sc5":         ("static", SYSTEM_INTERNAL_DIAGNOSIS, 256),
    "im_oracle":           ("dynamic", None,                      256),
    "im_diag_first":       ("static", SYSTEM_IM_DIAG_FIRST,      256),
    "im_diag_eqbudget":    ("static", SYSTEM_INTERNAL_DIAGNOSIS, 512),
}
NEW_CONDITIONS = ["im_diag_sc5", "im_oracle", "im_diag_first", "im_diag_eqbudget"]
BASELINE_CONDITIONS = ["reactive", "internal_diagnosis", "internal_cot"]


def system_for(condition: str, task: CodeTask) -> str:
    kind, static_prompt, _ = CONDITIONS[condition]
    if kind == "dynamic" and condition == "im_oracle":
        return system_im_oracle(task.bug_description)
    return static_prompt


def max_tokens_for(condition: str) -> int:
    return CONDITIONS[condition][2]


def build_user_prompt(task: CodeTask) -> str:
    return f"Here is a buggy Python function.  Fix it:\n\n{task.buggy_code}"


# ===========================================================================
# 2.  Self-consistency aggregator
# ===========================================================================
def _canonical_ast_hash(code: str) -> str | None:
    """Hash the canonicalised AST of code; whitespace/comments ignored."""
    try:
        tree = ast.parse(code)
        # ast.dump gives a deterministic structural representation
        canon = ast.dump(tree, annotate_fields=True, include_attributes=False)
        return hashlib.sha1(canon.encode()).hexdigest()
    except Exception:
        return None


def aggregate_sc(replies: list[str], task: CodeTask) -> tuple[int, list[str], int]:
    """Self-consistency aggregation over N sampled replies.
    Returns (chosen_index, candidate_codes, modal_count).
    Strategy: extract the def from each reply; group by canonical AST
    hash; pick the modal group's first member.  If every reply has a
    distinct hash (or all fail to extract), fall back to the first
    reply that yields a parseable def."""
    candidates = [extract_function_code(r, task.function_name) for r in replies]
    hashes = [_canonical_ast_hash(c) if c else None for c in candidates]
    # count modal hash among non-None hashes
    counts: dict[str, list[int]] = {}
    for i, h in enumerate(hashes):
        if h is None: continue
        counts.setdefault(h, []).append(i)
    if not counts:
        # no parseable defs; pick first reply (will be graded as no_def)
        return 0, candidates, 0
    # find modal hash
    modal_hash = max(counts.keys(), key=lambda h: len(counts[h]))
    modal_idx = counts[modal_hash][0]
    modal_count = len(counts[modal_hash])
    return modal_idx, candidates, modal_count


# ===========================================================================
# 3.  Cell runner
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
    temperature: float
    reply: str
    candidate_code: str | None
    n_pass: int
    n_total: int
    pass_rate: float
    error_kind: str | None
    sc_replies: list[str] = field(default_factory=list)
    sc_pass_rates: list[float] = field(default_factory=list)
    sc_modal_count: int = 0
    elapsed_s: float = 0.0


def run_cell(
    *,
    client: LLMClient,
    task: CodeTask,
    condition: str,
    seed: int,
    temperature: float,
    n_sc_samples: int = 5,
    grader_timeout_sec: float = 2.0,
) -> CellResult:
    sys_prompt = system_for(condition, task)
    user = build_user_prompt(task)
    msgs = [Message(role="system", content=sys_prompt),
            Message(role="user",   content=user)]
    max_tok = max_tokens_for(condition)
    t0 = time.time()

    if condition == "im_diag_sc5":
        replies = []
        for s in range(n_sc_samples):
            r = client.chat(msgs,
                            seed=seed * 100_000 + s,
                            temperature=max(temperature, 0.5),
                            max_tokens=max_tok)
            replies.append(r or "")
        chosen_idx, candidates, modal_count = aggregate_sc(replies, task)
        chosen_reply = replies[chosen_idx]
        # Grade each candidate so the JSONL has the full picture
        per_pass = []
        for c in candidates:
            if c is None:
                per_pass.append(0.0)
            else:
                gr = grade(c, task, test_timeout_sec=grader_timeout_sec)
                per_pass.append(gr.pass_rate)
        # Final reported result: the modal/chosen one
        result = grade(chosen_reply, task, test_timeout_sec=grader_timeout_sec)
        return CellResult(
            backend=client.name.split(":")[0], model=client.name,
            task=task.name, family=task.name.split("_", 1)[0],
            difficulty=task.difficulty, condition=condition, seed=seed,
            temperature=temperature, reply=chosen_reply,
            candidate_code=result.candidate_code,
            n_pass=result.n_pass, n_total=result.n_total,
            pass_rate=result.pass_rate, error_kind=result.error_kind,
            sc_replies=replies, sc_pass_rates=per_pass,
            sc_modal_count=modal_count,
            elapsed_s=time.time() - t0,
        )
    else:
        reply = client.chat(msgs,
                            seed=seed * 10_000,
                            temperature=temperature,
                            max_tokens=max_tok)
        result = grade(reply or "", task, test_timeout_sec=grader_timeout_sec)
        return CellResult(
            backend=client.name.split(":")[0], model=client.name,
            task=task.name, family=task.name.split("_", 1)[0],
            difficulty=task.difficulty, condition=condition, seed=seed,
            temperature=temperature, reply=reply or "",
            candidate_code=result.candidate_code,
            n_pass=result.n_pass, n_total=result.n_total,
            pass_rate=result.pass_rate, error_kind=result.error_kind,
            elapsed_s=time.time() - t0,
        )


# ===========================================================================
# 4.  Mock backends for smoke testing
# ===========================================================================
class MockReactiveFixer(LLMClient):
    """Always emits the canonical fix (perfect oracle).  Used to verify
    the harness end-to-end."""
    @property
    def name(self) -> str:
        return "mock:reactive_fixer"

    def chat(self, messages, *, seed=None, temperature=0.7, max_tokens=256):
        # The user prompt contains the buggy code; we cheat by looking
        # up the canonical from a global lookup keyed on function name.
        # (For smoke test only; not deployed.)
        user = next((m.content for m in messages if m.role == "user"), "")
        m = re.search(r"def\s+(\w+)\s*\(", user)
        if not m:
            return ""
        fn_name = m.group(1)
        # Find the task whose function_name matches
        for task in build_task_set():
            if task.function_name == fn_name:
                return task.canonical_code
        return ""


class MockBuggyFixer(LLMClient):
    """Always emits the buggy code unchanged.  Used to verify that the
    harness reports gaps from reactive when the model fails."""
    @property
    def name(self) -> str:
        return "mock:buggy_fixer"

    def chat(self, messages, *, seed=None, temperature=0.7, max_tokens=256):
        user = next((m.content for m in messages if m.role == "user"), "")
        m = re.search(r"def\s+(\w+)\s*\(", user)
        if not m:
            return ""
        fn_name = m.group(1)
        for task in build_task_set():
            if task.function_name == fn_name:
                return task.buggy_code
        return ""


# ===========================================================================
# 5.  IO + resume
# ===========================================================================
def cell_key(model: str, task: str, condition: str, seed: int) -> str:
    return f"{model}::{task}::{condition}::{seed}"


def load_done_keys(jsonl: Path) -> set[str]:
    if not jsonl.exists():
        return set()
    done = set()
    with jsonl.open("r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: d = json.loads(line)
            except: continue
            done.add(cell_key(d["model"], d["task"], d["condition"], d["seed"]))
    return done


def append_cell(jsonl: Path, r: CellResult) -> None:
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with jsonl.open("a") as f:
        f.write(json.dumps(asdict(r), default=str) + "\n")


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("phase_margin.imp_code")
    log.setLevel(logging.INFO); log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a"); fh.setFormatter(fmt); log.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout);       sh.setFormatter(fmt); log.addHandler(sh)
    return log


# ===========================================================================
# 6.  Statistics: paired t on arcsin(sqrt(pass_rate)) + cluster bootstrap
# ===========================================================================
def _arcsin_sqrt(p: float) -> float:
    p = min(max(p, 0.0), 1.0)
    return float(np.arcsin(np.sqrt(p)))


def paired_t_passrate(rows: list[CellResult], cond_a: str, cond_b: str
                      ) -> dict[str, Any]:
    by_key: dict[tuple[str, int], dict[str, float]] = {}
    for r in rows:
        by_key.setdefault((r.task, r.seed), {})[r.condition] = r.pass_rate
    paired = [(v[cond_a], v[cond_b]) for v in by_key.values()
              if cond_a in v and cond_b in v]
    if len(paired) < 2:
        return {"n_pairs": len(paired)}
    a = np.array([_arcsin_sqrt(x[0]) for x in paired])
    b = np.array([_arcsin_sqrt(x[1]) for x in paired])
    diff = b - a
    from scipy.stats import ttest_rel
    t, p = ttest_rel(a, b)
    return {
        "cond_a": cond_a, "cond_b": cond_b, "n_pairs": len(paired),
        "mean_pass_a": float(np.mean([x[0] for x in paired])),
        "mean_pass_b": float(np.mean([x[1] for x in paired])),
        "mean_diff_passrate_b_minus_a":
            float(np.mean([x[1] - x[0] for x in paired])),
        "mean_diff_arcsin_b_minus_a": float(diff.mean()),
        "t_stat": float(t), "p_value_two_sided": float(p),
        "b_helps_at_p_0_05": bool(p < 0.05 and b.mean() > a.mean()),
        "b_hurts_at_p_0_05": bool(p < 0.05 and b.mean() < a.mean()),
    }


def cluster_bootstrap_passrate(rows: list[CellResult], cond_a: str, cond_b: str,
                               n_boot: int = 5000, rng_seed: int = 42
                               ) -> dict[str, Any]:
    by_task: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        # Group reactive paired with cond_b on the same (task, seed)
        pass
    # Simpler: build (task, seed)-paired dict then resample tasks
    by_key: dict[tuple[str, int], dict[str, float]] = {}
    for r in rows:
        by_key.setdefault((r.task, r.seed), {})[r.condition] = r.pass_rate
    pairs_by_task: dict[str, list[tuple[float, float]]] = {}
    for (task, _seed), v in by_key.items():
        if cond_a in v and cond_b in v:
            pairs_by_task.setdefault(task, []).append((v[cond_a], v[cond_b]))
    tasks = list(pairs_by_task.keys())
    if len(tasks) < 2:
        return {"n_tasks": len(tasks)}
    rng = np.random.default_rng(rng_seed)
    medians = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(tasks), size=len(tasks))
        diffs = []
        for j in idx:
            for a_pass, b_pass in pairs_by_task[tasks[j]]:
                diffs.append(b_pass - a_pass)
        if diffs:
            medians.append(float(np.median(diffs)))
    arr = np.array(medians)
    return {
        "cond_a": cond_a, "cond_b": cond_b,
        "n_tasks": len(tasks), "n_boot": n_boot,
        "median_diff_point": float(np.median(arr)),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
        "ci_excludes_zero": bool(np.percentile(arr, 2.5) > 0
                                 or np.percentile(arr, 97.5) < 0),
    }


# ===========================================================================
# 7.  Backends
# ===========================================================================
def get_client(backend: str, model: str | None) -> LLMClient:
    if backend == "mock-fix":
        return MockReactiveFixer()
    if backend == "mock-buggy":
        return MockBuggyFixer()
    if backend == "ollama":
        from phase_margin.llm.ollama_client import OllamaClient
        return OllamaClient(model=model or "qwen2.5-coder:7b")
    if backend == "anthropic":
        from phase_margin.llm.anthropic_client import AnthropicClient
        return AnthropicClient(model=model or "claude-haiku-4-5-20251001",
                               cache_dir="data/cache/anthropic")
    raise ValueError(f"unknown backend: {backend}")


def fmt_eta(sec: float) -> str:
    if sec < 90: return f"{sec:.0f}s"
    if sec < 5400: return f"{sec/60:.1f}m"
    return f"{sec/3600:.1f}h"


# ===========================================================================
# 8.  Main
# ===========================================================================
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="mock-fix",
                   choices=["mock-fix", "mock-buggy", "ollama", "anthropic"])
    p.add_argument("--model", default=None)
    p.add_argument("--difficulties", nargs="+",
                   default=["easy", "medium", "hard"],
                   choices=["easy", "medium", "hard"])
    p.add_argument("--conditions", nargs="+",
                   default=NEW_CONDITIONS,
                   choices=BASELINE_CONDITIONS + NEW_CONDITIONS)
    p.add_argument("--include-baselines", action="store_true")
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--n-sc-samples", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--task-seed", type=int, default=2026)
    p.add_argument("--grader-timeout-sec", type=float, default=2.0)
    p.add_argument("--results-root", default="results")
    p.add_argument("--tag", default="")
    p.add_argument("--results-dir", default=None)
    args = p.parse_args(argv)

    conds = (list(dict.fromkeys(BASELINE_CONDITIONS + list(args.conditions)))
             if args.include_baselines else list(args.conditions))

    tasks = [t for t in build_task_set() if t.difficulty in args.difficulties]

    if args.results_dir:
        out_dir = Path(args.results_dir); out_dir.mkdir(parents=True, exist_ok=True)
    else:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        ftag = f"_{args.tag}" if args.tag else ""
        out_dir = Path(args.results_root) / f"experiment_imp_code_{ts}{ftag}"
        out_dir.mkdir(parents=True, exist_ok=True)
    cells_path = out_dir / "cells.jsonl"
    log = setup_logging(out_dir / "log.txt")

    # Manifest
    with (out_dir / "manifest.json").open("w") as f:
        json.dump({
            "argv": list(sys.argv),
            "args": vars(args),
            "conditions_actually_run": conds,
            "n_tasks_total": len(tasks),
            "tasks": [{"name": t.name, "difficulty": t.difficulty,
                       "function_name": t.function_name,
                       "n_tests": t.n_tests,
                       "bug_description": t.bug_description}
                      for t in tasks],
        }, f, indent=2, default=str)

    done = load_done_keys(cells_path)
    log.info("=" * 76)
    log.info("Apply-Don't-Articulate -- code-debugging benchmark")
    log.info(f"output dir   : {out_dir}")
    log.info("=" * 76)
    log.info(f"backend      : {args.backend}    model: {args.model}")
    log.info(f"tasks total  : {len(tasks)}")
    log.info(f"conditions   : {conds}")
    log.info(f"n_seeds      : {args.n_seeds}    sc_samples: {args.n_sc_samples}")
    log.info(f"temperature  : {args.temperature}    grader_timeout: {args.grader_timeout_sec}s")
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
                try:
                    r = run_cell(
                        client=client, task=task, condition=cond, seed=seed,
                        temperature=args.temperature,
                        n_sc_samples=args.n_sc_samples,
                        grader_timeout_sec=args.grader_timeout_sec,
                    )
                except Exception as e:                              # noqa: BLE001
                    log.error(f"  cell raised {type(e).__name__}: {e}")
                    continue
                rows.append(r); done.add(key); append_cell(cells_path, r); n_run += 1
                rate = n_run / max(time.time() - t_start, 1e-3)
                eta = (n_total - len(done)) / max(rate, 1e-9)
                log.info(
                    f"[{len(done):4d}/{n_total}] {task.difficulty:6s} "
                    f"{task.name:42s} {cond:20s} s={seed} "
                    f"pass={r.n_pass}/{r.n_total} err={r.error_kind!s:20s} "
                    f"({time.time()-t0:.1f}s) eta={fmt_eta(eta)}"
                )

    # Summary
    rows = []
    if cells_path.exists():
        with cells_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: d = json.loads(line)
                except: continue
                d.pop("backend", None); d.pop("model", None)
                # rebuild CellResult ignoring fields we don't need for stats
                rows.append(CellResult(
                    backend="", model="",
                    task=d["task"], family=d.get("family", ""),
                    difficulty=d["difficulty"], condition=d["condition"],
                    seed=d["seed"], temperature=d.get("temperature", 0.7),
                    reply="", candidate_code=None,
                    n_pass=d["n_pass"], n_total=d["n_total"],
                    pass_rate=d["pass_rate"],
                    error_kind=d.get("error_kind"),
                ))

    summary: dict[str, Any] = {
        "n_cells": len(rows),
        "by_difficulty_condition": {},
        "paired_ttests_vs_reactive": {},
        "bootstrap_ci_vs_reactive": {},
    }
    for diff in args.difficulties:
        for cond in conds:
            sub = [r for r in rows if r.difficulty == diff and r.condition == cond]
            if not sub: continue
            summary["by_difficulty_condition"][f"{diff}__{cond}"] = {
                "n": len(sub),
                "mean_pass_rate":   float(np.mean([r.pass_rate for r in sub])),
                "median_pass_rate": float(np.median([r.pass_rate for r in sub])),
                "frac_perfect":     float(np.mean([r.pass_rate >= 0.999 for r in sub])),
            }
    if any(r.condition == "reactive" for r in rows):
        for diff in args.difficulties:
            sub = [r for r in rows if r.difficulty == diff]
            for cond in [c for c in conds if c != "reactive"]:
                summary["paired_ttests_vs_reactive"][f"{diff}__reactive_vs_{cond}"] \
                    = paired_t_passrate(sub, "reactive", cond)
                summary["bootstrap_ci_vs_reactive"][f"{diff}__reactive_vs_{cond}"] \
                    = cluster_bootstrap_passrate(sub, "reactive", cond)

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info("\nmean pass-rate by difficulty x condition:")
    for diff in args.difficulties:
        line = f"  {diff:6s}"
        for cond in conds:
            v = summary["by_difficulty_condition"].get(
                f"{diff}__{cond}", {}).get("mean_pass_rate", float("nan"))
            line += f"  {cond}={v:.3f}"
        log.info(line)

    if any(r.condition == "reactive" for r in rows):
        log.info("\npaired t-tests vs reactive (positive diff = arm hurts):")
        for k, v in summary["paired_ttests_vs_reactive"].items():
            if v.get("n_pairs", 0) < 2:
                log.info(f"  {k:50s} n={v.get('n_pairs',0)}  (skipped)"); continue
            log.info(
                f"  {k:50s} n={v['n_pairs']:3d} "
                f"diff={-v['mean_diff_passrate_b_minus_a']:+.3f}_pass "
                f"t={v['t_stat']:+6.2f} p={v['p_value_two_sided']:.3g} "
                f"hurts={v['b_hurts_at_p_0_05']} helps={v['b_helps_at_p_0_05']}"
            )

    log.info(f"\nwall time this session: {fmt_eta(time.time() - t_start)}")
    log.info("=" * 76)
    log.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
