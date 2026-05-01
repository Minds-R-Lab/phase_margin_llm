#!/usr/bin/env python3
"""
analyze_hypothesis_correctness.py
=================================

Post-hoc analysis on benchmark2 cells.jsonl that decomposes the
IM-Hypothesis harm by whether the articulated hypothesis was
*correct* on each step.

For every cell in condition ``im_hypothesis``, we extract the
``HYPOTHESIS:`` line at each step, parse it (best-effort) into a
candidate rule, evaluate that rule at the next index, and compare
to (a) the model's own integer prediction (``rule-following``)
and (b) the true target (``rule-correctness``).  This decomposes
the asymptotic error of the cell into:

  * cells where the model articulated a CORRECT rule consistently;
  * cells where the model articulated an INCORRECT rule
    consistently;
  * cells where the rule could not be parsed.

The prediction made by the tokenized-commitment account is sharp:
the asymptotic error of "consistently correct rule" cells should
match or beat the reactive baseline; the gap to reactive should be
entirely concentrated in "consistently incorrect rule" cells.

Usage
-----
  python experiments/analyze_hypothesis_correctness.py \\
        results/experiment_imp_bench2_*_qwen7b_v2/ \\
        --reactive-cells results/experiment_imp_bench2_*_qwen7b_v2/cells.jsonl \\
        --out reports/hyp_correctness_qwen7b.json

If you point ``--cells`` at the same JSONL as ``--reactive-cells``
the script joins reactive and im_hypothesis cells from the same
run.  Pass multiple ``--cells`` paths to aggregate across models.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


# ===========================================================================
# 1.  Cheap rule parser
# ===========================================================================
# We handle the most common hypothesis phrasings produced by Qwen and Llama
# in benchmark2.  Anything we cannot parse is labelled 'unparseable'.

_INT = r"-?\d+"
_FLOAT = r"-?\d+(?:\.\d+)?"

_PATTERNS_LINEAR = [
    # y = a*k + b   /  y = ak + b  /  y = ak  /  y = a k + b
    re.compile(r"y\s*=\s*(" + _INT + r")\s*\*?\s*k\s*([+-]\s*" + _INT + r")?\s*$"),
    # y_k = a*k + b
    re.compile(r"y_?k\s*=\s*(" + _INT + r")\s*\*?\s*k\s*([+-]\s*" + _INT + r")?\s*$"),
]
_PAT_CONSTANT = re.compile(r"y\s*=\s*(" + _INT + r")\s*$")
_PAT_KMODN    = re.compile(r"y\s*=\s*k\s*(?:%|mod|\\bmod)\s*(" + _INT + r")\s*", re.I)
_PAT_QUAD_OFF = re.compile(r"y\s*=\s*\(\s*k\s*([+-]\s*" + _INT + r")\)\s*\^\s*2\s*$")
_PAT_QUAD_K2  = re.compile(r"y\s*=\s*k\s*\^?\*?\s*2\s*$")

_PAT_PERIOD   = re.compile(r"(?:cycle|period[^\[]*)\s*\[\s*([^\]]+)\]")
_PAT_FIB      = re.compile(r"fib", re.I)


def _safe_int(s: str) -> int:
    return int(s.replace(" ", "").replace("+", ""))


def parse_hypothesis(text: str) -> Callable[[int], int] | None:
    """Best-effort: turn a HYPOTHESIS line into a Python lambda f(k) -> int.

    Returns None if no pattern matches."""
    if not text:
        return None
    # Strip leading "HYPOTHESIS:" etc.
    body = text.strip()
    body = re.sub(r"^\s*HYPOTHESIS\s*:\s*", "", body, flags=re.I)
    body = body.strip().strip(".").strip()

    # constant
    m = _PAT_CONSTANT.match(body)
    if m:
        c = _safe_int(m.group(1))
        return lambda k, c=c: int(c)

    # linear   y = ak (+ b)
    for pat in _PATTERNS_LINEAR:
        m = pat.match(body)
        if m:
            a = _safe_int(m.group(1))
            b = _safe_int(m.group(2)) if m.group(2) else 0
            return lambda k, a=a, b=b: int(a*k + b)

    # k mod n
    m = _PAT_KMODN.search(body)
    if m:
        n = _safe_int(m.group(1))
        if n != 0:
            return lambda k, n=n: int(k % n)

    # quadratic  y = (k - a)^2  or y = k^2
    m = _PAT_QUAD_OFF.match(body)
    if m:
        a_signed = m.group(1).replace(" ", "")
        a = int(a_signed)         # already a signed integer string
        # body says "(k - a)" or "(k + a)"; the regex captures sign+digits
        return lambda k, a=a: int((k + a) ** 2)
    if _PAT_QUAD_K2.match(body):
        return lambda k: int(k * k)

    # period cycle
    m = _PAT_PERIOD.search(body)
    if m:
        try:
            elems = [int(x.strip()) for x in m.group(1).split(",")]
            if elems:
                return lambda k, e=elems: int(e[k % len(e)])
        except Exception:
            pass

    # Fibonacci -- with f(0), f(1) we'd need values; punt.
    if _PAT_FIB.search(body):
        return None

    return None


# ===========================================================================
# 2.  Cell-level correctness classification
# ===========================================================================
@dataclass
class CellLite:
    task: str; family: str; difficulty: str; condition: str; seed: int
    targets: list[int]; predictions: list[int | None]; raw_replies: list[str]
    asym_window: int; asym_error: float; K: int


def load_cells(jsonl: Path) -> list[CellLite]:
    out = []
    if not jsonl.exists():
        return out
    with jsonl.open("r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: d = json.loads(line)
            except: continue
            out.append(CellLite(
                task=d["task"], family=d["family"], difficulty=d["difficulty"],
                condition=d["condition"], seed=d["seed"],
                targets=d.get("targets", []), predictions=d.get("predictions", []),
                raw_replies=d.get("raw_replies", []),
                asym_window=d.get("asym_window", 5),
                asym_error=d.get("asym_error", float("inf")),
                K=d.get("K", len(d.get("targets", []))),
            ))
    return out


def extract_hypothesis_lines(cell: CellLite) -> list[str | None]:
    """Pull the HYPOTHESIS: line from each step's reply."""
    out = []
    for r in cell.raw_replies:
        if not r:
            out.append(None); continue
        # find first line that starts with HYPOTHESIS (case-insensitive)
        line = None
        for ln in r.splitlines():
            ln_s = ln.strip()
            if re.match(r"^\s*HYPOTHESIS\s*:", ln_s, flags=re.I):
                line = ln_s; break
        out.append(line)
    return out


def classify_cell(cell: CellLite) -> dict[str, Any]:
    """Classify the cell on three axes:
       (i)  fraction of steps where a hypothesis was emitted at all,
       (ii) fraction of steps where the parsed rule predicts the next y_k
            correctly (rule-correct rate),
       (iii) fraction of steps where the model's own integer prediction
            matches the rule's prediction (rule-following rate).
    """
    hyp_lines = extract_hypothesis_lines(cell)
    n = len(cell.targets)
    rule_emitted = 0
    rule_correct = 0
    rule_followed = 0
    rule_correct_in_asym = 0
    rule_correct_in_asym_n = 0

    asym_start = max(0, n - cell.asym_window)
    for k in range(n):
        line = hyp_lines[k]
        if line is None:
            continue
        rule_emitted += 1
        rule_fn = parse_hypothesis(line)
        if rule_fn is None:
            continue
        try:
            rule_pred = int(rule_fn(k))
        except Exception:
            continue
        true_y = int(cell.targets[k])
        model_pred = cell.predictions[k]
        if rule_pred == true_y:
            rule_correct += 1
            if k >= asym_start:
                rule_correct_in_asym += 1
        if k >= asym_start:
            rule_correct_in_asym_n += 1
        if model_pred is not None and rule_pred == model_pred:
            rule_followed += 1

    return {
        "task": cell.task, "family": cell.family, "difficulty": cell.difficulty,
        "condition": cell.condition, "seed": cell.seed,
        "asym_error": cell.asym_error,
        "rule_emitted_rate":  rule_emitted / max(n, 1),
        "rule_followed_rate": rule_followed / max(rule_emitted, 1)
                              if rule_emitted else 0.0,
        "rule_correct_rate":  rule_correct / max(rule_emitted, 1)
                              if rule_emitted else 0.0,
        "rule_correct_in_asym_rate":
            rule_correct_in_asym / max(rule_correct_in_asym_n, 1)
            if rule_correct_in_asym_n else 0.0,
        "rule_correct_in_asym_n": rule_correct_in_asym_n,
    }


# ===========================================================================
# 3.  Decomposition: error by hypothesis-correctness category
# ===========================================================================
def decompose(cells: list[CellLite],
              im_condition: str = "im_hypothesis",
              reactive_condition: str = "reactive",
              correct_threshold: float = 0.66,
              ) -> dict[str, Any]:
    """Group im_condition cells by hypothesis-correct-in-asym rate, and
    compare each group's mean asym_error to the same task's reactive
    asym_error."""
    # Group reactive cells by task for paired lookup
    reactive_by_task: dict[str, list[float]] = {}
    for c in cells:
        if c.condition == reactive_condition:
            reactive_by_task.setdefault(c.task, []).append(c.asym_error)
    reactive_mean: dict[str, float] = {
        k: float(np.mean(v)) for k, v in reactive_by_task.items()
    }

    classified = []
    for c in cells:
        if c.condition != im_condition:
            continue
        d = classify_cell(c)
        d["reactive_asym_error_for_task"] = reactive_mean.get(c.task)
        classified.append(d)

    # Bin by rule_correct_in_asym_rate
    bins = {"correct": [], "incorrect": [], "unparseable": [], "mixed": []}
    for d in classified:
        if d["rule_correct_in_asym_n"] == 0:
            bins["unparseable"].append(d)
        elif d["rule_correct_in_asym_rate"] >= correct_threshold:
            bins["correct"].append(d)
        elif d["rule_correct_in_asym_rate"] <= 1 - correct_threshold:
            bins["incorrect"].append(d)
        else:
            bins["mixed"].append(d)

    summary = {}
    for name, items in bins.items():
        if not items:
            summary[name] = {"n": 0}
            continue
        im_errs = np.array([d["asym_error"] for d in items])
        rx_errs = np.array([d["reactive_asym_error_for_task"]
                            for d in items
                            if d["reactive_asym_error_for_task"] is not None])
        summary[name] = {
            "n": len(items),
            "mean_im_asym_error":       float(np.mean(im_errs)),
            "median_im_asym_error":     float(np.median(im_errs)),
            "mean_reactive_asym_error": float(np.mean(rx_errs)) if len(rx_errs) else None,
            "n_paired_with_reactive":   int(len(rx_errs)),
        }
    summary["__bin_thresholds__"] = {
        "correct": f"rule_correct_in_asym_rate >= {correct_threshold}",
        "incorrect": f"rule_correct_in_asym_rate <= {1 - correct_threshold}",
        "mixed": f"between {1 - correct_threshold} and {correct_threshold}",
        "unparseable": "no parseable rule emitted in asym window",
    }
    return {
        "n_im_cells": len(classified),
        "n_reactive_tasks": len(reactive_mean),
        "decomposition": summary,
        "per_cell": classified,
    }


# ===========================================================================
# 4.  Main
# ===========================================================================
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", type=Path,
                   help="benchmark2 results directory (must contain cells.jsonl)")
    p.add_argument("--out", type=Path, default=None,
                   help="output JSON path; defaults to <results_dir>/hyp_correctness.json")
    p.add_argument("--correct-threshold", type=float, default=0.66,
                   help="rate above which a cell is binned as 'correct'")
    args = p.parse_args(argv)

    cells_path = args.results_dir / "cells.jsonl"
    cells = load_cells(cells_path)
    if not cells:
        print(f"No cells found in {cells_path}", file=sys.stderr)
        return 1

    out_path = args.out or (args.results_dir / "hyp_correctness.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = decompose(cells, correct_threshold=args.correct_threshold)
    out_path.write_text(json.dumps(result, indent=2, default=str))

    # Pretty print to stdout
    print(f"== Hypothesis-correctness decomposition ==")
    print(f"  loaded {len(cells)} cells from {cells_path}")
    print(f"  im_hypothesis cells classified: {result['n_im_cells']}")
    print(f"  reactive tasks paired:          {result['n_reactive_tasks']}")
    print()
    print(f"{'bin':14s}  {'n':>5s}  {'mean_im_err':>12s}  {'mean_rx_err':>12s}  gap")
    for bin_name in ("correct", "incorrect", "mixed", "unparseable"):
        s = result["decomposition"][bin_name]
        if s["n"] == 0:
            print(f"  {bin_name:12s}  {0:5d}  --")
            continue
        rx = s["mean_reactive_asym_error"]
        rx_str = f"{rx:12.3f}" if rx is not None else f"{'-':>12s}"
        gap = (s["mean_im_asym_error"] - rx) if rx is not None else None
        gap_str = f"{gap:+.3f}" if gap is not None else "-"
        print(f"  {bin_name:12s}  {s['n']:5d}  "
              f"{s['mean_im_asym_error']:12.3f}  {rx_str}  {gap_str}")
    print(f"\n  written {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
