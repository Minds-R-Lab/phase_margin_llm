#!/usr/bin/env python3
"""
analyze_hypothesis_correctness.py

Post-hoc analysis on benchmark2/3 cells.jsonl that decomposes the
IM-Hypothesis harm by whether the articulated hypothesis was
*correct* AND whether the integer prediction *followed* it.

Two failure modes are possible under the IM-Hypothesis prompt:
  (a) WRONG-RULE-COMMITTED:  model articulates an incorrect rule
                             and computes forward from it.  The
                             original tokenized-commitment story.
  (b) APPLY-STEP-FAILURE:    model articulates a correct rule but
                             emits an integer that does not match
                             the rule.  Discovered post-hoc.

Reactive has neither, since it never separates rule from
application.  This script bins each IM cell into a 2x2 grid on
(rule_correct, rule_followed) and reports the asymptotic-error gap
to reactive within each cell, plus a coarser correct-only ternary
for backwards-compatibility with the v1 paper draft.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


# ===========================================================================
# 1.  Rule parser
# ===========================================================================
_INT = r"-?\d+"

_LHS = (
    r"(?:y(?:_?k)?|y\s*\(\s*k\s*\)|f\s*\(\s*k\s*\)|f|"
    r"the\s+function(?:\s+is)?|the\s+rule(?:\s+is)?)"
)
_RE_LHS_EQ = re.compile(r"^\s*" + _LHS + r"\s*=\s*", re.I)
_RE_TEXT_PREFIX = re.compile(
    r"^\s*the\s+(?:function|rule|sequence|formula|generator|pattern)"
    r"(?:\s+(?:is|seems\s+to\s+be|appears\s+to\s+be|that\s+generates|equals|follows))?"
    r"\s*[:,]?\s*",
    re.I,
)


def _strip_lhs(body: str, _depth: int = 0) -> str:
    if _depth > 2:
        return body
    body0 = body
    m = _RE_TEXT_PREFIX.match(body)
    if m:
        body = body[m.end():].strip()
    m = _RE_LHS_EQ.match(body)
    if m:
        body = body[m.end():].strip()
    if body != body0 and _depth < 2:
        return _strip_lhs(body, _depth=_depth + 1)
    return body


def _safe_int(s: str) -> int:
    s = s.replace(" ", "").replace("+", "").replace(",", "")
    return int(s)


# ---------- RHS pattern grammar -------------------------------------------
_PAT_RHS_CONST = re.compile(r"^(" + _INT + r")\s*$")
_PAT_RHS_KMOD  = re.compile(r"^k\s*(?:%|mod|\\bmod)\s*(" + _INT + r")\s*$", re.I)
_PAT_RHS_K_SQ  = re.compile(r"^k\s*(?:\^|\*\*|²)\s*2\s*$|^k\s*\*\s*k\s*$|^k\s*²\s*$")
_PAT_RHS_K_OFFSET_SQ = re.compile(
    r"^\(\s*k\s*([+-])\s*(" + _INT + r")\s*\)\s*(?:\^|\*\*|²)\s*2\s*$"
)
_PAT_RHS_GEN_QUAD = re.compile(
    r"^(-?\d*)\s*\*?\s*k\s*(?:\^|\*\*|²)\s*2"
    r"\s*([+-]\s*\d+)?\s*\*?\s*k?\s*"
    r"([+-]\s*\d+)?\s*$"
)
_PAT_RHS_LINEAR = re.compile(r"^(-?\d*)\s*\*?\s*k\s*([+-]\s*\d+)?\s*$")
_PAT_RHS_K_TIMES_KMINUS1 = re.compile(r"^k\s*\*?\s*\(\s*k\s*-\s*1\s*\)\s*$")
_PAT_RHS_TRIANGLE = re.compile(r"^k\s*\*?\s*\(\s*k\s*\+\s*1\s*\)\s*/\s*2\s*$")
_PAT_RHS_CUBE_MOD = re.compile(
    r"^k\s*(?:\^|\*\*|³)\s*3\s*(?:%|mod|\\bmod)\s*(" + _INT + r")\s*$",
    re.I,
)
_PAT_RHS_GEOMETRIC2 = re.compile(r"^(\d+)?\s*\*?\s*2\s*(?:\^|\*\*)\s*k\s*$")
_PAT_RHS_TRI_WAVE = re.compile(
    r"^\|\s*\(?\s*k\s*(?:%|mod)\s*(\d+)\s*\)?\s*-\s*(\d+)\s*\|\s*$"
)
_PAT_RHS_FLOOR = re.compile(
    r"^(?:floor|⌊)\s*\(?\s*k\s*/\s*(" + _INT + r")\s*\)?\s*⌋?\s*$",
    re.I,
)
_PAT_RHS_ZIGZAG = re.compile(
    r"^(?:(-?\d*)\s*\*?\s*)?k\s*\*\s*\(?\s*-\s*1\s*\)?\s*(?:\^|\*\*)\s*k\s*$"
)
_PAT_RHS_FIB = re.compile(
    r"f\s*\(\s*k\s*-\s*1\s*\)\s*\+\s*f\s*\(\s*k\s*-\s*2\s*\)", re.I
)
_PAT_FIB_INIT = re.compile(
    r"f\s*\(\s*0\s*\)\s*=\s*(\d+).*?f\s*\(\s*1\s*\)\s*=\s*(\d+)",
    re.I | re.S,
)

# ---------- period / cycle phrasings (text-level, not RHS) ----------------
_PAT_PERIOD_BRACKET = re.compile(r"\[\s*(-?\d+(?:\s*,\s*-?\d+)+)\s*\]")
# fix: NO \b after the keyword so "repeats" and "cycles" both match.
# "every N" before the comma list is allowed but the leading N is NOT
# captured into the list (we use a tempered character class that excludes
# digits-followed-by-non-comma).
_PAT_PERIOD_LIST = re.compile(
    r"(?:cycles?|periods?|repeats?|repeating|patterns?|values?)"
    r"[^,\[\]\d]*(?:\d+\s*(?:terms?|elements?|steps?|values?)?)?[^,\[\]]*?"
    r"((?:-?\d+\s*,\s*){2,}-?\d+)"
)
# "alternates between A and B" -> period-2 [A, B]
_PAT_ALTERNATES = re.compile(
    r"alternat(?:e|es|ing)\s+between\s+(-?\d+)\s+and\s+(-?\d+)",
    re.I,
)


def _make_period_fn(elem_str: str):
    try:
        elems = [int(x.strip()) for x in elem_str.split(",") if x.strip()]
        if not elems:
            return None
        return lambda k, e=elems: int(e[k % len(e)])
    except Exception:
        return None


def parse_hypothesis(text: str):
    if not text:
        return None
    body = text.strip()
    body = re.sub(r"^\s*HYPOTHESIS\s*:\s*", "", body, flags=re.I)
    body = body.strip().strip(".").strip(",").strip(";").strip()

    # ----- whole-text patterns ------
    m = _PAT_PERIOD_BRACKET.search(body)
    if m:
        fn = _make_period_fn(m.group(1))
        if fn:
            return fn

    m = _PAT_ALTERNATES.search(body)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return lambda k, a=a, b=b: int([a, b][k % 2])

    m = _PAT_PERIOD_LIST.search(body)
    if m:
        fn = _make_period_fn(m.group(1))
        if fn:
            return fn

    if _PAT_RHS_FIB.search(body):
        m = _PAT_FIB_INIT.search(body)
        a, b = (int(m.group(1)), int(m.group(2))) if m else (1, 1)
        cache = {0: a, 1: b}
        def fib(k, cache=cache):
            if k in cache:
                return cache[k]
            for j in range(max(cache) + 1, k + 1):
                cache[j] = cache[j - 1] + cache[j - 2]
            return int(cache[k])
        return fib

    # ----- LHS-stripped RHS ------
    rhs = _strip_lhs(body).strip().strip(".").strip(",").strip()

    m = _PAT_RHS_CONST.match(rhs)
    if m:
        c = _safe_int(m.group(1))
        return lambda k, c=c: int(c)

    m = _PAT_RHS_KMOD.match(rhs)
    if m:
        n = _safe_int(m.group(1))
        if n != 0:
            return lambda k, n=n: int(k % n)

    m = _PAT_RHS_CUBE_MOD.match(rhs)
    if m:
        n = _safe_int(m.group(1))
        if n != 0:
            return lambda k, n=n: int((k ** 3) % n)

    m = _PAT_RHS_ZIGZAG.match(rhs)
    if m:
        s = m.group(1) or "1"
        s = _safe_int(s) if s and s != "-" else (1 if s != "-" else -1)
        return lambda k, s=s: int(s * k * ((-1) ** k))

    if _PAT_RHS_K_TIMES_KMINUS1.match(rhs):
        return lambda k: int(k * (k - 1))
    if _PAT_RHS_TRIANGLE.match(rhs):
        return lambda k: int(k * (k + 1) // 2)

    m = _PAT_RHS_TRI_WAVE.match(rhs)
    if m:
        n = int(m.group(1)); off = int(m.group(2))
        if n != 0:
            return lambda k, n=n, off=off: int(abs((k % n) - off))

    m = _PAT_RHS_FLOOR.match(rhs)
    if m:
        n = _safe_int(m.group(1))
        if n != 0:
            return lambda k, n=n: int(k // n)

    m = _PAT_RHS_GEOMETRIC2.match(rhs)
    if m:
        c = int(m.group(1)) if m.group(1) else 1
        return lambda k, c=c: int(c * (2 ** min(k, 30)))

    m = _PAT_RHS_K_OFFSET_SQ.match(rhs)
    if m:
        sign = +1 if m.group(1) == "+" else -1
        a = int(m.group(2))
        return lambda k, s=sign, a=a: int((k + s * a) ** 2)

    if _PAT_RHS_K_SQ.match(rhs):
        return lambda k: int(k * k)

    m = _PAT_RHS_GEN_QUAD.match(rhs)
    if m:
        a_str = (m.group(1) or "").strip()
        if a_str in ("", "+"):  a = 1
        elif a_str == "-":      a = -1
        else:                    a = int(a_str)
        b = _safe_int(m.group(2)) if m.group(2) else 0
        c = _safe_int(m.group(3)) if m.group(3) else 0
        return lambda k, a=a, b=b, c=c: int(a * k * k + b * k + c)

    m = _PAT_RHS_LINEAR.match(rhs)
    if m:
        a_str = (m.group(1) or "").strip()
        if a_str in ("", "+"):  a = 1
        elif a_str == "-":      a = -1
        else:                    a = int(a_str)
        b = _safe_int(m.group(2)) if m.group(2) else 0
        return lambda k, a=a, b=b: int(a * k + b)

    return None


# ===========================================================================
# 2.  Cells / classification
# ===========================================================================
@dataclass
class CellLite:
    task: str; family: str; difficulty: str; condition: str; seed: int
    targets: list; predictions: list; raw_replies: list
    asym_window: int; asym_error: float; K: int


def load_cells(jsonl: Path):
    out = []
    if not jsonl.exists():
        return out
    with jsonl.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.append(CellLite(
                task=d["task"], family=d["family"],
                difficulty=d["difficulty"], condition=d["condition"],
                seed=d["seed"],
                targets=d.get("targets", []),
                predictions=d.get("predictions", []),
                raw_replies=d.get("raw_replies", []),
                asym_window=d.get("asym_window", 5),
                asym_error=d.get("asym_error", float("inf")),
                K=d.get("K", len(d.get("targets", []))),
            ))
    return out


def extract_hypothesis_lines(cell: CellLite):
    out = []
    for r in cell.raw_replies:
        if not r:
            out.append(None); continue
        line = None
        for ln in r.splitlines():
            ln_s = ln.strip()
            if re.match(r"^\s*HYPOTHESIS\s*:", ln_s, flags=re.I):
                line = ln_s; break
        out.append(line)
    return out


def classify_cell(cell: CellLite):
    """Compute per-step (rule_correct, rule_followed) and aggregate over the
    asymptotic window."""
    hyp_lines = extract_hypothesis_lines(cell)
    n = len(cell.targets)
    rule_emitted = 0
    rule_correct_global = 0
    rule_followed_global = 0
    asym_n = 0
    asym_correct = 0
    asym_followed = 0
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
        model_pred = cell.predictions[k] if k < len(cell.predictions) else None
        if rule_pred == true_y:
            rule_correct_global += 1
            if k >= asym_start:
                asym_correct += 1
        if model_pred is not None and rule_pred == model_pred:
            rule_followed_global += 1
            if k >= asym_start:
                asym_followed += 1
        if k >= asym_start:
            asym_n += 1
    return {
        "task": cell.task, "family": cell.family,
        "difficulty": cell.difficulty, "condition": cell.condition,
        "seed": cell.seed, "asym_error": cell.asym_error,
        "rule_emitted_rate": rule_emitted / max(n, 1),
        "rule_correct_in_asym_n":      asym_n,
        "rule_correct_in_asym_rate":  asym_correct  / max(asym_n, 1) if asym_n else 0.0,
        "rule_followed_in_asym_rate": asym_followed / max(asym_n, 1) if asym_n else 0.0,
    }


# ===========================================================================
# 3.  Decompositions
# ===========================================================================
def _bin_3way(d, thr):
    """Original 3-way bin on rule_correct alone."""
    if d["rule_correct_in_asym_n"] == 0:
        return "unparseable"
    r = d["rule_correct_in_asym_rate"]
    if r >= thr:           return "correct"
    if r <= 1 - thr:       return "incorrect"
    return "mixed"


def _bin_4way(d, thr):
    """New 2x2 bin on (rule_correct, rule_followed)."""
    if d["rule_correct_in_asym_n"] == 0:
        return "unparseable"
    rc = d["rule_correct_in_asym_rate"]
    rf = d["rule_followed_in_asym_rate"]
    c_hi = rc >= thr
    f_hi = rf >= thr
    # Cells where either rate falls in the borderline get pooled into 'mixed'
    if (1 - thr) < rc < thr or (1 - thr) < rf < thr:
        return "mixed"
    if      c_hi and     f_hi:  return "correct_followed"     # commitment-faithful good
    if      c_hi and not f_hi:  return "correct_not_followed" # apply-step failure
    if not  c_hi and     f_hi:  return "wrong_followed"       # commitment story (bad)
    return "wrong_not_followed"


def _aggregate(items, name):
    if not items:
        return {"n": 0}
    im_errs = np.array([d["asym_error"] for d in items])
    rx_errs = np.array([d["reactive_asym_error_for_task"]
                        for d in items
                        if d["reactive_asym_error_for_task"] is not None])
    return {
        "n": len(items),
        "mean_im_asym_error": float(np.mean(im_errs)),
        "median_im_asym_error": float(np.median(im_errs)),
        "mean_reactive_asym_error":
            float(np.mean(rx_errs)) if len(rx_errs) else None,
        "n_paired_with_reactive": int(len(rx_errs)),
    }


def decompose(cells, im_condition=None, reactive_condition="reactive",
              correct_threshold=0.66):
    if im_condition is None:
        im_conditions = {"im_hypothesis", "internal_hypothesis"}
    elif isinstance(im_condition, str):
        im_conditions = {im_condition}
    else:
        im_conditions = set(im_condition)

    reactive_by_task = {}
    for c in cells:
        if c.condition == reactive_condition:
            reactive_by_task.setdefault(c.task, []).append(c.asym_error)
    reactive_mean = {k: float(np.mean(v)) for k, v in reactive_by_task.items()}

    classified = []
    for c in cells:
        if c.condition not in im_conditions:
            continue
        d = classify_cell(c)
        d["reactive_asym_error_for_task"] = reactive_mean.get(c.task)
        d["bin_3way"] = _bin_3way(d, correct_threshold)
        d["bin_4way"] = _bin_4way(d, correct_threshold)
        classified.append(d)

    bins3 = {k: [] for k in
             ("correct", "incorrect", "mixed", "unparseable")}
    bins4 = {k: [] for k in
             ("correct_followed", "correct_not_followed",
              "wrong_followed",   "wrong_not_followed",
              "mixed", "unparseable")}
    for d in classified:
        bins3[d["bin_3way"]].append(d)
        bins4[d["bin_4way"]].append(d)

    summary = {
        "decomposition_3way": {k: _aggregate(v, k) for k, v in bins3.items()},
        "decomposition_4way": {k: _aggregate(v, k) for k, v in bins4.items()},
        "__bin_thresholds__": {
            "correct (rate)":   ">= %g" % correct_threshold,
            "incorrect (rate)": "<= %g" % (1 - correct_threshold),
            "mixed":            "between %g and %g (either rate)"
                                 % (1 - correct_threshold, correct_threshold),
            "unparseable":      "no parseable rule emitted in asym window",
        },
    }
    return {
        "n_im_cells": len(classified),
        "n_reactive_tasks": len(reactive_mean),
        "summary": summary,
        "per_cell": classified,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results_dir", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--correct-threshold", type=float, default=0.66)
    args = p.parse_args(argv)

    cells_path = args.results_dir / "cells.jsonl"
    cells = load_cells(cells_path)
    if not cells:
        print("No cells found in %s" % cells_path, file=sys.stderr)
        return 1

    out_path = args.out or (args.results_dir / "hyp_correctness.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = decompose(cells, correct_threshold=args.correct_threshold)
    out_path.write_text(json.dumps(result, indent=2, default=str))

    print("== Hypothesis-correctness decomposition ==")
    print("  loaded %d cells from %s" % (len(cells), cells_path))
    print("  im_hypothesis cells classified: %d" % result["n_im_cells"])
    print("  reactive tasks paired:          %d" % result["n_reactive_tasks"])

    def _print_bin(name, s):
        if s["n"] == 0:
            print("  %-22s %5d  --" % (name, 0))
            return
        rx = s.get("mean_reactive_asym_error")
        rx_str = ("%12.3f" % rx) if rx is not None else "%12s" % "-"
        gap = (s["mean_im_asym_error"] - rx) if rx is not None else None
        gap_str = ("%+.3f" % gap) if gap is not None else "-"
        print("  %-22s %5d  %12.3f  %s  %s"
              % (name, s["n"], s["mean_im_asym_error"], rx_str, gap_str))

    print()
    print("--- 3-way bin (correctness only, backwards-compatible) ---")
    print("  %-22s %5s  %12s  %12s  %s"
          % ("bin", "n", "mean_im_err", "mean_rx_err", "gap"))
    for nm in ("correct", "incorrect", "mixed", "unparseable"):
        _print_bin(nm, result["summary"]["decomposition_3way"][nm])

    print()
    print("--- 4-way bin (correct x followed) ---")
    print("  %-22s %5s  %12s  %12s  %s"
          % ("bin", "n", "mean_im_err", "mean_rx_err", "gap"))
    for nm in ("correct_followed", "correct_not_followed",
               "wrong_followed", "wrong_not_followed",
               "mixed", "unparseable"):
        _print_bin(nm, result["summary"]["decomposition_4way"][nm])

    print("\n  written %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
