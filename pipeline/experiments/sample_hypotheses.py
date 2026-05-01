#!/usr/bin/env python3
"""
sample_hypotheses.py
====================

Pull a stratified sample of HYPOTHESIS lines out of a benchmark2
``cells.jsonl`` so we can see what real LLMs actually emit.  Used
to expand the parser in ``analyze_hypothesis_correctness.py`` to
cover formats the regex grammar currently misses.

Stratifies the sample by (difficulty, parser-status):
  * 5 lines from each of {easy, medium, hard} that the current
    parser CAN parse,
  * 10 lines from each of {easy, medium, hard} that it CANNOT.

Optionally splits parseables into "rule-correct" and "rule-incorrect"
based on whether the parsed rule predicts y_k correctly.

Usage
-----
  python experiments/sample_hypotheses.py results/experiment_imp_bench2_*_qwen7b_v2

By default prints to stdout; pass ``--out FILE`` to also save.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from analyze_hypothesis_correctness import (
    parse_hypothesis, extract_hypothesis_lines, load_cells,
)


def sample_hypotheses(cells, *,
                      im_conditions=("internal_hypothesis", "im_hypothesis"),
                      n_parseable_per_band: int = 5,
                      n_unparseable_per_band: int = 10,
                      seed: int = 0):
    rng = random.Random(seed)
    by_diff_status: dict[tuple[str, str], list[tuple[str, str, int, int, int]]] = {}
    # tuple = (task, hyp_line, k, true_y, model_pred)
    for c in cells:
        if c.condition not in im_conditions:
            continue
        hlines = extract_hypothesis_lines(c)
        for k, line in enumerate(hlines):
            if line is None:
                continue
            fn = parse_hypothesis(line)
            true_y = c.targets[k] if k < len(c.targets) else None
            mp = c.predictions[k] if k < len(c.predictions) else None
            if fn is None:
                status = "unparseable"
            else:
                try:
                    rule_pred = fn(k)
                except Exception:
                    rule_pred = None
                status = ("correct" if rule_pred == true_y else "incorrect")
            key = (c.difficulty, status)
            by_diff_status.setdefault(key, []).append(
                (c.task, line, k, true_y, mp)
            )

    out = []
    for diff in ("easy", "medium", "hard"):
        for status in ("correct", "incorrect", "unparseable"):
            pool = by_diff_status.get((diff, status), [])
            n_take = (n_parseable_per_band if status != "unparseable"
                      else n_unparseable_per_band)
            if len(pool) <= n_take:
                sample = pool
            else:
                sample = rng.sample(pool, n_take)
            for task, line, k, true_y, mp in sample:
                out.append({
                    "difficulty": diff, "status": status,
                    "task": task, "k": k, "true_y": true_y,
                    "model_pred": mp, "hypothesis": line,
                })
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", type=Path)
    p.add_argument("--n-parseable", type=int, default=5)
    p.add_argument("--n-unparseable", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    cells_path = args.results_dir / "cells.jsonl"
    cells = load_cells(cells_path)
    if not cells:
        print(f"No cells found at {cells_path}", file=sys.stderr)
        return 1

    samples = sample_hypotheses(
        cells,
        n_parseable_per_band=args.n_parseable,
        n_unparseable_per_band=args.n_unparseable,
        seed=args.seed,
    )

    bar = "=" * 76
    out_lines = [bar, f"Hypothesis samples from {cells_path}", bar]
    cur_section = None
    for s in samples:
        section = (s["difficulty"], s["status"])
        if section != cur_section:
            out_lines.append("")
            out_lines.append(f"--- {s['difficulty']:6s} / {s['status']:11s} ---")
            cur_section = section
        out_lines.append(
            f"  task={s['task']:30s} k={s['k']:2d} "
            f"true={s['true_y']!s:>6s} model_pred={s['model_pred']!s:>6s}"
        )
        out_lines.append(f"    hyp: {s['hypothesis']}")
    text = "\n".join(out_lines)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"\n  written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
