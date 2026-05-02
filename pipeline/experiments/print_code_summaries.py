#!/usr/bin/env python3
"""
print_code_summaries.py
=======================

Dump the summary.json from each of the three code-benchmark tags so
they can be copied to chat in one block.

By default looks for:
    qwen7b_coder_code_v1
    qwen14b_coder_code_v1
    qwen32b_coder_code_v1

Use --compact for a one-table-per-model view that fits in a terminal.
Use --full for the verbose JSON dump.  Default is --compact.

Examples
--------
    python experiments/print_code_summaries.py                 # compact
    python experiments/print_code_summaries.py --full          # verbose JSON
    python experiments/print_code_summaries.py --tags my_7b    # custom tags
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_TAGS = [
    "qwen7b_coder_code_v1",
    "qwen14b_coder_code_v1",
    "qwen32b_coder_code_v1",
]


def find_latest(results_root: Path, tag: str) -> Path | None:
    if not results_root.is_dir():
        return None
    candidates = sorted(
        results_root.glob(f"experiment_imp_code_*_{tag}"),
        key=lambda p: p.name,
    )
    return candidates[-1] if candidates else None


def _fmt_p(p):
    if p is None or p != p:    # NaN check
        return "  n/a"
    if p < 1e-3:
        return f"{p:.0e}"
    return f"{p:.3f}"


def _fmt_t(t):
    if t is None or t != t:
        return "  n/a"
    return f"{t:+6.2f}"


def _verdict(tt):
    if tt.get("b_hurts_at_p_0_05"):
        return "HURTS"
    if tt.get("b_helps_at_p_0_05"):
        return "HELPS"
    p = tt.get("p_value_two_sided")
    if p is None or p != p:
        return "tied"
    return "n.s."


def print_compact(tag: str, summary: dict) -> None:
    print(f"=== {tag}  (n_cells={summary.get('n_cells','?')}) ===")

    # Pass-rate table: rows = condition, columns = easy/medium/hard
    by_cell = summary.get("by_difficulty_condition", {})
    conditions = []
    seen = set()
    # Preserve insertion order so reactive comes first
    for k in by_cell.keys():
        cond = k.split("__", 1)[1] if "__" in k else k
        if cond not in seen:
            conditions.append(cond); seen.add(cond)

    print()
    print(f"  pass-rate by (difficulty, condition):")
    print(f"  {'condition':22s}  {'easy':>7s}  {'medium':>7s}  {'hard':>7s}")
    for cond in conditions:
        row = f"  {cond:22s} "
        for diff in ("easy", "medium", "hard"):
            v = by_cell.get(f"{diff}__{cond}", {}).get("mean_pass_rate")
            row += f"  {v:>7.3f}" if v is not None else "  " + "    -  "
        print(row)

    # Paired t-tests vs reactive
    tt_block = summary.get("paired_ttests_vs_reactive", {})
    if tt_block:
        print()
        print(f"  paired t-tests vs reactive:")
        print(f"  {'difficulty':>8s}  {'condition':22s} "
              f"{'n':>3s}  {'react':>6s}  {'arm':>6s}  {'diff':>6s} "
              f"{'t':>6s}  {'p':>7s}  verdict")
        for diff in ("easy", "medium", "hard"):
            for cond in conditions:
                if cond == "reactive":
                    continue
                key = f"{diff}__reactive_vs_{cond}"
                tt = tt_block.get(key)
                if not tt or tt.get("n_pairs", 0) < 2:
                    continue
                a = tt.get("mean_pass_a", float("nan"))
                b = tt.get("mean_pass_b", float("nan"))
                d = tt.get("mean_diff_passrate_b_minus_a", float("nan"))
                t = tt.get("t_stat")
                p = tt.get("p_value_two_sided")
                v = _verdict(tt)
                marker = "*" if v == "HURTS" else ("+" if v == "HELPS" else " ")
                print(f"  {marker} {diff:>6s}  {cond:22s} "
                      f"{tt['n_pairs']:>3d}  {a:>6.3f}  {b:>6.3f}  {d:>+6.3f} "
                      f"{_fmt_t(t):>6s}  {_fmt_p(p):>7s}  {v}")

    print()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", default="results")
    p.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    p.add_argument("--full", action="store_true",
                   help="full verbose JSON dump (was the previous default)")
    p.add_argument("--compact", action="store_true",
                   help="compact table (default)")
    p.add_argument("--also-log-tail", action="store_true",
                   help="also print last N lines of log.txt")
    p.add_argument("--n-tail", type=int, default=20)
    args = p.parse_args(argv)

    # Default to compact unless --full was passed.
    use_compact = args.compact or not args.full

    root = Path(args.results_root)
    found_any = False
    for tag in args.tags:
        d = find_latest(root, tag)
        if d is None:
            print(f"### {tag}: no run found under {root}/")
            continue
        sp = d / "summary.json"
        if not sp.exists():
            print(f"### {tag} ({d}): summary.json missing -- run still in progress?")
            continue
        try:
            obj = json.loads(sp.read_text())
        except json.JSONDecodeError:
            print(f"### {tag} ({d}): summary.json unparseable")
            continue
        found_any = True

        if use_compact:
            print_compact(tag, obj)
        else:
            print("=" * 78)
            print(f"### TAG: {tag}\n### DIR: {d}")
            print("=" * 78)
            print(json.dumps(obj, indent=2))
            print()

        if args.also_log_tail:
            lp = d / "log.txt"
            if lp.exists():
                print("-" * 60)
                print(f"  log.txt tail of {tag}:")
                print("-" * 60)
                lines = lp.read_text().splitlines()
                for ln in lines[-args.n_tail:]:
                    print(ln)
                print()

    if not found_any:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
