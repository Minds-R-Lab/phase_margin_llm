#!/usr/bin/env python3
"""
print_v2_summaries.py
=====================

Print the ``summary.json`` from each of the four benchmark2 runs to
stdout, with clear separators, so the whole batch can be copied in
one go.

By default it looks under ``pipeline/results/`` for the most recent
directory matching each of the four expected tags:

    qwen7b_v2   qwen14b_v2   qwen32b_v2   llama70b_v2

Override with ``--results-root <path>`` or ``--tags A B C D``.
Missing runs are noted but do not crash the script -- you can run it
mid-overnight to peek at whatever finished so far.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_TAGS = ["qwen7b_v2", "qwen14b_v2", "qwen32b_v2", "llama70b_v2"]


def find_latest_for_tag(results_root: Path, tag: str) -> Path | None:
    """Return the most-recent dir matching experiment_imp_bench2_*_<tag>."""
    if not results_root.is_dir():
        return None
    candidates = sorted(
        results_root.glob(f"experiment_imp_bench2_*_{tag}"),
        key=lambda p: p.name,
    )
    return candidates[-1] if candidates else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", default="results",
                   help="root directory containing experiment_imp_bench2_* "
                        "(default: ./results, i.e. relative to wherever you "
                        "invoke the script from)")
    p.add_argument("--tags", nargs="+", default=DEFAULT_TAGS,
                   help="tags to look for, in printing order")
    p.add_argument("--also-manifest", action="store_true",
                   help="also dump the manifest.json (verbose; off by default)")
    p.add_argument("--also-log-tail", action="store_true",
                   help="also print the last 20 lines of log.txt for each run")
    args = p.parse_args(argv)

    root = Path(args.results_root)
    bar = "=" * 78
    sub = "-" * 78

    found_any = False
    for tag in args.tags:
        d = find_latest_for_tag(root, tag)
        print(bar)
        print(f"### TAG: {tag}")
        if d is None:
            print(f"  (no run found under {root}/ matching this tag)")
            continue
        print(f"### DIR: {d}")
        print(bar)
        found_any = True

        if args.also_manifest:
            mp = d / "manifest.json"
            if mp.exists():
                print(sub)
                print("# manifest.json")
                print(sub)
                print(mp.read_text().rstrip())
            else:
                print(f"# manifest.json missing")

        sp = d / "summary.json"
        print(sub)
        print("# summary.json")
        print(sub)
        if sp.exists():
            try:
                obj = json.loads(sp.read_text())
                print(json.dumps(obj, indent=2))
            except json.JSONDecodeError:
                print("# (summary.json present but unparseable)")
                print(sp.read_text().rstrip())
        else:
            print("# summary.json missing -- run probably still in progress")

        if args.also_log_tail:
            lp = d / "log.txt"
            if lp.exists():
                print(sub)
                print("# log.txt (tail 20)")
                print(sub)
                lines = lp.read_text().splitlines()
                for ln in lines[-20:]:
                    print(ln)
        print()

    if not found_any:
        print()
        print(f"NOTHING FOUND under {root}/.")
        print("If your runs are elsewhere, pass --results-root <path>.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
