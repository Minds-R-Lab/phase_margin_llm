#!/usr/bin/env python3
"""
print_code_summaries.py
=======================

Dump the summary.json from each of the three code-benchmark tags so
they can be copied to chat in one block.

By default looks for the three tags:
    qwen7b_coder_code_v1
    qwen14b_coder_code_v1
    qwen32b_coder_code_v1

Usage
-----
    cd ~/phase_margin_llm/pipeline
    python experiments/print_code_summaries.py

Override the tags / results-root with --tags / --results-root if you
used different ones.

Missing tags are reported but do not crash.
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", default="results",
                   help="root containing experiment_imp_code_*  (default: ./results)")
    p.add_argument("--tags", nargs="+", default=DEFAULT_TAGS,
                   help="tags to look for, in print order")
    p.add_argument("--also-manifest", action="store_true",
                   help="also dump manifest.json for each run")
    p.add_argument("--also-log-tail", action="store_true",
                   help="also print last 30 lines of log.txt for each run")
    p.add_argument("--n-tail", type=int, default=30,
                   help="lines from log.txt tail when --also-log-tail (default 30)")
    args = p.parse_args(argv)

    root = Path(args.results_root)
    bar = "=" * 78
    sub = "-" * 78

    found_any = False
    for tag in args.tags:
        d = find_latest(root, tag)
        print(bar)
        print(f"### TAG: {tag}")
        if d is None:
            print(f"  (no run found under {root}/ matching this tag)")
            print(bar)
            print()
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
                print(f"# log.txt (tail {args.n_tail})")
                print(sub)
                lines = lp.read_text().splitlines()
                for ln in lines[-args.n_tail:]:
                    print(ln)
        print()

    if not found_any:
        print()
        print(f"NOTHING FOUND under {root}/.")
        print("If your runs are elsewhere, pass --results-root <path>.")
        print(f"If your tags are different, pass --tags A B C ...")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
