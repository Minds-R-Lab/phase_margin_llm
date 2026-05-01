"""
grader.py
=========

Sandboxed grader for code-debugging benchmark replies.

Given a model reply (raw text, possibly with markdown fences, prose,
hypotheses, multiple `def` blocks) and a CodeTask, this module:

  1. Extracts a parseable Python `def` block matching the target
     function name (handles markdown fences, takes the LAST
     syntactically-valid candidate when multiple defs are emitted).
  2. Exec's the extracted code in a restricted-builtins namespace
     (no `import`, `open`, `exec`, `eval`, `compile`, `__import__`,
     `globals`, `locals`, `vars`, etc.).
  3. Runs the task's `test_runner` against the extracted function
     under a wall-clock timeout (SIGALRM, Linux/macOS only; will
     raise on Windows main thread, which is acceptable since the
     H100 deployment target is Linux).
  4. Returns a GradeResult with n_pass / n_total / error_kind /
     candidate_code so downstream analysis can stratify by the
     failure modes ("no_def", "syntax", "timeout", etc.) as well as
     by pass-rate.

The grader is intentionally not bullet-proof against an adversarial
model; the goal is to prevent accidental damage from buggy fixes
(infinite loops, memory blowups, accidental file writes).  An
adversarial model could still construct a sandbox escape via Python
introspection.  For research-bench use this is acceptable.
"""
from __future__ import annotations

import ast
import builtins
import re
import signal
from contextlib import contextmanager
from dataclasses import dataclass


# ===========================================================================
# 1.  Restricted-builtins namespace
# ===========================================================================
# Allow common pure-Python operations; deny anything that touches I/O,
# imports, or compile/exec.
_SAFE_BUILTIN_NAMES = (
    # Fundamentals
    "True", "False", "None", "NotImplemented", "Ellipsis",
    # Predicates and casts
    "abs", "all", "any", "ascii", "bin", "bool", "callable", "chr",
    "complex", "divmod", "float", "format", "frozenset", "hash", "hex",
    "id", "int", "isinstance", "issubclass", "len", "max", "min", "oct",
    "ord", "pow", "repr", "round", "str",
    # Iteration / collections
    "dict", "enumerate", "filter", "iter", "list", "map", "next",
    "range", "reversed", "set", "slice", "sorted", "sum", "tuple",
    "type", "zip", "object",
    # Useful but harmless
    "print", "hasattr", "getattr", "setattr", "delattr",
    # Exceptions used in templates
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "AttributeError", "StopIteration", "ZeroDivisionError",
    "ArithmeticError", "RuntimeError", "RecursionError",
    # Decorators / class basics
    "property", "staticmethod", "classmethod", "super",
)


def _build_safe_builtins() -> dict:
    real = builtins.__dict__
    return {n: real[n] for n in _SAFE_BUILTIN_NAMES if n in real}


_SAFE_BUILTINS = _build_safe_builtins()


def safe_globals() -> dict:
    """Fresh restricted-globals dict for exec'ing model code."""
    return {"__builtins__": _SAFE_BUILTINS}


# ===========================================================================
# 2.  SIGALRM-based timeout context manager
# ===========================================================================
class GraderTimeout(Exception):
    pass


@contextmanager
def timeout(seconds: float):
    """Wall-clock timeout via SIGALRM.  Linux/macOS, main thread only.
    No-op if seconds <= 0."""
    if seconds is None or seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise GraderTimeout(f"timed out after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


# ===========================================================================
# 3.  Code extraction from model reply
# ===========================================================================
# Markdown fences come in many shapes; strip them before we look for defs.
_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*\s*\n?|\s*\n?)", re.MULTILINE)


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("\n", text)


def _candidate_def_blocks(text: str, function_name: str) -> list[str]:
    """Find every line starting with 'def <function_name>(' and gather it
    plus all subsequent indented (or blank) lines until a non-indented
    line."""
    lines = text.splitlines()
    head_re = re.compile(r"^def\s+" + re.escape(function_name) + r"\s*\(")
    blocks: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        if head_re.match(lines[i]):
            block = [lines[i]]
            j = i + 1
            while j < n:
                ln = lines[j]
                if ln.strip() == "":
                    block.append(ln)
                    j += 1
                elif ln.startswith(" ") or ln.startswith("\t"):
                    block.append(ln)
                    j += 1
                else:
                    break
            # Drop trailing blank lines from the block
            while block and block[-1].strip() == "":
                block.pop()
            if block:
                blocks.append("\n".join(block))
            i = j
        else:
            i += 1
    return blocks


def extract_function_code(reply: str, function_name: str) -> str | None:
    """Return a string containing a parseable `def function_name(...)` block,
    or None if no syntactically-valid candidate exists.  Strategy:
      1. Strip markdown fences from the whole reply.
      2. Try to ast.parse the entire stripped text.  If it parses and
         contains the target function, return the whole text (this is
         the cleanest case; helper functions defined alongside are
         preserved).
      3. Otherwise scan for `def <name>(` blocks and return the LAST
         one that ast.parses.  (LAST so that an earlier "thinking
         aloud" def gets superseded by the final answer.)"""
    text = _strip_fences(reply).strip()
    if not text:
        return None

    # Strategy 1: whole-text parse with helper-function preservation
    try:
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return text
    except SyntaxError:
        pass

    # Strategy 2: per-block extraction, return last-valid
    blocks = _candidate_def_blocks(text, function_name)
    for block in reversed(blocks):
        try:
            ast.parse(block)
            return block
        except SyntaxError:
            continue
    return None


# ===========================================================================
# 4.  Top-level grade()
# ===========================================================================
@dataclass
class GradeResult:
    n_pass: int
    n_total: int
    error_kind: str | None     # None | "no_def" | "syntax" | "timeout" |
                                #   "exec_error:<ExcName>" | "test_error:<ExcName>" |
                                #   "no_fn_after_exec"
    candidate_code: str | None  # the extracted def block (for inspection)

    @property
    def pass_rate(self) -> float:
        return self.n_pass / self.n_total if self.n_total else 0.0


def grade(reply: str, task, *,
          exec_timeout_sec: float = 2.0,
          test_timeout_sec: float = 2.0) -> GradeResult:
    """Grade a model reply against a CodeTask.

    Imports CodeTask lazily to avoid circular import; the grader does
    not need the bug_templates module loaded to function."""
    code = extract_function_code(reply, task.function_name)
    if code is None:
        return GradeResult(0, task.n_tests, "no_def", None)

    # Exec the candidate
    ns: dict = {"__builtins__": _SAFE_BUILTINS}
    try:
        with timeout(exec_timeout_sec):
            exec(code, ns)
    except GraderTimeout:
        return GradeResult(0, task.n_tests, "timeout", code)
    except SyntaxError:
        return GradeResult(0, task.n_tests, "syntax", code)
    except Exception as e:                        # noqa: BLE001
        return GradeResult(0, task.n_tests, f"exec_error:{type(e).__name__}", code)

    fn = ns.get(task.function_name)
    if fn is None:
        return GradeResult(0, task.n_tests, "no_fn_after_exec", code)

    # Run the test runner
    try:
        with timeout(test_timeout_sec):
            n_pass = task.test_runner(fn)
    except GraderTimeout:
        return GradeResult(0, task.n_tests, "timeout", code)
    except Exception as e:                        # noqa: BLE001
        return GradeResult(0, task.n_tests, f"test_error:{type(e).__name__}", code)

    n_pass = max(0, min(int(n_pass), task.n_tests))
    return GradeResult(n_pass, task.n_tests, None, code)
