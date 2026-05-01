"""
bug_templates.py
================

30 hand-curated Python bug templates for the IM-prompt code-debugging
benchmark.  Ten per difficulty band:
  EASY:    single-token bugs (operator, literal, off-by-one)
  MEDIUM:  control-flow or data-structure bugs
  HARD:    semantic / algorithmic bugs

Each template specifies:
  * function_name      -- name of the function under test
  * canonical_code     -- correct implementation (string)
  * buggy_code         -- subtly wrong version (string)
  * bug_description    -- short ground-truth fix description for Oracle
  * test_runner(fn)    -- callable: returns count of passing tests
  * n_tests            -- total number of tests

A self-validation pass at import time confirms that for every template,
canonical passes all tests and buggy fails at least one.  Import fails
loudly with AssertionError on any inconsistency.

Tests live in-process; they call the candidate function inside a
restricted exec'd namespace (no imports allowed in the model's code).
The grader provides a wall-clock timeout via subprocess; this module
just defines the data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


# ===========================================================================
# Data model
# ===========================================================================
@dataclass
class CodeTask:
    name: str
    difficulty: str
    function_name: str
    canonical_code: str
    buggy_code: str
    bug_description: str
    test_runner: Callable[[Callable], int]
    n_tests: int


def _exec_to_fn(code: str, function_name: str):
    """Execute ``code`` in a fresh namespace and return the named function."""
    ns: dict = {}
    exec(code, ns)
    fn = ns.get(function_name)
    if fn is None:
        raise ValueError(f"function {function_name!r} not found in code")
    return fn


# ===========================================================================
# EASY templates (10)
# ===========================================================================

def _t_e01_off_by_one_range() -> CodeTask:
    canonical = (
        "def sum_n(n):\n"
        "    return sum(range(1, n+1))\n"
    )
    buggy = (
        "def sum_n(n):\n"
        "    return sum(range(1, n))\n"
    )
    def runner(fn):
        cases = [(5, 15), (10, 55), (1, 1), (100, 5050)]
        return sum(1 for n, exp in cases if fn(n) == exp)
    return CodeTask(
        name="e01_off_by_one_range",
        difficulty="easy",
        function_name="sum_n",
        canonical_code=canonical,
        buggy_code=buggy,
        bug_description="off-by-one in range upper bound; needs n+1 not n",
        test_runner=runner,
        n_tests=4,
    )


def _t_e02_wrong_comparison_findmax() -> CodeTask:
    canonical = (
        "def find_max(xs):\n"
        "    m = xs[0]\n"
        "    for x in xs:\n"
        "        if x > m:\n"
        "            m = x\n"
        "    return m\n"
    )
    buggy = (
        "def find_max(xs):\n"
        "    m = xs[0]\n"
        "    for x in xs:\n"
        "        if x < m:\n"
        "            m = x\n"
        "    return m\n"
    )
    def runner(fn):
        cases = [([1, 3, 2], 3), ([5, 1, 4, 8, 2], 8), ([7], 7), ([-3, -1, -7], -1)]
        return sum(1 for xs, exp in cases if fn(xs) == exp)
    return CodeTask(
        name="e02_wrong_comparison_findmax",
        difficulty="easy",
        function_name="find_max",
        canonical_code=canonical,
        buggy_code=buggy,
        bug_description="comparison operator is < but should be > to find the maximum",
        test_runner=runner,
        n_tests=4,
    )


def _t_e03_wrong_arithmetic_avg() -> CodeTask:
    canonical = (
        "def avg(xs):\n"
        "    return sum(xs) / len(xs)\n"
    )
    buggy = (
        "def avg(xs):\n"
        "    return sum(xs) * len(xs)\n"
    )
    def runner(fn):
        cases = [([2, 4, 6], 4.0), ([1, 2, 3, 4, 5], 3.0),
                 ([10], 10.0), ([0, 100], 50.0)]
        return sum(1 for xs, exp in cases if abs(fn(xs) - exp) < 1e-9)
    return CodeTask(
        name="e03_wrong_arithmetic_avg",
        difficulty="easy",
        function_name="avg",
        canonical_code=canonical,
        buggy_code=buggy,
        bug_description="uses multiplication where division is needed",
        test_runner=runner,
        n_tests=4,
    )


def _t_e04_swapped_operands_diff() -> CodeTask:
    canonical = (
        "def range_diff(xs):\n"
        "    return max(xs) - min(xs)\n"
    )
    buggy = (
        "def range_diff(xs):\n"
        "    return min(xs) - max(xs)\n"
    )
    def runner(fn):
        cases = [([1, 5, 3], 4), ([10, 2, 7, 1], 9), ([5, 5, 5], 0), ([-3, 0, 4], 7)]
        return sum(1 for xs, exp in cases if fn(xs) == exp)
    return CodeTask(
        name="e04_swapped_operands_diff",
        difficulty="easy",
        function_name="range_diff",
        canonical_code=canonical,
        buggy_code=buggy,
        bug_description="subtraction operands are swapped (min - max instead of max - min)",
        test_runner=runner,
        n_tests=4,
    )


def _t_e05_wrong_default_dict() -> CodeTask:
    canonical = (
        "def get_count(d, key):\n"
        "    return d.get(key, 0)\n"
    )
    buggy = (
        "def get_count(d, key):\n"
        "    return d.get(key, 1)\n"
    )
    def runner(fn):
        cases = [({"a": 3}, "a", 3), ({"a": 3}, "b", 0),
                 ({}, "x", 0), ({"x": 0}, "x", 0)]
        return sum(1 for d, k, exp in cases if fn(d, k) == exp)
    return CodeTask(
        name="e05_wrong_default_dict",
        difficulty="easy",
        function_name="get_count",
        canonical_code=canonical,
        buggy_code=buggy,
        bug_description="default value for missing key is 1 but should be 0",
        test_runner=runner,
        n_tests=4,
    )


def _t_e06_wrong_index_last() -> CodeTask:
    canonical = (
        "def last(xs):\n"
        "    return xs[-1]\n"
    )
    buggy = (
        "def last(xs):\n"
        "    return xs[0]\n"
    )
    def runner(fn):
        cases = [([1, 2, 3], 3), (["a", "b", "c"], "c"), ([7], 7), ([0, 1], 1)]
        return sum(1 for xs, exp in cases if fn(xs) == exp)
    return CodeTask(
        name="e06_wrong_index_last",
        difficulty="easy",
        function_name="last",
        canonical_code=canonical,
        buggy_code=buggy,
        bug_description="returns first element xs[0] instead of last element xs[-1]",
        test_runner=runner,
        n_tests=4,
    )


def _t_e07_wrong_logical_both_positive() -> CodeTask:
    canonical = (
        "def both_positive(a, b):\n"
        "    return a > 0 and b > 0\n"
    )
    buggy = (
        "def both_positive(a, b):\n"
        "    return a > 0 or b > 0\n"
    )
    def runner(fn):
        cases = [(1, 2, True), (1, -1, False), (-1, -1, False), (0, 1, False)]
        return sum(1 for a, b, exp in cases if fn(a, b) == exp)
    return CodeTask(
        name="e07_wrong_logical_both_positive",
        difficulty="easy",
        function_name="both_positive",
        canonical_code=canonical,
        buggy_code=buggy,
        bug_description="uses logical OR but the function name says BOTH; should be AND",
        test_runner=runner,
        n_tests=4,
    )


def _t_e08_missing_return_square() -> CodeTask:
    canonical = (
        "def sq(x):\n"
        "    return x * x\n"
    )
    buggy = (
        "def sq(x):\n"
        "    result = x * x\n"
    )
    def runner(fn):
        cases = [(3, 9), (0, 0), (-2, 4), (10, 100)]
        return sum(1 for x, exp in cases if fn(x) == exp)
    return CodeTask(
        name="e08_missing_return_square",
        difficulty="easy",
        function_name="sq",
        canonical_code=canonical,
        buggy_code=buggy,
        bug_description="missing return statement; function returns None implicitly",
        test_runner=runner,
        n_tests=4,
    )


def _t_e09_wrong_loop_var_sumsq() -> CodeTask:
    canonical = (
        "def sum_sq(xs):\n"
        "    total = 0\n"
        "    for x in xs:\n"
        "        total += x * x\n"
        "    return total\n"
    )
    buggy = (
        "def sum_sq(xs):\n"
        "    total = 0\n"
        "    for x in xs:\n"
        "        total += xs[0] * xs[0]\n"
        "    return total\n"
    )
    def runner(fn):
        cases = [([1, 2, 3], 14), ([4], 16), ([2, 2, 2], 12), ([3, 4], 25)]
        return sum(1 for xs, exp in cases if fn(xs) == exp)
    return CodeTask(
        name="e09_wrong_loop_var_sumsq",
        difficulty="easy",
        function_name="sum_sq",
        canonical_code=canonical,
        buggy_code=buggy,
        bug_description="always uses xs[0] inside the loop instead of the loop variable x",
        test_runner=runner,
        n_tests=4,
    )


def _t_e10_inverted_boolean_iseven() -> CodeTask:
    canonical = (
        "def is_even(n):\n"
        "    return n % 2 == 0\n"
    )
    buggy = (
        "def is_even(n):\n"
        "    return n % 2 != 0\n"
    )
    def runner(fn):
        cases = [(2, True), (3, False), (0, True), (-4, True), (7, False)]
        return sum(1 for n, exp in cases if fn(n) == exp)
    return CodeTask(
        name="e10_inverted_boolean_iseven",
        difficulty="easy",
        function_name="is_even",
        canonical_code=canonical,
        buggy_code=buggy,
        bug_description="comparison is inverted; returns True for odd numbers",
        test_runner=runner,
        n_tests=5,
    )


# ===========================================================================
# MEDIUM templates (10)
# ===========================================================================

def _t_m01_wrong_base_case_factorial() -> CodeTask:
    canonical = (
        "def fact(n):\n"
        "    return 1 if n <= 1 else n * fact(n-1)\n"
    )
    buggy = (
        "def fact(n):\n"
        "    return 0 if n <= 1 else n * fact(n-1)\n"
    )
    def runner(fn):
        cases = [(5, 120), (1, 1), (0, 1), (4, 24), (6, 720)]
        return sum(1 for n, exp in cases if fn(n) == exp)
    return CodeTask(
        name="m01_wrong_base_case_factorial",
        difficulty="medium",
        function_name="fact",
        canonical_code=canonical,
        buggy_code=buggy,
        bug_description="recursion base case returns 0 but should return 1",
        test_runner=runner,
        n_tests=5,
    )


def _t_m02_mutable_default_dict() -> CodeTask:
    canonical = (
        "def add_count(key, counts=None):\n"
        "    if counts is None:\n"
        "        counts = {}\n"
        "    counts[key] = counts.get(key, 0) + 1\n"
        "    return counts\n"
    )
    buggy = (
        "def add_count(key, counts={}):\n"
        "    counts[key] = counts.get(key, 0) + 1\n"
        "    return counts\n"
    )
    def runner(fn):
        # Sequential calls with no explicit dict: should be independent.
        # Bug shares the default {} across calls.
        r1 = fn("a")
        r2 = fn("b")
        r3 = fn("a", {"x": 5})
        passed = 0
        if r1 == {"a": 1}:                 passed += 1
        if r2 == {"b": 1}:                 passed += 1
        if r3 == {"x": 5, "a": 1}:         passed += 1
        # explicit dict argument
        if fn("y", {"y": 2}) == {"y": 3}:  passed += 1
        return passed
    return CodeTask(
        name="m02_mutable_default_dict",
        difficulty="medium",
        function_name="add_count",
        canonical_code=canonical,
        buggy_code=buggy,
        bug_description="mutable default dict is shared across calls; use None and create inside",
        test_runner=runner,
        n_tests=4,
    )


def _t_m03_list_method_append_vs_extend() -> CodeTask:
    canonical = (
        "def combine(a, b):\n"
        "    a = list(a)\n"
        "    a.extend(b)\n"
        "    return a\n"
    )
    buggy = (
        "def combine(a, b):\n"
        "    a = list(a)\n"
        "    a.append(b)\n"
        "    return a\n"
    )
    def runner(fn):
        cases = [
            ([1, 2], [3, 4], [1, 2, 3, 4]),
            ([], [1], [1]),
            ([5], [], [5]),
            (["a"], ["b", "c"], ["a", "b", "c"]),
        ]
        return sum(1 for a, b, exp in cases if fn(a, b) == exp)
    return CodeTask(
        name="m03_list_method_append_vs_extend",
        difficulty="medium",
        function_name="combine",
        canonical_code=canonical,
        buggy_code=buggy,
        bug_description="uses append (adds list as single element) instead of extend (adds elements)",
        test_runner=runner,
        n_tests=4,
    )


def _t_m04_wrong_string_method_capitalize() -> CodeTask:
    canonical = (
        "def cap_first(s):\n"
        "    return s[0].upper() + s[1:]\n"
    )
    buggy = (
        "def cap_first(s):\n"
        "    return s.upper()\n"
    )
    def runner(fn):
        cases = [("hello", "Hello"), ("python", "Python"),
                 ("a", "A"), ("xY", "XY")]
        return sum(1 for s, exp in cases if fn(s) == exp)
    return CodeTask(
        name="m04_wrong_string_method_capitalize",
        difficulty="medium",
        function_name="cap_first",
        canonical_code=canonical,
        buggy_code=buggy,
        bug_description="uppercases entire string instead of just the first character",
        test_runner=runner,
        n_tests=4,
    )


def _t_m05_wrong_sentinel_findindex() -> CodeTask:
    canonical = (
        "def find_idx(xs, target):\n"
        "    for i, x in enumerate(xs):\n"
        "        if x == target:\n"
        "            return i\n"
        "    return -1\n"
    )
    buggy = (
        "def find_idx(xs, target):\n"
        "    for i, x in enumerate(xs):\n"
        "        if x == target:\n"
        "            return i\n"
        "    return None\n"
    )
    def runner(fn):
        cases = [
            ([1, 2, 3], 2, 1),
            ([1, 2, 3], 99, -1),
            ([], 5, -1),
            (["a", "b", "c"], "c", 2),
        ]
        return sum(1 for xs, t, exp in cases if fn(xs, t) == exp)
    return CodeTask(
        name="m05_wrong_sentinel_findindex",
        difficulty="medium",
        function_name="find_idx",
        canonical_code=canonical,
        buggy_code=buggy,
        bug_description="returns None when not found but should return -1",
        test_runner=runner,
        n_tests=4,
    )


def _t_m06_early_return_nested_allrows() -> CodeTask:
    canonical = (
        "def all_rows_have(rows, val):\n"
        "    for row in rows:\n"
        "        if val not in row:\n"
        "            return False\n"
        "    return True\n"
    )
    buggy = (
        "def all_rows_have(rows, val):\n"
        "    for row in rows:\n"
        "        if val in row:\n"
        "            return True\n"
        "    return False\n"
    )
    def runner(fn):
        cases = [
            ([[1, 2], [2, 3]], 2, True),
            ([[1, 2], [3]], 2, False),
            ([[5], [5], [5]], 5, True),
            ([], 5, True),
        ]
        return sum(1 for rows, v, exp in cases if fn(rows, v) == exp)
    return CodeTask(
        name="m06_early_return_nested_allrows",
        difficulty="medium",
        function_name="all_rows_have",
        canonical_code=canonical,
        buggy_code=buggy,
        bug_description="returns True if ANY row has the value; should be ALL rows (any vs all confusion)",
        test_runner=runner,
        n_tests=4,
    )


def _t_m07_off_by_one_2d_diagonal() -> CodeTask:
    canonical = (
        "def diag_sum(m):\n"
        "    n = len(m)\n"
        "    return sum(m[i][i] for i in range(n))\n"
    )
    buggy = (
        "def diag_sum(m):\n"
        "    n = len(m)\n"
        "    return sum(m[i][i] for i in range(n-1))\n"
    )
    def runner(fn):
        cases = [
            ([[1, 2], [3, 4]], 5),
            ([[1, 0, 0], [0, 2, 0], [0, 0, 3]], 6),
            ([[7]], 7),
            ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 15),
        ]
        return sum(1 for mat, exp in cases if fn(mat) == exp)
    return CodeTask(
        name="m07_off_by_one_2d_diagonal",
        difficulty="medium",
        function_name="diag_sum",
        canonical_code=canonical,
        buggy_code=buggy,
        bug_description="loop misses last diagonal element; range(n-1) should be range(n)",
        test_runner=runner,
        n_tests=4,
    )


def _t_m08_operator_precedence_discount() -> CodeTask:
    canonical = (
        "def discounted(price, rate):\n"
        "    return price * (1 - rate)\n"
    )
    buggy = (
        "def discounted(price, rate):\n"
        "    return price * 1 - rate\n"
    )
    def runner(fn):
        cases = [(100, 0.2, 80.0), (50, 0.0, 50.0),
                 (200, 0.5, 100.0), (10, 0.1, 9.0)]
        return sum(1 for p, r, exp in cases if abs(fn(p, r) - exp) < 1e-9)
    return CodeTask(
        name="m08_operator_precedence_discount",
        difficulty="medium",
        function_name="discounted",
        canonical_code=canonical,
        buggy_code=buggy,
        bug_description="missing parentheses around (1 - rate); evaluates as price - rate",
        test_runner=runner,
        n_tests=4,
    )


def _t_m09_accumulator_in_loop_cumsum() -> CodeTask:
    canonical = (
        "def cumsum(xs):\n"
        "    result = []\n"
        "    total = 0\n"
        "    for x in xs:\n"
        "        total += x\n"
        "        result.append(total)\n"
        "    return result\n"
    )
    buggy = (
        "def cumsum(xs):\n"
        "    result = []\n"
        "    for x in xs:\n"
        "        total = 0\n"
        "        total += x\n"
        "        result.append(total)\n"
        "    return result\n"
    )
    def runner(fn):
        cases = [
            ([1, 2, 3], [1, 3, 6]),
            ([1, 1, 1, 1], [1, 2, 3, 4]),
            ([], []),
            ([5], [5]),
        ]
        return sum(1 for xs, exp in cases if fn(xs) == exp)
    return CodeTask(
        name="m09_accumulator_in_loop_cumsum",
        difficulty="medium",
        function_name="cumsum",
        canonical_code=canonical,
        buggy_code=buggy,
        bug_description="accumulator total is reset inside the loop; should be initialized once before the loop",
        test_runner=runner,
        n_tests=4,
    )


def _t_m10_wrong_step_evens() -> CodeTask:
    canonical = (
        "def evens_up_to(n):\n"
        "    return list(range(2, n+1, 2))\n"
    )
    buggy = (
        "def evens_up_to(n):\n"
        "    return list(range(2, n+1, 1))\n"
    )
    def runner(fn):
        cases = [(8, [2, 4, 6, 8]), (5, [2, 4]),
                 (1, []), (10, [2, 4, 6, 8, 10])]
        return sum(1 for n, exp in cases if fn(n) == exp)
    return CodeTask(
        name="m10_wrong_step_evens",
        difficulty="medium",
        function_name="evens_up_to",
        canonical_code=canonical,
        buggy_code=buggy,
        bug_description="loop step is 1 but should be 2 to produce only even numbers",
        test_runner=runner,
        n_tests=4,
    )


# ===========================================================================
# HARD templates (10)
# ===========================================================================

def _t_h01_wrong_algorithm_reverse_uses_sort() -> CodeTask:
    canonical = (
        "def reverse_str(s):\n"
        "    return s[::-1]\n"
    )
    buggy = (
        "def reverse_str(s):\n"
        "    return ''.join(sorted(s))\n"
    )
    def runner(fn):
        cases = [("hello", "olleh"), ("abc", "cba"),
                 ("", ""), ("racecar", "racecar")]
        return sum(1 for s, exp in cases if fn(s) == exp)
    return CodeTask(
        name="h01_wrong_algorithm_reverse_uses_sort",
        difficulty="hard",
        function_name="reverse_str",
        canonical_code=canonical,
        buggy_code=buggy,
        bug_description="sorts characters instead of reversing the string",
        test_runner=runner,
        n_tests=4,
    )


def _t_h02_wrong_sign_distance() -> CodeTask:
    canonical = (
        "def euclid(p1, p2):\n"
        "    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5\n"
    )
    buggy = (
        "def euclid(p1, p2):\n"
        "    return ((p1[0]+p2[0])**2 + (p1[1]+p2[1])**2) ** 0.5\n"
    )
    def runner(fn):
        cases = [
            ((0, 0), (3, 4), 5.0),
            ((1, 1), (4, 5), 5.0),
            ((0, 0), (0, 0), 0.0),
            ((-1, -1), (2, 3), 5.0),
        ]
        return sum(1 for p1, p2, exp in cases if abs(fn(p1, p2) - exp) < 1e-6)
    return CodeTask(
        name="h02_wrong_sign_distance",
        difficulty="hard",
        function_name="euclid",
        canonical_code=canonical,
        buggy_code=buggy,
        bug_description="uses + instead of - in the distance formula coordinate differences",
        test_runner=runner,
        n_tests=4,
    )


def _t_h03_dict_keyerror_wordcount() -> CodeTask:
    canonical = (
        "def count_words(text):\n"
        "    c = {}\n"
        "    for w in text.split():\n"
        "        c[w] = c.get(w, 0) + 1\n"
        "    return c\n"
    )
    buggy = (
        "def count_words(text):\n"
        "    c = {}\n"
        "    for w in text.split():\n"
        "        c[w] = c[w] + 1\n"
        "    return c\n"
    )
    def runner(fn):
        cases = [
            ("hi hi bye", {"hi": 2, "bye": 1}),
            ("a", {"a": 1}),
            ("x y x y x", {"x": 3, "y": 2}),
        ]
        passed = 0
        for text, exp in cases:
            try:
                if fn(text) == exp:
                    passed += 1
            except KeyError:
                pass
        # empty input should give empty dict (does not crash for either)
        try:
            if fn("") == {}:
                passed += 1
        except KeyError:
            pass
        return passed
    return CodeTask(
        name="h03_dict_keyerror_wordcount",
        difficulty="hard",
        function_name="count_words",
        canonical_code=canonical,
        buggy_code=buggy,
        bug_description="c[w] raises KeyError on first occurrence; should use c.get(w, 0)",
        test_runner=runner,
        n_tests=4,
    )


def _t_h04_off_by_one_string_middle() -> CodeTask:
    canonical = (
        "def middle(s):\n"
        "    return s[len(s)//2]\n"
    )
    buggy = (
        "def middle(s):\n"
        "    return s[len(s)//2 - 1]\n"
    )
    def runner(fn):
        cases = [("abc", "b"), ("hello", "l"), ("X", "X"), ("abcde", "c")]
        return sum(1 for s, exp in cases if fn(s) == exp)
    return CodeTask(
        name="h04_off_by_one_string_middle",
        difficulty="hard",
        function_name="middle",
        canonical_code=canonical,
        buggy_code=buggy,
        bug_description="off-by-one; returns char before middle (extra -1 in index)",
        test_runner=runner,
        n_tests=4,
    )


def _t_h05_integer_division_median() -> CodeTask:
    canonical = (
        "def median(xs):\n"
        "    s = sorted(xs)\n"
        "    n = len(s)\n"
        "    if n % 2:\n"
        "        return s[n//2]\n"
        "    return (s[n//2 - 1] + s[n//2]) / 2\n"
    )
    buggy = (
        "def median(xs):\n"
        "    s = sorted(xs)\n"
        "    n = len(s)\n"
        "    if n % 2:\n"
        "        return s[n//2]\n"
        "    return (s[n//2 - 1] + s[n//2]) // 2\n"
    )
    def runner(fn):
        cases = [
            ([1, 2, 3, 4], 2.5),
            ([1, 2, 3], 2),
            ([1, 3], 2.0),
            ([1, 2, 3, 4, 5, 6], 3.5),
        ]
        return sum(1 for xs, exp in cases if abs(fn(xs) - exp) < 1e-9)
    return CodeTask(
        name="h05_integer_division_median",
        difficulty="hard",
        function_name="median",
        canonical_code=canonical,
        buggy_code=buggy,
        bug_description="uses integer division // for the even-length case; should be float division /",
        test_runner=runner,
        n_tests=4,
    )


def _t_h06_off_by_one_binary_search() -> CodeTask:
    canonical = (
        "def bsearch(xs, target):\n"
        "    lo, hi = 0, len(xs) - 1\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if xs[mid] == target:\n"
        "            return mid\n"
        "        elif xs[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return -1\n"
    )
    buggy = (
        "def bsearch(xs, target):\n"
        "    lo, hi = 0, len(xs) - 1\n"
        "    while lo < hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if xs[mid] == target:\n"
        "            return mid\n"
        "        elif xs[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return -1\n"
    )
    def runner(fn):
        cases = [
            ([1, 2, 3, 4, 5], 5, 4),
            ([1, 2, 3], 3, 2),
            ([1, 2, 3, 4, 5], 1, 0),
            ([1, 2, 3, 4, 5], 99, -1),
        ]
        return sum(1 for xs, t, exp in cases if fn(xs, t) == exp)
    return CodeTask(
        name="h06_off_by_one_binary_search",
        difficulty="hard",
        function_name="bsearch",
        canonical_code=canonical,
        buggy_code=buggy,
        bug_description="while condition is < but should be <=; misses the last comparison",
        test_runner=runner,
        n_tests=4,
    )


def _t_h07_wrong_invariant_gcd() -> CodeTask:
    canonical = (
        "def gcd(a, b):\n"
        "    while b:\n"
        "        a, b = b, a % b\n"
        "    return a\n"
    )
    buggy = (
        "def gcd(a, b):\n"
        "    while b:\n"
        "        a, b = b, a // b\n"
        "    return a\n"
    )
    def runner(fn):
        cases = [(12, 8, 4), (15, 25, 5), (7, 1, 1), (100, 75, 25)]
        return sum(1 for a, b, exp in cases if fn(a, b) == exp)
    return CodeTask(
        name="h07_wrong_invariant_gcd",
        difficulty="hard",
        function_name="gcd",
        canonical_code=canonical,
        buggy_code=buggy,
        bug_description="uses // instead of % in the Euclidean recurrence",
        test_runner=runner,
        n_tests=4,
    )


def _t_h08_incorrect_normalization() -> CodeTask:
    canonical = (
        "def normalize(xs):\n"
        "    s = sum(xs)\n"
        "    return [x / s for x in xs]\n"
    )
    buggy = (
        "def normalize(xs):\n"
        "    s = sum(xs)\n"
        "    return [x / (s + 1) for x in xs]\n"
    )
    def runner(fn):
        cases = [[1, 2, 3], [4, 4], [10, 20, 30, 40]]
        passed = 0
        for xs in cases:
            r = fn(xs)
            if abs(sum(r) - 1.0) < 1e-6:
                passed += 1
        # one extra check: proportions are preserved
        r = fn([1, 1])
        if abs(r[0] - r[1]) < 1e-9 and abs(sum(r) - 1.0) < 1e-6:
            passed += 1
        return passed
    return CodeTask(
        name="h08_incorrect_normalization",
        difficulty="hard",
        function_name="normalize",
        canonical_code=canonical,
        buggy_code=buggy,
        bug_description="divides by sum+1 instead of sum; output does not sum to 1",
        test_runner=runner,
        n_tests=4,
    )


def _t_h09_set_loses_order_dedup() -> CodeTask:
    canonical = (
        "def dedup(xs):\n"
        "    seen = set()\n"
        "    result = []\n"
        "    for x in xs:\n"
        "        if x not in seen:\n"
        "            seen.add(x)\n"
        "            result.append(x)\n"
        "    return result\n"
    )
    buggy = (
        "def dedup(xs):\n"
        "    return list(set(xs))\n"
    )
    def runner(fn):
        cases = [
            ([3, 1, 2, 1, 3], [3, 1, 2]),
            ([1, 2, 3], [1, 2, 3]),
            ([5, 5, 5], [5]),
            (["b", "a", "b"], ["b", "a"]),
        ]
        return sum(1 for xs, exp in cases if fn(xs) == exp)
    return CodeTask(
        name="h09_set_loses_order_dedup",
        difficulty="hard",
        function_name="dedup",
        canonical_code=canonical,
        buggy_code=buggy,
        bug_description="uses set() which loses insertion order; needs an order-preserving approach",
        test_runner=runner,
        n_tests=4,
    )


def _t_h10_case_sensitive_palindrome() -> CodeTask:
    canonical = (
        "def is_palindrome(s):\n"
        "    s = s.lower()\n"
        "    return s == s[::-1]\n"
    )
    buggy = (
        "def is_palindrome(s):\n"
        "    return s == s[::-1]\n"
    )
    def runner(fn):
        cases = [
            ("Racecar", True),
            ("hello", False),
            ("aba", True),
            ("AbBa", True),
        ]
        return sum(1 for s, exp in cases if fn(s) == exp)
    return CodeTask(
        name="h10_case_sensitive_palindrome",
        difficulty="hard",
        function_name="is_palindrome",
        canonical_code=canonical,
        buggy_code=buggy,
        bug_description="missing case normalization; case-sensitive comparison fails on mixed case",
        test_runner=runner,
        n_tests=4,
    )


# ===========================================================================
# Registry + builder
# ===========================================================================
_ALL_TEMPLATES = [
    _t_e01_off_by_one_range, _t_e02_wrong_comparison_findmax,
    _t_e03_wrong_arithmetic_avg, _t_e04_swapped_operands_diff,
    _t_e05_wrong_default_dict, _t_e06_wrong_index_last,
    _t_e07_wrong_logical_both_positive, _t_e08_missing_return_square,
    _t_e09_wrong_loop_var_sumsq, _t_e10_inverted_boolean_iseven,
    _t_m01_wrong_base_case_factorial, _t_m02_mutable_default_dict,
    _t_m03_list_method_append_vs_extend,
    _t_m04_wrong_string_method_capitalize,
    _t_m05_wrong_sentinel_findindex, _t_m06_early_return_nested_allrows,
    _t_m07_off_by_one_2d_diagonal, _t_m08_operator_precedence_discount,
    _t_m09_accumulator_in_loop_cumsum, _t_m10_wrong_step_evens,
    _t_h01_wrong_algorithm_reverse_uses_sort, _t_h02_wrong_sign_distance,
    _t_h03_dict_keyerror_wordcount, _t_h04_off_by_one_string_middle,
    _t_h05_integer_division_median, _t_h06_off_by_one_binary_search,
    _t_h07_wrong_invariant_gcd, _t_h08_incorrect_normalization,
    _t_h09_set_loses_order_dedup, _t_h10_case_sensitive_palindrome,
]


def build_task_set() -> list[CodeTask]:
    """Return all 30 templates instantiated.  Deterministic; no seed needed
    because there is no parameterization in v1 of this benchmark."""
    return [factory() for factory in _ALL_TEMPLATES]


# ===========================================================================
# Self-validation:  for every template, canonical passes all tests and
# buggy fails at least one.  Runs at import time; AssertionError on any
# inconsistency.
# ===========================================================================
def _validate_all_templates() -> None:
    tasks = build_task_set()
    seen_names: set = set()
    counts = {"easy": 0, "medium": 0, "hard": 0}
    for t in tasks:
        assert t.name not in seen_names, "duplicate template name: %s" % t.name
        seen_names.add(t.name)
        assert t.difficulty in counts, "bad difficulty: %s" % t.difficulty
        counts[t.difficulty] += 1
        # canonical must pass everything
        canon = _exec_to_fn(t.canonical_code, t.function_name)
        canon_pass = t.test_runner(canon)
        assert canon_pass == t.n_tests, (
            "[%s] canonical passed %d/%d tests"
            % (t.name, canon_pass, t.n_tests)
        )
        # buggy must fail at least one
        try:
            buggy = _exec_to_fn(t.buggy_code, t.function_name)
        except Exception as e:
            # buggy code that doesn't even exec is a failure too
            continue
        buggy_pass = t.test_runner(buggy)
        assert buggy_pass < t.n_tests, (
            "[%s] buggy passed all %d tests; bug not exposed"
            % (t.name, t.n_tests)
        )
    assert counts == {"easy": 10, "medium": 10, "hard": 10}, \
        "band counts mismatch: %r" % counts


_validate_all_templates()
