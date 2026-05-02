# Papers

Three iterations of the *Apply, Don't Articulate* paper, kept here for
traceability.  The recommended entry points are **v3** (NeurIPS-style,
7 pages) and **v2** (detailed treatment, 23 pages, intended as the
v3 supplementary).

| Version | Pages | Description | TeX | PDF |
|--------:|------:|-------------|-----|-----|
| **v3**  |     7 | NeurIPS-format submission with the full empirical story compressed.  Recommended starting point. | [imp\_paper\_v3.tex](imp_paper_v3.tex) | [imp\_paper\_v3.pdf](imp_paper_v3.pdf) |
| **v2**  |    23 | Full detailed write-up: per-condition tables, 4-way `(rule_correct × rule_followed)` decomposition, inverted-*U* analysis, raw failure-mode replies, full appendices. | [imp\_paper\_v2.tex](imp_paper_v2.tex) | [imp\_paper\_v2.pdf](imp_paper_v2.pdf) |
| **v1**  |    13 | Initial discovery (sequence prediction only, single-seed greedy, 30 hand-curated tasks).  Superseded by v2/v3 in scope and statistics; kept for narrative traceability. | [imp\_paper\_v1.tex](imp_paper_v1.tex) | [imp\_paper\_v1.pdf](imp_paper_v1.pdf) |

## How the empirical claim grew

* **v1** — establishes the harm in sequence prediction with $n = 10$ tasks
  per band, single-seed greedy decoding, four model scales.
* **v2** — programmatic task generator ($260$ tasks, $30$ parametric families
  covering $20$+ patterns), multi-seed sampled decoding, paired-$t$ + cluster
  bootstrap CI.  Introduces the inverted-$U$ for chain-of-thought.
* **v3** — adds four causal-mechanism arms (self-consistency, oracle,
  hypothesis-first, equal-budget) and the 4-way
  `(rule_correct × rule_followed)` decomposition.  Refutes the simplest
  "tokenized commitment" account and supports the sharper
  rule-generation account.
* **Cross-paradigm replication (Section 9 of v2)** — the same five
  mechanism predictions tested again on a code-debugging benchmark with
  hidden test-suite grading.  All five transfer; the catastrophic
  $97$-percentage-point collapse at Qwen2.5-Coder-14B is the largest
  single prompt-induced regression in the study.
* **NeurIPS submission (v3 of this folder)** — the cross-paradigm
  result compressed into $7$ pages with the detailed treatment kept as
  v2 supplementary.

## Compilation

Each `.tex` is self-contained (single-file, no \texttt{.bib} dependency,
no external \texttt{.sty} beyond standard CTAN packages).

```bash
cd papers/
pdflatex -interaction=nonstopmode imp_paper_v3.tex
pdflatex -interaction=nonstopmode imp_paper_v3.tex   # second pass for cross-refs
```

The submitted-version `imp_paper_v3.tex` is currently formatted with a
NeurIPS-approximating preamble (geometry/titlesec); for actual NeurIPS
submission, replace lines 1--19 with `\usepackage[final]{neurips_2026}`
and update the author block per the official template's
double-blind submission instructions.
