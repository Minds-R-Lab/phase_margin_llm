# phase-margin

Black-box phase-margin certification of LLM agentic loops.

This repository is the validation pipeline for the white paper
*Phase-Margin Certification of Multi-Agent LLM Systems: A Behavioural
Frequency-Response Approach to Loop-Stability Diagnosis*.

It implements:
1. The behavioural phase-response identifier (complex least-squares fit
   from probe perturbations) — `phase_margin.identification`.
2. The phase-margin formula and the contractive / oscillatory /
   exploratory regime classifier — `phase_margin.margin`.
3. Pluggable LLM backends — `phase_margin.llm.{mock, anthropic, ollama}`.
4. Pluggable loop topologies — `phase_margin.loops` (synthetic LTI shadow,
   paraphrase loop, ReAct-style, multi-agent debate).
5. End-to-end orchestration — `phase_margin.pipeline.run_certification`.

It validates the framework in three stages of increasing realism:

| stage | what is validated | backend | runtime |
|-------|-------------------|---------|---------|
| 1     | identification math vs. analytical phase | synthetic LTI shadow | seconds |
| 2     | regime classifier vs. spectral-radius ground truth | nonlinear shadow | seconds |
| 3     | 2-cycle prediction vs. observed paraphrasing limit cycle | mock or Claude / Ollama | minutes |

See `notebooks/` for the runnable demonstrations.

## Install

```bash
# core only (synthetic experiments)
pip install -e .

# + Anthropic
pip install -e ".[anthropic]"

# + Ollama (local models)
pip install -e ".[ollama]"

# + jupyter notebooks
pip install -e ".[notebooks,dev]"
```

## Quick start

```python
from phase_margin import run_certification
from phase_margin.llm.mock import SyntheticLTILoop
from phase_margin.probe import default_basis

loop = SyntheticLTILoop.from_random(d=8, seed=0)
basis = default_basis(d=8, n_directions=4, seed=0)

report = run_certification(
    loop=loop,
    basis=basis,
    horizon=32,
    n_seeds=8,
    n_frequencies=8,
    epsilon=0.1,
)

print(f"phase margin = {report.phase_margin:.3f}")
print(f"predicted regime = {report.regime}")
```

## CLI

```bash
phase-margin synthetic --d 8 --horizon 32 --frequencies 8
phase-margin paraphrase --backend mock --topic "Deep learning" --steps 20
```

## Repo layout

```
src/phase_margin/        library
  ├── types.py
  ├── embedder.py
  ├── identification.py
  ├── margin.py
  ├── probe.py
  ├── loops.py
  ├── ground_truth.py
  ├── pipeline.py
  ├── cli.py
  └── llm/
      ├── base.py
      ├── mock.py
      ├── anthropic_client.py
      └── ollama_client.py
tests/                   pytest tests
notebooks/               runnable validation notebooks
data/                    cached probe rollouts
results/                 figures, csvs, reports
```

## Validation status

Stage 1 (synthetic LTI) is fully validated by `tests/test_synthetic_lti.py`
and notebook `01`.  Stages 2 and 3 are demonstrated in notebooks `02` and
`03`.  Real-LLM experiments require an API key (`ANTHROPIC_API_KEY`) or a
local Ollama daemon at `http://localhost:11434`.
