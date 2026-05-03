# Apply, Don't Articulate

> A multi-paradigm, multi-scale study of internal-model prompting in LLMs.
> Asking an LLM to *articulate* its internal model before acting hurts.
> *Providing* the model with the rule and asking it to apply it helps.

This repository contains the benchmarks, prompts, graders, raw outputs,
and analysis pipeline for the paper
**Apply, Don't Articulate**.

## Headline finding

We tested a popular agentic-LLM design pattern — *"state your hypothesis,
then predict"*, *"identify the bug, then fix"*, *"state your plan, then
execute"* — across two distinct symbolic paradigms (deterministic integer
sequence prediction and Python code debugging) and seven open-weight
model scales (Qwen2.5 7B/14B/32B, Llama3.1-70B-q4, Qwen2.5-Coder
7B/14B/32B), totalling more than **16,200** prompted trials.

Three findings stand together:

1. **Articulating an internal model never significantly helps** — on no
   (paradigm, scale, difficulty) cell, under any of the articulation
   phrasings tested.
2. **The IM-Hypothesis prompt significantly hurts** in both paradigms.
   The largest single regression is a **97-percentage-point pass-rate
   collapse** at Qwen2.5-Coder-14B on code debugging
   (*t* = 25–42, *p* < 10⁻²⁰), from a single-line prompt change.
3. **Providing the rule directly (the Oracle arm) helps**, with
   monotonically growing magnitude on hard tasks — up to an **84%**
   asymptotic-error reduction at Qwen-32B-hard sequence prediction.

Four causal-mechanism arms (self-consistency, oracle, hypothesis-first,
equal-budget) refute the simplest "tokenized commitment" account but
support a sharper one:  the harm is in **rule-generation**, not
rule-application.  The model can apply a stated rule reliably; it
cannot reliably write a correct one.

The full numbers, statistics (paired *t* + cluster-bootstrap CI),
mechanism analysis, and limitations are in
the NeurIPS-format paper accompanying this repository (kept privately by the author for the time being).

## Repository structure

```
phase_margin_llm/
├── README.md                           ← you are here
└── pipeline/                           ← all code
    ├── README.md                       ← pipeline-level details
    ├── pyproject.toml
    ├── requirements.txt
    ├── src/phase_margin/
    │   ├── coding/                     ← code-debugging benchmark
    │   │   ├── bug_templates.py        (30 hand-curated bug templates)
    │   │   └── grader.py               (sandboxed exec + timeout + def extraction)
    │   ├── llm/                        (Ollama / Anthropic / mock backends)
    │   └── ...                         (loops, embedder, identification, margin, probe)
    ├── experiments/                    ← top-level entry points
    │   ├── run_imp_benchmark.py        (v1 — discovery; 30 hand-curated)
    │   ├── run_imp_benchmark2.py       (v2 — confirmation; 260 programmatic + multi-seed)
    │   ├── run_imp_benchmark3.py       (v3 — mechanism arms)
    │   ├── run_imp_code_benchmark.py   (code-debugging)
    │   ├── analyze_hypothesis_correctness.py    (post-hoc rule-correct × rule-followed)
    │   ├── sample_hypotheses.py                 (inspect emitted HYPOTHESIS lines)
    │   ├── print_v2_summaries.py                (compact summary tables)
    │   ├── print_code_summaries.py              (compact summary tables for code)
    │   └── run_32b_v3.sh                        (overnight wrapper for Qwen-32B)
    ├── tests/                          ← pytest
    └── notebooks/                      ← exploratory notebooks
```

## Reproducing the empirical results

The experiments were run on a single NVIDIA H100 80GB with Ollama serving
the open-weight models. Total compute is approximately 50 H100-hours
across all benchmarks and arms.

### Quick start

```bash
git clone https://github.com/Minds-R-Lab/phase_margin_llm.git
cd phase_margin_llm/pipeline
pip install -e .
ollama pull qwen2.5:7b
ollama pull qwen2.5-coder:7b
```

### v1 — sequence-prediction discovery (~10 min on a 7B)

```bash
python experiments/run_imp_benchmark.py --backend ollama --model qwen2.5:7b --tag qwen7b
```

### v2 — sequence-prediction confirmation (multi-seed, ~12 h on 7B at default settings)

```bash
python experiments/run_imp_benchmark2.py \
    --backend ollama --model qwen2.5:7b \
    --tasks-per-band 100 --n-seeds 5 --temperature 0.7 \
    --task-seed 2026 --tag qwen7b_v2
```

### v3 — mechanism arms (sc5, oracle, hyp_first, eqbudget; ~6 h on 7B)

```bash
python experiments/run_imp_benchmark3.py \
    --backend ollama --model qwen2.5:7b \
    --tasks-per-band 30 --n-seeds 3 --n-sc-samples 5 --temperature 0.7 \
    --conditions im_hyp_sc5 im_oracle im_hyp_first im_hyp_eqbudget \
    --include-baselines --task-seed 2026 --tag qwen7b_v3
```

### Code-debugging benchmark (~3 h on a 14B)

```bash
python experiments/run_imp_code_benchmark.py \
    --backend ollama --model qwen2.5-coder:14b \
    --conditions im_diag_sc5 im_oracle im_diag_first im_diag_eqbudget \
    --include-baselines --n-seeds 3 --n-sc-samples 5 --temperature 0.7 \
    --task-seed 2026 --tag qwen14b_coder_code_v1
```

### Inspecting the results

```bash
# Sequence-prediction summaries
python experiments/print_v2_summaries.py

# Code-debugging summaries (compact tables by default)
python experiments/print_code_summaries.py

# Hypothesis-correctness post-hoc decomposition
python experiments/analyze_hypothesis_correctness.py results/experiment_imp_bench2_*_qwen14b_v2

# Sample emitted HYPOTHESIS lines, stratified by parser status
python experiments/sample_hypotheses.py results/experiment_imp_bench2_*_qwen14b_v2
```

All scripts support **resume-on-crash** via `--results-dir <existing>`:
the JSONL output is append-only, and re-invoking the same command after
an interruption picks up at the next un-recorded cell.

## Citation

Citation details for the accompanying paper will be added once it is
publicly released.  This repository ships with a
[`CITATION.cff`](CITATION.cff) for the code/benchmarks themselves
(GitHub renders this as the "Cite this repository" widget on the
repo home page).

## Background and lineage

This repository began as the validation pipeline for an earlier paper on
*phase-margin certification of multi-agent LLM systems* (a control-theoretic
diagnosis of loop stability in LLM agentic loops); the LLM clients,
loop topologies, and identification machinery in
[`pipeline/src/phase_margin/`](pipeline/src/phase_margin/) come from
that lineage and are documented in
[`pipeline/README.md`](pipeline/README.md).  The IMP paper presented
here (*Apply, Don't Articulate*) shares the infrastructure but is a
self-contained empirical investigation.

## License

The code in `pipeline/` is released for research use. Please see the
headers of individual files for any file-level licensing or
attribution notes.

## Contact

Mohamed A. Mabrok &nbsp;·&nbsp; m.a.mabrok@gmail.com
