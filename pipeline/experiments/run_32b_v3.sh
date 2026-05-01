#!/usr/bin/env bash
# run_32b_v3.sh
# =============
# Wrapper around run_imp_benchmark3.py for Qwen2.5-32B.
#
# Defaults are sized for ~6-10h on a single H100 with Ollama:
#   * 20 tasks per difficulty band  (60 total tasks)
#   * 2 seeds per cell
#   * sc5 uses 3 hypothesis samples (not 5)
#   * Two new arms only:  im_hyp_sc5 + im_oracle
#     (the two most diagnostic;  hyp_first + eqbudget are skipped
#     to fit overnight)
#   * --include-baselines so the run is self-contained for analysis
#
# The script supports crash-resume: if ~/phase_margin_llm/pipeline/
# results/experiment_imp_bench3_*_qwen32b_v3 exists, it resumes into
# the most-recent matching dir.  Otherwise a fresh dir is created.
#
# Usage
# -----
#   bash run_32b_v3.sh                 # run with defaults
#   bash run_32b_v3.sh resume          # explicitly resume the latest
#   bash run_32b_v3.sh --tasks-per-band 30 --n-seeds 3   # override
#
# To kill cleanly:  Ctrl-C then re-invoke; resume picks up where it
# stopped from cells.jsonl.

set -e

cd "$(dirname "$0")/.."   # cd into pipeline/

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="qwen2.5:32b"
TAG="qwen32b_v3"

# Defaults; can be overridden by argv (after possible 'resume').
TASKS_PER_BAND=20
N_SEEDS=2
N_SC_SAMPLES=3
TEMPERATURE=0.7
K_STEPS=12
ASYM=5
TASK_SEED=2026
CONDITIONS="im_hyp_sc5 im_oracle"

# Parse args.  If first arg is 'resume', force resume mode.
FORCE_RESUME=0
if [[ "${1:-}" == "resume" ]]; then
  FORCE_RESUME=1; shift
fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks-per-band) TASKS_PER_BAND="$2"; shift 2 ;;
    --n-seeds)        N_SEEDS="$2";        shift 2 ;;
    --n-sc-samples)   N_SC_SAMPLES="$2";   shift 2 ;;
    --conditions)     shift; CONDITIONS=""; while [[ $# -gt 0 && "$1" != --* ]]; do CONDITIONS="$CONDITIONS $1"; shift; done ;;
    --K)              K_STEPS="$2";        shift 2 ;;
    --asym-window)    ASYM="$2";           shift 2 ;;
    --task-seed)      TASK_SEED="$2";      shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Resume detection
EXISTING_DIR=""
if compgen -G "results/experiment_imp_bench3_*_${TAG}" > /dev/null; then
  EXISTING_DIR=$(ls -td results/experiment_imp_bench3_*_${TAG} | head -1)
fi
RESUME_FLAG=""
if [[ $FORCE_RESUME -eq 1 || -n "$EXISTING_DIR" ]]; then
  if [[ -z "$EXISTING_DIR" ]]; then
    echo "no existing dir to resume; running fresh" >&2
  else
    echo "resuming into $EXISTING_DIR"
    RESUME_FLAG="--results-dir $EXISTING_DIR"
  fi
fi

# Cell-count and walltime estimate
N_CONDS=$(echo "reactive internal_hypothesis internal_cot $CONDITIONS" | wc -w)
N_BANDS=3
N_CELLS=$(( N_BANDS * TASKS_PER_BAND * N_CONDS * N_SEEDS ))
# Weighted call-count per cell (sc5 = N_SC_SAMPLES; others = 1)
NC_PER_CELL_SCALAR=$(( N_CONDS - 1 + N_SC_SAMPLES ))
N_CALLS=$(( N_CELLS * K_STEPS * NC_PER_CELL_SCALAR / N_CONDS ))
SECS_PER_CALL=8        # 32B at ollama ~ 6-10s for non-CoT calls; CoT is longer
EST_SECS=$(( N_CALLS * SECS_PER_CALL ))
EST_HRS=$(( EST_SECS / 3600 ))

echo "============================================================"
echo "  Qwen2.5-32B benchmark3 run plan"
echo "============================================================"
echo "  conditions       : reactive internal_hypothesis internal_cot$CONDITIONS"
echo "  tasks-per-band   : $TASKS_PER_BAND  (total tasks: $((N_BANDS * TASKS_PER_BAND)))"
echo "  n-seeds          : $N_SEEDS"
echo "  n-sc-samples     : $N_SC_SAMPLES"
echo "  K, asym-window   : $K_STEPS, $ASYM"
echo "  task-seed        : $TASK_SEED"
echo "  total cells      : $N_CELLS"
echo "  estimated calls  : ~$N_CALLS"
echo "  estimated time   : ~${EST_HRS}h  (at ${SECS_PER_CALL}s/call avg; first call slower for warmup)"
[[ -n "$RESUME_FLAG" ]] && echo "  resume into      : $EXISTING_DIR"
echo "============================================================"
echo ""

# Sanity: confirm Ollama has the model pulled
if command -v ollama &>/dev/null; then
  if ! ollama list 2>/dev/null | awk '{print $1}' | grep -q "^${MODEL}$"; then
    echo "WARNING: '${MODEL}' not found in 'ollama list'." >&2
    echo "         Pulling now..." >&2
    ollama pull "${MODEL}" || { echo "ollama pull failed" >&2; exit 1; }
  fi
fi

# Launch
exec "$PYTHON_BIN" experiments/run_imp_benchmark3.py \
  --backend ollama \
  --model "$MODEL" \
  --tasks-per-band "$TASKS_PER_BAND" \
  --n-seeds "$N_SEEDS" \
  --n-sc-samples "$N_SC_SAMPLES" \
  --temperature "$TEMPERATURE" \
  --K "$K_STEPS" \
  --asym-window "$ASYM" \
  --task-seed "$TASK_SEED" \
  --conditions $CONDITIONS \
  --include-baselines \
  $RESUME_FLAG \
  --tag "$TAG"
