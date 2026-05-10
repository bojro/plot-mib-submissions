#!/usr/bin/env bash
# scripts/overnight.sh — chained RAVEL overnight job (~10 hr).
#
# Phase ordering is defensive: smokes for 21/23 first to OOM-canary all
# three vars within the first hour, then the three full runs.
#
#   1. Cell 21 Country smoke    (n_features=64 ds=128)            ~30 min
#   2. Cell 23 Language smoke   (n_features=64 ds=128)            ~30 min
#   3. Cell 22 Continent full   (default n_features=288 ds=256)   ~3 hr
#   4. Cell 21 Country full     (default n_features=288 ds=256)   ~3 hr
#   5. Cell 23 Language full    (default n_features=288 ds=256)   ~3 hr
#
# Smokes intentionally skip the eval step — they're OOM canaries only;
# the full-run overwrites the cell folder anyway. Full runs are followed
# by eval, which archives the results JSON to mib_submission/results/.
#
# Failure of any phase does not kill the chain. Each phase logs to its
# own file under logs/overnight_*.log so issues can be inspected after.
#
# Run from the repo root inside tmux:
#
#     tmux new -s mib-overnight
#     bash scripts/overnight.sh

set -u

cd "$(dirname "$0")/.."

PY=.venv-mib/bin/python
LOG_DIR=logs
mkdir -p "$LOG_DIR"

START=$(date +%s)
echo "[overnight] start: $(date -Iseconds)"
echo "[overnight] gpu state at launch:"
nvidia-smi --query-gpu=name,memory.total,memory.used,power.draw,temperature.gpu --format=csv 2>&1 | sed 's/^/[overnight]   /'

elapsed () { printf '%dh%02dm' $((($(date +%s) - START) / 3600)) $((($(date +%s) - START) % 3600 / 60)); }

run_plot () {
  local phase="$1" cell="$2"; shift 2
  local log="$LOG_DIR/overnight_${phase}.log"
  echo "[overnight] $(elapsed) PLOT phase=${phase} cell=${cell} → $log"
  "$PY" -u -m mib_submission.plot.run "$@" > "$log" 2>&1
  local rc=$?
  echo "[overnight] $(elapsed)   PLOT rc=${rc}"
  return $rc
}

run_eval () {
  local phase="$1" cell="$2"
  local log="$LOG_DIR/overnight_${phase}_eval.log"
  echo "[overnight] $(elapsed) EVAL phase=${phase} cell=${cell} → $log"
  "$PY" -u scripts/eval_cell.py --cell "$cell" > "$log" 2>&1
  local rc=$?
  echo "[overnight] $(elapsed)   EVAL rc=${rc}"
  return $rc
}

# Common args. RAVEL Gemma defaults: n_features=288, dataset_size=256,
# epochs=1, max_new_tokens=2 — set in mib_submission/plot/configs.py.
RAVEL_BASE=(--task ravel_task --model google/gemma-2-2b)
SMOKE_OVERRIDES=(--n-features 64 --dataset-size 128)
# 8 GB OOM guard: forces train batch 16. Continent smoke ran at 32 with
# n_features=64; full at n_features=288 needs the smaller batch.
FULL_OVERRIDES=(--train-batch-size 16)

# ---- Phase 1: Country smoke (OOM canary, no eval) -----------------
run_plot 01_country_smoke ravel_task_Gemma2ForCausalLM_Country \
  "${RAVEL_BASE[@]}" --variable Country "${SMOKE_OVERRIDES[@]}" \
  || echo "[overnight]   ❌ Country smoke PLOT failed (see logs/overnight_01_country_smoke.log)"

# ---- Phase 2: Language smoke (OOM canary, no eval) ----------------
run_plot 02_language_smoke ravel_task_Gemma2ForCausalLM_Language \
  "${RAVEL_BASE[@]}" --variable Language "${SMOKE_OVERRIDES[@]}" \
  || echo "[overnight]   ❌ Language smoke PLOT failed (see logs/overnight_02_language_smoke.log)"

# ---- Phase 3: Continent full ---------------------------------------
if run_plot 03_continent_full ravel_task_Gemma2ForCausalLM_Continent \
     "${RAVEL_BASE[@]}" --variable Continent "${FULL_OVERRIDES[@]}"; then
  run_eval 03_continent_full ravel_task_Gemma2ForCausalLM_Continent \
    || echo "[overnight]   ❌ Continent full eval failed (see logs/overnight_03_continent_full_eval.log)"
else
  echo "[overnight]   ❌ Continent full PLOT failed; skipping eval"
fi

# ---- Phase 4: Country full -----------------------------------------
if run_plot 04_country_full ravel_task_Gemma2ForCausalLM_Country \
     "${RAVEL_BASE[@]}" --variable Country "${FULL_OVERRIDES[@]}"; then
  run_eval 04_country_full ravel_task_Gemma2ForCausalLM_Country \
    || echo "[overnight]   ❌ Country full eval failed (see logs/overnight_04_country_full_eval.log)"
else
  echo "[overnight]   ❌ Country full PLOT failed; skipping eval"
fi

# ---- Phase 5: Language full ----------------------------------------
if run_plot 05_language_full ravel_task_Gemma2ForCausalLM_Language \
     "${RAVEL_BASE[@]}" --variable Language "${FULL_OVERRIDES[@]}"; then
  run_eval 05_language_full ravel_task_Gemma2ForCausalLM_Language \
    || echo "[overnight]   ❌ Language full eval failed (see logs/overnight_05_language_full_eval.log)"
else
  echo "[overnight]   ❌ Language full PLOT failed; skipping eval"
fi

# ---- Aggregate -----------------------------------------------------
echo "[overnight] $(elapsed) regenerating RESULTS.md"
"$PY" -m mib_submission.results._aggregate \
  --write mib_submission/results/RESULTS.md \
  > "$LOG_DIR/overnight_99_aggregate.log" 2>&1 \
  && echo "[overnight]   aggregate OK" \
  || echo "[overnight]   ❌ aggregate failed (see logs/overnight_99_aggregate.log)"

END=$(date +%s)
TOTAL=$((END - START))
echo "[overnight] end: $(date -Iseconds), total $((TOTAL / 3600))h$((TOTAL % 3600 / 60))m"
echo "[overnight] gpu state at end:"
nvidia-smi --query-gpu=name,memory.used,power.draw,temperature.gpu --format=csv 2>&1 | sed 's/^/[overnight]   /'
