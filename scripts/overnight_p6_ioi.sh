#!/usr/bin/env bash
# scripts/overnight_p6_ioi.sh — IOI 13, 14 GPT-2 scale-up.
#
# Runs after overnight.sh (RAVEL) completes if the budget allows.
# Both cells use the same default IOI preset; we only override
# --dataset-size 512 to scale up from the 128-example smoke.
#
# Linear params (ioi_linear_params.json) are expected at
# submissions/plot/ioi_linear_params.json (already there from
# the smoke run; not regenerated).

set -u

cd "$(dirname "$0")/.."

PY=.venv-mib/bin/python
LOG_DIR=logs
mkdir -p "$LOG_DIR"

START=$(date +%s)
echo "[p6] start: $(date -Iseconds)"

elapsed () { printf '%dh%02dm' $((($(date +%s) - START) / 3600)) $((($(date +%s) - START) % 3600 / 60)); }

run_plot () {
  local phase="$1" cell="$2"; shift 2
  local log="$LOG_DIR/overnight_${phase}.log"
  echo "[p6] $(elapsed) PLOT phase=${phase} cell=${cell} → $log"
  "$PY" -u -m mib_submission.plot.run "$@" > "$log" 2>&1
  local rc=$?
  echo "[p6] $(elapsed)   PLOT rc=${rc}"
  return $rc
}

run_eval () {
  local phase="$1" cell="$2"
  local log="$LOG_DIR/overnight_${phase}_eval.log"
  echo "[p6] $(elapsed) EVAL phase=${phase} cell=${cell} → $log"
  "$PY" -u scripts/eval_cell.py --cell "$cell" > "$log" 2>&1
  local rc=$?
  echo "[p6] $(elapsed)   EVAL rc=${rc}"
  return $rc
}

# ---- Phase 6: IOI cell 13 (output_token) full ----------------------
if run_plot 06_ioi13_full ioi_task_GPT2LMHeadModel_output_token \
     --task ioi_task --model gpt2 --variable output_token --dataset-size 512; then
  run_eval 06_ioi13_full ioi_task_GPT2LMHeadModel_output_token \
    || echo "[p6]   ❌ IOI 13 eval failed"
else
  echo "[p6]   ❌ IOI 13 PLOT failed; skipping eval"
fi

# ---- Phase 7: IOI cell 14 (output_position) full -------------------
if run_plot 07_ioi14_full ioi_task_GPT2LMHeadModel_output_position \
     --task ioi_task --model gpt2 --variable output_position --dataset-size 512; then
  run_eval 07_ioi14_full ioi_task_GPT2LMHeadModel_output_position \
    || echo "[p6]   ❌ IOI 14 eval failed"
else
  echo "[p6]   ❌ IOI 14 PLOT failed; skipping eval"
fi

# ---- Final aggregate (overwrites overnight.sh's earlier write) -----
echo "[p6] $(elapsed) regenerating RESULTS.md"
"$PY" -m mib_submission.results._aggregate \
  --write mib_submission/results/RESULTS.md \
  > "$LOG_DIR/overnight_99b_aggregate.log" 2>&1 \
  && echo "[p6]   aggregate OK" \
  || echo "[p6]   ❌ aggregate failed"

END=$(date +%s)
TOTAL=$((END - START))
echo "[p6] end: $(date -Iseconds), total $((TOTAL / 3600))h$((TOTAL % 3600 / 60))m"
