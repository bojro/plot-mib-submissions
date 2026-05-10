#!/usr/bin/env bash
# Overnight 2 (2026-05-09): D.7 ARC tweak → A.2 priority → C.6 arithmetic
# → A.2 stretch → aggregate. Total ~12 hr GPU.
#
# Phase decision rules:
#   - D.7 + C.6 update the shipping submission (the new config / scale-up
#     becomes the canonical cell). Original backed up under
#     submissions/_plot_backups/.
#   - A.2 seed sweeps DON'T change shipping state — each seed is run,
#     evaluated, results saved to a tagged log file, then the baseline is
#     restored. Goal is variance estimation, not re-shipping.
#
# Failures don't kill the chain (`;` semicolons + explicit fallthrough).
# Run inside tmux for survivability:
#     tmux attach -t mib-overnight   # or new -s

set -u
cd "$(dirname "$0")/.."

PY=.venv-mib/bin/python
LOG=logs
BAK=submissions/_plot_backups

start=$(date +%s)
mkdir -p "$LOG" "$BAK"

elapsed () { printf '%dh%02dm' $((($(date +%s) - start) / 3600)) $((($(date +%s) - start) % 3600 / 60)); }

run_plot () {
  local name="$1"; shift
  local log="$LOG/exp_o2_${name}.log"
  echo "[o2] $(elapsed) PLOT ${name} → $log"
  "$PY" -u -m mib_submission.plot.run "$@" > "$log" 2>&1
  echo "[o2] $(elapsed)   PLOT rc=$?"
}
run_eval () {
  local name="$1" cell="$2"
  local log="$LOG/exp_o2_${name}_eval.log"
  echo "[o2] $(elapsed) EVAL ${name} cell=${cell} → $log"
  "$PY" -u scripts/eval_cell.py --cell "$cell" --no-archive > "$log" 2>&1
  local rc=$?
  echo "[o2] $(elapsed)   EVAL rc=${rc}"
  if [ $rc -eq 0 ]; then
    cp submissions/plot/${cell}/*results.json "$LOG/exp_o2_${name}_results.json" 2>/dev/null && echo "[o2]   results saved"
  fi
}

backup_cell () {
  local cell="$1" tag="$2"
  local dest="$BAK/${cell}_${tag}_$(date +%Y%m%d_%H%M%S)"
  echo "[o2] backing up $cell → $dest"
  mkdir -p "$dest" && cp -r submissions/plot/${cell}/* "$dest/" && echo "$dest" > "$LOG/_o2_lastbak_${cell}"
}

restore_cell () {
  local cell="$1"
  local bak; bak=$(cat "$LOG/_o2_lastbak_${cell}")
  echo "[o2] restoring $cell from $bak"
  rm -rf submissions/plot/${cell}
  cp -r "$bak" submissions/plot/${cell}
}

# ============================================================
# Phase 1 — D.7 ARC tweak (cells 7, 8 with stage_b_top_k_grid=(1,))
# ============================================================
echo "[o2] $(elapsed) ===== Phase 1: D.7 ARC tweak ====="

backup_cell ARC_easy_Gemma2ForCausalLM_answer_pointer pre_d7
run_plot d7_arc_pointer --task ARC_easy --model google/gemma-2-2b --variable answer_pointer --train-batch-size 16
run_eval d7_arc_pointer ARC_easy_Gemma2ForCausalLM_answer_pointer
# Archive the tweaked result for the leaderboard pipeline
cp submissions/plot/ARC_easy_Gemma2ForCausalLM_answer_pointer/*results.json mib_submission/results/ARC_easy_Gemma2ForCausalLM_answer_pointer.json 2>/dev/null

backup_cell ARC_easy_Gemma2ForCausalLM_answer pre_d7
run_plot d7_arc_answer --task ARC_easy --model google/gemma-2-2b --variable answer --train-batch-size 16
run_eval d7_arc_answer ARC_easy_Gemma2ForCausalLM_answer
cp submissions/plot/ARC_easy_Gemma2ForCausalLM_answer/*results.json mib_submission/results/ARC_easy_Gemma2ForCausalLM_answer.json 2>/dev/null

# ============================================================
# Phase 2 — A.2 priority: cells 1, 4 seed sweeps (3 seeds each)
# ============================================================
echo "[o2] $(elapsed) ===== Phase 2: A.2 priority seed sweeps (cells 1, 4) ====="

# Cell 1: MCQA × Qwen × answer_pointer
backup_cell 4_answer_MCQA_Qwen2ForCausalLM_answer_pointer pre_seedsweep
for s in 1 2 3; do
  run_plot a2_cell1_seed${s} --task 4_answer_MCQA --model Qwen/Qwen2.5-0.5B --variable answer_pointer --seed $s
  run_eval a2_cell1_seed${s} 4_answer_MCQA_Qwen2ForCausalLM_answer_pointer
done
restore_cell 4_answer_MCQA_Qwen2ForCausalLM_answer_pointer

# Cell 4: MCQA × Gemma × answer
backup_cell 4_answer_MCQA_Gemma2ForCausalLM_answer pre_seedsweep
for s in 1 2 3; do
  run_plot a2_cell4_seed${s} --task 4_answer_MCQA --model google/gemma-2-2b --variable answer --seed $s
  run_eval a2_cell4_seed${s} 4_answer_MCQA_Gemma2ForCausalLM_answer
done
restore_cell 4_answer_MCQA_Gemma2ForCausalLM_answer

# ============================================================
# Phase 3 — C.6 arithmetic ds=1024 scale-up
# ============================================================
echo "[o2] $(elapsed) ===== Phase 3: C.6 arithmetic ds=1024 ====="

backup_cell arithmetic_Gemma2ForCausalLM_ones_carry pre_c6
run_plot c6_arithmetic --task arithmetic --model google/gemma-2-2b --variable ones_carry --dataset-size 1024 --train-batch-size 16
run_eval c6_arithmetic arithmetic_Gemma2ForCausalLM_ones_carry
cp submissions/plot/arithmetic_Gemma2ForCausalLM_ones_carry/*results.json mib_submission/results/arithmetic_Gemma2ForCausalLM_ones_carry.json 2>/dev/null

# ============================================================
# Phase 4 — A.2 stretch: cells 3, 8 seed sweeps
# ============================================================
echo "[o2] $(elapsed) ===== Phase 4: A.2 stretch seed sweeps (cells 3, 8) ====="

# Cell 3: MCQA × Gemma × answer_pointer
backup_cell 4_answer_MCQA_Gemma2ForCausalLM_answer_pointer pre_seedsweep
for s in 1 2 3; do
  run_plot a2_cell3_seed${s} --task 4_answer_MCQA --model google/gemma-2-2b --variable answer_pointer --seed $s
  run_eval a2_cell3_seed${s} 4_answer_MCQA_Gemma2ForCausalLM_answer_pointer
done
restore_cell 4_answer_MCQA_Gemma2ForCausalLM_answer_pointer

# Cell 8: ARC × Gemma × answer (uses the D.7 tweaked config implicitly)
backup_cell ARC_easy_Gemma2ForCausalLM_answer pre_seedsweep
for s in 1 2 3; do
  run_plot a2_cell8_seed${s} --task ARC_easy --model google/gemma-2-2b --variable answer --seed $s --train-batch-size 16
  run_eval a2_cell8_seed${s} ARC_easy_Gemma2ForCausalLM_answer
done
restore_cell ARC_easy_Gemma2ForCausalLM_answer

# ============================================================
# Phase 5 — Aggregate: regenerate RESULTS.md + write seed stats
# ============================================================
echo "[o2] $(elapsed) ===== Phase 5: aggregate ====="

"$PY" -m mib_submission.results._aggregate \
  --write mib_submission/results/RESULTS.md \
  > "$LOG/exp_o2_aggregate.log" 2>&1 \
  && echo "[o2]   aggregate OK" \
  || echo "[o2]   ❌ aggregate failed"

# Per-cell seed mean/std
"$PY" - <<'PYEOF' > "$LOG/exp_o2_seed_stats.log" 2>&1 || echo "[o2]   ❌ seed stats failed"
import json, glob, statistics
from collections import defaultdict
results = defaultdict(list)
for path in sorted(glob.glob("logs/exp_o2_a2_*_results.json")):
    name = path.split("/")[-1].replace("exp_o2_", "").replace("_results.json", "")
    cell = name.split("_seed")[0]
    seed = int(name.split("_seed")[1])
    d = json.loads(open(path).read())
    splits = list(d['dataset'].keys())
    sites = list(d['dataset'][splits[0]]['model_unit'].keys())
    per_site_avgs = {}
    for site in sites:
        layer = d['dataset'][splits[0]]['model_unit'][site]['metadata']['layer']
        scores = []
        for s in splits:
            sv = d['dataset'][s]['model_unit'][site]
            for k, v in sv.items():
                if isinstance(v, dict) and 'average_score' in v:
                    scores.append(v['average_score']); break
        if scores:
            per_site_avgs.setdefault(layer, []).append(sum(scores)/len(scores))
    if per_site_avgs:
        highest = max(max(vs) for vs in per_site_avgs.values())
        results[cell].append((seed, highest))

print(f'{"cell":50s} | {"seed":4s} | highest-view')
print('-'*80)
for cell, runs in sorted(results.items()):
    runs.sort()
    for seed, score in runs:
        print(f'{cell:50s} | {seed:4d} | {score:.4f}')
    if len(runs) > 1:
        scores = [s for _, s in runs]
        print(f'{cell:50s} |  mean | {statistics.mean(scores):.4f}')
        print(f'{cell:50s} |   std | {statistics.stdev(scores):.4f}')
        print()
PYEOF

end=$(date +%s)
TOTAL=$((end - start))
echo "[o2] end: $(date -Iseconds), total $((TOTAL / 3600))h$((TOTAL % 3600 / 60))m"
