# PLOT cell tracker

26 valid cells per `MIB/MIB-causal-variable-track/verify_submission.py:VALID_TASK_MODELS × TASK_VARIABLES`.

This file tracks **status only**. For per-cell IIA scores, picked sites,
methodology, and comparisons to baseline DAS, see
[`RESULTS.md`](RESULTS.md) — regenerated from raw eval archives by
`_aggregate.py`.

For methodological narrative (what we tried, what didn't work, why),
see [`../../JOURNAL.md`](../../JOURNAL.md).

Status legend: ☐ todo · ◐ in progress · ☑ shipped · ✗ blocked

| # | task | model | variable | status |
|---|---|---|---|---|
| 1  | 4_answer_MCQA | Qwen2ForCausalLM    | answer_pointer | ☑ — seed sweep (3 seeds) gave 1.000 ± 0.000 highest-view; gap to DAS LB 1.000 was seed noise (original 0.8915) |
| 2  | 4_answer_MCQA | Qwen2ForCausalLM    | answer         | ☑ — structural per ../../PLOT_SHORTCOMINGS.md §2; -0.125 gap accepted |
| 3  | 4_answer_MCQA | Gemma2ForCausalLM   | answer_pointer | ☑ — seed sweep gave 0.923 ± 0.006 highest-view; -0.051 vs DAS LB 0.974 (outside seed band, structural) |
| 4  | 4_answer_MCQA | Gemma2ForCausalLM   | answer         | ☑ — seed sweep gave 0.904 ± 0.010 highest-view; -0.070 vs DAS LB 0.974 (outside seed band, structural) |
| 5  | 4_answer_MCQA | LlamaForCausalLM    | answer_pointer | ✗ — needs ≥16 GB VRAM (8B at fp16) |
| 6  | 4_answer_MCQA | LlamaForCausalLM    | answer         | ✗ — needs ≥16 GB VRAM |
| 7  | ARC_easy      | Gemma2ForCausalLM   | answer_pointer | ☑ — D.7 config (`stage_b_top_k_grid=(1,)` per ../../PLOT_SHORTCOMINGS.md §8) gave 0.827 highest-view (unchanged from pre-D.7 0.827); -0.009 vs DAS LB 0.836 (tied) |
| 8  | ARC_easy      | Gemma2ForCausalLM   | answer         | ☑ — D.7 config jumped 0.849 → **0.999 highest-view** (+0.058 vs DAS LB 0.941); win driven by harness identity-fallback at L25 last_token, where DAS rotation was actively *subtractive* (../../PLOT_SHORTCOMINGS.md §15) |
| 9  | ARC_easy      | LlamaForCausalLM    | answer_pointer | ✗ — needs ≥16 GB VRAM |
| 10 | ARC_easy      | LlamaForCausalLM    | answer         | ✗ — needs ≥16 GB VRAM |
| 11 | arithmetic    | Gemma2ForCausalLM   | ones_carry     | ☑ — **smoke restored after ds=1024 scale-up regression**. Smoke (ds=128) IIA 0.440 highest-view; baseline DAS ~0.53. ds=1024 rerun on 2026-05-10 picked L17+L19 instead of smoke's L16+L21 and dropped to 0.265 (ones_carry_test ~0). Future work: rerun with `--signature-dataset ones_carry_train`. |
| 12 | arithmetic    | LlamaForCausalLM    | ones_carry     | ✗ — needs ≥16 GB VRAM |
| 13 | ioi_task      | GPT2LMHeadModel     | output_token   | ☑ — full (V=3 splits-as-rows, ds=512, all 12 layers); MSE 5.16; picks L9H11/H2/H3; baseline DAS 2.08; **structural per ../../PLOT_SHORTCOMINGS.md §13 — accepted (2026-05-09)**, signature picks loud Name Movers, misses indirect-effect heads |
| 14 | ioi_task      | GPT2LMHeadModel     | output_position| ☑ — full (V=3 splits-as-rows, ds=512, all 12 layers); MSE 16.0; picks L9H10/H5/H1; baseline DAS 2.20; **structural per ../../PLOT_SHORTCOMINGS.md §13 — accepted (2026-05-09)**; bypass diagnostic to S-Inhibition heads gave 4.12 (in `_plot_backups/ioi14_baseline_*`) but pure PLOT ships |
| 15 | ioi_task      | Qwen2ForCausalLM    | output_token   | ☐ — same |
| 16 | ioi_task      | Qwen2ForCausalLM    | output_position| ☐ — same |
| 17 | ioi_task      | Gemma2ForCausalLM   | output_token   | ☐ — same |
| 18 | ioi_task      | Gemma2ForCausalLM   | output_position| ☐ — same |
| 19 | ioi_task      | LlamaForCausalLM    | output_token   | ✗ — needs ≥16 GB VRAM |
| 20 | ioi_task      | LlamaForCausalLM    | output_position| ✗ — needs ≥16 GB VRAM |
| 21 | ravel_task    | Gemma2ForCausalLM   | Country        | ☑ — full (n_features=288, ds=256, 1 epoch); IIA 0.615; picks (L6 + L25, entity_last_token); baseline DAS 0.957; **structural per ../../PLOT_SHORTCOMINGS.md §14 — accepted (2026-05-09)**; E-R-4 confirmed picks are reasonable; L25 is a 0.615 ceiling site even with identity featurizer |
| 22 | ravel_task    | Gemma2ForCausalLM   | Continent      | ☑ — full (n_features=288, ds=256, 1 epoch); IIA 0.856 highest-view, +0.008 vs baseline DAS 0.848 🏆; picks same as 21/23 (L6 + L25) |
| 23 | ravel_task    | Gemma2ForCausalLM   | Language       | ☑ — full (n_features=288, ds=256, 1 epoch); IIA 0.629; picks same as 21/22; baseline DAS 0.812; **structural per ../../PLOT_SHORTCOMINGS.md §14 — accepted (2026-05-09)**; same ceiling as Country, plus alphabet compaction (13 collision groups, max 9 labels) |
| 24 | ravel_task    | LlamaForCausalLM    | Country        | ✗ — needs ≥16 GB VRAM |
| 25 | ravel_task    | LlamaForCausalLM    | Continent      | ✗ — needs ≥16 GB VRAM |
| 26 | ravel_task    | LlamaForCausalLM    | Language       | ✗ — needs ≥16 GB VRAM |

## Summary

| status | count | fraction |
|---|---|---|
| ☑ shipped | 13 | 50.0% |
| ◐ in progress | 0 | — |
| ☐ todo (fits on 8 GB) | 4 | 15.4% |
| ✗ blocked on Llama VRAM | 9 | 34.6% |

**8 GB box: 12 of 12 reachable cells now have submissions. 11 at full quality, 1 (cell 11 arithmetic) still smoke.** Remaining `☐` cells are the 4 IOI Qwen/Gemma cells that need ≥16 GB VRAM — cloud-only.

## How to update

When a cell ships:
1. Change ☐ → ☑ in the table above.
2. Archive the eval JSON to `<task>_<model>_<variable>.json` in this directory.
3. If the submission folder under `submissions/plot/` is preserved,
   `_aggregate.py` will pick up the picked sites automatically. Otherwise
   it falls back to a heuristic on the eval JSON (marked with † in
   `RESULTS.md`).
4. Regenerate `RESULTS.md`:
   `python -m mib_submission.results._aggregate --write mib_submission/results/RESULTS.md`
5. If the run revealed something methodologically interesting, append a
   short note to `../../JOURNAL.md`.

When a cell becomes blocked, change ☐ → ✗ and add a one-line reason.
