# PLOT cell tracker

26 valid cells per `MIB/MIB-causal-variable-track/verify_submission.py:VALID_TASK_MODELS × TASK_VARIABLES`.

This file tracks **status only**. For per-cell IIA scores, picked sites,
methodology, and comparisons to baseline DAS, see
[`RESULTS.md`](RESULTS.md) — regenerated from raw eval archives by
`_aggregate.py`.

For methodological narrative (what we tried, what didn't work, why),
see [`JOURNAL.md`](JOURNAL.md).

Status legend: ☐ todo · ◐ in progress · ☑ shipped · ✗ blocked

| # | task | model | variable | status |
|---|---|---|---|---|
| 1  | 4_answer_MCQA | Qwen2ForCausalLM    | answer_pointer | ☑ |
| 2  | 4_answer_MCQA | Qwen2ForCausalLM    | answer         | ☑ |
| 3  | 4_answer_MCQA | Gemma2ForCausalLM   | answer_pointer | ☑ |
| 4  | 4_answer_MCQA | Gemma2ForCausalLM   | answer         | ☑ |
| 5  | 4_answer_MCQA | LlamaForCausalLM    | answer_pointer | ✗ — needs ≥16 GB VRAM (8B at fp16) |
| 6  | 4_answer_MCQA | LlamaForCausalLM    | answer         | ✗ — needs ≥16 GB VRAM |
| 7  | ARC_easy      | Gemma2ForCausalLM   | answer_pointer | ☑ |
| 8  | ARC_easy      | Gemma2ForCausalLM   | answer         | ☑ |
| 9  | ARC_easy      | LlamaForCausalLM    | answer_pointer | ✗ — needs ≥16 GB VRAM |
| 10 | ARC_easy      | LlamaForCausalLM    | answer         | ✗ — needs ≥16 GB VRAM |
| 11 | arithmetic    | Gemma2ForCausalLM   | ones_carry     | ☐ — V=1 collapse risk; needs adjacent-variable workaround |
| 12 | arithmetic    | LlamaForCausalLM    | ones_carry     | ✗ — needs ≥16 GB VRAM |
| 13 | ioi_task      | GPT2LMHeadModel     | output_token   | ☐ — IOI bootstrap (linear params) needed first |
| 14 | ioi_task      | GPT2LMHeadModel     | output_position| ☐ — same |
| 15 | ioi_task      | Qwen2ForCausalLM    | output_token   | ☐ — same |
| 16 | ioi_task      | Qwen2ForCausalLM    | output_position| ☐ — same |
| 17 | ioi_task      | Gemma2ForCausalLM   | output_token   | ☐ — same |
| 18 | ioi_task      | Gemma2ForCausalLM   | output_position| ☐ — same |
| 19 | ioi_task      | LlamaForCausalLM    | output_token   | ✗ — needs ≥16 GB VRAM |
| 20 | ioi_task      | LlamaForCausalLM    | output_position| ✗ — needs ≥16 GB VRAM |
| 21 | ravel_task    | Gemma2ForCausalLM   | Country        | ☐ — unblocked; pipeline validated by Continent smoke |
| 22 | ravel_task    | Gemma2ForCausalLM   | Continent      | ☑ — smoke at 0.845 (n_features=64, dataset_size=128, 1 epoch); scale up later |
| 23 | ravel_task    | Gemma2ForCausalLM   | Language       | ☐ — same; first-token collisions on multi-token Language values may noise IIA |
| 24 | ravel_task    | LlamaForCausalLM    | Country        | ✗ — needs ≥16 GB VRAM |
| 25 | ravel_task    | LlamaForCausalLM    | Continent      | ✗ — needs ≥16 GB VRAM |
| 26 | ravel_task    | LlamaForCausalLM    | Language       | ✗ — needs ≥16 GB VRAM |

## Summary

| status | count | fraction |
|---|---|---|
| ☑ shipped | 7 | 26.9% |
| ◐ in progress | 0 | — |
| ☐ todo (fits on 8 GB) | 10 | 38.5% |
| ✗ blocked on Llama VRAM | 9 | 34.6% |

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
   short note to `JOURNAL.md`.

When a cell becomes blocked, change ☐ → ✗ and add a one-line reason.
