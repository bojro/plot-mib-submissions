# PLOT cell tracker

26 valid cells per `MIB/MIB-causal-variable-track/verify_submission.py:VALID_TASK_MODELS × TASK_VARIABLES`.

Status legend: ☐ todo · ◐ in progress · ☑ shipped · ✗ blocked

| # | task | model | variable | status | mean IIA | sites | DAS baseline | notes |
|---|---|---|---|---|---|---|---|---|
| 1 | 4_answer_MCQA | Qwen2ForCausalLM | answer_pointer | ☑ | 0.956 | 4 | 0.96 best / 0.76 avg | V=4 choices, picks L23/last_token; essentially ties baseline DAS best |
| 2 | 4_answer_MCQA | Qwen2ForCausalLM | answer | ☑ | 0.801 | 4 | 0.97 best / 0.86 avg | PLOT picked correctly: L23/last_token full-replace IIA = 1.000 on calibration; gap is DAS@n_features=16 not generalizing to random-letter splits, not site-selection |
| 3 | 4_answer_MCQA | Gemma2ForCausalLM | answer_pointer | ☐ | — | — | — | |
| 4 | 4_answer_MCQA | Gemma2ForCausalLM | answer | ☐ | — | — | — | |
| 5 | 4_answer_MCQA | LlamaForCausalLM | answer_pointer | ☐ | — | — | — | Llama 8B, cost-bound |
| 6 | 4_answer_MCQA | LlamaForCausalLM | answer | ☐ | — | — | — | Llama 8B, cost-bound |
| 7 | ARC_easy | Gemma2ForCausalLM | answer_pointer | ☐ | — | — | — | |
| 8 | ARC_easy | Gemma2ForCausalLM | answer | ☐ | — | — | — | |
| 9 | ARC_easy | LlamaForCausalLM | answer_pointer | ☐ | — | — | — | Llama 8B |
| 10 | ARC_easy | LlamaForCausalLM | answer | ☐ | — | — | — | Llama 8B |
| 11 | arithmetic | Gemma2ForCausalLM | ones_carry | ☐ | — | — | — | single var, V=1 collapse risk |
| 12 | arithmetic | LlamaForCausalLM | ones_carry | ☐ | — | — | — | Llama 8B |
| 13 | ioi_task | GPT2LMHeadModel | output_token | ☐ | — | — | — | smallest model |
| 14 | ioi_task | GPT2LMHeadModel | output_position | ☐ | — | — | — | smallest model |
| 15 | ioi_task | Qwen2ForCausalLM | output_token | ☐ | — | — | — | |
| 16 | ioi_task | Qwen2ForCausalLM | output_position | ☐ | — | — | — | |
| 17 | ioi_task | Gemma2ForCausalLM | output_token | ☐ | — | — | — | |
| 18 | ioi_task | Gemma2ForCausalLM | output_position | ☐ | — | — | — | |
| 19 | ioi_task | LlamaForCausalLM | output_token | ☐ | — | — | — | Llama 8B |
| 20 | ioi_task | LlamaForCausalLM | output_position | ☐ | — | — | — | Llama 8B |
| 21 | ravel_task | Gemma2ForCausalLM | Country | ☐ | — | — | — | best PLOT candidate (V≥3 distinct vars) |
| 22 | ravel_task | Gemma2ForCausalLM | Continent | ☐ | — | — | — | |
| 23 | ravel_task | Gemma2ForCausalLM | Language | ☐ | — | — | — | |
| 24 | ravel_task | LlamaForCausalLM | Country | ☐ | — | — | — | Llama 8B |
| 25 | ravel_task | LlamaForCausalLM | Continent | ☐ | — | — | — | Llama 8B |
| 26 | ravel_task | LlamaForCausalLM | Language | ☐ | — | — | — | Llama 8B |

## Progress

- **Shipped**: 2 / 26 (7.7%)
- **In progress**: 0
- **Blocked**: 0
- **Todo**: 24

## How to update

When a cell ships:
1. Change the status box from ☐ to ☑.
2. Fill `mean IIA`, `sites` (count of (layer, position) pairs trained), `DAS baseline` (from leaderboard if available).
3. Append a one-line note (config tweaks, surprises, gap reason).
4. Add a row to `EVAL_LOG.md` with per-split IIA breakdown.
5. Archive the eval JSON to `mib_submission/results/<task>_<model>_<variable>.json`.

Cells expected to be straightforward (PLOT works as designed): RAVEL × Gemma × {Country, Continent, Language}.

Cells expected to need workarounds (V=1 collapse risk or single-variable): arithmetic × * × ones_carry.

Cells with known site-selection limits (similar structure to shipped MCQA cell): ARC_easy × * × {answer_pointer, answer}, 4_answer_MCQA × {Gemma, Llama} × *.
