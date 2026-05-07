# PLOT MIB submission — eval log

Cell: `4_answer_MCQA × Qwen2ForCausalLM × answer_pointer`
Eval command: `evaluate_submission.py --no-private_data --public_data` (public test sets only).

Filtered test-set sizes after the harness's correctness filter:
- `answerPosition_test`: 50 / 50 kept (100%)
- `randomLetter_test`: 26 / 50 kept (52%)
- `answerPosition_randomLetter_test`: 30 / 50 kept (60%)

Mean IIA = unweighted mean of per-split max IIA (the "best" leaderboard convention).

| run | OT rows | split | sites | answerPosition | randomLetter | answerPosition_randomLetter | **mean** |
|---|---|---|---|---|---|---|---|
| Single-layer (initial) | `symbol0..3` | answerPosition_randomLetter_train | 1 (L12, correct_symbol) | 0.94 | 1.00 | 0.63 | **0.857** |
| Single-layer (choices) | `choice0..3` | answerPosition_randomLetter_train | 1 (L8, correct_symbol) | 0.94 | 1.00 | 0.63 | **0.857** |
| V=4 multi-row | `choice0..3` | answerPosition_randomLetter_train | 4 ([L0, L2, L8, L23]) | 1.00 | 1.00 | 0.83 | **0.944** |
| V=8 mixed | `choice0..3 + symbol0..3` | answerPosition_randomLetter_train | 7 ([L2, L4, L7, L8, L13, L23] × token positions) | 1.00 | 1.00 | 0.83 | **0.944** |
| **Off-PLOT (L15, L20)** | **bypass — hardcoded sites** | **n/a** | **2 (L15, L20 × last_token)** | **1.00** | **1.00** | **1.00** | **1.000** |
| Baseline DAS (leaderboard) | n/a | n/a | 72 (all (layer, position)) | — | — | — | **1.000** |
| `causal-submission-non-linearity` (leaderboard) | n/a | n/a | — | — | — | — | 1.000 |
| DBM (leaderboard) | n/a | n/a | — | — | — | — | 0.989 |

## Per-site breakdown of the latest two runs

### V=4 multi-row — best site per split
- `answerPosition_test`: **L23, last_token** = 1.000
- `randomLetter_test`: many tied at 1.000 (identity-territory; pointer didn't change)
- `answerPosition_randomLetter_test`: **L23, last_token** = 0.833 — same winner as V=8

### V=8 mixed — best site per split
- `answerPosition_test`: **L23, last_token** = 1.000
- `randomLetter_test`: many tied
- `answerPosition_randomLetter_test`: **L23, last_token** = 0.833

The added V=8 layers (L4, L7, L13) didn't beat L23 on the hard split:
```
L23, last_token        0.833  ← winner
L13, correct_symbol    0.633
L 7, correct_symbol    0.567
L 8, correct_symbol    0.533
L 2, correct_symbol    0.267
L 4, correct_symbol    0.233
```

## Disambiguation result (resolved)

Trained DAS at `(L15, last_token)` and `(L20, last_token)` — both *missed* by PLOT. Both score **1.000** on `answerPosition_test`; **L15, last_token scores 1.000 on the hard split** (vs 0.833 for PLOT's L23 pick). L20 hard-split = 0.900.

Per-site IIA on hard split (`answerPosition_randomLetter_test`):
- **L15, last_token = 1.000** ← winner
- L20, last_token = 0.900
- L23, last_token = 0.833 (previous PLOT pick)
- All other (L15/L20, non-last_token) = 0.000

**Conclusion: site-selection is the bottleneck.** PLOT's Stage A/B with current OT rows + sq_l2 cost systematically picks L23 over the better L15. L23 is downstream — closer to the output but past the layer where the pointer representation is most causally complete.

## Next question — make PLOT pick L15

Now that we know the right site, the question is why Stage A's layer OT prefers L23 over L15. Likely candidates:
1. Effect-signature magnitude grows monotonically with layer (later layers dominate cost matrix). Try cosine cost or row-normalisation tweak.
2. The OT rows we use happen to peak at L23. Try alternate rows (e.g., `answer` directly, or per-pointer rows from a balanced subset).
3. Stage A picks per-row top-1; L15 may sit at row 2 in mass while L23 sits at row 1. Try top-2 / weighted union.
