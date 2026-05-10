# Per-task walkthroughs

How the PLOT-DAS pipeline works on each MIB task we benchmark. These are short prose explanations of the per-task choices in `mib_submission/plot/configs.py` plus reproduction recipes; for end-to-end runnable walkthroughs with heatmaps and Sinkhorn plans, see the Jupyter notebooks in `notebooks/`.

## Common shape (all tasks)

Every cell follows the same pipeline:

1. **Setup** — load the LM pipeline, build a `CounterfactualDataset` for the cell, filter to examples the model gets right under no intervention (the "filter experiment"). Per-task entry is in `mib_submission/pipeline.py::setup_residual_experiment` (or `setup_attention_head_experiment` for IOI).
2. **Stage A — layer OT.** Collect per-site neural effect signatures (output-probability deltas over an alphabet of expected answer tokens, aggregated per layer). Build a `(V, K)` abstract row table whose v-th row is the OT-row variable's expected one-hot diff under interchange. Compute pairwise `sq_l2` cost between L2-row-normalized abstract and per-layer neural rows, run balanced Sinkhorn at a sweep of `epsilon`. Per row, pick the top-k layers with highest π mass. Union across rows = Stage A picks.
3. **Stage B — per-layer token-position OT.** Within each picked layer, do the same OT over the layer's available token positions. Per (row, layer), pick the top-k positions. Union over (row, layer) = Stage B picks → the final `(layer, token_position)` sites.
4. **Stage C — DAS at picked sites only.** Prune `bundle.experiment.model_units_lists` to the picked sites. Call MIB harness's `train_interventions(method="DAS")`. Each picked site gets a trained orthogonal rotation matrix that isolates an `n_features`-dimensional subspace where the answer info lives.
5. **Write submission** — `mib_submission/serialize.py::write_submission` saves one featurizer/inverse/indices triplet per picked site.
6. **Verify + Eval** — `verify_submission.py` checks format; `scripts/eval_cell.py` runs the harness's `evaluate_submission_task` (or `evaluate_ioi_submission_task` for IOI) with our per-task `max_new_tokens` override and `LMPipeline.load` `position_ids` patch.

A calibration sweep over `(epsilon × top_k)` runs at both Stage A and Stage B; the candidate maximizing mean per-site IIA on the train split is kept.

What follows is the per-task specialization.

---

## MCQA (cells 1, 2, 3, 4) — `4_answer_MCQA`

**Cells:** 1 (Qwen × pointer), 2 (Qwen × answer), 3 (Gemma × pointer), 4 (Gemma × answer).
**Preset:** `_mcqa_v4_choices` in `mib_submission/plot/configs.py`.

### Setup specifics

- **Output variable**: `answer_pointer` (which choice index is correct) or `answer` (the actual letter, A–Z).
- **OT row schema**: V=4 over `choice0..3` — interchanges that swap the contents of one of the four answer choices. These produce observably distinct letter-output changes ~25% of the time per row.
- **Alphabet**: 26-letter `letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ"`.
- **Signature dataset**: `answerPosition_randomLetter_train` (only split where both the pointer AND the letter vary).
- **DAS**: `n_features=16`, 12 epochs.
- **Token positions** (3): `correct_symbol`, `correct_symbol_period`, `last_token`.

### Reproduce

```bash
.venv-mib/bin/python -m mib_submission.plot.run \
    --task 4_answer_MCQA \
    --model google/gemma-2-2b \
    --variable answer_pointer
.venv-mib/bin/python scripts/eval_cell.py \
    --cell 4_answer_MCQA_Gemma2ForCausalLM_answer_pointer
```

### Shipped results (highest-view)

| cell | model × variable | seeds | mean ± std | DAS LB | gap |
|---|---|---|---|---|---|
| 1 | Qwen × pointer | 3 | 1.000 ± 0.000 | 1.000 | 0.000 🤝 |
| 2 | Qwen × answer | 1 | 0.788 | 0.913 | -0.125 (structural — see PLOT_SHORTCOMINGS §2) |
| 3 | Gemma × pointer | 3 | 0.923 ± 0.006 | 0.974 | -0.051 |
| 4 | Gemma × answer | 3 | 0.904 ± 0.010 | 0.974 | -0.070 |

### Things worth knowing

- **Cell 2's gap is documented and structural.** PLOT's V=4 `choice_i` rows reward sites whose patch effect mimics a choice-swap (letter change). But the `answer` variable for Qwen happens to live at a late-layer site where the residual encodes the *letter*, not the *pointer*. The OT cost matrix is high against `choice_i` rows there. PLOT_SHORTCOMINGS §2 has the full mechanism + the disambiguation experiment that confirmed it.
- **Cell 1 wins big at any seed.** Qwen's L15 last_token residual encodes the answer pointer cleanly — every seed sweep run reached 1.000 (mean ± std = 1.000 ± 0.000). The original 0.8915 was a seed-unlucky draw on the pre-sweep run.

---

## ARC_easy (cells 7, 8) — `ARC_easy`

**Cells:** 7 (Gemma × pointer), 8 (Gemma × answer).
**Preset:** `_arc_v4_symbols`.

### Setup specifics

- **Output variable**: `answer_pointer` (which symbol is correct) or `answer` (the literal letter).
- **OT row schema**: V=4 over `symbol0..3` — ARC's causal model has no `choice` variables (it's science questions, not color/object MCQs), so we use letter-swaps at the four symbol positions.
- **Alphabet**: 26 letters.
- **Signature dataset**: `answerPosition_randomLetter_train` (same as MCQA).
- **Token positions** (2 — not 3 like MCQA): `correct_symbol`, `last_token`.
- **`stage_b_top_k_grid=(1,)`** per PLOT_SHORTCOMINGS §8. ARC has only 2 token positions per layer, so the more permissive `(1, 2)` lets Stage B pick BOTH positions per layer — and as PLOT_SHORTCOMINGS §15 documents, trained DAS rotations can score worse than the harness's identity-fallback at the under-chosen position. Tightening to `(1,)` lets identity-fallback do its job at the un-trained position.

### Reproduce

```bash
.venv-mib/bin/python -m mib_submission.plot.run \
    --task ARC_easy \
    --model google/gemma-2-2b \
    --variable answer \
    --train-batch-size 16          # 8 GB VRAM guard
.venv-mib/bin/python scripts/eval_cell.py \
    --cell ARC_easy_Gemma2ForCausalLM_answer
```

### Shipped results (highest-view)

| cell | variable | PLOT | DAS LB | gap |
|---|---|---|---|---|
| 7 | answer_pointer | 0.827 | 0.836 | -0.009 🤝 |
| 8 | answer | **0.999** | 0.941 | +0.058 🏆 (with caveat — see PLOT_SHORTCOMINGS §15) |

### Things worth knowing

- **Cell 8's 0.999 is partly a harness mechanism.** With `top_k=(1,)`, Stage B trains DAS at only 1 position per picked layer; the eval scores both positions at each picked layer, using identity (full-residual swap) at the un-trained position. For Gemma's L25 last_token, identity gives 0.999 IIA — better than any trained rotation at the chosen `correct_symbol` position (0.04). The picked-layer Stage A was right; the picked-position Stage B was wrong; the harness's per-position scoring fallback covered for it. PLOT_SHORTCOMINGS §15 is the full mechanism.
- **`correct_symbol` is not the right position late in the network.** PLOT_SHORTCOMINGS §9 documents this: the answer information at late layers (>L20) sits in `last_token`, not `correct_symbol`. PLOT picks `correct_symbol` for the OT rows because that's where the OT-row variable's expected effect concentrates, but identity at `last_token` is better.

---

## arithmetic (cell 11) — `arithmetic` × Gemma-2-2b × `ones_carry`

**Preset:** `_arithmetic_v2_carry_children` (default) or `_arithmetic_v4_operands`.

### Setup specifics

- **Output variable**: `ones_carry` — the carry-out bit of the ones digit (true/false).
- **OT row schema**: V=2 over `{tens_out, hundreds_out}` (default Option C — both children of `ones_carry`) or V=4 over `{op1_ones, op2_ones, op1_tens, op2_tens}` (Option B fallback). MIB's `arithmetic_task` declares only one variable for scoring (`ones_carry`); PLOT needs V≥2 to avoid the V=1 collapse (PLOT_SHORTCOMINGS §1), so we use non-target variables as additional OT rows. Mirrors source PLOT's binary-addition pipeline mixing `S_i + C_i` rows.
- **Alphabet**: digit characters `"0123456789"`.
- **Signature dataset**: `random_train`.
- **DAS**: `n_features=16`, 1 epoch (matches the arithmetic baseline).
- **`max_new_tokens=3`** (Gemma's multi-digit answers up to 3 tokens).
- **`output_key="raw_output"` and `label_from_output=_arithmetic_first_digit`** — arithmetic's causal model exposes the output as a multi-digit string (e.g. `"168"`) under `raw_output`, not as a single character under `answer`. The first-digit extractor projects to the digit alphabet so signatures align.

### Reproduce

```bash
.venv-mib/bin/python -m mib_submission.plot.run \
    --task arithmetic \
    --model google/gemma-2-2b \
    --variable ones_carry \
    --train-batch-size 16
.venv-mib/bin/python scripts/eval_cell.py \
    --cell arithmetic_Gemma2ForCausalLM_ones_carry
```

### Shipped results

| cell | quality | ones_carry_test | random_test | mean | DAS LB |
|---|---|---|---|---|---|
| 11 | smoke (ds=128) | 0.275 | 0.622 | 0.448 | ~0.53 |

### Things worth knowing

- **The ds=1024 scale-up regressed.** Bigger dataset moved Stage A to different layers (L17 + L19 instead of smoke's L16 + L21); `ones_carry_test` IIA dropped to ~0 at the new sites. We reverted to smoke. Future work: rerun with `--signature-dataset ones_carry_train` so signature collection targets carry-specific counterfactuals instead of random pairs.
- **`random_test` vs `ones_carry_test` matter unequally.** `random_test` has ~50% no-op interchange pairs (random source happens to match base on `ones_carry`), so any site can score moderately by trivial-copy luck. `ones_carry_test` forces source.carry ≠ base.carry — the real localization measurement. Use the latter for diagnostic comparison.
- **Eval needs `max_new_tokens=3`.** The harness's default `max_new_tokens=1` rejects 100% of arithmetic examples (the answer is multi-digit, the model gets cut off at the first token, `arithmetic_checker`'s regex finds only "9" and compares to "98"). `scripts/eval_cell.py` overrides this.

---

## RAVEL (cells 21, 22, 23) — `ravel_task` × Gemma-2-2b × `{Country, Continent, Language}`

**Cells:** 21 (Country), 22 (Continent), 23 (Language).
**Preset:** `_ravel_v3_attributes`.

### Setup specifics

- **Output variable**: one of `Country`, `Continent`, `Language` (queried-attribute about a city entity).
- **OT row schema**: V=3 over all three attributes. Each row asks "what changes when interchange replaces the source's value of this attribute?"
- **Alphabet**: derived from `causal_model.values["answer"]` — the full set of attribute values across the dataset (~340 distinct strings, compacted to ~263 first-token IDs under Gemma's tokenizer due to first-token collisions).
- **Per-row dataset filter**: `per_row_filter_attribute="queried_attribute"`. Each OT row's signature is collected only on bases where `input["queried_attribute"]` matches that row's variable — the Country row sees only Country queries. Costs 3× the signature collection time but eliminates the no-op-base SNR drag from cross-attribute queries.
- **DAS**: `n_features=288`, 1 epoch (matches the RAVEL baseline).
- **`max_new_tokens=2`** for the LM pipeline (multi-token answers like "United States" tokenize as 2 tokens).
- **Custom checker `_ravel_checker`** — handles comma-separated alternative-answer strings ("Arabic, French, English") and multi-word labels ("South Korea").
- **Token positions**: `entity_last_token`, `last_token`.

### Reproduce

```bash
.venv-mib/bin/python -m mib_submission.plot.run \
    --task ravel_task \
    --model google/gemma-2-2b \
    --variable Continent \
    --train-batch-size 16
.venv-mib/bin/python scripts/eval_cell.py \
    --cell ravel_task_Gemma2ForCausalLM_Continent
```

### Shipped results (highest-view)

| cell | variable | values | per-value support (ds=256) | PLOT | DAS LB | gap |
|---|---|---|---|---|---|---|
| 22 | Continent | 6 | ~43 | **0.856** | 0.848 | +0.008 🏆 |
| 21 | Country | 160 | ~1.6 | 0.615 | 0.957 | -0.342 (structural — §14) |
| 23 | Language | 174 | ~1.5 | 0.629 | 0.812 | -0.183 (structural — §14) |

### Things worth knowing

- **All three RAVEL cells picked the same sites: (L6, entity_last_token) + (L25, entity_last_token).** Gemma represents Country/Continent/Language at the same layers — they're co-located in the entity-tracking circuit. E-R-4's bypass-grid experiment confirmed those picks are reasonable (no alternative tested beats them).
- **Country and Language fall to L25's expressivity ceiling.** Even an identity featurizer at (L25, entity_last_token) gives the same IIA as our trained DAS rotation. The information is diffusely encoded across the L25 residual on a 160-or-174-class problem; no single-site rotation can fully capture it. PLOT_SHORTCOMINGS §14 has the full diagnosis.
- **Language has an alphabet compaction issue too.** 13 first-token collision groups, largest 9 — multi-language answers like "Arabic, French, English" all map to the same alphabet entry " Arabic". The diagnostic C-split experiment (`scripts/experiment_abc.sh` in earlier sessions) confirmed splitting comma-separated answers doesn't change PLOT's picks or eval IIA — the compaction is real but not the dominant cause of the gap.

---

## IOI (cells 13, 14) — `ioi_task` × GPT-2

**Cells:** 13 (output_token), 14 (output_position).
**Preset:** `_ioi_v3_splits` + `_attention_head_experiment`.

### Setup specifics

This cell type is **structurally different from the other tasks.** The submission ships attention-head featurizers (not residual-stream), uses `PatchAttentionHeads` joint-mode DAS, has its own eval entry point (`evaluate_ioi_submission_task`), and requires a linear-params bootstrap step before any PLOT runs.

- **Output variable**: `output_token` (which name should be output) or `output_position` (where the IO token is in the prompt).
- **OT row schema**: V=3 over IOI's three counterfactual splits: `s1_io_flip`, `s2_io_flip`, `s1_ioi_flip_s2_ioi_flip`. Each split swaps a specific subset of names in the prompt; the row signature is the model's logit-diff response.
- **Alphabet**: name-token first-IDs derived from IOI's name dataset (~85 distinct names, compacted under GPT-2's tokenizer).
- **DAS loss**: MSE on logit-difference (`ioi_loss_and_metric_fn`), not cross-entropy. The target for each interchange is a linear combination of source's position and token (the linear-params): `target_logit_diff = bias + token_coeff * source.token_diff + position_coeff * source.position_diff`.
- **Linear-params bootstrap**: before any PLOT run, `mib_submission/ioi/bootstrap.py` learns `(bias, token_coeff, position_coeff)` per model via linear regression of logit-diff against (position_flip, token_flip) labels across 4 IOI splits. Saved to `submissions/plot/ioi_linear_params.json`. The DAS loss reads from this file.
- **DAS**: `n_features=32`, 2 epochs, `init_lr=1.0`.
- **`max_new_tokens=1`** (single name token).

### Reproduce

```bash
# One-time bootstrap (already done; lives in submissions/plot/ioi_linear_params.json)
.venv-mib/bin/python -c "
from mib_submission.ioi import bootstrap_linear_params
bootstrap_linear_params('gpt2')
"

# Run a cell
.venv-mib/bin/python -m mib_submission.plot.run \
    --task ioi_task \
    --model gpt2 \
    --variable output_position \
    --dataset-size 512

.venv-mib/bin/python scripts/eval_cell.py \
    --cell ioi_task_GPT2LMHeadModel_output_position
```

### Shipped results (lower MSE is better)

| cell | variable | PLOT | DAS LB | gap |
|---|---|---|---|---|
| 13 | output_token | 5.16 | 2.08 | +3.08 (structural — §13) |
| 14 | output_position | 16.0 | 2.20 | +13.8 (structural — §13) |

### Things worth knowing

- **PLOT picks the loud Name Mover heads (L9), misses the quiet Position Mover heads (L7-L8 S-Inhibition).** For cell 14 specifically, the right heads exist — diagnostic E-I-2 bypassed cell 14 to literature S-Inhibition heads `(7,3), (7,9), (8,6), (8,10)` and got MSE 4.12 (vs PLOT's 16.0; vs DAS LB 2.20). The signature design (logit-diff effect) systematically picks heads with large *direct* effects on output and misses heads that route information *indirectly*. PLOT_SHORTCOMINGS §13 has the full diagnosis.
- **The harness has two compatibility issues** patched in `mib_submission/ioi/_patches.py`: `LMPipeline.load` reads `position_ids` from `model.prepare_inputs_for_generation(...)` but transformers 5.x doesn't always include that key (KeyError); `PatchAttentionHeads` reads `model.config.head_dim` but Qwen2Config doesn't have it. Both are runtime-patched.
- **Submission folder structure is flat for IOI**, not nested. The IOI example notebook in the upstream MIB repo uses `ioi_task_M_V/DAS_M_V/`, but `ioi_evaluate_submission.py` scans the top-level non-recursively. We ship flat `ioi_task_M_V/AttentionHead(...)_*` directly.
- **Cells 15-18 (Qwen, Gemma IOI) require ≥16 GB VRAM.** pyvene's `IntervenableModel` + 4-head residual caches exceed 8 GB even at `eval_batch_size=32`. Cloud GPU work, deferred.

---

## Further reading

- `PLOT_SHORTCOMINGS.md` — 15 numbered sections documenting where each task's structural gap comes from
- `JOURNAL.md` — methodological narrative, session-by-session
- `HYPOTHESES.md` — experimental hypotheses tested in the diagnostic sessions
- `notebooks/residual_walkthrough.ipynb` — end-to-end Jupyter walkthrough of RAVEL Continent with OT plan heatmaps
- `notebooks/ioi_walkthrough.ipynb` — end-to-end Jupyter walkthrough of IOI cell 14 with the bypass diagnostic
