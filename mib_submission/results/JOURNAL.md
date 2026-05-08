# PLOT MIB submission — methodological journal

Narrative log of methodological decisions, surprises, and what we tried.
Per-cell IIA scores live in [`RESULTS.md`](RESULTS.md); cell status lives
in [`CELLS.md`](CELLS.md). This file is the place for *why*, not *what*.

Append-only by date; oldest first.

---

## Cell 1 (`MCQA × Qwen-2.5-0.5B × answer_pointer`) — porting + tuning

The first ported cell. Several methodological turns before the final
mean IIA = 0.956:

- **Initial bug — flattened (N·K) signatures.** Squared-L2 cost over
  thousands of dims with magnitudes O(1) gave costs O(10³); `exp(−10³/ε)`
  underflows for any ε ≪ 100, producing uniform Sinkhorn plans. Fix:
  aggregate across examples to a single (K,) vector per OT row, mirroring
  `aggregate_mean` in the source PLOT branch.
- **Logit space → probability space.** Logits amplified noise; `softmax`
  bounded magnitudes and behaved well under L2 row-normalization.
- **L2-normalize abstract and neural rows before cost.** Brings sq_l2 cost
  into [0, 4] regardless of K, well-conditioned for any reasonable ε.
- **Multi-row Stage A.** The source's `_stage_a_timesteps` returns one
  layer *per OT row*. Earlier attempts collapsed all V rows into one
  layer, killing diversity. Fix: per-row top-k.
- **V=1 collapse on `(answer_pointer, answer)` rows.** Both interchanges
  produce identical observable letter changes in this dataset's fixed
  letter ordering. Balanced Sinkhorn with V=1 forces uniform plan
  regardless of cost. Fix: switch OT row variables to `choice_i` (probes
  pointer mechanism) or `symbol_i` (probes letter copy). Final config used
  V=4 `choice0..3` on the `answerPosition_randomLetter_train` split.

The final 0.044 gap to baseline DAS landed on `answerPosition_randomLetter_test`
(5/30 wrong at the picked `(L23, last_token)` site). Off-PLOT disambiguation
trained DAS at `(L15, last_token)` and `(L20, last_token)`; both scored
1.000 on the hard split, confirming the gap is **site selection**, not DAS
quality. PLOT's Stage A consistently prefers L23 over L15 with V=4
`choice` rows. Open question: tweak the cost / row schema to shift mass
from L23 to L15 without sacrificing the easy splits. See `PLOT_SHORTCOMINGS.md`
for structural reasons this is hard.

## Cell 2 (`MCQA × Qwen × answer`) — DAS-quality bottleneck, not site selection

PLOT picked `(L23, last_token)` correctly — full-replace IIA on the
calibration set was 1.000. The 0.169 gap to baseline DAS concentrated
on the random-letter splits. **Conclusion: the gap is DAS at
`n_features=16` not generalizing across the random-letter distribution
shift, not a site-selection failure.** Trying `n_features=32` /
`epochs=24` is the obvious next move but hasn't been done.

## Cells 3 & 4 (`MCQA × Gemma`) — ports

Both cells transferred cleanly with the same V=4 config from cell 1.

- Cell 3 (`answer_pointer`) ties cell 1 essentially exactly (0.955 vs 0.956).
- Cell 4 (`answer`) **beats cell 2 by +0.107** (0.908 vs 0.801). Gemma's
  `answer` representation concentrates more cleanly on `correct_symbol`
  positions than Qwen's, generalising better at the same `n_features=16`.
- Picked sites differ across models: Qwen `answer_pointer` lands on
  L23/last_token; Gemma `answer_pointer` lands on L17/last_token. Both
  are the deepest residual-stream position before unembedding.

## Cell 4 — wall-clock anomaly (May 8, 2026)

Cell 4 ran at ~50 s/batch for the entire first DAS site (78 min for 12
epochs), then 12–30× faster for sites 2–4. I initially hypothesised a
structural difference between `answer` and `answer_pointer` in the
training-data construction; D1 (read pyvene's training loop source code)
and D2 (probe the labeled counterfactual datasets) ruled this out — both
variables produce byte-identical training tensor shapes.

The actual cause: laptop was on **battery** during site 1, drawing 22 W
of the 4060 Laptop's 125 W envelope. Plugging in mid-run + setting Windows
"Best Performance" + NVIDIA Control Panel "Prefer maximum performance"
restored expected speed. **Sites 2–4 trained on AC at ~2 s/batch.**

Lesson: GPU power state can produce 30× slowdowns that don't correlate
with model state. Sanity-check `nvidia-smi power.draw` against
`power.max_limit` early in any long run.

## Eval CLI (`--no-private_data`)

CLAUDE.md and EVAL_LOG.md (the predecessor of this file) referenced
`evaluate_submission.py --no-private_data --public_data`. That flag does
not exist in the pinned MIB harness commit — `--private_data` is
`store_true, default=True` and cannot be disabled from the CLI. The
runtime fix is to invoke `evaluate_submission_task(..., private_data=False,
public_data=True)` directly from a small wrapper. CLAUDE.md still has
the stale reference and should be updated.

## ARC preset bug (May 8, 2026)

Original `_mcqa_arc_v4_choices` used OT rows `("choice0..3")`. **ARC's
causal model has no `choice` variables** — its prompts are science
questions, not the MCQA color/object format. `causal_model.run_interchange`
would raise KeyError on first signature collection. Replaced with
`("symbol0..3")`; CPU-probed before any GPU run — full-rank (4/4)
abstract table, 0% skip rate, clear per-row letter concentration
(row 0 → A, row 1 → B, ...).

## RAVEL — preset blocked

The natural V=3 schema (`Country`, `Continent`, `Language`) hits two
structural blockers in the existing `features.py`:

1. **Token-vs-letter signature space.** PLOT's signature is a one-hot
   delta over a `letters` alphabet. RAVEL answers are word tokens
   ("Bulgaria", "France") — sometimes multi-token, never letters.
2. **Per-base attribute selection.** RAVEL's `answer` mechanism reads
   only the attribute matching `queried_attribute`. So row `Continent`
   on a base where `queried_attribute=Country` is a no-op. On a mixed
   split, at most 1 of 3 rows has signal per base.

Plan to land both extensions before any RAVEL cell run: in the per-cell
`features.py` extension proposal (chat history; not yet committed).

## Storage of results (May 8, 2026)

Earlier setup mixed status, IIA tables, methodology, and per-cell
narrative in `EVAL_LOG.md`. Restructured into:

- `RESULTS.md` — research-style results document, generated by
  `_aggregate.py` from the per-cell JSON archives. Don't edit by hand.
- `CELLS.md` — pure status tracker, manual.
- `JOURNAL.md` (this file) — methodological narrative, manual.
- Per-cell `*.json` — raw eval output, immutable.

`_aggregate.py` falls back to a heuristic when a submission folder is
absent (e.g. cells 1–2, shipped before the current local environment
existed): trained sites are the (layer, position) units whose IIA on the
informative test split exceeds 0.3. This recovered 4 picked sites for
both Qwen cells. Inferred picks are flagged with † in the headline table.

## RAVEL extension (May 8, 2026)

Initial RAVEL config was a placeholder that raised NotImplementedError.
Deep investigation revealed several wrinkles that needed real engineering
before any RAVEL cell could run.

**Causal model + dataset reality check:**

- 3122 cities (not 50 as initially assumed) with 6 attributes; the causal
  model uses 3 (Continent, Country, Language) but `causal_model.values["answer"]`
  is the union of *all* 6 attributes' values — 928 distinct strings.
- HF dataset rows carry 3 counterfactuals per base: `prompt_template_*`,
  `attribute_*`, `wikipedia_*`. The baseline uses only `attribute_*` +
  `wikipedia_*`; `prompt_template_*` swaps the template format keeping the
  attribute, and isn't useful for site-level localization.
- Each base specifies a `queried_attribute`. The `answer` mechanism
  selects only the matching attribute's value: patching any other
  attribute is a no-op for that base. So per-row interchange has signal
  only on bases where `queried_attribute == row_variable`.

**Tokenization analysis (Gemma):**

- 928 attribute strings collapse to **271 unique first tokens** under
  Gemma's tokenizer (71% collision). 28 within-attribute collisions are
  the harmful kind: `"United States"`, `"United Kingdom"`, `"United Arab
  Emirates"` all share `" United"`; 7 different `"French..."` languages
  share `" French"`; 17 different `"Arabic..."` languages share
  `" Arabic"`.
- 110 of 174 Language values are multi-token (most because they're
  comma-separated alternative-answer lists like `"English,Gaeli,Kymri"`).
- The harness's checker accepts any of the comma-separated alternatives,
  so the LM only needs to output one of them. Our PLOT signature uses the
  first-token of the *whole comma-list string* — accepting some noise on
  Language for now.

**Code changes:**

- New `mib_submission/plot/_alphabets.py` with `LabelAlphabet`: encapsulates
  label → signature-dim mapping + LM-vocab token IDs. Three constructors:
  `from_letters` (legacy MCQA/ARC), `from_labels` (general), and
  `from_causal_model_answers` (reads `values["answer"]`). Token resolution
  is lazy and compacts collisions to fewer dims.
- Extended `features.py` with `alphabet` kwarg (alongside legacy `letters`),
  `per_row_dataset_filter` for per-OT-row dataset filtering, and
  `on_unknown_label="skip"` for tolerating LM outputs not pre-registered
  in the alphabet.
- Extended `PlotConfig` with `answer_strings`, `answer_alphabet_from_causal_model`,
  `per_row_filter_attribute`, `on_unknown_label`. Existing MCQA/ARC configs
  unchanged (defaults preserve old behaviour).
- New RAVEL preset uses `per_row_filter_attribute="queried_attribute"` so
  each OT row collects neural signatures only on bases where
  `queried_attribute == row_variable`. With per-row filter, `select_sites_via_plot`
  runs `collect_neural_outputs` V times instead of once — 3× the GPU work
  for V=3, but eliminates the 67% no-op-base SNR drag.
- `setup_residual_experiment` gained `max_new_tokens` parameter (default 1
  for MCQA/ARC, 2 for RAVEL). Custom checker now propagates from `RunConfig`
  through `run.py` → `setup_residual_experiment`.
- RAVEL preset uses `n_features=288`, `training_epochs=1`, `max_new_tokens=2`,
  and `_ravel_checker` (ported from `baselines/ravel_baselines.py:checker`)
  to handle multi-word answers and comma-list alternatives.

**CPU smoke test results (real RAVEL data, 200 examples):**

- Alphabet: 928 labels → 271 dims (token collisions).
- Per-row filter on `attribute_train`: 51 / 63 / 86 examples for
  Country / Continent / Language (matches manual probe; 25–43% of bases
  per row).
- Abstract table: shape `(3, 271)`, full rank 3/3, per-row L2 norm = 1.0.
- Top dims per row are interpretable (Country row → country tokens,
  modulo first-token collisions).

**What's NOT yet validated:**

- End-to-end GPU run (deferred until cell 7 finishes).
- Whether `n_features=288` actually trains within 8 GB VRAM (a 26-layer ×
  ~6-site simultaneous DAS at this size may OOM on the laptop).
- Whether the harness's `evaluate_submission_task` correctly scores RAVEL
  with `n_features=288` featurizers.
- The first-token collision impact on Language cell IIA — Language has
  the most multi-token / comma-list answers so its signature alphabet is
  the noisiest.

The RAVEL config now produces a non-degenerate abstract table on real
data, all 28 new unit tests pass, and the existing 54 tests still pass.
First GPU run is the natural next step but should wait until after cell 7
+ cell 8 to avoid VRAM contention.

## Cell 7 (`ARC × Gemma × answer_pointer`) — first ARC ship at 0.884

ARC's first cell on the new preset (`_arc_v4_symbols`). Required two
operational tweaks:

- **OOM at first launch.** Cell 7 started with the default
  `--train-batch-size 32` and OOM'd at site 1 / epoch 0 of DAS. ARC's
  filtered train pool (~430 examples vs MCQA's ~256) yields 14
  batches/epoch × 6 picked sites jointly trained → exceeded 8 GB VRAM.
  Fix: bypass restart with `--bypass-sites` (skipping ~80 min of redundant
  signature collection) + `--train-batch-size 16`. Recovered cleanly.
- Picked sites cluster late: Stage A → `[16, 17, 22, 25]`; Stage B picked
  `top_k=1` (so 6 sites total, mostly `last_token`). Hard split
  (`answerPosition_randomLetter_test`) = 0.667, 0.20 below MCQA Gemma's
  0.865 on the analogous split. Plausible reasons for the gap:
  - ARC has only 2 token positions per layer (vs MCQA's 3) — less site
    diversity for Stage B.
  - ARC's `symbol_i` OT rows have ~25% informative-example density
    (vs MCQA's `choice_i` ~75%) → noisier site selection.
  - ARC's science-question prompts stress the pointer mechanism more
    than MCQA's color-matching prompts.

## Cell 8 (`ARC × Gemma × answer`) — better than expected at 0.923

Same ARC pipeline, target variable `answer`. Stage A picked
`[17, 22, 24, 25]`, Stage B chose `top_k=2` (both positions per
layer-row), giving **7 sites trained**. Final mean IIA = 0.923 — beats
cell 7 (0.884) and *almost* matches cell 4 (MCQA × Gemma × answer = 0.908).

Three observations worth tracking forward.

### 1. PLOT picked correctly *and* incorrectly at the same time

Per-site IIA at the picked sites:

| site | aP | rL | aPrL | trained well? |
|---|---|---|---|---|
| **L22/correct_symbol** | 0.609 | **0.890** | **0.881** | ✓ (the headline winner) |
| L22/last_token | 0.998 | 0.800 | 0.749 | ✓ |
| L24/last_token | 0.996 | 0.771 | 0.732 | ✓ |
| L25/last_token | 0.998 | 0.671 | 0.625 | ✓ |
| L17/last_token | 0.971 | 0.326 | 0.339 | ✓ (good on easy only) |
| L17/correct_symbol | 0.176 | 0.147 | 0.093 | ✗ (DAS didn't converge) |
| L24/correct_symbol | 0.056 | 0.306 | 0.305 | ✗ |
| L25/correct_symbol | 0.000 | 0.048 | 0.058 | ✗ (loss *climbed* during training) |

All 4 `last_token` sites converged; only 1 of 4 `correct_symbol` sites did.

### 2. Why the failures — diagnosis

Two coupled hypotheses:

- **Stage B's `top_k_grid=(1, 2)` degeneracy on 2-position tasks.** ARC has
  only 2 token positions per layer. `top_k=2` lets Stage B pick *both*
  positions whenever the sum-of-per-site-IIA calibration score is
  non-decreasing. Adding a marginally-positive site never lowers the sum,
  so calibration prefers the larger pick. Effectively, Stage B picked
  layers via Stage A and took whatever existed at each layer — site
  selection wasn't actually *selecting* among token positions.
- **`correct_symbol` doesn't carry the answer in late ARC layers.** The
  answer letter is generated at `last_token` (position right before
  next-token output). At `correct_symbol` (position of the correct letter
  in the prompt), late-layer residual streams have moved on to
  downstream-context processing. Patching there at L17/L24/L25 doesn't
  propagate to the output. DAS at those sites can't find a useful
  rotation — the optimizer fits training noise (site 6's loss climbed
  from 5.95 → 6.55 → 8.7 mid-training, classic noise-fitting signature).

Together: Stage B's loose top_k forces non-causal sites into Stage C,
and DAS at non-causal sites genuinely can't be rescued by a 16-dim
rotation. Best-per-split scoring saves the headline IIA, but the wasted
~40% DAS time was real.

### 3. Mitigations to try

- **`stage_b_top_k_grid=(1,)` for ARC**: forces real Stage B selection.
  Should drop trained sites from 7 → ~4 and save ~40% wall-clock with
  little IIA risk (the failed sites contributed 0 to best-per-split).
- **Stricter calibration filter**: require per-site Stage-B IIA > 0.3
  before shipping a site. Cleans up submissions; orthogonal to the
  underlying IIA.
- **Layer-aware Stage B**: at late layers, prefer `last_token` over
  `correct_symbol` by default. Task-specific prior, but well-justified
  by the cell-8 evidence.

## Methodological gap — seed effects not measured

Every shipped cell so far is a *single seed run*. DAS training is
stochastic: orthogonal init of the rotation matrix, DataLoader shuffle,
and (potentially) CUDA non-determinism in matmul backward all introduce
run-to-run variance. We have no estimate of that variance.

The differences between shipped cells need to be read with this caveat:

- Cell 4 (Gemma MCQA answer) = 0.908 vs cell 8 (Gemma ARC answer) = 0.923.
  A 0.015 gap could plausibly be within seed noise.
- Cell 1 (Qwen MCQA pointer) = 0.956 vs cell 3 (Gemma MCQA pointer) = 0.955.
  Within 0.001 — either a remarkable cross-model match or coincidence
  inside the noise floor.
- Cell 7 (ARC pointer) = 0.884 vs cell 8 (ARC answer) = 0.923. 0.04 gap;
  big enough to be a real effect *if* seed variance is small.

**To validate any cell's IIA as a "result," we should run it 3 times with
distinct seeds and report mean ± std.** Not done so far. This is the
single biggest methodological gap in the current results.

`PlotConfig` doesn't currently have a seed field; DAS uses whatever torch
default RNG state exists at the time of training. Adding `--seed` to the
CLI + threading through to `train_interventions` is straightforward but
hasn't been done.

## Cell 22 (RAVEL × Gemma × Continent, smoke) — first end-to-end RAVEL run

Smoke-test settings: `n_features=64`, `dataset_size=128`, `epochs=1`,
`train_batch_size=16`. Goal: validate the new alphabet + per-row-filter
machinery on a real GPU without committing 3+ hours to a competitive run.
Result: **mean IIA = 0.845** — better than expected for a tiny config.

### Bug caught: alphabet dim mismatch (token resolution)

First launch crashed at Stage A's `cost_matrix` call:
`ValueError: feature dim mismatch: A has 928, S has 271`.

Root cause: `from_causal_model_answers(cm)` builds a 928-label
`LabelAlphabet` (one dim per distinct answer string in
`causal_model.values["answer"]`). When the alphabet is later resolved
via `resolve_tokens(alpha, tokenizer)`, Gemma's tokenizer compacts
collisions ("United States"/"United Kingdom"/"United Arab Emirates" all
share the leading-space `" United"` token, etc.) → **271 unique first
tokens**.

The bug: `_resolve_config_alphabet` returned the unresolved alphabet
(928 dims). `build_abstract_table` ran with that, producing a (V, 928)
table. `collect_neural_outputs` lazily resolved tokens internally, then
sliced softmax to (N, 271). When Stage A's `cost_matrix` tried to match
a (V, 928) abstract table against a (V, 271) neural table, dim mismatch.

Fix: resolve tokens **eagerly** in `_resolve_config_alphabet` when a
real tokenizer is available (left lazy when stub bundles have no
tokenizer, so unit tests that mock `collect_neural_outputs` still
work). Added regression tests
(`test_alphabets_and_ravel.py::TokenResolutionEagerInPipeline`)
covering both paths.

### Stage A picked early-mid layers: [L6, L18]

A surprise. ARC and MCQA on Gemma both pick late layers (L7–L25);
RAVEL × Continent picked layers L6 and L18. Stage A pre-DAS IIA = 0.319;
Stage B pre-DAS IIA = **0.889** (highest of any cell so far).

The layer profile makes sense post-hoc: RAVEL prompts have varied
templates (`"city to country: Paris is in France. Varna is in"`,
`'[{"city": "Kuala Lumpur", "country": "Malaysia"}, ...]'`,
`"What is the continent of Varna?"`). The LM "looks up" the entity
early — the residual stream right after the entity name (token
position `entity_last_token`) carries continent/country/language info
regardless of which template is being processed. Localizing there is
template-invariant; localizing at `last_token` would be
template-dependent.

### `entity_last_token` is the standout site

| site | attribute_test | prompt_template_test | wikipedia_test |
|---|---|---|---|
| L6/last_token | 0.561 | 0.784 | **0.027** |
| L6/entity_last_token | 0.851 | 0.837 | 0.821 |
| L18/last_token | 0.005 | 0.329 | **0.000** |
| **L18/entity_last_token** | **0.851** | **0.853** | **0.832** |

Both `entity_last_token` sites score ≥0.83 on all three splits.
Both `last_token` sites collapse to ~0 on `wikipedia_test` because the
wikipedia answer is `""` — there's no continent token at the end of
the prompt to match.

### What this validates

- Alphabet construction (928 → 271 dims with collision compaction) works.
- Per-row dataset filter (`queried_attribute=Country|Continent|Language`)
  works — abstract and neural rows both restrict cleanly.
- Multi-token labels work via `max_new_tokens=2`.
- Custom checker (`_ravel_checker`) handles multi-word + comma-list
  answers — filter retention rates were healthy (50–75% per split).
- The whole machinery runs end-to-end on real GPU at smoke settings in
  ~10 min. Full-config (`n_features=288 dataset_size=256 epochs=1`)
  should land 0.90+ but takes longer (~3-4 h projected).

### Open

- Run RAVEL × Continent at full settings to land a competitive submission.
- Run RAVEL × Country (160 values, 28 collisions) and RAVEL × Language
  (174 values, 110 multi-token, heavy collisions). Country should
  resemble Continent; Language is the riskiest cell of the three.
- The per-row filter pays a 3× signature-collection cost. For RAVEL
  this is ~3 min total at smoke scale. At full scale it's ~15-20 min,
  acceptable.

## Sessions of record

- Cells 1, 2 shipped pre-session (Qwen MCQA × {pointer, answer}).
- Session 2026-05-07: cells 3, 4 shipped (Gemma MCQA × {pointer, answer}).
  Most of the environment setup (Python 3.12, MIB harness, .venv-mib,
  HF token, NVIDIA driver upgrade) happened here.
- Session 2026-05-08: cells 7, 8 shipped (Gemma ARC); RAVEL extension
  landed (alphabet, per-row filter, custom checker, max_new_tokens);
  cell 22 smoke shipped at 0.845. RESULTS.md generator (`_aggregate.py`)
  written. Test count grew 26 → 84.
