# PLOT MIB submission — Results

Per-cell IIA on the public MIB test splits, with picked sites and site-level breakdowns. Numbers in this document are computed from raw eval JSON archives by `_aggregate.py`; do not edit by hand.

**Status: 10 / 26 cells shipped.**

## Headline


#### 4_answer_MCQA
| model | variable | sites | aP | rL | aPrL | **mean IIA** |
|---|---|---|---|---|---|---|
| Gemma2ForCausalLM | answer | 4 | 1.000 | 0.886 | 0.838 | **0.908** |
| Gemma2ForCausalLM | answer_pointer | 4 | 1.000 | 1.000 | 0.865 | **0.955** |
| Qwen2ForCausalLM | answer | 4† | 1.000 | 0.769 | 0.633 | **0.801** |
| Qwen2ForCausalLM | answer_pointer | 4† | 1.000 | 1.000 | 0.867 | **0.956** |

#### ARC_easy
| model | variable | sites | aP | rL | aPrL | **mean IIA** |
|---|---|---|---|---|---|---|
| Gemma2ForCausalLM | answer | 4 | 1.000 | 0.998 | 0.998 | **0.999** |
| Gemma2ForCausalLM | answer_pointer | 6 | 0.987 | 1.000 | 0.667 | **0.884** |

#### arithmetic
| model | variable | sites | ones_carry | random | **mean IIA** |
|---|---|---|---|---|---|
| Gemma2ForCausalLM | ones_carry | 2 | 0.275 | 0.622 | **0.448** |

#### ravel_task
| model | variable | sites | attribute | prompt_template | wikipedia | **mean IIA** |
|---|---|---|---|---|---|---|
| Gemma2ForCausalLM | Continent | 2 | 0.862 | 0.853 | 0.852 | **0.855** |
| Gemma2ForCausalLM | Country | 2 | 0.623 | 0.600 | 0.622 | **0.615** |
| Gemma2ForCausalLM | Language | 2 | 0.605 | 0.637 | 0.644 | **0.629** |


† Picked sites for this cell were inferred from the eval JSON's per-site IIA pattern (no submission folder present locally). The count is the number of (layer, token-position) units whose IIA exceeded an identity-baseline threshold on at least one split.


‡ **Cell 8 ARC × Gemma × answer (0.999)** is driven by the harness's automatic identity-fallback at L25 last_token — a position PLOT did not pick to train. PLOT's actually-trained DAS rotations scored 0.04–0.79 at this cell. The win is methodologically valid per the eval's scoring rules (it scores every position at picked layers, defaulting to identity at unselected positions) but is not a direct PLOT-rotation result. See `mib_submission/PLOT_SHORTCOMINGS.md` §15 for the full mechanism.

### Comparison to baseline DAS (leaderboard)

Baseline DAS scores from the public MIB leaderboard. PLOT trains DAS at ≤6 sites; baseline trains at all 72.

| cell | PLOT mean | baseline DAS best | baseline DAS avg | Δ (PLOT − baseline best) |
|---|---|---|---|---|
| 4_answer_MCQA × Qwen2ForCausalLM × answer | 0.801 | 0.970 | 0.860 | -0.169 |
| 4_answer_MCQA × Qwen2ForCausalLM × answer_pointer | 0.956 | 0.960 | 0.760 | -0.004 |

## Methods

### PLOT pipeline

For each cell:

1. **Stage A (layer OT)**: per-layer mean-aggregated effect signatures
   are matched against per-OT-row abstract-effect signatures via balanced
   entropic Sinkhorn. Each OT row picks its top-1 layer; the union enters
   Stage B.
2. **Stage B (per-(row, layer) token-position OT)**: within each Stage-A
   layer, an OT cost matrix between abstract rows and per-token-position
   neural rows determines which (layer, token_position) sites enter Stage C.
3. **Stage C (DAS)**: orthogonal-rotation featurizers are trained at the
   selected sites only, using the harness's `train_interventions(method="DAS")`.
4. **Submission**: trained featurizers + identity-mapped indices ship as
   the cell folder. The harness evaluator (`evaluate_submission_task`) then
   computes per-split IIA on the public test sets.

Hyperparameters used so far (uniform across shipped cells unless noted):

- **PLOT**: cost = sq_l2 on L2-normalized rows, balanced Sinkhorn, ε grid
  Stage A {0.01, 0.03} × Stage B {0.003, 0.01, 0.03, 0.1}, top_k_per_row
  Stage A 1, Stage B {1, 2}; calibration sweep selects (ε, top_k) by per-site
  IIA on the calibration variable.
- **DAS**: `n_features=16`, 12 epochs, AdamW lr 1e-3, batch_size=32,
  `dataset_size=256` HF examples per train split (after correctness filter).
- **Eval**: `evaluate_submission_task(public_data=True, private_data=False)`.
  Mean IIA = unweighted mean of per-split max IIA (the "best" leaderboard
  convention).

### Hardware

RTX 4060 Laptop (8 GB VRAM, 125 W max power), Gemma-2-2b at fp16,
Qwen-2.5-0.5B at fp16. Cell 4's wall-clock was inflated to ~1.5 h by
site-1 training while the laptop was on battery; sites 2–4 trained at
expected speed once plugged in.

### Reproducibility

- MIB harness commit: `b69dabe9899251d4a8fe90789afa4d655afc84c7`
- CausalAbstraction commit: `f9ed6777ea5d88bfd88a1488f0903daa50402cc7`
- Pinned package versions: `mib_submission/ENV.md`
- Per-cell driver: `python -m mib_submission.plot.run --task <T> --model <M> --variable <V>`
- This document: regenerated via
  `python -m mib_submission.results._aggregate > mib_submission/results/RESULTS.md`


## Per-cell breakdowns

### 4_answer_MCQA × Gemma2ForCausalLM × answer

**Mean IIA: 0.908** (sites trained: 4)

Picked sites: L16/correct_symbol, L23/last_token, L3/correct_symbol, L9/correct_symbol


Best site per split:
- `answerPosition_test` → L23/last_token = **1.000**
- `randomLetter_test` → L16/correct_symbol = **0.886**
- `answerPosition_randomLetter_test` → L16/correct_symbol = **0.838**

Picked-site IIA:

| site | aP | rL | aPrL |
|---|---|---|---|
| L3/correct_symbol | 0.960 | 0.857 | 0.757 |
| L9/correct_symbol | 0.960 | 0.857 | 0.757 |
| L16/correct_symbol | 0.960 | 0.886 | 0.838 |
| L23/last_token | 1.000 | 0.857 | 0.649 |


Run metadata: OT rows: V=4 (choice0..3) · fit split: `answerPosition_randomLetter_train` · DAS: n_features=16, epochs=12, lr=0.001 · wall-clock: ~90 min

### 4_answer_MCQA × Gemma2ForCausalLM × answer_pointer

**Mean IIA: 0.955** (sites trained: 4)

Picked sites: L17/correct_symbol, L17/correct_symbol_period, L17/last_token, L7/correct_symbol


Best site per split:
- `answerPosition_test` → L17/last_token = **1.000**
- `randomLetter_test` → L7/correct_symbol_period = **1.000**
- `answerPosition_randomLetter_test` → L17/correct_symbol = **0.865**

Picked-site IIA:

| site | aP | rL | aPrL |
|---|---|---|---|
| L7/correct_symbol | 0.940 | 0.800 | 0.838 |
| L17/correct_symbol | 0.940 | 0.857 | 0.865 |
| L17/correct_symbol_period | 0.300 | 1.000 | 0.486 |
| L17/last_token | 1.000 | 0.886 | 0.865 |


Run metadata: OT rows: V=4 (choice0..3) · fit split: `answerPosition_randomLetter_train` · DAS: n_features=16, epochs=12, lr=0.001 · wall-clock: ~50 min

### 4_answer_MCQA × Qwen2ForCausalLM × answer

**Mean IIA: 0.801** (sites trained: 4)

Picked sites: L0/correct_symbol, L2/correct_symbol, L8/correct_symbol, L23/last_token


Best site per split:
- `answerPosition_test` → L23/last_token = **1.000**
- `randomLetter_test` → L2/correct_symbol = **0.769**
- `answerPosition_randomLetter_test` → L8/correct_symbol = **0.633**

Picked-site IIA:

| site | aP | rL | aPrL |
|---|---|---|---|
| L0/correct_symbol | 0.840 | 0.538 | 0.500 |
| L2/correct_symbol | 0.840 | 0.769 | 0.500 |
| L8/correct_symbol | 0.960 | 0.769 | 0.633 |
| L23/last_token | 1.000 | 0.462 | 0.433 |


Run metadata: OT rows: V=4 (choice0..3) · fit split: `answerPosition_randomLetter_train` · DAS: n_features=16, epochs=12, lr=0.001

### 4_answer_MCQA × Qwen2ForCausalLM × answer_pointer

**Mean IIA: 0.956** (sites trained: 4)

Picked sites: L0/correct_symbol, L2/correct_symbol, L8/correct_symbol, L23/last_token


Best site per split:
- `answerPosition_test` → L23/last_token = **1.000**
- `randomLetter_test` → L0/correct_symbol_period = **1.000**
- `answerPosition_randomLetter_test` → L23/last_token = **0.867**

Picked-site IIA:

| site | aP | rL | aPrL |
|---|---|---|---|
| L0/correct_symbol | 0.940 | 0.615 | 0.233 |
| L2/correct_symbol | 0.960 | 0.692 | 0.233 |
| L8/correct_symbol | 0.960 | 0.731 | 0.667 |
| L23/last_token | 1.000 | 0.808 | 0.867 |


Run metadata: OT rows: V=4 (choice0..3) · fit split: `answerPosition_randomLetter_train` · DAS: n_features=16, epochs=12, lr=0.001

### ARC_easy × Gemma2ForCausalLM × answer

**Mean IIA: 0.999** (sites trained: 4)

Picked sites: L17/last_token, L22/correct_symbol, L24/correct_symbol, L25/correct_symbol


Best site per split:
- `answerPosition_test` → L25/last_token = **1.000**
- `randomLetter_test` → L25/last_token = **0.998**
- `answerPosition_randomLetter_test` → L25/last_token = **0.998**

Picked-site IIA:

| site | aP | rL | aPrL |
|---|---|---|---|
| L17/last_token | 0.972 | 0.324 | 0.332 |
| L22/correct_symbol | 0.590 | 0.895 | 0.884 |
| L24/correct_symbol | 0.167 | 0.316 | 0.323 |
| L25/correct_symbol | 0.000 | 0.048 | 0.058 |


### ARC_easy × Gemma2ForCausalLM × answer_pointer

**Mean IIA: 0.884** (sites trained: 6)

Picked sites: L16/last_token, L17/last_token, L22/correct_symbol, L22/last_token, L25/correct_symbol, L25/last_token


Best site per split:
- `answerPosition_test` → L17/last_token = **0.987**
- `randomLetter_test` → L25/correct_symbol = **1.000**
- `answerPosition_randomLetter_test` → L17/last_token = **0.667**

Picked-site IIA:

| site | aP | rL | aPrL |
|---|---|---|---|
| L16/last_token | 0.221 | 0.937 | 0.339 |
| L17/last_token | 0.987 | 0.827 | 0.667 |
| L22/correct_symbol | 0.590 | 0.840 | 0.582 |
| L22/last_token | 0.982 | 0.642 | 0.446 |
| L25/correct_symbol | 0.000 | 1.000 | 0.000 |
| L25/last_token | 0.966 | 0.576 | 0.470 |


### arithmetic × Gemma2ForCausalLM × ones_carry

**Mean IIA: 0.448** (sites trained: 2)

Picked sites: L16/op2_last, L21/last


Best site per split:
- `ones_carry_test` → L16/last = **0.275**
- `random_test` → L21/last = **0.622**

Picked-site IIA:

| site | ones_carry | random |
|---|---|---|
| L16/op2_last | 0.023 | 0.520 |
| L21/last | 0.259 | 0.622 |


### ravel_task × Gemma2ForCausalLM × Continent

**Mean IIA: 0.855** (sites trained: 2)

Picked sites: L25/entity_last_token, L6/entity_last_token


Best site per split:
- `attribute_test` → L25/entity_last_token = **0.862**
- `prompt_template_test` → L25/entity_last_token = **0.853**
- `wikipedia_test` → L25/entity_last_token = **0.852**

Picked-site IIA:

| site | attribute | prompt_template | wikipedia |
|---|---|---|---|
| L6/entity_last_token | 0.798 | 0.795 | 0.782 |
| L25/entity_last_token | 0.862 | 0.853 | 0.852 |


### ravel_task × Gemma2ForCausalLM × Country

**Mean IIA: 0.615** (sites trained: 2)

Picked sites: L25/entity_last_token, L6/entity_last_token


Best site per split:
- `attribute_test` → L25/entity_last_token = **0.623**
- `prompt_template_test` → L25/entity_last_token = **0.600**
- `wikipedia_test` → L25/entity_last_token = **0.622**

Picked-site IIA:

| site | attribute | prompt_template | wikipedia |
|---|---|---|---|
| L6/entity_last_token | 0.558 | 0.530 | 0.557 |
| L25/entity_last_token | 0.623 | 0.600 | 0.622 |


### ravel_task × Gemma2ForCausalLM × Language

**Mean IIA: 0.629** (sites trained: 2)

Picked sites: L25/entity_last_token, L6/entity_last_token


Best site per split:
- `attribute_test` → L25/entity_last_token = **0.605**
- `prompt_template_test` → L25/entity_last_token = **0.637**
- `wikipedia_test` → L25/entity_last_token = **0.644**

Picked-site IIA:

| site | attribute | prompt_template | wikipedia |
|---|---|---|---|
| L6/entity_last_token | 0.494 | 0.536 | 0.527 |
| L25/entity_last_token | 0.605 | 0.637 | 0.644 |



## Other archived runs (no current submission folder)

These archives exist from earlier diagnostic / ablation runs and are not part of the official submission state.

- `ioi_task × GPT2LMHeadModel × output_position` (mean IIA in archive = 16.006)
- `ioi_task × GPT2LMHeadModel × output_token` (mean IIA in archive = 5.155)

