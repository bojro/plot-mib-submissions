# PLOT MIB submission — Results

Per-cell IIA on the public MIB test splits, with picked sites and site-level breakdowns. Numbers in this document are computed from raw eval JSON archives by `_aggregate.py`; do not edit by hand.

**Status: 7 / 26 cells shipped.**

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
| Gemma2ForCausalLM | answer | 7 | 0.998 | 0.890 | 0.881 | **0.923** |
| Gemma2ForCausalLM | answer_pointer | 6 | 0.987 | 1.000 | 0.667 | **0.884** |

#### ravel_task
| model | variable | sites | attribute | prompt_template | wikipedia | **mean IIA** |
|---|---|---|---|---|---|---|
| Gemma2ForCausalLM | Continent | 2 | 0.851 | 0.853 | 0.832 | **0.845** |


† Picked sites for this cell were inferred from the eval JSON's per-site IIA pattern (no submission folder present locally). The count is the number of (layer, token-position) units whose IIA exceeded an identity-baseline threshold on at least one split.

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

**Mean IIA: 0.923** (sites trained: 7)

Picked sites: L17/last_token, L22/correct_symbol, L22/last_token, L24/correct_symbol, L24/last_token, L25/correct_symbol, L25/last_token


Best site per split:
- `answerPosition_test` → L22/last_token = **0.998**
- `randomLetter_test` → L22/correct_symbol = **0.890**
- `answerPosition_randomLetter_test` → L22/correct_symbol = **0.881**

Picked-site IIA:

| site | aP | rL | aPrL |
|---|---|---|---|
| L17/last_token | 0.971 | 0.326 | 0.339 |
| L22/correct_symbol | 0.609 | 0.890 | 0.881 |
| L22/last_token | 0.998 | 0.800 | 0.749 |
| L24/correct_symbol | 0.056 | 0.306 | 0.305 |
| L24/last_token | 0.996 | 0.771 | 0.732 |
| L25/correct_symbol | 0.000 | 0.048 | 0.058 |
| L25/last_token | 0.998 | 0.671 | 0.625 |


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


### ravel_task × Gemma2ForCausalLM × Continent

**Mean IIA: 0.845** (sites trained: 2)

Picked sites: L18/entity_last_token, L6/entity_last_token


Best site per split:
- `attribute_test` → L6/entity_last_token = **0.851**
- `prompt_template_test` → L18/entity_last_token = **0.853**
- `wikipedia_test` → L18/entity_last_token = **0.832**

Picked-site IIA:

| site | attribute | prompt_template | wikipedia |
|---|---|---|---|
| L6/entity_last_token | 0.851 | 0.837 | 0.821 |
| L18/entity_last_token | 0.851 | 0.853 | 0.832 |


