# CLAUDE.md — PLOT MIB Submissions

Project guidance for Claude Code (claude.ai/code) when working in this repository.

## Repository purpose

Standalone repo for PLOT (Progressive Localized Optimal Transport) submissions to the MIB Causal Variable Localization Track. PLOT picks (layer, token-position) sites via two-stage Sinkhorn OT, then trains DAS rotations only at the picked sites — targeting baseline-DAS-comparable accuracy at ~10× fewer rotations trained.

- Method narrative + cell-1 port story: `mib_submission/JOURNEY.md`.
- Structural limits of the method: `mib_submission/PLOT_SHORTCOMINGS.md`.
- Per-cell IIA results (auto-generated): `mib_submission/results/RESULTS.md`.
- Methodological narrative + per-session decisions: `mib_submission/results/JOURNAL.md`.
- 26-cell status tracker: `mib_submission/results/CELLS.md`.

Source-of-truth PLOT (binary-addition GRU origin) is preserved offline at `reference/source_plot/`.

## MIB Causal Variable Track — submission plan

Goal: benchmark **PLOT only** (Progressive Localized Optimal Transport — OT site selection + DAS rotation training at picked sites) on the MIB Causal Variable Localization Track. Earlier scope included raw OT / GW / FGW / UOT / OT+gradient / OT+PCA; those are dropped. Circuit track is out of scope.

Total target cells = **26** (constrained by `verify_submission.py:VALID_TASK_MODELS`):
- ioi_task: 2 vars × 4 models = 8
- 4_answer_MCQA: 2 vars × 3 models = 6
- ARC_easy: 2 vars × 2 models = 4
- arithmetic: 1 var × 2 models = 2
- ravel_task: 3 vars × 2 models = 6

Reference points:
- Harness: `https://github.com/aaronmueller/MIB`, submodule `MIB-causal-variable-track/CausalAbstraction` (`https://github.com/atticusg/CausalAbstraction`).
- `Featurizer` interface in `CausalAbstraction/neural/featurizers.py`: paired `featurizer (x → (features, error))` + `inverse_featurizer ((features, error) → x̂)`, plus `n_features` and string `id`.
- Submission unit: a folder per `{TASK}_{MODEL}_{VARIABLE}` containing `{ModelUnit}_featurizer`, `{ModelUnit}_inverse_featurizer`, `{ModelUnit}_indices`. Verify with `verify_submission.py`; private eval runs `evaluate_submission.py`.
- Tasks: IOI, simple_MCQA, ARC, two_digit_addition, RAVEL. Models: GPT-2 Small, Qwen-2.5-0.5B, Gemma-2-2B, Llama-3.1-8B.
- Average leaderboard requires every layer; "best" leaderboard accepts a single layer.

All MIB harness code lives under `MIB/` (gitignored on `/mnt/c`; symlinked from `~/MIB` on WSL because 9p mounts can't `chmod` git lockfiles). Our submission-side code lives under `mib_submission/`.

## Current status (as of 2026-05-08)

**7 / 26 cells shipped (26.9%).** Live numbers in `mib_submission/results/RESULTS.md`. Headline:

| cell | task × model × variable | mean IIA | sites | notes |
|---|---|---|---|---|
| 1 | MCQA × Qwen × answer_pointer | 0.956 | 4 (inferred) | shipped pre-session |
| 2 | MCQA × Qwen × answer | 0.801 | 4 (inferred) | shipped pre-session; gap = DAS@n_features=16 generalization |
| 3 | MCQA × Gemma × answer_pointer | 0.955 | 4 | ties Qwen cell 1 |
| 4 | MCQA × Gemma × answer | 0.908 | 4 | beats Qwen cell 2 by +0.107 |
| 7 | ARC × Gemma × answer_pointer | 0.884 | 6 | OOM forced bypass + batch_size=16 |
| 8 | ARC × Gemma × answer | 0.923 | 7 | several picks failed during DAS but didn't hurt headline |
| 22 | RAVEL × Gemma × Continent | 0.845 (smoke) | 2 | n_features=64, dataset_size=128, 1 epoch |

Repository layout (current, as code now exists):

```
mib_submission/
├── pipeline.py                 # setup_residual_experiment + ExperimentBundle
│                                # _TASK_MODULES: 4_answer_MCQA, ARC_easy, ravel_task, arithmetic
│                                # max_new_tokens parameter (default 1, 2 for RAVEL)
├── serialize.py                # write_submission for on-disk MIB triplets
├── apply_results.py            # alternative save path
├── method_to_featurizer.py     # MethodResult → Featurizer encoding
├── featurizers.py              # re-exports of upstream module classes
├── signatures.py               # alphabet_token_ids helper (legacy MCQA path)
├── site_keys.py                # (layer, token_position) key helper
├── activations.py              # base activation collection (unused by PLOT)
├── plot/
│   ├── _alphabets.py           # NEW. LabelAlphabet for letter / multi-string /
│   │                            # causal-model alphabets. Lazy token resolution
│   │                            # with collision compaction (e.g. RAVEL 928→271 dims).
│   ├── features.py             # signatures + abstract table.
│   │                            # Now accepts alphabet kwarg (alongside legacy letters)
│   │                            # + per_row_dataset_filter + on_unknown_label.
│   ├── transport.py            # Sinkhorn solvers (verbatim port).
│   ├── pipeline.py             # select_sites_via_plot. Per-row neural collection
│   │                            # when per_row_filter_attribute is set; otherwise
│   │                            # single shared collection (back-compat with cells 1-8).
│   ├── configs.py              # NEW. RunConfig + per-task PlotConfig presets.
│   │                            # _mcqa_v4_choices, _arc_v4_symbols, _ravel_v3_attributes,
│   │                            # _ravel_checker (multi-word/comma-list answers).
│   ├── bucketed.py             # Parked variant (see PLOT_SHORTCOMINGS §1).
│   ├── diagnose_costs.py       # Granular cost-matrix dump.
│   └── run.py                  # CLI driver. argparse: --task --model --variable
│                                # plus overrides --epochs --n-features --bypass-sites
│                                # --train-batch-size --dataset-size --signature-dataset.
├── results/
│   ├── _aggregate.py           # NEW. Single source of truth for RESULTS.md.
│   ├── RESULTS.md              # AUTO-GENERATED. Don't edit by hand.
│   ├── CELLS.md                # Status tracker (status only, no IIA numbers).
│   ├── JOURNAL.md              # Methodological narrative (append-only).
│   └── *.json                  # Archived eval outputs, one per cell.
├── JOURNEY.md                  # Cell-1 port story (historical).
├── PLOT_SHORTCOMINGS.md        # Method limits + per-cell expectations.
└── ENV.md                      # Pinned commits, package versions.

tests/
├── test_mib_plot.py
├── test_mib_submission_cross_equiv.py
├── test_mib_submission_roundtrip.py
├── test_mib_submission_signatures.py
├── test_results_aggregate.py     # _aggregate.py coverage (28 tests)
└── test_alphabets_and_ravel.py   # alphabets + per-row filter + RAVEL config (30 tests)
```

84 tests passing (was 26 at session start).

## How to run a cell (CLI)

```bash
.venv-mib/bin/python -u -m mib_submission.plot.run \
    --task <TASK> \
    --model <MODEL_NAME> \
    --variable <VARIABLE> \
    [--train-batch-size 16]      # Use for ARC/RAVEL on 8GB to avoid OOM
    [--n-features N]              # Override the per-task default
    [--epochs N]
    [--dataset-size N]
    [--bypass-sites "L:tok,L:tok"]  # Skip Stage A/B with hardcoded picks
    > logs/<cell>.log 2>&1
```

After PLOT writes `submissions/plot/<cell>/` and `verify_submission.py` says "Perfect submission":

```bash
# Eval. The harness's --no-private_data CLI flag DOES NOT EXIST in this commit;
# call evaluate_submission_task() directly. See logs/cell*_eval.log for examples.
.venv-mib/bin/python -u -c "
import sys
from pathlib import Path
ROOT = Path('.')
TRACK = ROOT / 'MIB' / 'MIB-causal-variable-track'
sys.path.insert(0, str(TRACK)); sys.path.insert(0, str(TRACK / 'CausalAbstraction'))
from evaluate_submission import evaluate_submission_task
evaluate_submission_task(
    task_folder_path=str(ROOT / 'submissions/plot/<cell>'),
    submission_base_path=str(ROOT / 'submissions/plot'),
    private_data=False, public_data=True,
)
"

# Archive
cp submissions/plot/<cell>/*results.json mib_submission/results/<cell>.json

# Update CELLS.md status (☐→☑) and regenerate RESULTS.md
.venv-mib/bin/python -m mib_submission.results._aggregate --write mib_submission/results/RESULTS.md
```

Append a note to `JOURNAL.md` if the run revealed something methodologically interesting.

## Critical implementation choices (do not regress)

These survived multiple cells of validation:

1. **Output-space signatures, not feature-space.** S and A are length-K (alphabet size) per row, aggregated across examples — not (N · K) flattened. Flattened form caused uniform Sinkhorn plans because sq_l2 over thousands of dims with magnitudes O(1) gives costs O(10³); `exp(−10³/ε)` underflows for any ε ≪ 100.
2. **L2-normalise rows of A and S before cost.** Brings sq_l2 cost into [0, 4], well-conditioned for any ε.
3. **V ≥ 2 OT variables.** Balanced Sinkhorn with V=1 forces uniform plan mathematically. MCQA/ARC use V=4 `choice0..3` or `symbol0..3`; RAVEL uses V=3 `(Country, Continent, Language)`.
4. **Stage A is per-row top-1 layer pick.** Each OT row picks its own best layer; the union goes to Stage B. Faithful to source `_stage_a_timesteps`.
5. **Stage B uses Stage-A-cached signatures.** No new forward passes between stages.
6. **DAS only on selected sites.** Prune `experiment.model_units_lists` rather than rebuild the bundle.
7. **Token-set signature alphabet for word-token tasks (RAVEL).** `LabelAlphabet` resolves answer strings to LM-vocab first-token IDs, compacting collisions (RAVEL: 928 labels → 271 dims under Gemma's tokenizer). Eagerly resolved in `select_sites_via_plot` so abstract and neural tables share K.
8. **Per-row dataset filter for RAVEL.** When `per_row_filter_attribute` is set in `PlotConfig`, `select_sites_via_plot` runs `collect_neural_outputs` once per OT row on the matching subset of bases (e.g. row `Country` only sees bases where `queried_attribute=Country`). Costs 3× signature collection time but eliminates the no-op-base SNR drag.
9. **Custom checker for multi-word answers.** RAVEL's HF-correctness filter needs `_ravel_checker` (in `configs.py`) that handles "United States", "South Korea", and comma-separated alternative-answer lists. The default `expected in output_text` rejects too many examples.
10. **`max_new_tokens=2` for RAVEL.** Gemma tokenizes "United States" → 2 tokens, "France" → 1. The DAS loss function indexes `logits[:, -labels.shape[-1] - 1 : -1]`, which works correctly for multi-token labels but the LM pipeline must be configured to generate that many tokens.

## Per-task config notes

**MCQA (cells 1–6).** V=4 `choice0..3`. Letters="A..Z". `signature_dataset="answerPosition_randomLetter_train"` (only split where both pointer and letter vary). `n_features=16`, 12 epochs. Custom checker: not needed (single-letter answers).

**ARC (cells 7–10).** V=4 `symbol0..3` — ARC has no `choice` variables (different from MCQA). Letters="A..Z". Same signature dataset as MCQA. **Known issue**: only 2 token positions per layer, so `stage_b_top_k_grid=(1,2)` makes Stage B essentially keep both positions per picked layer. Cells 7+8 picked 6 and 7 sites where 4 would have sufficed; the extras failed to converge but didn't hurt best-per-split IIA. Mitigation for future: switch to `stage_b_top_k_grid=(1,)`.

**RAVEL (cells 21–23).** V=3 `(Country, Continent, Language)`. Alphabet from causal model (≈928 labels → ~271 first-token dims under Gemma). Per-row filter on `queried_attribute`. `signature_dataset="attribute_train"` (cross-attribute counterfactuals). `n_features=288`, **1 epoch** (matches baseline). `max_new_tokens=2`. `_ravel_checker` for filter. Cell 22 (Continent) shipped at smoke settings (n_features=64, dataset_size=128) at 0.845 — full settings should land 0.90+. **Open**: scale up; run Country and Language with full settings.

**arithmetic (cells 11, 12).** Single variable `ones_carry`. **V=1 collapse risk**, deferred. Needs adjacent-variable workaround or bucketing.

**IOI (cells 13–20).** **Bootstrap blocked**: `tasks/IOI_task/ioi_task.get_causal_model(parameters)` requires per-model `{bias, token_coeff, position_coeff}` learned via `baselines/ioi_baselines/ioi_learn_linear_params.py`. Run that bootstrap once per model and ship `ioi_linear_params.json` with the submission. Not yet wired in `pipeline.py`.

**Llama-8B cells (5, 6, 9, 10, 12, 19, 20, 24, 25, 26).** Won't fit in 8 GB VRAM at fp16. Need ≥16 GB GPU (cloud A100 / L4 etc.) or quantization. Deferred until a bigger box is available.

## Rollout order (post-session)

1. ✅ MCQA × {Qwen, Gemma} × {pointer, answer} — cells 1–4 shipped.
2. ✅ ARC × Gemma × {pointer, answer} — cells 7, 8 shipped.
3. ◐ RAVEL × Gemma × Continent — smoke shipped at 0.845; **needs full-settings rerun**.
4. ☐ RAVEL × Gemma × {Country, Language} — same pipeline; Language has the most token collisions and may underperform.
5. ☐ Scale up RAVEL × Continent: re-run with `n_features=288 dataset_size=256` to land a competitive number.
6. ☐ ARC config tweak: switch `stage_b_top_k_grid=(1,)` and re-run cells 7+8 to validate the H1 mitigation (~halves DAS cost).
7. ☐ Seed sweeps: re-run one shipped cell 3× with different seeds to estimate IIA variance. **Single-seed runs are samples not results.** Add `--seed` to the CLI.
8. ☐ IOI bootstrap (linear params learning) → IOI × {GPT-2, Qwen, Gemma} — cells 13–18.
9. ☐ arithmetic V=1 workaround design + cell 11.
10. ☐ Llama cells — only after a ≥16 GB GPU is available.

## Hard constraints

- The official MIB `Featurizer` must be invertible. Any "selection" featurizer routes the unselected dims through `error` rather than discarding them.
- Use upstream `evaluate_submission.py` / `verify_submission.py` / `aggregate_results.py` verbatim. Do not reimplement IIA.
- `reference/source_plot/` is read-only. Don't import from or modify it.
- All run outputs (`logs/`, `submissions/`, `models/`, `signatures/`, `MIB/`, `.venv-mib/`) are gitignored. Curated results land in `mib_submission/results/`.
- `RESULTS.md` is generated, not hand-written. Edit `_aggregate.py` if formatting needs changing, then regenerate.
- Adding a new cell type → write a preset in `configs.py` and a smoke test in `tests/test_alphabets_and_ravel.py` (or analogous). Don't edit `run.py` constants — use the CLI.

## Operational gotchas

- `evaluate_submission.py --no-private_data` **does not exist** in the pinned harness commit. `--private_data` is `store_true, default=True` and can't be disabled from CLI. Workaround: call `evaluate_submission_task(..., private_data=False, public_data=True)` directly from a wrapper script.
- WSL `/mnt/c` mounts can't `chmod` lockfiles, so `git clone` fails for repos with hooks. Clone to `~/MIB` and symlink `MIB → ~/MIB`. Same for `.venv-mib`.
- Laptop GPUs throttle on battery. Power state changes can produce 30× speed swings during a single run (cell 4 site 1 ran on battery; sites 2–4 ran on AC after we plugged in mid-run). Always confirm `nvidia-smi power.draw` is at expected level before launching long runs.
- ARC has only 2 token positions per layer; MCQA has 3. Stage B's `top_k_grid=(1,2)` degenerates on ARC. Cells 7+8 picked too many sites but it didn't hurt IIA.
- Gemma is gated. HF token must be saved to `~/.cache/huggingface/token` and the user must accept the license at https://huggingface.co/google/gemma-2-2b.
