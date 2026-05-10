# PLOT MIB submission

PLOT (**Progressive Localization via Optimal Transport**) submissions to the [MIB Causal Variable Localization Track](https://github.com/aaronmueller/MIB). PLOT picks `(layer, token-position)` sites via two-stage Sinkhorn OT, then trains DAS rotations only at the picked sites — targeting baseline-DAS-comparable accuracy at ≤10× fewer rotations trained.

What this repo ships is the **PLOT-DAS** variant from the source paper (Stage A + Stage B + DAS at picked sites). Other PLOT variants in the paper — `PLOT` (localization only), `PLOT-native` / `PLOT-PCA` (Stage B handles in native or PCA coords), `Full DAS` — aren't implemented here. Source repo for the paper: <https://github.com/jchang153/causal-abstractions-ot>.

## Headline status

**12 of 26 cells with submissions** (46.2%). 11 at full quality, 1 (arithmetic) at smoke.

Of the 12 shipped:

| status | cells | mechanism |
|---|---|---|
| 🏆 win/tie vs DAS leaderboard | 1 (Qwen pointer), 7 (ARC pointer), 8 (ARC answer)\*, 22 (RAVEL Continent) | PLOT picks well |
| 📏 small structural gap (~5–7%) | 3 (Gemma pointer), 4 (Gemma answer) | confirmed outside seed band |
| ❌ documented structural gap | 2 (Qwen answer), 13/14 (IOI), 21/23 (RAVEL Country/Language) | each diagnosed in `PLOT_SHORTCOMINGS.md` |
| ⚠ smoke quality | 11 (arithmetic) | scale-up regressed; reverted to smoke |

\* Cell 8's 0.999 score includes a non-obvious mechanism — see `PLOT_SHORTCOMINGS.md` §15.

The other 14 of 26 cells require ≥16 GB GPU (Qwen/Gemma IOI + 10 Llama cells); deferred to cloud.

## Relationship to the source paper

The PLOT algorithm is from <https://github.com/jchang153/causal-abstractions-ot>. This repo ships the **PLOT-DAS** variant of the paper (Stage A localizes layers, Stage B localizes positions within those layers, DAS rotations are trained at the picked sites). Algorithm-shape-wise we match the paper's binary-addition pipeline `experiments/binary_addition/` (formerly the `codex/binary-addition-two-stage-plot` branch). What is **not** a direct port of paper code:

| benchmark | paper status | what we did |
|---|---|---|
| Binary addition | in paper | not in scope here (not a MIB cell) |
| MCQA (cells 1–4) | in paper | re-implemented PLOT-DAS against MIB's MCQA dataset using the binary-addition framework; **not** a port of `experiments/mcqa/` scripts |
| Two-digit addition (cell 11) | in paper | re-implemented against MIB's `arithmetic` task; **not** a port of `experiments/two_digit_addition/` |
| ARC (cells 7, 8) | not in paper | **new** application — V=4 OT rows over `symbol0..3` letter swaps |
| RAVEL (cells 21, 22, 23) | not in paper | **new** application — V=3 attribute rows + per-row dataset filter + causal-model-derived alphabet |
| IOI (cells 13, 14) | not in paper | **new** application — attention-head featurizers, `PatchAttentionHeads` joint DAS, IOI-specific linear-params bootstrap |

**Things in this repo that aren't in the source paper:**
- Per-row dataset filter for RAVEL (each OT row's signature collected only on bases where `queried_attribute == row_variable`)
- Attention-head dispatch for IOI: 3-tuple site keys `(layer, head, token_pos)`, joint DAS across picked heads, MSE-on-logit-diff loss
- Eval-driver patches (`scripts/eval_cell.py`): per-task `max_new_tokens` override, `LMPipeline.load` `position_ids` fallback for transformers 5.x, Qwen2 `head_dim` injection — these are MIB-harness-specific
- `--seed` flag + seed-variance sweep methodology
- ARC `stage_b_top_k_grid=(1,)` tweak (PLOT_SHORTCOMINGS §8) and the resulting DAS-vs-identity finding (§15)

**Scoring difference:** we report MIB's IIA aggregation (per-split → per-layer max → highest-view across layers). Numbers in `mib_submission/results/RESULTS.md` are directly comparable to MIB's DAS leaderboard but **not directly comparable to paper tables**, which use task-specific metrics.

**Things the paper has that this repo doesn't implement:** `PLOT` (localization-only, no DAS), `PLOT-native` and `PLOT-PCA` (Stage B handles in native or PCA basis), `PLOT-native-DAS` and `PLOT-PCA-DAS` (DAS guided by Stage B support), `Full DAS` (DAS over all sites). We only ship `PLOT-DAS`.

## Where to look

- **`CLAUDE.md`** — project context, status table, leaderboard comparison, rollout plan. Dense; engineer-oriented.
- **`PLOT_SHORTCOMINGS.md`** — 15-section catalog of diagnosed limitations. Read this for a calibrated view of where PLOT works vs doesn't, and *why*.
- **`mib_submission/results/RESULTS.md`** — auto-generated per-cell IIA tables.
- **`mib_submission/results/CELLS.md`** — per-cell status tracker.
- **`JOURNAL.md`** — methodological narrative, append-only by date. The full engineering record.
- **`HYPOTHESES.md`** — experimental hypotheses and outcomes from the diagnostic sessions.

## What's the value proposition

PLOT trains DAS rotations at **2–7 picked sites per cell** vs the baseline's **72 sites** (every layer × token position). On cells where PLOT's site selection is well-matched to the task, scores are competitive at 10–25× fewer trained rotations. On cells where PLOT's signature design picks the wrong sites, the gap to baseline DAS is structural and documented.

A surprise finding from the diagnostic sessions: PLOT's value-add is concentrated in **layer selection** (Stage A). Stage B (position selection) and DAS training can be subtractive on some cells — see `PLOT_SHORTCOMINGS.md` §15. A leaner "Stage A only" PLOT remains an open follow-up.

## Setup from a fresh clone

The `MIB/` submodule, `submissions/`, `logs/`, `models/`, and `.venv-mib/` are gitignored — they need to be created locally.

```bash
git clone https://github.com/bojro/plot-mib-submissions.git
cd plot-mib-submissions

# Pull the MIB harness (gitignored). On WSL clone to ~ and symlink because
# /mnt/c can't chmod the git lockfiles in MIB hooks.
git clone https://github.com/aaronmueller/MIB.git ~/MIB
cd ~/MIB && git checkout b69dabe9899251d4a8fe90789afa4d655afc84c7
git submodule update --init --recursive MIB-causal-variable-track
cd -
ln -sf ~/MIB MIB

# Python 3.12 venv (sae_lens dep requires it). Install uv if needed:
#   curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uv venv --python 3.12 ~/.venv-mib
ln -sf ~/.venv-mib .venv-mib
~/.local/bin/uv pip install --python ~/.venv-mib/bin/python \
    -r MIB/MIB-causal-variable-track/requirements.txt

# HuggingFace token (Gemma is gated — accept the license at huggingface.co/google/gemma-2-2b first)
mkdir -p ~/.cache/huggingface
echo -n 'hf_<your_token>' > ~/.cache/huggingface/token
```

Sanity check the install:

```bash
.venv-mib/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available())"
.venv-mib/bin/python -c "from huggingface_hub import whoami; print(whoami()['name'])"
.venv-mib/bin/python -m pytest tests/ -q   # should report 126 passed
```

## How to work with the codebase

### Inspect what's shipped

```bash
# Verify the 10 shipped triplets pass the harness's submission format check
.venv-mib/bin/python MIB/MIB-causal-variable-track/verify_submission.py submissions/plot

# Look at per-cell scores
cat mib_submission/results/RESULTS.md
cat mib_submission/results/CELLS.md
```

### Run a cell end-to-end

```bash
# Run PLOT on one cell. Defaults are loaded from mib_submission/plot/configs.py
# (per-task PlotConfig + RunConfig presets). Common overrides shown.
.venv-mib/bin/python -m mib_submission.plot.run \
    --task <TASK> \
    --model <HF_MODEL_NAME> \
    --variable <VARIABLE_NAME> \
    [--n-features N] \
    [--epochs N] \
    [--dataset-size N] \
    [--train-batch-size 16]   # use for ARC/RAVEL on 8 GB to avoid OOM
    [--seed N]                # for variance sweeps
    [--bypass-sites "L:tok,L:tok"]   # skip Stage A/B with manual picks
    > logs/<cell>.log 2>&1
```

`<TASK>` is one of `4_answer_MCQA`, `ARC_easy`, `arithmetic`, `ravel_task`, `ioi_task`. The cell folder is created at `submissions/plot/<task>_<modelClass>_<variable>/`.

### Evaluate a cell

The MIB harness's `evaluate_submission.py` has two harness-specific quirks (no `--no-private_data` flag, `max_new_tokens=1` hardcoded for all tasks) that break arithmetic, RAVEL, and IOI evals. We ship a patched driver:

```bash
.venv-mib/bin/python scripts/eval_cell.py \
    --cell <cell_folder_name> \
    > logs/<cell>_eval.log 2>&1
```

The driver auto-dispatches to `evaluate_submission_task` for residual-stream cells and `evaluate_ioi_submission_task` for IOI cells, applies the right `max_new_tokens` per task, and patches the `LMPipeline.load` `position_ids` fallback for transformers 5.x.

### Archive and update results docs

```bash
# Copy the cell's results JSON into the curated results folder
cp submissions/plot/<cell>/*results.json mib_submission/results/<cell>.json

# Bump the cell row in CELLS.md (☐ → ☑) by hand
# Regenerate RESULTS.md from the curated JSONs
.venv-mib/bin/python -m mib_submission.results._aggregate \
    --write mib_submission/results/RESULTS.md

# Append a session entry to JOURNAL.md if the run revealed something
# methodologically interesting
```

### Repo map

```
plot-mib-submissions/
├── README.md                          # this file
├── CLAUDE.md                          # engineer-oriented project context
├── PLOT_SHORTCOMINGS.md               # 15-section catalog of method limits
├── HYPOTHESES.md                      # experimental hypotheses + outcomes
├── JOURNAL.md                         # methodological narrative, append-only
├── mib_submission/
│   ├── pipeline.py                    # ExperimentBundle + setup_residual / attention head
│   ├── serialize.py, site_keys.py     # MIB Featurizer triplet I/O
│   ├── plot/
│   │   ├── pipeline.py                # select_sites_via_plot (Stage A + B + calibration)
│   │   ├── transport.py               # Sinkhorn solvers
│   │   ├── features.py                # signatures + abstract table
│   │   ├── _alphabets.py              # LabelAlphabet (letter / multi-string / causal-model)
│   │   ├── configs.py                 # per-task PlotConfig + RunConfig presets
│   │   ├── run.py                     # CLI driver
│   │   └── bucketed.py                # parked variant (see §1)
│   ├── ioi/
│   │   ├── bootstrap.py               # IOI linear-params bootstrap
│   │   ├── submission.py              # cell_dir, write_ioi_submission
│   │   └── _patches.py, _runner.py    # transformers/pyvene compat patches
│   └── results/
│       ├── _aggregate.py              # generates RESULTS.md from JSONs
│       ├── RESULTS.md                 # AUTO-GENERATED, don't edit
│       ├── CELLS.md                   # 26-cell status tracker
│       └── *.json                     # archived eval outputs (one per cell)
├── scripts/
│   ├── eval_cell.py                   # patched MIB eval driver
│   └── overnight*.sh                  # overnight launcher patterns
├── tests/                             # 126 tests, pytest
└── reference/source_plot/             # read-only snapshot of paper code
                                       # (binary-addition Stage A+B branch)
```

The `MIB/`, `submissions/`, `logs/`, `models/`, and `.venv-mib/` directories are gitignored — created on first run. `submissions/plot/` is where verified submissions live; `submissions/_plot_backups/` (also gitignored) preserves pre-modification baselines for cells we re-ran during diagnostic sessions.

### Adding a new cell or task

1. **New cell of an existing task** — no code changes needed. Run `mib_submission.plot.run` with the new `--task --model --variable` triple. Defaults come from the per-task `PlotConfig` preset in `mib_submission/plot/configs.py`.
2. **New task** — add a per-task preset function in `configs.py` (a `PlotConfig` returning the OT row schema, alphabet, and signature dataset) plus a branch in `default_config()`. See `_mcqa_v4_choices`, `_arc_v4_symbols`, `_ravel_v3_attributes`, `_arithmetic_v2_carry_children`, `_ioi_v3_splits` for working patterns. Don't edit `run.py` constants — use the CLI.
3. **New cell type** (e.g. attention-head instead of residual-stream) — needs a `setup_<type>_experiment` in `mib_submission/pipeline.py` and a `main_<type>` branch in `run.py`. See `mib_submission/ioi/` for the attention-head precedent.

## Hardware

Developed on an 8 GB RTX 4060 Laptop. 12 of 26 cells fit at this scale. The other 14 (4 Qwen/Gemma IOI cells via pyvene's `IntervenableModel` + 10 Llama-8B cells) need ≥16 GB VRAM — cloud GPU work, deferred.

## Caveats for a careful reader

- **Cell 8's 0.999 leaderboard-relative win comes from an interaction with the eval harness's identity-fallback** at unselected positions, not from PLOT-trained DAS rotations. Methodologically valid per the harness's scoring rules. Full mechanism documented in `PLOT_SHORTCOMINGS.md` §15.
- **5 of 12 reachable cells have real structural gaps to DAS baseline.** Each is diagnosed in `PLOT_SHORTCOMINGS.md` (§2 cell 2, §13 cells 13/14, §14 cells 21/23). Closing them is out of scope for this submission.
- **Cell 11 arithmetic ds=1024 scale-up regressed.** Reverted to the smoke result; the failed scale-up's submission is preserved at `submissions/_plot_backups/arithmetic_*_pre_c6_*` for reference.
