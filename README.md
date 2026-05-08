# PLOT MIB Submissions

PLOT (Progressive Localized Optimal Transport) submissions for the MIB Causal Variable Localization Track.

PLOT picks a small set of (layer, token-position) sites via two-stage optimal transport, then trains DAS rotations only at the picked sites. Goal: comparable accuracy to baseline DAS at roughly an order-of-magnitude reduction in DAS rotation training.

## Status

**7 / 26 cells shipped** (as of 2026-05-08).

- Per-cell results table (auto-generated): [`mib_submission/results/RESULTS.md`](mib_submission/results/RESULTS.md)
- Status tracker: [`mib_submission/results/CELLS.md`](mib_submission/results/CELLS.md)
- Methodological narrative: [`mib_submission/results/JOURNAL.md`](mib_submission/results/JOURNAL.md)
- Cell-1 port story (historical): [`mib_submission/JOURNEY.md`](mib_submission/JOURNEY.md)
- Method limits: [`mib_submission/PLOT_SHORTCOMINGS.md`](mib_submission/PLOT_SHORTCOMINGS.md)
- Project plan + AI assistant guidance: [`CLAUDE.md`](CLAUDE.md)

## Setup (fresh machine)

Tested on Python 3.12. Linux/WSL/macOS. Requires CUDA for Gemma/Llama at reasonable speed. Tested on RTX 4060 Laptop (8 GB VRAM) — that's enough for Gemma-2-2B but not Llama-3.1-8B.

```bash
git clone <this-repo> plot-mib-submissions
cd plot-mib-submissions

# Pull the MIB harness (gitignored)
# WSL note: clone to ~/MIB and symlink — /mnt/c can't chmod git lockfiles.
git clone https://github.com/aaronmueller/MIB.git ~/MIB
cd ~/MIB && git checkout b69dabe9899251d4a8fe90789afa4d655afc84c7
git submodule update --init --recursive MIB-causal-variable-track  # not the circuit-track submodule
cd -
ln -sf ~/MIB MIB

# Python venv (3.12 specifically — sae_lens requires it)
# Use uv if you don't have system 3.12: `curl -LsSf https://astral.sh/uv/install.sh | sh && uv python install 3.12`
~/.local/bin/uv venv --python 3.12 ~/.venv-mib
ln -sf ~/.venv-mib .venv-mib
~/.local/bin/uv pip install --python ~/.venv-mib/bin/python \
    -r MIB/MIB-causal-variable-track/requirements.txt

# HF token (Gemma is gated)
mkdir -p ~/.cache/huggingface
echo -n 'hf_<your_token>' > ~/.cache/huggingface/token
```

Verify CUDA + dataset access:

```bash
.venv-mib/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available())"
.venv-mib/bin/python -c "from huggingface_hub import whoami; print(whoami()['name'])"
```

## Running a cell

Use the CLI:

```bash
.venv-mib/bin/python -u -m mib_submission.plot.run \
    --task <TASK> \
    --model <MODEL_NAME> \
    --variable <VARIABLE> \
    [--train-batch-size 16]   # use for ARC/RAVEL on 8GB VRAM
    [--epochs N]
    [--n-features N]
    [--dataset-size N]
    [--bypass-sites "L:tok,L:tok"]   # skip Stage A/B with hardcoded picks
    > logs/<cell>.log 2>&1
```

Per-task defaults (`PlotConfig` + DAS hyperparameters) live in `mib_submission/plot/configs.py` — adding a new cell is a one-line addition there, not editing `run.py`.

After PLOT writes `submissions/plot/<cell>/` and `verify_submission.py` says "Perfect submission":

```bash
# Eval (call evaluate_submission_task directly — the harness's --no-private_data CLI flag
# does not exist in the pinned commit)
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
```

Archive + regenerate the results doc:

```bash
cp submissions/plot/<cell>/*results.json mib_submission/results/<cell>.json
# Bump the cell row in CELLS.md (☐ → ☑)
.venv-mib/bin/python -m mib_submission.results._aggregate \
    --write mib_submission/results/RESULTS.md
```

If the run revealed something methodologically interesting, append a short note to `JOURNAL.md`.

## Repository layout

```
mib_submission/
├── pipeline.py                 # setup_residual_experiment + ExperimentBundle
├── serialize.py                # write_submission for on-disk MIB triplets
├── apply_results.py            # alternative save path
├── method_to_featurizer.py     # MethodResult → Featurizer encoding
├── featurizers.py              # re-exports of upstream module classes
├── signatures.py               # alphabet_token_ids helper (legacy MCQA path)
├── site_keys.py                # (layer, token_position) key helper
├── activations.py              # base activation collection (unused by PLOT)
├── plot/
│   ├── _alphabets.py           # LabelAlphabet (letter / multi-string / causal-model)
│   ├── features.py             # signatures + abstract table; per-row filter support
│   ├── transport.py            # Sinkhorn solvers (verbatim port)
│   ├── pipeline.py             # select_sites_via_plot (Stage A + B + calibration)
│   ├── configs.py              # per-task PlotConfig presets + RunConfig
│   ├── run.py                  # CLI driver
│   ├── bucketed.py             # parked variant (see PLOT_SHORTCOMINGS §1)
│   └── diagnose_costs.py       # granular cost-matrix dump
├── results/
│   ├── _aggregate.py           # generates RESULTS.md from raw JSON archives
│   ├── RESULTS.md              # AUTO-GENERATED — don't edit by hand
│   ├── CELLS.md                # 26-cell status tracker
│   ├── JOURNAL.md              # methodological narrative
│   └── *.json                  # archived eval outputs, one per cell
├── JOURNEY.md
├── PLOT_SHORTCOMINGS.md
└── ENV.md

tests/                          # 84 tests, all passing
├── test_mib_plot.py
├── test_mib_submission_cross_equiv.py
├── test_mib_submission_roundtrip.py
├── test_mib_submission_signatures.py
├── test_results_aggregate.py
└── test_alphabets_and_ravel.py
```

`MIB/`, `submissions/`, `logs/`, and `.venv-mib/` are gitignored — generated locally.
