# PLOT MIB Submissions

PLOT (Progressive Localized Optimal Transport) submissions for the MIB Causal Variable Localization Track.

PLOT picks a small set of (layer, token-position) sites via two-stage optimal transport, then trains DAS rotations only at the picked sites. Goal: comparable accuracy to baseline DAS at roughly an order-of-magnitude reduction in DAS rotation training.

## Status

See `mib_submission/results/CELLS.md` for the per-cell tracker (1/26 shipped at 0.956, 2/26 shipped at 0.801 as of repo init).

Process narrative: `mib_submission/JOURNEY.md`.

Method limits: `mib_submission/PLOT_SHORTCOMINGS.md`.

Per-run results: `mib_submission/results/EVAL_LOG.md`.

## Setup (fresh machine)

Tested on Python 3.12. Linux/macOS. Requires CUDA for Gemma/Llama at reasonable speed.

```bash
git clone <this-repo> plot-mib-submissions
cd plot-mib-submissions

# Pull the MIB harness (gitignored — not in this repo)
git clone --recurse-submodules https://github.com/aaronmueller/MIB.git
cd MIB && git checkout <pinned-sha>  # see mib_submission/ENV.md
cd ..

# Python venv
python3.12 -m venv .venv-mib
.venv-mib/bin/pip install -r MIB/MIB-causal-variable-track/requirements.txt

# HF token (gated models)
mkdir -p ~/.cache/huggingface
echo -n 'hf_<your_token>' > ~/.cache/huggingface/token
```

Verify CUDA + dataset access:

```bash
.venv-mib/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available())"
.venv-mib/bin/python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])"
```

## Running a cell

Edit the config block at the top of `mib_submission/plot/run.py`:

- `TASK`, `MODEL_NAME`, `MODEL_CLASS_NAME`, `VARIABLE` — the cell.
- `PLOT_CONFIG.variables` — OT row variables (V≥2, observably distinct).
- `PLOT_SIGNATURE_DATASET` — train split to fit signatures on.
- `BYPASS_SITES = None` for normal runs (set to a list to hardcode sites for diagnostics).
- `USE_BUCKETED_PLOT = False` (the bucketed variant is parked, see PLOT_SHORTCOMINGS.md §1).

Run:

```bash
.venv-mib/bin/python -u -m mib_submission.plot.run > logs/<cell>.log 2>&1
```

After the script writes `submissions/plot/<cell>/`, evaluate:

```bash
.venv-mib/bin/python MIB/MIB-causal-variable-track/evaluate_submission.py \
    --submission_folder submissions/plot \
    --no-private_data --public_data > logs/eval_<cell>.log 2>&1
```

Archive and update tracker:

```bash
cp submissions/plot/<cell>/*results.json mib_submission/results/<cell>.json
# Then edit mib_submission/results/CELLS.md and EVAL_LOG.md by hand.
```

## Repository layout

```
mib_submission/
├── pipeline.py              # ExperimentBundle setup, task module dispatch
├── serialize.py             # write_submission for on-disk MIB triplets
├── apply_results.py         # alternative save path via experiment.save_featurizers
├── method_to_featurizer.py  # MethodResult → Featurizer encoding
├── featurizers.py           # re-exports of upstream module classes
├── activations.py           # base activation collection (for non-PLOT methods)
├── signatures.py            # alphabet token id helpers
├── site_keys.py             # (layer, token_position) key helper
├── plot/
│   ├── features.py          # output-prob-delta signatures
│   ├── transport.py         # Sinkhorn solvers (verbatim from source PLOT)
│   ├── pipeline.py          # select_sites_via_plot (Stage A + B + calibration)
│   ├── bucketed.py          # parked: bucketed-by-source-value variant
│   ├── diagnose_costs.py    # granular cost-matrix dump
│   └── run.py               # driver
├── results/
│   ├── CELLS.md             # 26-cell tracker
│   ├── EVAL_LOG.md          # per-run results
│   └── *.json               # archived eval outputs
├── JOURNEY.md
├── PLOT_SHORTCOMINGS.md
└── ENV.md

tests/
├── test_mib_plot.py
├── test_mib_submission_cross_equiv.py
├── test_mib_submission_roundtrip.py
└── test_mib_submission_signatures.py
```

`MIB/`, `submissions/`, `logs/`, and `.venv-mib/` are gitignored — generated locally.
