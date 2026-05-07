# MIB submission environment

Reproducibility record for the MIB causal-variable-track submission pipeline.
All commands below assume the project venv at `../.venv-mib/`.

## Python

- python 3.12.13 (homebrew `/opt/homebrew/opt/python@3.12`)
- venv: `./.venv-mib/` (PEP 668 prevents system-wide installs into homebrew Python)

## Pinned upstream commits

- `aaronmueller/MIB`: `b69dabe9899251d4a8fe90789afa4d655afc84c7` (cloned at `./MIB/`)
- `atticusg/CausalAbstraction` (submodule): `f9ed6777ea5d88bfd88a1488f0903daa50402cc7`
  - At this SHA the layout is `CausalAbstraction/neural/featurizers.py` (no `causalab/` wrapper).
- `MIB-circuit-track` submodule cloned but its EAP-IG sub-submodule fails (SSH). Irrelevant — we ignore the circuit track.

## Key package versions (from `.venv-mib/`)

- torch 2.11.0
- transformers 5.7.0
- pyvene 0.1.8
- sae_lens 6.43.0
- datasets 4.8.5

## Reproducing the install

```bash
git clone --recurse-submodules https://github.com/aaronmueller/MIB.git MIB || true
python3.12 -m venv .venv-mib
.venv-mib/bin/pip install --upgrade pip
.venv-mib/bin/pip install -r MIB/MIB-causal-variable-track/requirements.txt
```

(The `|| true` accommodates the EAP-IG SSH failure.)

## HF auth

Required for gated models (Gemma-2-2B, Llama-3.1-8B). Public datasets in
`mib-bench/*` and Qwen-2.5-0.5B don't strictly need it.

```bash
export HF_TOKEN=hf_...
```

## Smoke checks performed (Step 1)

- Harness imports succeed (`tasks.simple_MCQA`, `experiments.aggregate_experiments`,
  `neural.pipeline.LMPipeline`, `CausalAbstraction.neural.featurizers.Featurizer`).
- HF dataset load works without auth on public splits
  (`get_counterfactual_datasets(hf=True, size=10, load_private_data=False)`).
- `verify_submission.py` passes on the stock `MIB/MIB-causal-variable-track/mock_submission/` folder.

## Valid (task, model) pairs

From `MIB/MIB-causal-variable-track/verify_submission.py:VALID_TASK_MODELS`. Use
this table to pick submission cells; do not waste effort on combinations
outside it.

| Task            | GPT2 | Qwen2.5 | Gemma-2 | Llama-3.1 |
|-----------------|:----:|:-------:|:-------:|:---------:|
| ioi_task        |  Y   |    Y    |    Y    |     Y     |
| 4_answer_MCQA   |  -   |    Y    |    Y    |     Y     |
| ARC_easy        |  -   |    -    |    Y    |     Y     |
| arithmetic      |  -   |    -    |    Y    |     Y     |
| ravel_task      |  -   |    -    |    Y    |     Y     |

Cheapest valid starting cells: `4_answer_MCQA × Qwen2ForCausalLM` and
`ioi_task × GPT2LMHeadModel`.
