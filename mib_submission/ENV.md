# MIB submission environment

Reproducibility record for the MIB causal-variable-track submission pipeline.
All commands below assume the project venv at `../.venv-mib/`.

## Python

- python 3.12.13 (originally homebrew on macOS; on Ubuntu/WSL installed via `uv python install 3.12`)
- venv: `./.venv-mib/` (symlinked to `~/.venv-mib` on WSL since `/mnt/c` 9p mounts can't atomic-rename certain pip files)

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

WSL note: clone MIB outside `/mnt/c` (9p mounts reject git's chmod on lockfiles).

```bash
# MIB (skip the SSH-only sub-submodule under MIB-circuit-track)
git clone https://github.com/aaronmueller/MIB.git ~/MIB
cd ~/MIB && git checkout b69dabe9899251d4a8fe90789afa4d655afc84c7
git submodule update --init --recursive MIB-causal-variable-track
cd -
ln -sf ~/MIB MIB

# Python 3.12 + venv (also outside /mnt/c)
~/.local/bin/uv venv --python 3.12 ~/.venv-mib
ln -sf ~/.venv-mib .venv-mib
~/.local/bin/uv pip install --python ~/.venv-mib/bin/python \
    -r MIB/MIB-causal-variable-track/requirements.txt
```

## NVIDIA driver requirement (WSL)

`requirements.txt` pins `torch==2.11.0+cu130`, which needs **NVIDIA driver ≥ 555** on the Windows host (CUDA 13). On older drivers (12.3 = driver 546) `torch.cuda.is_available()` returns False and runs fall back to CPU. Update via NVIDIA App / GeForce Experience, then `wsl --shutdown` and reopen.

Tested hardware: RTX 4060 Laptop (8 GB VRAM, 125 W max). Sufficient for Gemma-2-2B at fp16 with `train_batch_size=16` (some cells OOM at batch_size=32). **Insufficient** for Llama-3.1-8B at fp16 (would need ~16 GB VRAM).

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
