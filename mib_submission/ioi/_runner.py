"""In-process IOI bootstrap.

Re-implements `baselines/ioi_baselines/ioi_learn_linear_params.py:__main__`
inline so we can apply runtime monkey-patches (see `_patches.py`) BEFORE
the harness loads its model / pipeline / pyvene experiment.

Why inline (vs subprocess as the wrapper does for non-blocked models):
  - GPT-2 needs `LMPipeline.load` patched for transformers 5.x position_ids.
  - Qwen needs `model.config.head_dim` injected for pyvene 0.1.8.
  - Both patches must be in-process; subprocess + monkey-patch combo is
    fragile (PYTHONSTARTUP, etc.).

The output JSON shape matches the subprocess script exactly so downstream
loaders work unchanged.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Optional

import torch

from ..pipeline import MIB_TRACK, add_mib_to_syspath


def run_inline(
    model_short: str,
    *,
    output_path: Path,
    heads_list: Optional[list[tuple[int, int]]] = None,
    quick_test: bool = False,
    eval_batch_size: Optional[int] = None,
) -> dict:
    """Run the bootstrap pipeline in-process. Returns the parsed output JSON."""
    add_mib_to_syspath()
    sys.path.insert(0, str(MIB_TRACK / "baselines" / "ioi_baselines"))

    # Apply patches before any harness module loads its pipeline /
    # config-dependent code. patch_lm_pipeline_load() takes effect
    # against the LMPipeline class once it's imported.
    from . import _patches
    _patches.patch_lm_pipeline_load()

    # Now safe to import the harness modules.
    from tasks.IOI_task.ioi_task import (  # type: ignore[import-not-found]
        get_causal_model,
        get_counterfactual_datasets,
        get_token_positions,
    )
    from CausalAbstraction.experiments.filter_experiment import FilterExperiment  # type: ignore[import-not-found]
    from CausalAbstraction.experiments.attention_head_experiment import PatchAttentionHeads  # type: ignore[import-not-found]
    from ioi_utils import (  # type: ignore[import-not-found]
        log_diff,
        clear_memory,
        checker as ioi_checker,
        filter_checker,
        setup_pipeline,
        get_model_config,
    )
    from sklearn.linear_model import LinearRegression
    import numpy as np

    if heads_list is None:
        heads_list = [(7, 3), (7, 9), (8, 6), (8, 10)]

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    causal_model = get_causal_model(
        {"bias": 0.0, "token_coeff": 0.0, "position_coeff": 0.0}
    )
    dataset_size = 10 if quick_test else None
    counterfactual_datasets = get_counterfactual_datasets(
        hf=True, size=dataset_size,
    )
    print("Available datasets:", list(counterfactual_datasets.keys()), flush=True)

    print(f"\n===== Computing parameters for model: {model_short} =====", flush=True)
    pipeline, batch_size = setup_pipeline(model_short, device, eval_batch_size)
    print(f"DEVICE: {pipeline.model.device}", flush=True)

    # Inject head_dim onto the model config if pyvene will need it (Qwen).
    _patches.patch_model_config_head_dim(pipeline.model.config)

    # Filter datasets through the model's correctness checker.
    print("\nFiltering datasets based on model performance...", flush=True)
    exp = FilterExperiment(pipeline, causal_model, filter_checker)
    filtered = exp.filter(counterfactual_datasets, verbose=True, batch_size=batch_size)

    token_positions = get_token_positions(pipeline, causal_model)

    if quick_test and len(heads_list) > 1:
        heads_list = heads_list[:1]

    pipeline.return_scores = True

    data_to_X = {
        "same_train": {"position": 1, "token": 1},
        "s1_io_flip_train": {"position": -1, "token": 1},
        "s2_io_flip_train": {"position": -1, "token": -1},
        "s1_ioi_flip_s2_ioi_flip_train": {"position": 1, "token": -1},
    }
    if quick_test:
        data_to_X = dict(list(data_to_X.items())[:2])

    X, y = [], []
    for cf_name, signals in data_to_X.items():
        if cf_name not in filtered:
            print(f"Warning: {cf_name} missing from filtered_datasets, skipping", flush=True)
            continue
        experiment = PatchAttentionHeads(
            pipeline=pipeline,
            causal_model=causal_model,
            layer_head_list=heads_list,
            token_positions=token_positions,
            checker=lambda logits, params: ioi_checker(logits, params, pipeline),
            config={
                "evaluation_batch_size": batch_size,
                "output_scores": True,
                "check_raw": True,
            },
        )
        raw_results = experiment.perform_interventions(
            {cf_name: filtered[cf_name]},
            target_variables_list=[["output_token"]],
            verbose=False,
        )
        # Drill through the harness's nested dict to get the raw_outputs.
        raw_outputs = None
        for v in raw_results["dataset"][cf_name].values():
            for v2 in v.values():
                raw_outputs = v2["raw_outputs"][0]

        for raw_logits, input_data in zip(raw_outputs, filtered[cf_name]):
            actual_diff = log_diff(
                raw_logits,
                causal_model.run_forward(input_data["input"]),
                pipeline,
            )
            y.append(actual_diff)
            X.append((signals["position"], signals["token"]))
        clear_memory()

    if not X:
        raise RuntimeError("No data points collected; bootstrap failed.")

    model = LinearRegression()
    X_t = torch.tensor(X)
    y_t = torch.tensor(y)
    model.fit(X_t, y_t)
    score = float(model.score(X_t, y_t))
    intercept = float(model.intercept_)
    position_coef = float(model.coef_[0])
    token_coef = float(model.coef_[1])

    model_config = get_model_config(model_short)
    new_entry = {
        "bias": intercept,
        "position_coeff": position_coef,
        "token_coeff": token_coef,
        "score": score,
        "model_name": model_config["model_path"],
    }

    # Merge with any existing JSON so multiple-model bootstraps accumulate
    # rather than overwrite. Match the example notebook's shape: top-level
    # entry per model_short, with optional "model_class" field.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file():
        try:
            existing = json.loads(output_path.read_text())
            if not isinstance(existing, dict):
                existing = {}
        except json.JSONDecodeError:
            existing = {}
    else:
        existing = {}
    existing[model_short] = new_entry
    results = existing

    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nLinear parameters saved to {output_path}", flush=True)
    print(
        f"  bias={intercept:.4f}  position_coeff={position_coef:.4f}  "
        f"token_coeff={token_coef:.4f}  R²={score:.4f}",
        flush=True,
    )
    return results
