"""
Serialise ``MethodResult`` objects into MIB's expected submission layout::

    submission_root/
        token_position.py               (task-specific; only if caller supplies it)
        {TASK}_{MODEL}_{VARIABLE}/
            ResidualStream(Layer-{L},Token-{T})_featurizer
            ResidualStream(Layer-{L},Token-{T})_inverse_featurizer
            ResidualStream(Layer-{L},Token-{T})_indices
            ...                                          (one triplet per layer)

We deliberately do NOT ship a custom ``featurizer.py``. Every method we
benchmark maps onto upstream's ``IdentityFeaturizerModule`` /
``SubspaceFeaturizerModule``, both of which ``Featurizer.load_modules``
already deserialises; shipping a custom file would only force shipping a
matching ``token_position.py`` (verify_submission errors otherwise) without
adding any capability.

The cell folder name and model-unit id strings must match exactly what
``verify_submission.py`` and ``ResidualStream.load_modules`` expect — see
``MIB/MIB-causal-variable-track/CausalAbstraction/neural/LM_units.py:94``
for the id format and ``verify_submission.py:VALID_TASK_MODELS`` for the
allowed (task, model) pairs.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional

from .method_to_featurizer import MethodResult


VALID_TASK_MODELS = {
    ("ioi_task", "GPT2LMHeadModel"),
    ("ioi_task", "Qwen2ForCausalLM"),
    ("ioi_task", "Gemma2ForCausalLM"),
    ("ioi_task", "LlamaForCausalLM"),
    ("4_answer_MCQA", "Qwen2ForCausalLM"),
    ("4_answer_MCQA", "Gemma2ForCausalLM"),
    ("4_answer_MCQA", "LlamaForCausalLM"),
    ("ARC_easy", "Gemma2ForCausalLM"),
    ("ARC_easy", "LlamaForCausalLM"),
    ("arithmetic", "Gemma2ForCausalLM"),
    ("arithmetic", "LlamaForCausalLM"),
    ("ravel_task", "Gemma2ForCausalLM"),
    ("ravel_task", "LlamaForCausalLM"),
}

TASK_VARIABLES = {
    "ioi_task": {"output_token", "output_position"},
    "4_answer_MCQA": {"answer_pointer", "answer"},
    "ARC_easy": {"answer_pointer", "answer"},
    "arithmetic": {"ones_carry"},
    "ravel_task": {"Country", "Continent", "Language"},
}


def model_unit_id(layer: int, token_position_id: str) -> str:
    """Mirror ``ResidualStream.__init__``'s id format."""
    return f"ResidualStream(Layer-{layer},Token-{token_position_id})"


def cell_folder_name(task: str, model_class_name: str, variable: str) -> str:
    if (task, model_class_name) not in VALID_TASK_MODELS:
        raise ValueError(
            f"({task!r}, {model_class_name!r}) is not in MIB's "
            f"VALID_TASK_MODELS — submission would be silently skipped."
        )
    if variable not in TASK_VARIABLES.get(task, set()):
        raise ValueError(
            f"variable {variable!r} is not declared for task {task!r}."
        )
    return f"{task}_{model_class_name}_{variable}"


def write_submission(
    submission_root: str | os.PathLike,
    *,
    task: str,
    model_class_name: str,
    variable: str,
    results: Iterable[MethodResult],
    token_position_py: Optional[str | os.PathLike] = None,
    overwrite: bool = False,
) -> Path:
    """Write all ``results`` for a single (task, model, variable) cell.

    Parameters
    ----------
    submission_root :
        Top-level submission directory. Created if missing.
    task, model_class_name, variable :
        Used to build the cell folder name. Validated against MIB's allowed
        combinations.
    results :
        ``MethodResult`` objects, one per (layer, token_position) pair.
        ``result.variable`` must equal ``variable``; mismatched results are an error.
    token_position_py :
        Optional path to a ``token_position.py`` file to copy into the
        submission root. If omitted, the evaluator falls back to baseline
        token positions for the task (per the upstream README).
    overwrite :
        If True, an existing cell folder is removed before writing. False by
        default — fails loudly on collision.

    Returns
    -------
    Path
        Path to the cell folder that was written.
    """
    submission_root = Path(submission_root)
    submission_root.mkdir(parents=True, exist_ok=True)

    # We deliberately do NOT ship a custom featurizer.py: every method maps onto
    # upstream's IdentityFeaturizerModule / SubspaceFeaturizerModule, which the
    # evaluator already knows how to deserialise. verify_submission.py errors
    # if any .py file lacks `get_token_positions`, so we only ship a
    # token_position.py when the caller explicitly supplies one — and never
    # without an accompanying valid token_position.py.
    if token_position_py is not None:
        shutil.copy(token_position_py, submission_root / "token_position.py")

    cell_dir = submission_root / cell_folder_name(task, model_class_name, variable)
    if cell_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{cell_dir} already exists; pass overwrite=True.")
        shutil.rmtree(cell_dir)
    cell_dir.mkdir(parents=True)

    written = 0
    for result in results:
        if result.variable != variable:
            raise ValueError(
                f"MethodResult variable {result.variable!r} does not match "
                f"target variable {variable!r}."
            )
        base = cell_dir / model_unit_id(result.layer, result.token_position_id)
        # Featurizer.save_modules writes "{base}_featurizer" and "{base}_inverse_featurizer".
        result.featurizer.save_modules(str(base))
        with open(str(base) + "_indices", "w") as f:
            json.dump(
                None if result.indices is None else [int(i) for i in result.indices],
                f,
            )
        written += 1

    if written == 0:
        raise ValueError("No MethodResults provided; cell folder would be empty.")

    return cell_dir
