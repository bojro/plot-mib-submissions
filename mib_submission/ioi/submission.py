"""IOI-specific submission writing helpers.

Differences from the residual-stream cells:

- **Nested folder layout**: featurizers go in
  ``{submission_root}/ioi_task_{MODEL}_{VAR}/DAS_{MODEL}_{VAR}/``,
  one level deeper than other cells. Verified by the example
  ``ioi_example_submission.ipynb`` notebook.
- **`ioi_linear_params.json`** must be present at the submission root
  (sibling of the per-cell folders). The harness's
  ``ioi_evaluate_submission.load_linear_params`` searches there first.
- The featurizers are saved by the upstream
  ``PatchAttentionHeads.save_featurizers(path)`` method itself — we
  don't construct the file names manually.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from . import LINEAR_PARAMS_FILENAME
from .bootstrap import LinearParams


def cell_dir(submission_root: Path, model_class_name: str, variable: str) -> Path:
    """Return the cell directory for one IOI submission.

    ``ioi_evaluate_submission.load_attention_head_featurizers`` does a flat
    ``os.listdir(task_folder_path)`` scan (no recursion), so we keep the
    AttentionHead files at the top of ``ioi_task_M_V/`` rather than under a
    nested ``DAS_M_V/`` subfolder. The example notebook's nested layout is
    for a different code path (``attention_head_baselines``-style training
    that wraps method name into the path); ``ioi_evaluate_submission_task``
    expects this flatter shape.
    """
    if variable not in ("output_token", "output_position"):
        raise ValueError(
            f"IOI variable must be 'output_token' or 'output_position'; got {variable!r}"
        )
    return Path(submission_root) / f"ioi_task_{model_class_name}_{variable}"


def ensure_linear_params_json(
    submission_root: Path,
    *,
    model_short: str,
    model_class_name: str,
    params: LinearParams,
) -> Path:
    """Write ``ioi_linear_params.json`` to the submission root.

    Format mirrors the example notebook (model-keyed entry + ``model_class``
    key at the top level for harness eval to pick up).
    """
    submission_root = Path(submission_root)
    submission_root.mkdir(parents=True, exist_ok=True)
    out_path = submission_root / LINEAR_PARAMS_FILENAME

    blob: dict = {
        model_short: {
            "bias": float(params.bias),
            "token_coeff": float(params.token_coeff),
            "position_coeff": float(params.position_coeff),
        },
        "model_class": model_class_name,
    }
    if params.score is not None:
        blob[model_short]["score"] = float(params.score)
    if params.model_name is not None:
        blob[model_short]["model_name"] = str(params.model_name)

    out_path.write_text(json.dumps(blob, indent=2))
    return out_path


def write_ioi_submission(
    submission_root: Path,
    *,
    experiment,
    model_class_name: str,
    variable: str,
    overwrite: bool = True,
) -> Path:
    """Write the AttentionHead featurizers for a trained IOI experiment.

    Parameters
    ----------
    submission_root : Path
        Top-level submission directory (e.g. ``submissions/plot``).
    experiment : PatchAttentionHeads (post-DAS-training)
        The experiment whose featurizers we save. The upstream
        ``save_featurizers(method_name=None, path=str)`` writes one
        ``AttentionHead(Layer-X,Head-Y,Token-T)_{featurizer,inverse_featurizer,indices}``
        triplet per active head.
    model_class_name : str
        e.g. ``"Gemma2ForCausalLM"``.
    variable : str
        ``"output_token"`` or ``"output_position"``.
    overwrite : bool
        If the cell directory already exists, remove it first.

    Returns
    -------
    Path
        The cell directory that was written.
    """
    import shutil

    out = cell_dir(submission_root, model_class_name, variable)
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} already exists; pass overwrite=True.")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # Upstream API: save_featurizers(method_name, path) writes per-head triplets.
    experiment.save_featurizers(None, str(out))
    return out
