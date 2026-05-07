"""
Install ``MethodResult``s onto a ``PatchResidualStream`` experiment, then save
the resulting featurizers in MIB's expected on-disk layout.

This is an alternative to ``serialize.write_submission`` that goes through
upstream's ``experiment.save_featurizers`` instead of writing the triplet
files ourselves. Use it when you already have an ``ExperimentBundle``
in memory (the typical post-step-4 path); use ``serialize.write_submission``
when you only have ``MethodResult``s and want to write a folder without
spinning up the LM.

Both paths produce the same on-disk layout — the upstream
``intervention_experiment.save_featurizers`` writes
``{ResidualStream(Layer-L,Token-T)}_{featurizer,inverse_featurizer,indices}``
under the supplied directory, which is exactly what
``verify_submission.py`` and the evaluator expect.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable

from .method_to_featurizer import MethodResult
from .pipeline import ExperimentBundle
from .serialize import VALID_TASK_MODELS, TASK_VARIABLES, cell_folder_name
from .site_keys import site_key_for_unit


def apply_method_results(
    bundle: ExperimentBundle,
    results: Iterable[MethodResult],
    *,
    submission_root: str | os.PathLike,
    variable: str,
    overwrite: bool = False,
) -> Path:
    """Wire ``results`` onto matching units in ``bundle.experiment`` and save.

    Each ``MethodResult.layer`` × ``MethodResult.token_position_id`` pair must
    match a unit declared by the experiment — otherwise we fail loudly. After
    installing the featurizer / indices on each unit we call
    ``experiment.save_featurizers`` so upstream owns the file naming and JSON
    formatting (matching what the leaderboard evaluator expects).

    Parameters
    ----------
    overwrite :
        If True and the cell folder already exists, it is removed before
        writing — matches ``serialize.write_submission``'s collision behavior.
        If False, the cell folder is created with ``exist_ok=True`` and any
        previously written triplets in it are left alone (and may be silently
        replaced file-by-file as ``save_featurizers`` writes).
    """
    if (bundle.task, bundle.model_class_name) not in VALID_TASK_MODELS:
        raise ValueError(
            f"({bundle.task!r}, {bundle.model_class_name!r}) not in VALID_TASK_MODELS — "
            "submission would be silently rejected."
        )
    if variable not in TASK_VARIABLES.get(bundle.task, set()):
        raise ValueError(
            f"variable {variable!r} not declared for task {bundle.task!r}."
        )

    cell_dir = Path(submission_root) / cell_folder_name(
        bundle.task, bundle.model_class_name, variable
    )
    if cell_dir.exists() and overwrite:
        shutil.rmtree(cell_dir)
    cell_dir.mkdir(parents=True, exist_ok=True)

    units_by_key = {}
    for outer in bundle.experiment.model_units_lists:
        for group in outer:
            for unit in group:
                units_by_key[site_key_for_unit(unit)] = unit

    touched_units = []
    for r in results:
        if r.variable != variable:
            raise ValueError(
                f"MethodResult variable {r.variable!r} != target {variable!r}."
            )
        key = (r.layer, r.token_position_id)
        unit = units_by_key.get(key)
        if unit is None:
            raise KeyError(
                f"No model_unit at {key} on this experiment. Available: "
                f"{sorted(units_by_key.keys())}"
            )
        unit.set_featurizer(r.featurizer)
        unit.set_feature_indices(r.indices)
        touched_units.append(unit)

    if not touched_units:
        raise ValueError("No MethodResults provided.")

    bundle.experiment.save_featurizers(touched_units, str(cell_dir))
    return cell_dir
