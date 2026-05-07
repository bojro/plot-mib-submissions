"""
Collect residual-stream activations on the *factual* inputs of a dataset, at
every (layer, token_position) pair declared by a ``PatchResidualStream``
experiment.

Used to fit data-driven featurizers that don't depend on counterfactuals — the
canonical case is PCA fitting for the OT+PCA method. For methods that need
per-site logit-shift signatures (OT, GW, FGW, UOT), use ``signatures.py``
instead (added in Step 5); upstream's ``_collect_features`` returns base and
counterfactual activations interleaved into a single tensor in a way that's
hard to recover post hoc, so we deliberately do not expose a
collect-with-counterfactuals path here.

Shape contract::

    {(layer: int, token_position_id: str): Tensor of shape (n_samples, hidden_dim)}
"""

from __future__ import annotations

from itertools import chain
from typing import Dict, Tuple

import torch

from .pipeline import ExperimentBundle, add_mib_to_syspath
from .site_keys import site_key_for_unit


SiteKey = Tuple[int, str]
ActivationDict = Dict[SiteKey, torch.Tensor]


def _zipped_units(experiment) -> list:
    """Mirror ``build_SVD_feature_interventions``'s flattening pattern.

    The triple-nested ``model_units_lists`` layout (experiments × counterfactual
    groups × units) gets transposed and flattened so ``_collect_features``
    receives one list of lists ready for batching. We replicate exactly so
    output indexing stays identical.
    """
    return [
        list(chain.from_iterable(units_per_group))
        for units_per_group in zip(*experiment.model_units_lists)
    ]


def collect_base_activations(
    bundle: ExperimentBundle,
    dataset,
    *,
    verbose: bool = False,
) -> ActivationDict:
    """Run the LM over ``dataset`` and gather residual-stream activations at
    every (layer, token_position) site declared by ``bundle.experiment``.

    Only the factual inputs are processed; counterfactuals are ignored.

    Parameters
    ----------
    bundle :
        Output of ``setup_residual_experiment``.
    dataset :
        A single ``CounterfactualDataset`` (e.g.
        ``bundle.train_data["answerPosition_train"]``).

    Returns
    -------
    dict[(layer, tok_id), Tensor]
        Hidden states on the factual inputs, one tensor per declared site.
    """
    add_mib_to_syspath()
    from experiments.pyvene_core import _collect_features  # type: ignore[import-not-found]

    experiment = bundle.experiment
    zipped = _zipped_units(experiment)

    raw = _collect_features(
        dataset,
        bundle.pipeline,
        zipped,
        experiment.config,
        collect_counterfactuals=False,
        verbose=verbose,
    )

    out: ActivationDict = {}
    for group_idx, units in enumerate(zipped):
        cell = raw[group_idx]
        for unit_idx, unit in enumerate(units):
            out[site_key_for_unit(unit)] = cell[unit_idx]
    return out
