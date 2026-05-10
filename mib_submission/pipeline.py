"""
Wire MIB datasets, the LM pipeline, token positions, and a
``PatchResidualStream`` experiment so our alignment methods consume real
LM activations rather than our private ``mcqa_experiment/`` pair banks.

Usage::

    from mib_submission.pipeline import setup_residual_experiment

    bundle = setup_residual_experiment(
        task="4_answer_MCQA",
        model_name="Qwen/Qwen2.5-0.5B",
        layers=range(24),                      # all layers of the chosen LM
        dtype=torch.float16,
        target_variables=["answer_pointer"],
    )
    bundle.experiment   # PatchResidualStream wired to the LM
    bundle.train_data   # filtered counterfactual train splits
    bundle.test_data    # filtered counterfactual test splits

The MIB harness lives outside this repo (under ``MIB/``); this module assumes
``MIB/MIB-causal-variable-track`` and its ``CausalAbstraction`` submodule are
on ``sys.path``. The caller is responsible for that — either run from
inside the MIB venv with the right ``PYTHONPATH``, or call
``mib_submission.pipeline.add_mib_to_syspath()`` first.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
MIB_TRACK = REPO_ROOT / "MIB" / "MIB-causal-variable-track"


# Task module paths keyed by the task names in
# ``MIB/MIB-causal-variable-track/verify_submission.py:TASKS``. IOI is wired
# only at the dataset level here; ``setup_residual_experiment`` cannot build
# IOI bundles directly because ``ioi_task.get_causal_model`` requires a
# learned ``{bias, token_coeff, position_coeff}`` dict — see
# ``baselines/ioi_baselines/ioi_learn_linear_params.py`` for the bootstrap.
_TASK_MODULES = {
    "4_answer_MCQA": "tasks.simple_MCQA.simple_MCQA",
    "arithmetic": "tasks.two_digit_addition_task.arithmetic",
    "ARC_easy": "tasks.ARC.ARC",
    "ravel_task": "tasks.RAVEL.ravel",
}

# Tasks whose ``get_causal_model`` is parameterless. IOI is intentionally not
# in this set — its causal model depends on per-model linear params that must
# be learned first. See the IOI bootstrap script (out of scope here).
_PARAMETERLESS_CAUSAL_MODEL_TASKS = set(_TASK_MODULES)


# Map our task-name keys to the HF model-class names declared in
# ``verify_submission.py:VALID_TASK_MODELS``. The second tuple element is what
# ``pipeline.model.__class__.__name__`` will report once the model is loaded;
# we assert it matches what verify expects.
_HF_MODEL_TO_CLASS_NAME = {
    "gpt2": "GPT2LMHeadModel",
    "Qwen/Qwen2.5-0.5B": "Qwen2ForCausalLM",
    "google/gemma-2-2b": "Gemma2ForCausalLM",
    "meta-llama/Llama-3.1-8B": "LlamaForCausalLM",
}


def add_mib_to_syspath() -> None:
    """Idempotently insert MIB harness paths at the front of ``sys.path``."""
    for p in (str(MIB_TRACK), str(MIB_TRACK / "CausalAbstraction")):
        if p not in sys.path:
            sys.path.insert(0, p)


@dataclass
class ExperimentBundle:
    """Everything ``setup_residual_experiment`` returns.

    All fields are forward references to upstream classes — we keep this
    typed loosely (``object``) so this module can be imported without the
    MIB harness on the path.
    """

    task: str
    model_name: str
    model_class_name: str
    target_variables: List[str]
    layers: List[int]
    causal_model: object
    pipeline: object
    token_positions: list
    filtered_datasets: dict
    train_data: dict
    test_data: dict
    experiment: object


def _default_checker(output_text, expected) -> bool:
    return expected in output_text


def _split_filtered_datasets(filtered):
    """Mirror what ``example_submission.ipynb`` does to bucket datasets.

    The HF dataset keys come back like ``"answerPosition_train"`` /
    ``"answerPosition_test"`` / ``"answerPosition_validation"`` /
    ``"answerPosition_testprivate"``. We bucket by the trailing split name.
    """
    train = {k: v for k, v in filtered.items() if k.endswith("_train")}
    # Both public test and (when downloaded) private test go under test_data.
    test = {
        k: v
        for k, v in filtered.items()
        if k.endswith("_test") or k.endswith("_testprivate")
    }
    return train, test


def setup_residual_experiment(
    *,
    task: str,
    model_name: str,
    layers: Iterable[int],
    target_variables: List[str],
    dtype: torch.dtype = torch.float32,
    device: Optional[str] = None,
    dataset_size: Optional[int] = None,
    load_private_data: bool = False,
    config_overrides: Optional[dict] = None,
    checker=None,
    filter_batch_size: int = 32,
    verbose: bool = False,
    max_new_tokens: int = 1,
) -> ExperimentBundle:
    """Load dataset + model, filter, and build a PatchResidualStream experiment.

    The experiment is constructed without training any featurizer — each
    ResidualStream model_unit starts with the default identity Featurizer
    and ``feature_indices=None``. After running our alignment methods, call
    ``mib_submission.apply_results.apply_method_results`` to install
    Featurizer / indices, then ``experiment.save_featurizers(None, dir)`` to
    write the submission triplet.
    """
    add_mib_to_syspath()

    if task not in _TASK_MODULES:
        raise ValueError(f"Unknown task {task!r}; extend _TASK_MODULES.")

    task_mod = importlib.import_module(_TASK_MODULES[task])
    counterfactual_datasets = task_mod.get_counterfactual_datasets(
        hf=True, size=dataset_size, load_private_data=load_private_data
    )
    causal_model = task_mod.get_causal_model()

    from neural.pipeline import LMPipeline  # type: ignore[import-not-found]
    from experiments.filter_experiment import FilterExperiment  # type: ignore[import-not-found]
    from experiments.residual_stream_experiment import PatchResidualStream  # type: ignore[import-not-found]

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipeline = LMPipeline(model_name, max_new_tokens=max_new_tokens, device=device, dtype=dtype)
    pipeline.tokenizer.padding_side = "left"

    actual_class = pipeline.model.__class__.__name__
    expected_class = _HF_MODEL_TO_CLASS_NAME.get(model_name, actual_class)
    if expected_class != actual_class:
        raise RuntimeError(
            f"Model {model_name!r} loaded as {actual_class!r}, "
            f"but MIB expects {expected_class!r}. Submission will be rejected."
        )

    if checker is None:
        checker = _default_checker

    filter_exp = FilterExperiment(pipeline, causal_model, checker)
    filtered_datasets = filter_exp.filter(
        counterfactual_datasets,
        verbose=verbose,
        batch_size=filter_batch_size,
    )

    token_positions = task_mod.get_token_positions(pipeline, causal_model)
    train_data, test_data = _split_filtered_datasets(filtered_datasets)

    layers_list = sorted(int(l) for l in layers)
    config = {
        "batch_size": 64,
        "evaluation_batch_size": 1024,
        "training_epoch": 1,
        "n_features": 16,
        "regularization_coefficient": 0.0,
        "output_scores": False,
    }
    if config_overrides:
        config.update(config_overrides)

    experiment = PatchResidualStream(
        pipeline,
        causal_model,
        layers_list,
        token_positions,
        checker,
        config=config,
    )

    return ExperimentBundle(
        task=task,
        model_name=model_name,
        model_class_name=actual_class,
        target_variables=list(target_variables),
        layers=layers_list,
        causal_model=causal_model,
        pipeline=pipeline,
        token_positions=token_positions,
        filtered_datasets=filtered_datasets,
        train_data=train_data,
        test_data=test_data,
        experiment=experiment,
    )


# --------------------------------------------------------------------------- #
# IOI: attention-head experiment setup                                        #
# --------------------------------------------------------------------------- #

def setup_attention_head_experiment(
    *,
    model_name: str,
    layer_head_list: Iterable[Tuple[int, int]],
    target_variables: List[str],
    linear_params: dict,
    dtype: torch.dtype = torch.float16,
    device: Optional[str] = None,
    dataset_size: Optional[int] = None,
    load_private_data: bool = False,
    config_overrides: Optional[dict] = None,
    verbose: bool = False,
    per_site_units: bool = True,
) -> ExperimentBundle:
    """Parallel to ``setup_residual_experiment`` but for IOI cells.

    Differences from the residual-stream variant:

    1. **Causal model needs linear parameters** — the IOI causal model's
       ``logit_diff`` mechanism reads ``{bias, token_coeff, position_coeff}``.
       Pass them via ``linear_params``; bootstrap with
       ``mib_submission.ioi.bootstrap.bootstrap_linear_params``.

    2. **IOI-specific pipeline config** — `max_length=32`, `logit_labels=True`,
       `max_new_tokens=1` (matches `ioi_utils.setup_pipeline`'s non-special
       branch). The GPT-2 special branch (`position_ids=True`, fp32) is not
       wired here — see CLAUDE.md operational gotchas.

    3. **Sites = (layer, head)** — built from ``layer_head_list``. Token
       positions come from ``ioi_task.get_token_positions`` which returns a
       single ``id="all"`` position covering the full sequence.

    4. **Filter uses ``filter_checker`` from ``ioi_utils``** (not the
       default ``expected in output_text``).

    5. **Per-site sweep mode** (``per_site_units=True``, default) — the
       upstream ``PatchAttentionHeads`` builds ``model_units_lists`` as a
       single joint entry covering all heads. We re-shape it to one entry
       per (layer, head, position) so ``collect_neural_outputs`` can
       collect per-site signatures. Set ``per_site_units=False`` for joint
       interventions (used at DAS-train time once PLOT has picked sites).
    """
    add_mib_to_syspath()

    task = "ioi_task"
    task_mod = importlib.import_module("tasks.IOI_task.ioi_task")
    counterfactual_datasets = task_mod.get_counterfactual_datasets(
        hf=True, size=dataset_size, load_private_data=load_private_data,
    )
    # IOI's get_causal_model REQUIRES the parameters dict (bias / coeffs).
    causal_model = task_mod.get_causal_model(linear_params)

    from neural.pipeline import LMPipeline  # type: ignore[import-not-found]
    from experiments.filter_experiment import FilterExperiment  # type: ignore[import-not-found]
    from experiments.attention_head_experiment import PatchAttentionHeads  # type: ignore[import-not-found]

    # Use the harness IOI checker for filtering (substring match on output text).
    sys.path.insert(0, str(MIB_TRACK / "baselines" / "ioi_baselines"))
    from ioi_utils import filter_checker  # type: ignore[import-not-found]

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipeline = LMPipeline(
        model_name,
        max_new_tokens=1,
        device=device,
        dtype=dtype,
        max_length=32,
        logit_labels=True,
    )
    pipeline.tokenizer.padding_side = "left"

    actual_class = pipeline.model.__class__.__name__
    expected_class = _HF_MODEL_TO_CLASS_NAME.get(model_name, actual_class)
    if expected_class != actual_class:
        raise RuntimeError(
            f"Model {model_name!r} loaded as {actual_class!r}, "
            f"but MIB expects {expected_class!r}. Submission will be rejected."
        )

    filter_exp = FilterExperiment(pipeline, causal_model, filter_checker)
    filtered_datasets = filter_exp.filter(
        counterfactual_datasets,
        verbose=verbose,
        batch_size=64,
    )

    token_positions = task_mod.get_token_positions(pipeline, causal_model)
    train_data, test_data = _split_filtered_datasets(filtered_datasets)

    layer_head_list = list(layer_head_list)

    config = {
        "batch_size": 128,
        "evaluation_batch_size": 1024,
        "training_epoch": 2,
        "n_features": 32,
        "regularization_coefficient": 0.0,
        "output_scores": True,
        "shuffle": True,
        "temperature_schedule": (1.0, 0.01),
        "init_lr": 1.0,
        "check_raw": True,
    }
    if config_overrides:
        config.update(config_overrides)

    # Wire the IOI loss/metric function from the harness.
    from ioi_utils import ioi_loss_and_metric_fn, checker as ioi_checker  # type: ignore[import-not-found]
    config.setdefault(
        "loss_and_metric_fn",
        lambda pipe, intervenable, batch, units: ioi_loss_and_metric_fn(
            pipe, intervenable, batch, units,
        ),
    )

    experiment = PatchAttentionHeads(
        pipeline=pipeline,
        causal_model=causal_model,
        layer_head_list=layer_head_list,
        token_positions=token_positions,
        checker=lambda logits, params: ioi_checker(logits, params, pipeline),
        config=config,
    )

    if per_site_units:
        # Re-shape: PatchAttentionHeads ships with `model_units_lists =
        # [[ all_units ]]` — a single joint entry that intervenes on all
        # heads at once. PLOT needs per-site signatures; flatten to one
        # entry per (layer, head, token_pos).
        flat = []
        for entry in experiment.model_units_lists:
            for inner in entry:
                for unit in inner:
                    flat.append([[unit]])
        experiment.model_units_lists = flat

    return ExperimentBundle(
        task=task,
        model_name=model_name,
        model_class_name=actual_class,
        target_variables=list(target_variables),
        layers=sorted({L for L, _ in layer_head_list}),
        causal_model=causal_model,
        pipeline=pipeline,
        token_positions=token_positions,
        filtered_datasets=filtered_datasets,
        train_data=train_data,
        test_data=test_data,
        experiment=experiment,
    )
