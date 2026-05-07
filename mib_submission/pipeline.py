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
# ``MIB/MIB-causal-variable-track/verify_submission.py:TASKS``. Only the two
# we plan to submit on are wired here; extend as we grow.
_TASK_MODULES = {
    "4_answer_MCQA": "tasks.simple_MCQA.simple_MCQA",
    "arithmetic": "tasks.two_digit_addition_task.arithmetic",
}


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
    pipeline = LMPipeline(model_name, max_new_tokens=1, device=device, dtype=dtype)
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
