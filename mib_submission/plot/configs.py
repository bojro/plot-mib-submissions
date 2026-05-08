"""Per-task run configuration presets for the PLOT submission driver.

Indexed by ``(task, variable)``. Each preset returns the full ``RunConfig``
that ``run.py:main`` consumes — model class name, OT row schema, signature
fit split, DAS hyperparameters. The point is that adding a new cell to the
rollout reduces to writing one new entry here, not editing ``run.py``.

Per-task notes
--------------

``4_answer_MCQA`` and ``ARC_easy`` share the same shape: 4 choice positions
(``A``..``D``), letter labels, and the ``answerPosition_*`` /
``randomLetter_*`` train splits. The OT row schema that worked for cell 1
(V=4 ``choice0..3``) ports directly. Same calibration variable. Note that
ARC's causal model has no ``choice`` variables — see ``_arc_v4_symbols``.

``ravel_task`` has a different shape and required real engineering before
it could be configured (see commit history + JOURNAL.md). Key facts:

- 3122 city entities; ~340 distinct attribute strings across
  ``Continent`` (6), ``Country`` (160), ``Language`` (174). Many countries
  and most languages are multi-token under Gemma's tokenizer.
- Each base example specifies a ``queried_attribute``; ``answer = attr_values
  [idx_of_queried_attribute]`` selects only that attribute's value. Patching
  any other attribute is a no-op for that base.
- The HF dataset's ``attribute_train`` split provides cross-attribute
  counterfactuals: each base is paired with a source whose
  ``queried_attribute`` differs. Bases are mixed across all 3 attributes.
- The baseline (``baselines/ravel_baselines.py``) uses
  ``n_features=288`` for Gemma, ``max_new_tokens=2``, a custom checker
  that handles multi-word answers and comma-separated alternative answers,
  and a per-attribute filtered train pool.

This config wires PLOT to use:
- ``answer_strings`` from ``causal_model.values["answer"]`` (the 340-element
  attribute vocabulary) instead of the 26-letter alphabet.
- ``per_row_filter_attribute="queried_attribute"`` so each OT row's signature
  is collected on the subset of bases where ``queried_attribute == row_variable``
  — eliminating the no-op-base SNR drag.
- ``n_features=288`` for the rotation matrix (matching baseline).
- ``max_new_tokens=2`` for the LM pipeline (multi-token answers).
- A custom checker (passed through ``setup_residual_experiment``) that
  handles multi-word answer matching.

``arithmetic`` has a single causal variable (``ones_carry``) and is at risk
of V=1 collapse — needs a workaround per CLAUDE.md, deferred.

``ioi_task`` is intentionally absent: ``get_causal_model`` requires learned
linear params (bootstrap step lives in
``baselines/ioi_baselines/ioi_learn_linear_params.py``). Wire after the
bootstrap is run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

from .pipeline import PlotConfig


_HF_MODEL_TO_CLASS_NAME = {
    "gpt2": "GPT2LMHeadModel",
    "Qwen/Qwen2.5-0.5B": "Qwen2ForCausalLM",
    "google/gemma-2-2b": "Gemma2ForCausalLM",
    "meta-llama/Llama-3.1-8B": "LlamaForCausalLM",
}


@dataclass(frozen=True)
class RunConfig:
    """Everything ``run.main`` needs to launch one cell."""

    task: str
    model_name: str
    model_class_name: str
    variable: str

    # PLOT site selection
    plot_config: PlotConfig
    signature_dataset: Optional[str] = None  # train-split key; None ⇒ first

    # DAS training
    n_features: int = 16
    training_epochs: int = 12
    init_lr: float = 1e-3
    train_batch_size: int = 32
    eval_batch_size: int = 256
    dataset_size: Optional[int] = 256

    # Layer set; None ⇒ all layers of the model
    layers: Optional[Tuple[int, ...]] = None

    # Pipeline / filter
    max_new_tokens: int = 1
    # ``checker`` is a callable taking (output_text, expected) → bool. ``None``
    # uses ``setup_residual_experiment``'s default (``expected in output_text``).
    # RAVEL needs a custom checker that handles multi-word + comma-list answers.
    checker: Optional[Callable[[Optional[str], str], bool]] = None

    # Diagnostic / advanced knobs
    bypass_sites: Optional[Tuple[Tuple[int, str], ...]] = None
    use_bucketed_plot: bool = False


# --------------------------------------------------------------------------- #
# RAVEL-specific helpers                                                      #
# --------------------------------------------------------------------------- #

def _ravel_checker(output_text: Optional[str], expected: str) -> bool:
    """Port of ``baselines/ravel_baselines.py:checker`` — handles RAVEL's
    multi-word answers, comma-separated alternative answers, and country-name
    edge cases. Required for the LM-correctness filter to keep enough examples
    when multi-token answers like "United States" or "South Korea" appear.
    """
    if output_text is None:
        return False
    output_clean = re.sub(r"[^\w\s]+", "", output_text.lower()).strip()
    expected_list = [e.strip().lower() for e in expected.split(",")]
    if any(part in output_clean for part in expected_list):
        return True
    if re.search(r"united states|united kingdom|czech republic", expected, re.IGNORECASE):
        raw_expected = expected.strip().lower().replace("the ", "")
        raw_output = output_text.strip().lower().replace("the ", "")
        if raw_output.startswith(raw_expected) or raw_output.startswith("england") or raw_output == "us":
            return True
    if re.search(r"south korea", expected, re.IGNORECASE):
        if output_clean.startswith("korea") or output_clean.startswith("south korea"):
            return True
    if re.search(r"persian|farsi", expected, re.IGNORECASE):
        if output_clean.startswith("persian") or output_clean.startswith("farsi"):
            return True
    if re.search(r"oceania", expected, re.IGNORECASE):
        if output_clean.startswith("australia"):
            return True
    if re.search(r"north america", expected, re.IGNORECASE):
        if "north america" in output_clean or output_clean == "na" or output_clean.startswith("america"):
            return True
    if re.search(r"mandarin|chinese", expected, re.IGNORECASE):
        if "chinese" in output_clean or "mandarin" in output_clean:
            return True
    return False


# --------------------------------------------------------------------------- #
# Preset PlotConfigs                                                          #
# --------------------------------------------------------------------------- #

def _mcqa_v4_choices(variable: str) -> PlotConfig:
    """MCQA V=4 OT rows on ``choice0..3`` (color-word swaps).

    Probes the pointer mechanism by swapping color words at each choice
    position. This is the same shape that hit 0.956 on cell 1 and 0.955 on
    cell 3. ARC does NOT have ``choice`` variables — see ``_arc_v4_symbols``.
    """
    return PlotConfig(
        variables=("choice0", "choice1", "choice2", "choice3"),
        calibration_variable=variable,
        letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        cost_metric="sq_l2",
        normalize_signatures=True,
        stage_a_solver="ot",
        stage_b_solver="ot",
        sinkhorn_iters=200,
        target_row_index=0,
        stage_a_epsilon_grid=(0.01, 0.03),
        stage_b_epsilon_grid=(0.003, 0.01, 0.03, 0.1),
        stage_a_top_k_grid=(1,),
        stage_b_top_k_grid=(1, 2),
    )


def _arc_v4_symbols(variable: str) -> PlotConfig:
    """ARC V=4 OT rows on ``symbol0..3`` (letter-label swaps).

    ARC's causal model has no ``choice`` variables — its prompts are science
    questions, not color/object MCQs. The natural V=4 schema is the four
    symbol positions, each a random letter A–Z. Each row's swap changes the
    answer letter only when ``answer_pointer == i`` (≈ 25% of examples per
    row), so signal magnitude is lower than MCQA's ``choice`` rows but the
    swap never breaks the causal model — 0% skip rate vs ~25% for MCQA.

    L2 row-normalisation in PLOT preserves direction at the cost of magnitude,
    so the lower row magnitude shouldn't hurt site selection materially.
    """
    return PlotConfig(
        variables=("symbol0", "symbol1", "symbol2", "symbol3"),
        calibration_variable=variable,
        letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        cost_metric="sq_l2",
        normalize_signatures=True,
        stage_a_solver="ot",
        stage_b_solver="ot",
        sinkhorn_iters=200,
        target_row_index=0,
        stage_a_epsilon_grid=(0.01, 0.03),
        stage_b_epsilon_grid=(0.003, 0.01, 0.03, 0.1),
        stage_a_top_k_grid=(1,),
        stage_b_top_k_grid=(1, 2),
    )


def _ravel_v3_attributes(variable: str) -> PlotConfig:
    """RAVEL V=3 OT rows on ``("Country", "Continent", "Language")``.

    Uses the new alphabet + per-row dataset filter machinery:

    - ``answer_alphabet_from_causal_model=True``: signature space is the
      LM-vocab first-token set of ``causal_model.values["answer"]`` (the 340
      attribute strings; collapsed to ~263 distinct first tokens after
      tokenizer collisions).
    - ``per_row_filter_attribute="queried_attribute"``: each row's signature
      is collected only on bases where ``input["queried_attribute"]`` matches
      the row's variable, eliminating the no-op-base SNR drag.
    - ``on_unknown_label="skip"``: tolerate rare LM outputs whose first token
      isn't in the attribute alphabet (rather than crashing — the train pool
      may include comma-list answers whose first token wasn't pre-registered).
    """
    if variable not in ("Country", "Continent", "Language"):
        raise ValueError(
            f"RAVEL variable {variable!r} must be one of (Country, Continent, Language)"
        )
    return PlotConfig(
        variables=("Country", "Continent", "Language"),
        calibration_variable=variable,
        # Don't use letters; alphabet comes from causal_model.values["answer"]
        letters="",
        answer_alphabet_from_causal_model=True,
        per_row_filter_attribute="queried_attribute",
        on_unknown_label="skip",
        cost_metric="sq_l2",
        normalize_signatures=True,
        stage_a_solver="ot",
        stage_b_solver="ot",
        sinkhorn_iters=200,
        # target_row_index points at the cell's variable for π-row reporting
        target_row_index=("Country", "Continent", "Language").index(variable),
        stage_a_epsilon_grid=(0.01, 0.03),
        stage_b_epsilon_grid=(0.003, 0.01, 0.03, 0.1),
        stage_a_top_k_grid=(1,),
        stage_b_top_k_grid=(1, 2),
    )


# --------------------------------------------------------------------------- #
# Public factory                                                              #
# --------------------------------------------------------------------------- #

def default_config(
    task: str,
    model_name: str,
    variable: str,
    *,
    overrides: Optional[dict] = None,
) -> RunConfig:
    """Build a ``RunConfig`` for the given cell.

    ``overrides`` keys may be: ``training_epochs``, ``n_features``,
    ``init_lr``, ``train_batch_size``, ``eval_batch_size``, ``dataset_size``,
    ``layers``, ``bypass_sites``, ``use_bucketed_plot``,
    ``signature_dataset``. Anything not in this set is rejected.
    """
    overrides = dict(overrides or {})

    if model_name not in _HF_MODEL_TO_CLASS_NAME:
        raise ValueError(
            f"Unknown model {model_name!r}. Add to "
            f"_HF_MODEL_TO_CLASS_NAME in mib_submission.plot.configs."
        )
    model_class_name = _HF_MODEL_TO_CLASS_NAME[model_name]

    if task == "4_answer_MCQA":
        plot_config = _mcqa_v4_choices(variable)
        signature_dataset = overrides.pop(
            "signature_dataset", "answerPosition_randomLetter_train"
        )
    elif task == "ARC_easy":
        plot_config = _arc_v4_symbols(variable)
        signature_dataset = overrides.pop(
            "signature_dataset", "answerPosition_randomLetter_train"
        )
    elif task == "ravel_task":
        plot_config = _ravel_v3_attributes(variable)
        signature_dataset = overrides.pop("signature_dataset", "attribute_train")
        # Mirror the RAVEL baseline's per-task overrides: bigger DAS subspace,
        # longer LM generation, custom checker for multi-word answers.
        # Caller can still override via ``overrides``.
        overrides.setdefault("n_features", 288)
        overrides.setdefault("training_epochs", 1)  # baseline uses 1 epoch
        overrides.setdefault("max_new_tokens", 2)
        overrides.setdefault("checker", _ravel_checker)
    elif task == "arithmetic":
        raise NotImplementedError(
            "arithmetic preset not implemented — single var, V=1 collapse risk; "
            "needs bucketed PLOT or adjacent-variable workaround. Defer."
        )
    elif task == "ioi_task":
        raise NotImplementedError(
            "ioi_task requires learned linear params bootstrap; not wired here."
        )
    else:
        raise ValueError(f"Unknown task {task!r}.")

    allowed = {
        "training_epochs", "n_features", "init_lr",
        "train_batch_size", "eval_batch_size", "dataset_size",
        "layers", "bypass_sites", "use_bucketed_plot",
        "max_new_tokens", "checker",
    }
    bad = set(overrides) - allowed
    if bad:
        raise ValueError(f"Unknown overrides: {sorted(bad)}")

    return RunConfig(
        task=task,
        model_name=model_name,
        model_class_name=model_class_name,
        variable=variable,
        plot_config=plot_config,
        signature_dataset=signature_dataset,
        **overrides,
    )
