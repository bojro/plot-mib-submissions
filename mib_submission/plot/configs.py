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

``arithmetic`` has a single causal variable (``ones_carry``) for scoring,
but the causal model exposes 10 nodes total. We satisfy V≥2 by picking OT
rows from non-target CM variables (allowed — RAVEL does the same; source
PLOT's default rows are intermediate carry bits, not the target output).

Two variants supported:

- ``arithmetic_variant="C"`` (default): V=2 from ``{tens_out, hundreds_out}``.
  Both are direct children of ``ones_carry`` in the SCM. This mirrors source
  PLOT's S_i + C_i row mixing on the binary GRU adder (S and C rows are
  always downstream of the carry chain). Patching a site that the model
  uses to compute the carry will perturb both rows; sites that compute
  only the ones digit will perturb neither (since ``ones_out`` is decoupled
  from the carry). Best causal alignment.

- ``arithmetic_variant="B"``: V=4 from ``{op1_ones, op2_ones, op1_tens,
  op2_tens}``. Operand-digit interchanges; analogous to source PLOT's
  ``flip_Ai`` / ``flip_Bi`` family rows. PLOT_SHORTCOMINGS §2 risk: these
  rows pick sites that *represent* operands, not sites that compute the
  carry. Kept as a fallback / diagnostic.

Calibration is on ``ones_carry`` regardless of variant.

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
        # Was (1, 2). Per PLOT_SHORTCOMINGS §8: ARC has only 2 token
        # positions per layer, so top_k_grid=(1,2) lets Stage B select
        # both positions per (row, layer), polluting the joint result
        # with weak/non-converging extras. Tightened to (1,) on
        # 2026-05-09 for D.7 — single best position per layer.
        stage_b_top_k_grid=(1,),
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


def _arithmetic_first_digit(out: str) -> str:
    """Take the first digit of an arithmetic output. Used as PlotConfig's
    ``label_from_output`` so signatures lookup against the digit alphabet
    works regardless of whether the answer is 1, 2, or 3 characters long.
    """
    s = out.strip()
    return s[:1] if s else ""


def _arithmetic_v2_carry_children(variable: str) -> PlotConfig:
    """Arithmetic V=2 OT rows on ``{tens_out, hundreds_out}`` (children of
    ``ones_carry``). Default variant.

    Both rows are SCM-downstream of the target ``ones_carry``. A site whose
    residual encodes the carry will perturb both rows when patched; a site
    that encodes only the ones-digit computation (decoupled from carry) will
    perturb neither — providing causal contrast.

    Two-token-position task (``op2_last``, ``last``) so we follow PLOT_
    SHORTCOMINGS §8: ``stage_b_top_k_grid=(1,)`` to force Stage B to actually
    select rather than degenerating to "keep both positions per layer".
    """
    if variable != "ones_carry":
        raise ValueError(
            f"arithmetic variable must be 'ones_carry' (only valid scoring "
            f"variable per verify_submission.py:21); got {variable!r}"
        )
    return PlotConfig(
        variables=("tens_out", "hundreds_out"),
        calibration_variable="ones_carry",
        letters="0123456789",
        # Arithmetic outputs are multi-char ("68", "168"); the alphabet is
        # the digit chars; project output to its first char.
        output_key="raw_output",
        label_from_output=_arithmetic_first_digit,
        cost_metric="sq_l2",
        normalize_signatures=True,
        stage_a_solver="ot",
        stage_b_solver="ot",
        sinkhorn_iters=200,
        target_row_index=0,
        stage_a_epsilon_grid=(0.01, 0.03),
        stage_b_epsilon_grid=(0.003, 0.01, 0.03, 0.1),
        stage_a_top_k_grid=(1,),
        stage_b_top_k_grid=(1,),  # 2-position task; (1,2) degenerates per §8
    )


def _arithmetic_v4_operands(variable: str) -> PlotConfig:
    """Arithmetic V=4 OT rows on ``{op1_ones, op2_ones, op1_tens, op2_tens}``.

    Diagnostic / fallback variant. Each row is an operand-digit interchange,
    analogous to source PLOT's ``flip_Ai`` / ``flip_Bi`` family rows. Risk
    per PLOT_SHORTCOMINGS §2: picks sites that represent operand digits,
    not sites that compute the carry — but the carry sits downstream of
    ``op{1,2}_ones`` so it should still appear in those rows' top picks.
    """
    if variable != "ones_carry":
        raise ValueError(
            f"arithmetic variable must be 'ones_carry'; got {variable!r}"
        )
    return PlotConfig(
        variables=("op1_ones", "op2_ones", "op1_tens", "op2_tens"),
        calibration_variable="ones_carry",
        letters="0123456789",
        output_key="raw_output",
        label_from_output=_arithmetic_first_digit,
        cost_metric="sq_l2",
        normalize_signatures=True,
        stage_a_solver="ot",
        stage_b_solver="ot",
        sinkhorn_iters=200,
        target_row_index=0,  # op1_ones — a parent of ones_carry
        stage_a_epsilon_grid=(0.01, 0.03),
        stage_b_epsilon_grid=(0.003, 0.01, 0.03, 0.1),
        stage_a_top_k_grid=(1,),
        stage_b_top_k_grid=(1,),
    )


def _ioi_names_alphabet(*, scan_size: int = 2000) -> Tuple[str, ...]:
    """Return the IOI name vocabulary as a tuple of strings — the alphabet
    the LM's first-token softmax is projected onto for IOI signatures.

    Sourced from the actual HF dataset (`mib-bench/ioi`) by scanning
    ``scan_size`` train examples and collecting every distinct
    ``name_A``/``name_B``/``name_C`` value, *unioned* with the harness's
    ``tasks/IOI_task/names.json`` for completeness.

    Why both: ``names.json`` ships ~99 names but the real ``mib-bench/ioi``
    dataset uses a different set (~44 distinct names across the splits we
    found, with only 10/44 overlapping ``names.json``). Scanning catches
    the actually-used names; ``names.json`` adds belt-and-suspenders.

    After ``resolve_tokens`` first-token compaction, this typically
    reduces to ~50–80 dims under modern tokenisers.
    """
    from ..pipeline import MIB_TRACK
    import json

    names: set[str] = set()

    # Pull from names.json (best-effort).
    names_path = MIB_TRACK / "tasks" / "IOI_task" / "names.json"
    if names_path.is_file():
        try:
            with names_path.open() as f:
                names.update(str(n).strip() for n in json.load(f) if str(n).strip())
        except Exception:
            pass

    # Pull from the actual dataset.
    try:
        import sys
        sys.path.insert(0, str(MIB_TRACK))
        sys.path.insert(0, str(MIB_TRACK / "CausalAbstraction"))
        from tasks.IOI_task.ioi_task import get_counterfactual_datasets  # type: ignore[import-not-found]

        datasets = get_counterfactual_datasets(hf=True, size=scan_size)
        # Pick any one split (they all share the same name vocab).
        ds = next(iter(datasets.values()))
        for ex in ds:
            inp = ex["input"]
            for key in ("name_A", "name_B", "name_C"):
                v = inp.get(key)
                if isinstance(v, str) and v.strip():
                    names.add(v.strip())
    except Exception as e:
        # If dataset scan fails (no HF cache, no network), fall back to
        # whatever names.json gave us. The alphabet may be incomplete but
        # the run can still proceed with on_unknown_label="skip".
        print(f"[ioi-config] dataset scan failed ({e}); using names.json only.")

    if not names:
        raise RuntimeError(
            "Could not assemble IOI name alphabet — both names.json and "
            "dataset scan failed."
        )
    return tuple(sorted(names))


def _ioi_v3_splits(variable: str) -> PlotConfig:
    """IOI V=3 OT rows on the 3 non-``same`` counterfactual splits.

    Each row interchanges ``variable`` (``output_token`` or
    ``output_position``) on a different split's source distribution. The
    splits sit at three distinct corners of the
    ``(token_signal, position_signal)`` grid (see
    ``ioi_learn_linear_params.py:88-93``):

    | split                      | (pos, tok)  |
    |----------------------------|-------------|
    | s1_io_flip_train           | (-1, +1)    |
    | s2_io_flip_train           | (-1, -1)    |
    | s1_ioi_flip_s2_ioi_flip_t. | (+1, -1)    |

    The fourth corner (``same``, +1 +1) gives a zero abstract row and is
    excluded.

    Signature alphabet: the IOI name vocabulary (~40 names). After
    ``resolve_tokens`` compaction this typically lands at ~30–40 dims.

    DAS hyperparameters match ``ioi_baselines.py``: ``n_features=32``,
    ``epochs=2``, ``init_lr=1.0``, ``loss_and_metric_fn=ioi_loss_and_metric_fn``
    (set in run.py's IOI branch since it imports the harness module at
    runtime).
    """
    if variable not in ("output_token", "output_position"):
        raise ValueError(
            f"IOI variable must be 'output_token' or 'output_position'; "
            f"got {variable!r}"
        )
    splits = ("s1_io_flip_train", "s2_io_flip_train", "s1_ioi_flip_s2_ioi_flip_train")
    return PlotConfig(
        # ``variables`` here are the OT-row LABELS (split short names) for
        # reporting; the actual interchange uses ``calibration_variable``.
        variables=tuple(s.replace("_train", "") for s in splits),
        per_row_split_datasets=splits,
        calibration_variable=variable,
        letters="",
        answer_strings=_ioi_names_alphabet(),
        # IOI's causal model exposes its output as ``raw_output`` (the
        # predicted name string), not ``answer``. The label is the full
        # name verbatim — no projection needed (no ``label_from_output``).
        output_key="raw_output",
        on_unknown_label="skip",  # rare LM outputs may fall outside name vocab
        cost_metric="sq_l2",
        normalize_signatures=True,
        stage_a_solver="ot",
        stage_b_solver="ot",
        sinkhorn_iters=200,
        target_row_index=0,
        stage_a_epsilon_grid=(0.01, 0.03),
        stage_b_epsilon_grid=(0.003, 0.01, 0.03, 0.1),
        stage_a_top_k_grid=(1,),
        # IOI's Stage B picks heads within layer; with a single TokenPosition
        # ``id="all"`` and many heads, top_k=1 forces the OT to actually
        # select instead of keeping every head.
        stage_b_top_k_grid=(1,),
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
        variant = overrides.pop("arithmetic_variant", "C")
        if variant == "C":
            plot_config = _arithmetic_v2_carry_children(variable)
        elif variant == "B":
            plot_config = _arithmetic_v4_operands(variable)
        else:
            raise ValueError(
                f"Unknown arithmetic_variant {variant!r}; expected 'C' or 'B'."
            )
        signature_dataset = overrides.pop("signature_dataset", "random_train")
        # Baseline (arithmetic_baselines.py:78-82,119-125): n_features=16,
        # epochs=1, max_new_tokens=3 for Gemma / 1 for Llama. We default to
        # max_new_tokens=3 (covers Gemma's multi-token answers); Llama can
        # override down to 1 via CLI when we get to that cell.
        overrides.setdefault("n_features", 16)
        overrides.setdefault("training_epochs", 1)
        overrides.setdefault("max_new_tokens", 3)
        # Default smoke dataset_size; baseline uses 10_000 but smoke is fine
        # for first cell. Override via --dataset-size for the real run.
        overrides.setdefault("dataset_size", 256)
    elif task == "ioi_task":
        plot_config = _ioi_v3_splits(variable)
        # Signature dataset is overridden by the per-row split mode; pass
        # any train split as the placeholder ``fit_dataset`` argument.
        # The IOI variant of run.py routes to setup_attention_head_experiment
        # which uses the per-row datasets directly.
        signature_dataset = overrides.pop(
            "signature_dataset", "s1_io_flip_train",
        )
        # Match ioi_baselines.py:144-157 hyperparameters.
        overrides.setdefault("n_features", 32)
        overrides.setdefault("training_epochs", 2)
        overrides.setdefault("init_lr", 1.0)
        overrides.setdefault("max_new_tokens", 1)
        overrides.setdefault("dataset_size", 256)  # smoke; baseline=full
        # Conservative train batch size for 8 GB VRAM.
        overrides.setdefault("train_batch_size", 32)
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
