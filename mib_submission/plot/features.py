"""Output-space effect signatures for the PLOT pipeline.

For each (variable, site) pair we measure the *output-prob delta* the
intervention produces, then aggregate across examples. The result is a small
(num_letters,) vector per row — bounded magnitude, well-suited to L2
normalisation and squared-L2 cost. This is the MIB analog of
``experiments/binary_addition_rnn/features.py`` on the
``codex/binary-addition-two-stage-plot`` branch.

Two essential differences from earlier ``answer_logit_delta``-based attempts:

1. **Aggregate across examples**: each signature row is a *single*
   (num_letters,)-length vector — not a (N · num_letters,) flattened tensor.
   Averaging collapses the batch axis the way the source's
   ``aggregate_mean`` does, dropping per-example noise and shrinking the
   cost-matrix dimension by orders of magnitude.

2. **Probability space, not logit space**: we read ``softmax(logits)`` and
   the abstract row is a one-hot diff. Both live in ``[-1, 1]`` per element,
   making L2 normalisation produce well-conditioned costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ..pipeline import ExperimentBundle, add_mib_to_syspath
from ..site_keys import site_key_for_unit
from ._alphabets import LabelAlphabet, from_letters, resolve_tokens


SiteKey = Tuple[int, str]


def _resolve_alphabet(
    alphabet: Optional[LabelAlphabet], letters: Optional[str]
) -> LabelAlphabet:
    """Coerce the legacy ``letters: str`` arg into a ``LabelAlphabet`` if needed.

    Functions accept either ``alphabet`` (the new explicit form) or ``letters``
    (legacy MCQA/ARC convenience). At least one must be provided.
    """
    if alphabet is not None:
        if letters:
            raise ValueError("provide either `alphabet` or `letters`, not both")
        return alphabet
    if letters:
        return from_letters(letters)
    raise ValueError("must provide either `alphabet` or `letters`")


def _iter_input_dicts(dataset) -> List[dict]:
    """Yield raw input dicts from a CounterfactualDataset (for filtering)."""
    inner = getattr(dataset, "dataset", None)
    if inner is not None and hasattr(inner, "__len__") and hasattr(inner, "__getitem__"):
        return [inner[i] for i in range(len(inner))]
    return list(dataset)


def _filter_dataset(dataset, predicate: Callable[[dict], bool]):
    """Return a new CounterfactualDataset with examples passing ``predicate``.

    Falls back to a list-of-dicts when the underlying object isn't a real HF
    dataset (used in tests that pass python dicts directly).
    """
    inner = getattr(dataset, "dataset", None)
    if inner is None or not (hasattr(inner, "filter") or hasattr(inner, "select")):
        return [ex for ex in dataset if predicate(ex)]
    if hasattr(inner, "filter"):
        from copy import copy as _copy
        new = _copy(dataset)
        new.dataset = inner.filter(predicate)
        return new
    indices = [i for i in range(len(inner)) if predicate(inner[i])]
    from copy import copy as _copy
    new = _copy(dataset)
    new.dataset = inner.select(indices)
    return new


@dataclass
class NeuralOutputs:
    """Cached per-site outputs needed by both signature and IIA computations.

    Shapes::

        base_alpha_probs:   (N, K)
        base_alpha_argmax:  (N,)
        cf_alpha_probs:     {(layer, tok_id): (N, K)}
        cf_alpha_argmax:    {(layer, tok_id): (N,)}

    Letter indices are into ``letters`` (the alphabet string). Computing the
    argmax over the alphabet rather than the full vocab matches the
    source PLOT's evaluation, which compares to the model's task-output
    distribution rather than its raw next-token distribution.
    """

    base_alpha_probs: torch.Tensor
    base_alpha_argmax: torch.Tensor
    cf_alpha_probs: Dict[SiteKey, torch.Tensor]
    cf_alpha_argmax: Dict[SiteKey, torch.Tensor]
    alphabet: LabelAlphabet

    @property
    def letters(self) -> str:
        """Legacy accessor: only meaningful if the alphabet is a single-char one."""
        if all(len(lab) == 1 for lab in self.alphabet.labels):
            return "".join(self.alphabet.labels)
        return ""


def aggregate_mean(rows: Sequence[torch.Tensor]) -> torch.Tensor:
    """Mirror ``features.aggregate_mean`` from the source branch."""
    if not rows:
        raise ValueError("need at least one row to aggregate")
    return torch.stack([r.to(torch.float32) for r in rows], dim=0).mean(dim=0)


def normalize_rows(M: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    """L2-normalise each row; rows with norm ≤ eps are left at zero."""
    norms = M.norm(dim=-1, keepdim=True).clamp_min(eps)
    return M / norms


def _causal_letter_pairs(
    causal_model, dataset, *, variable: str, output_key: str = "answer",
    label_from_output: Optional[Callable[[str], str]] = None,
) -> Tuple[List[str], List[str]]:
    """Per-example (base_label, source_label) under interchange of ``variable``.

    Skips examples where the interchange leaves a downstream variable
    undefined (e.g. ``choice_i`` swap that removes the question's color from
    the choice list, leaving pointer = None). This loses some examples per
    row but is the only safe option without filtering the dataset upstream.

    Despite the legacy "letter" naming, the returned strings are stripped
    versions of ``causal_model.run_*()[output_key]`` — for MCQA/ARC these are
    single letters from ``"answer"``; for RAVEL these are word strings from
    ``"answer"``; for arithmetic these are digit strings from ``"raw_output"``.
    """
    base_letters: List[str] = []
    source_letters: List[str] = []
    n_skipped = 0
    for example in dataset:
        try:
            base_out = causal_model.run_forward(example["input"])
            cf_setting = causal_model.run_interchange(
                example["input"],
                {variable: example["counterfactual_inputs"][0]},
            )
            base_letter = str(base_out[output_key]).strip()
            source_letter = str(cf_setting[output_key]).strip()
            if label_from_output is not None:
                base_letter = label_from_output(base_letter)
                source_letter = label_from_output(source_letter)
        except (TypeError, KeyError, IndexError):
            n_skipped += 1
            continue
        base_letters.append(base_letter)
        source_letters.append(source_letter)
    if n_skipped:
        # Light-touch reporting; not a hard error since the row can still be
        # computed from the surviving examples. If the skip-rate is huge,
        # the caller should pick a different variable.
        print(f"[features] _causal_letter_pairs(variable={variable!r}): "
              f"skipped {n_skipped} / {n_skipped + len(base_letters)} examples")
    return base_letters, source_letters


def build_abstract_effect_row(
    causal_model,
    dataset,
    *,
    variable: str,
    alphabet: Optional[LabelAlphabet] = None,
    letters: Optional[str] = None,
    normalize: bool = True,
    dataset_filter: Optional[Callable[[dict], bool]] = None,
    on_unknown_label: str = "raise",  # "raise" | "skip"
    output_key: str = "answer",
    label_from_output: Optional[Callable[[str], str]] = None,
) -> torch.Tensor:
    """One-row abstract-effect signature for a single OT variable.

    For each example: compute ``one_hot(source_label) - one_hot(base_label)``
    over the alphabet, then average. Optionally L2-normalise the result.
    Output shape: ``(alphabet.num_dims,)``.

    Parameters
    ----------
    alphabet, letters :
        Use either. ``letters`` is the legacy MCQA/ARC convenience.
    dataset_filter :
        If provided, examples are filtered before computing the row. Used
        for per-row dataset slicing (e.g. RAVEL: row ``Country`` only sees
        bases where ``queried_attribute == "Country"``).
    on_unknown_label :
        ``"raise"`` is the strict legacy behaviour. ``"skip"`` drops examples
        whose answer string isn't in the alphabet (useful when the alphabet
        was built from the causal model's declared answer set but real LM
        outputs include rare unseen variants).
    """
    alpha = _resolve_alphabet(alphabet, letters)
    ds = dataset if dataset_filter is None else _filter_dataset(dataset, dataset_filter)
    base_letters, source_letters = _causal_letter_pairs(
        causal_model, ds, variable=variable, output_key=output_key,
        label_from_output=label_from_output,
    )
    K = alpha.num_dims
    rows: List[torch.Tensor] = []
    n_unknown = 0
    for b, s in zip(base_letters, source_letters):
        b_idx = alpha.label_to_dim.get(b)
        s_idx = alpha.label_to_dim.get(s)
        if b_idx is None or s_idx is None:
            if on_unknown_label == "raise":
                raise ValueError(
                    f"Label {b!r} or {s!r} not in alphabet ({K} dims); "
                    "widen the alphabet or pass on_unknown_label='skip'."
                )
            n_unknown += 1
            continue
        row = torch.zeros(K, dtype=torch.float32)
        row[s_idx] += 1.0
        row[b_idx] -= 1.0
        rows.append(row)
    if n_unknown:
        print(f"[features] build_abstract_effect_row(variable={variable!r}): "
              f"skipped {n_unknown} examples with labels outside the alphabet")
    if not rows:
        # Fallback: return a zero row rather than raising. The caller will
        # see an L2-normless row and can decide whether to drop it.
        zero = torch.zeros(K, dtype=torch.float32)
        return zero
    aggregated = aggregate_mean(rows)
    if normalize:
        aggregated = normalize_rows(aggregated.unsqueeze(0)).squeeze(0)
    return aggregated


def build_abstract_table(
    causal_model,
    dataset,
    *,
    variables: Sequence[str],
    alphabet: Optional[LabelAlphabet] = None,
    letters: Optional[str] = None,
    normalize: bool = True,
    per_row_dataset_filter: Optional[Sequence[Optional[Callable[[dict], bool]]]] = None,
    on_unknown_label: str = "raise",
    output_key: str = "answer",
    label_from_output: Optional[Callable[[str], str]] = None,
) -> torch.Tensor:
    """Stack one abstract row per variable. Shape ``(V, alphabet.num_dims)``.

    ``per_row_dataset_filter`` (if given) is a sequence of per-row predicates
    aligned with ``variables`` — each row's signature is computed on the
    examples passing its predicate. Use ``None`` for a row to disable
    filtering on that row only.
    """
    if per_row_dataset_filter is not None and len(per_row_dataset_filter) != len(variables):
        raise ValueError(
            f"per_row_dataset_filter length {len(per_row_dataset_filter)} "
            f"!= len(variables) {len(variables)}"
        )
    rows = []
    for i, v in enumerate(variables):
        ds_filter = per_row_dataset_filter[i] if per_row_dataset_filter is not None else None
        rows.append(build_abstract_effect_row(
            causal_model, dataset,
            variable=v,
            alphabet=alphabet, letters=letters,
            normalize=normalize,
            dataset_filter=ds_filter,
            on_unknown_label=on_unknown_label,
            output_key=output_key,
            label_from_output=label_from_output,
        ))
    return torch.stack(rows, dim=0)


def collect_neural_outputs(
    bundle: ExperimentBundle,
    dataset,
    *,
    alphabet: Optional[LabelAlphabet] = None,
    letters: Optional[str] = None,
    batch_size: int = 32,
    verbose: bool = False,
) -> NeuralOutputs:
    """Single forward-pass collection of base + per-site interchange outputs.

    Every downstream PLOT primitive (effect signatures, IIA-based
    calibration) reads from the cached probabilities here, so we only run
    the model once per site per dataset.
    """
    add_mib_to_syspath()
    from experiments.pyvene_core import _run_interchange_interventions  # type: ignore[import-not-found]

    alpha = _resolve_alphabet(alphabet, letters)
    if not alpha.has_tokens:
        alpha = resolve_tokens(alpha, bundle.pipeline.tokenizer)
    tok_ids = alpha.token_ids_tensor()

    base_chunks: List[torch.Tensor] = []
    n = len(dataset.dataset)
    for start in range(0, n, batch_size):
        batch = [dataset.dataset[i] for i in range(start, min(start + batch_size, n))]
        bases = [ex["input"] for ex in batch]
        out = bundle.pipeline.generate(bases)
        last_logits = out["scores"][0]
        base_chunks.append(torch.softmax(last_logits, dim=-1)[:, tok_ids])
    base_probs = torch.cat(base_chunks, dim=0)                # (N, K)
    base_argmax = torch.argmax(base_probs, dim=-1)             # (N,)

    cf_probs_by_site: Dict[SiteKey, torch.Tensor] = {}
    cf_argmax_by_site: Dict[SiteKey, torch.Tensor] = {}
    # Dispatch site-key extraction by unit type — ResidualStream uses
    # ``(layer, token_pos)``; AttentionHead uses ``(layer, head, token_pos)``.
    from ..site_keys import attention_head_site_key_for_unit
    for model_units_list in bundle.experiment.model_units_lists:
        unit = model_units_list[0][0]
        per_batch = _run_interchange_interventions(
            pipeline=bundle.pipeline,
            counterfactual_dataset=dataset,
            model_units_list=model_units_list,
            verbose=verbose,
            batch_size=batch_size,
            output_scores=True,
        )
        cf_logits = torch.cat([b[:, 0, :] for b in per_batch], dim=0)
        cf_probs = torch.softmax(cf_logits, dim=-1)[:, tok_ids]
        if hasattr(unit, "head"):
            key = attention_head_site_key_for_unit(unit)
        else:
            key = site_key_for_unit(unit)
        cf_probs_by_site[key] = cf_probs
        cf_argmax_by_site[key] = torch.argmax(cf_probs, dim=-1)
    return NeuralOutputs(
        base_alpha_probs=base_probs,
        base_alpha_argmax=base_argmax,
        cf_alpha_probs=cf_probs_by_site,
        cf_alpha_argmax=cf_argmax_by_site,
        alphabet=alpha,
    )


def signatures_from_outputs(
    outputs: NeuralOutputs,
    *,
    normalize: bool = True,
) -> Dict[SiteKey, torch.Tensor]:
    """Aggregate cached per-site cf/base probability deltas into (K,) rows.

    Mirror of the source's per-site ``aggregate_mean(intervened_probs - base_probs)``.
    """
    out: Dict[SiteKey, torch.Tensor] = {}
    base = outputs.base_alpha_probs
    for key, cf in outputs.cf_alpha_probs.items():
        delta = (cf - base).mean(dim=0)
        if normalize:
            delta = normalize_rows(delta.unsqueeze(0)).squeeze(0)
        out[key] = delta
    return out


def collect_neural_effect_signatures(
    bundle: ExperimentBundle,
    dataset,
    *,
    alphabet: Optional[LabelAlphabet] = None,
    letters: Optional[str] = None,
    normalize: bool = True,
    batch_size: int = 32,
    verbose: bool = False,
) -> Dict[SiteKey, torch.Tensor]:
    """Backwards-compatible wrapper: returns just the (K,) per-site rows."""
    outputs = collect_neural_outputs(
        bundle, dataset,
        alphabet=alphabet, letters=letters,
        batch_size=batch_size, verbose=verbose,
    )
    return signatures_from_outputs(outputs, normalize=normalize)


def _iter_examples(dataset):
    """Iterate examples from either a MIB ``CounterfactualDataset`` (has
    ``.dataset`` indexable) or a plain iterable of dicts (used in tests)."""
    inner = getattr(dataset, "dataset", None)
    if inner is not None and hasattr(inner, "__len__") and hasattr(inner, "__getitem__"):
        for i in range(len(inner)):
            yield inner[i]
        return
    for ex in dataset:
        yield ex


def expected_cf_letter_indices(
    causal_model,
    dataset,
    *,
    variable: str,
    alphabet: Optional[LabelAlphabet] = None,
    letters: Optional[str] = None,
    on_unknown_label: str = "raise",  # "raise" | "skip"
    output_key: str = "answer",
    label_from_output: Optional[Callable[[str], str]] = None,
) -> torch.Tensor:
    """Per-example expected counterfactual label index after interchanging
    ``variable`` from the source. This is the IIA target.

    With ``on_unknown_label="skip"``, examples whose CF answer is outside the
    alphabet are mapped to ``-1`` so callers can mask them out of IIA scoring.
    """
    alpha = _resolve_alphabet(alphabet, letters)
    indices: List[int] = []
    n_unknown = 0
    for i, example in enumerate(_iter_examples(dataset)):
        cf_setting = causal_model.run_interchange(
            example["input"],
            {variable: example["counterfactual_inputs"][0]},
        )
        letter = str(cf_setting[output_key]).strip()
        if label_from_output is not None:
            letter = label_from_output(letter)
        idx = alpha.label_to_dim.get(letter)
        if idx is None:
            if on_unknown_label == "raise":
                raise ValueError(
                    f"Counterfactual label {letter!r} at example {i} not in "
                    f"alphabet ({alpha.num_dims} dims); widen the alphabet."
                )
            n_unknown += 1
            indices.append(-1)
            continue
        indices.append(idx)
    if n_unknown:
        print(f"[features] expected_cf_letter_indices(variable={variable!r}): "
              f"{n_unknown} examples had CF labels outside the alphabet (idx=-1)")
    return torch.tensor(indices, dtype=torch.long)


def per_site_iia(
    outputs: NeuralOutputs,
    expected_cf_indices: torch.Tensor,
) -> Dict[SiteKey, float]:
    """Interchange-intervention accuracy per site over the alphabet.

    IIA[s] = mean_j ( argmax_alphabet(cf_logits[s, j]) == expected_cf[j] ).
    Examples with ``expected_cf[j] == -1`` are treated as masked-out (their
    CF label was outside the alphabet) and excluded from the mean.
    """
    out: Dict[SiteKey, float] = {}
    target = expected_cf_indices.to(torch.long)
    mask = target >= 0
    n_valid = int(mask.sum().item())
    for key, argmax in outputs.cf_alpha_argmax.items():
        if n_valid == 0:
            out[key] = 0.0
            continue
        correct = (argmax.to(torch.long) == target) & mask
        out[key] = float(correct.to(torch.float32).sum().item() / n_valid)
    return out
