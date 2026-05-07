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
from typing import Dict, List, Sequence, Tuple

import torch

from ..pipeline import ExperimentBundle, add_mib_to_syspath
from ..signatures import alphabet_token_ids
from ..site_keys import site_key_for_unit


SiteKey = Tuple[int, str]


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
    letters: str


def aggregate_mean(rows: Sequence[torch.Tensor]) -> torch.Tensor:
    """Mirror ``features.aggregate_mean`` from the source branch."""
    if not rows:
        raise ValueError("need at least one row to aggregate")
    return torch.stack([r.to(torch.float32) for r in rows], dim=0).mean(dim=0)


def normalize_rows(M: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    """L2-normalise each row; rows with norm ≤ eps are left at zero."""
    norms = M.norm(dim=-1, keepdim=True).clamp_min(eps)
    return M / norms


def _causal_letter_pairs(causal_model, dataset, *, variable: str) -> Tuple[List[str], List[str]]:
    """Per-example (base_letter, source_letter) under interchange of ``variable``.

    Skips examples where the interchange leaves a downstream variable
    undefined (e.g. ``choice_i`` swap that removes the question's color from
    the choice list, leaving pointer = None). This loses some examples per
    row but is the only safe option without filtering the dataset upstream.
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
            base_letter = str(base_out["answer"]).strip()
            source_letter = str(cf_setting["answer"]).strip()
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
    letters: str,
    normalize: bool = True,
) -> torch.Tensor:
    """One-row abstract-effect signature for a single OT variable.

    For each example: compute ``one_hot(source_letter) - one_hot(base_letter)``
    over ``letters``, then average. Optionally L2-normalise the result.
    Output shape: ``(len(letters),)``.
    """
    base_letters, source_letters = _causal_letter_pairs(
        causal_model, dataset, variable=variable
    )
    K = len(letters)
    letter_to_idx = {ch: i for i, ch in enumerate(letters)}
    rows: List[torch.Tensor] = []
    for b, s in zip(base_letters, source_letters):
        if b not in letter_to_idx or s not in letter_to_idx:
            raise ValueError(
                f"Letter {b!r} or {s!r} not in alphabet {letters!r}; "
                "widen `letters` to cover all observed answers."
            )
        row = torch.zeros(K, dtype=torch.float32)
        row[letter_to_idx[s]] += 1.0
        row[letter_to_idx[b]] -= 1.0
        rows.append(row)
    aggregated = aggregate_mean(rows)
    if normalize:
        aggregated = normalize_rows(aggregated.unsqueeze(0)).squeeze(0)
    return aggregated


def build_abstract_table(
    causal_model,
    dataset,
    *,
    variables: Sequence[str],
    letters: str,
    normalize: bool = True,
) -> torch.Tensor:
    """Stack one abstract row per variable. Shape ``(V, len(letters))``."""
    return torch.stack(
        [
            build_abstract_effect_row(
                causal_model, dataset,
                variable=v, letters=letters, normalize=normalize,
            )
            for v in variables
        ],
        dim=0,
    )


def collect_neural_outputs(
    bundle: ExperimentBundle,
    dataset,
    *,
    letters: str,
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

    tok_ids = alphabet_token_ids(bundle.pipeline.tokenizer, letters=letters)

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
        key = site_key_for_unit(unit)
        cf_probs_by_site[key] = cf_probs
        cf_argmax_by_site[key] = torch.argmax(cf_probs, dim=-1)
    return NeuralOutputs(
        base_alpha_probs=base_probs,
        base_alpha_argmax=base_argmax,
        cf_alpha_probs=cf_probs_by_site,
        cf_alpha_argmax=cf_argmax_by_site,
        letters=letters,
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
    letters: str,
    normalize: bool = True,
    batch_size: int = 32,
    verbose: bool = False,
) -> Dict[SiteKey, torch.Tensor]:
    """Backwards-compatible wrapper: returns just the (K,) per-site rows."""
    outputs = collect_neural_outputs(
        bundle, dataset, letters=letters, batch_size=batch_size, verbose=verbose,
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
    letters: str,
) -> torch.Tensor:
    """Per-example expected counterfactual letter index after interchanging
    ``variable`` from the source. This is the IIA target.
    """
    letter_to_idx = {ch: i for i, ch in enumerate(letters)}
    indices: List[int] = []
    for i, example in enumerate(_iter_examples(dataset)):
        cf_setting = causal_model.run_interchange(
            example["input"],
            {variable: example["counterfactual_inputs"][0]},
        )
        letter = str(cf_setting["answer"]).strip()
        if letter not in letter_to_idx:
            raise ValueError(
                f"Counterfactual letter {letter!r} at example {i} not in alphabet "
                f"{letters!r}; widen `letters`."
            )
        indices.append(letter_to_idx[letter])
    return torch.tensor(indices, dtype=torch.long)


def per_site_iia(
    outputs: NeuralOutputs,
    expected_cf_indices: torch.Tensor,
) -> Dict[SiteKey, float]:
    """Interchange-intervention accuracy per site over the alphabet.

    IIA[s] = mean_j ( argmax_alphabet(cf_logits[s, j]) == expected_cf[j] ).
    Defined on the alphabet (not the full vocab) to match the source PLOT,
    which scored against the task's discrete output set.
    """
    out: Dict[SiteKey, float] = {}
    target = expected_cf_indices.to(torch.long)
    for key, argmax in outputs.cf_alpha_argmax.items():
        out[key] = float((argmax.to(torch.long) == target).to(torch.float32).mean().item())
    return out
