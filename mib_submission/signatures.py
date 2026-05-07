"""
Per-site logit-shift signatures for OT / GW / FGW / UOT.

For each ``(layer, token_position)`` site declared by ``bundle.experiment``, we
patch the residual stream from a counterfactual source into the base run and
record the resulting logits. The "signature" of a site is the per-example
shift between intervened and factual logits — that's the row that an OT solver
matches against the abstract-variable signatures from the causal model.

This is the MIB-data analog of ``mcqa_experiment/signatures.py``: same idea
(measure each site's intervention effect on the LM's prediction distribution),
but consuming a ``CounterfactualDataset`` from MIB rather than our private
``MCQAPairBank``.

Shape contract::

    base_logits:                     Tensor (N, vocab)
    intervention_logits[(L, T)]:     Tensor (N, vocab)
    signatures[(L, T)]:              Tensor (N,)        # collapsed by signature_mode
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from .pipeline import ExperimentBundle, add_mib_to_syspath
from .site_keys import site_key_for_unit


SiteKey = Tuple[int, str]
LogitsDict = Dict[SiteKey, torch.Tensor]
SignatureDict = Dict[SiteKey, torch.Tensor]


def alphabet_token_ids(tokenizer, letters: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ") -> torch.Tensor:
    """Return per-letter next-token ids using the leading-space variant.

    The MIB MCQA causal model produces ``answer = " A"`` (with leading space) so
    the matching LM token is the leading-space variant for every letter. Each
    letter must encode to a single token; if it doesn't (rare for unicode
    tokenizers but possible) we fall back to the no-space variant. We do not
    take a max over variants the way ``mcqa_experiment/data.py`` does — we ship
    one token per letter, which is enough for ``answer_logit_delta``.
    """
    ids = []
    for letter in letters:
        for variant in (f" {letter}", letter):
            enc = tokenizer.encode(variant, add_special_tokens=False)
            if len(enc) == 1:
                ids.append(int(enc[0]))
                break
        else:
            raise ValueError(f"Letter {letter!r} did not encode to a single token.")
    return torch.tensor(ids, dtype=torch.long)


def causal_answer_letters(causal_model, dataset, *, variable: str = "answer_pointer") -> Tuple[List[str], List[str]]:
    """Per-example (base_letter, source_letter) lists, stripping leading spaces.

    The simple_MCQA causal model returns ``answer = " A"`` etc.; we strip so the
    return values are plain letters, ready for ``alphabet_token_ids`` lookup.
    """
    base_letters: List[str] = []
    source_letters: List[str] = []
    for example in dataset:
        base_out = causal_model.run_forward(example["input"])
        base_letters.append(str(base_out["answer"]).strip())
        source_setting = causal_model.run_interchange(
            example["input"],
            {variable: example["counterfactual_inputs"][0]},
        )
        source_letters.append(str(source_setting["answer"]).strip())
    return base_letters, source_letters


def _last_token_logits(scores_list) -> torch.Tensor:
    """Pull the next-token logits out of an HF ``generate`` result.

    ``pipeline.generate`` returns ``{"scores": [Tensor (B, vocab) per new
    token], ...}``. With ``max_new_tokens=1`` (our default) ``scores`` has
    length 1 — we take that single tensor.
    """
    if not scores_list:
        raise RuntimeError(
            "pipeline.generate returned no scores; check max_new_tokens > 0 "
            "and that output_scores=True is set."
        )
    return scores_list[0]


def collect_base_logits(
    bundle: ExperimentBundle,
    dataset,
    *,
    batch_size: int = 32,
) -> torch.Tensor:
    """Factual next-token logits over ``dataset``'s base inputs.

    Returns
    -------
    Tensor of shape ``(N, vocab)``, CPU.
    """
    pipeline = bundle.pipeline
    chunks = []
    n = len(dataset.dataset)
    for start in range(0, n, batch_size):
        batch = [dataset.dataset[i] for i in range(start, min(start + batch_size, n))]
        bases = [ex["input"] for ex in batch]
        out = pipeline.generate(bases)
        chunks.append(_last_token_logits(out["scores"]))
    return torch.cat(chunks, dim=0)


def collect_site_intervention_logits(
    bundle: ExperimentBundle,
    dataset,
    *,
    batch_size: int = 32,
    verbose: bool = False,
) -> LogitsDict:
    """Run an interchange intervention at each site and record next-token logits.

    For each ``model_units_list`` in ``bundle.experiment.model_units_lists``
    (one per site, shape ``[[unit]]`` as constructed by ``PatchResidualStream``),
    we call upstream's ``_run_interchange_interventions`` with
    ``output_scores=True``. The returned per-batch tensors have shape
    ``(B, max_new_tokens, vocab)``; we keep the next-token slice and concat.

    Returns
    -------
    dict ``{(layer, tok_id): Tensor (N, vocab)}``.
    """
    add_mib_to_syspath()
    from experiments.pyvene_core import _run_interchange_interventions  # type: ignore[import-not-found]

    out: LogitsDict = {}
    for model_units_list in bundle.experiment.model_units_lists:
        # Each model_units_list is [[unit]] for PatchResidualStream.
        unit = model_units_list[0][0]
        per_batch = _run_interchange_interventions(
            pipeline=bundle.pipeline,
            counterfactual_dataset=dataset,
            model_units_list=model_units_list,
            verbose=verbose,
            batch_size=batch_size,
            output_scores=True,
        )
        # Each batch tensor has shape (B, max_new_tokens, vocab); take next token.
        cat = torch.cat([b[:, 0, :] for b in per_batch], dim=0)
        out[site_key_for_unit(unit)] = cat
    return out


def signature_from_logits(
    *,
    intervention_logits: torch.Tensor,
    base_logits: torch.Tensor,
    mode: str = "whole_vocab_kl",
) -> torch.Tensor:
    """Collapse a (N, vocab) intervention/base pair into a per-example scalar.

    Modes
    -----
    ``whole_vocab_kl`` — KL(intervention || base), the same default as
        ``mcqa_experiment/signatures.py``. Task-agnostic; works on every MIB
        cell without needing to know which vocab tokens are "answers".
    ``logit_l2`` — ``||intervention - base||_2`` over the vocab axis. Cheaper
        and avoids softmax saturation in fp16.
    """
    if intervention_logits.shape != base_logits.shape:
        raise ValueError(
            f"shape mismatch: intervention {tuple(intervention_logits.shape)} "
            f"vs base {tuple(base_logits.shape)}"
        )
    if mode == "whole_vocab_kl":
        base_lp = torch.log_softmax(base_logits, dim=-1)
        cf_lp = torch.log_softmax(intervention_logits, dim=-1)
        cf_p = cf_lp.exp()
        return torch.sum(cf_p * (cf_lp - base_lp), dim=-1)
    if mode == "logit_l2":
        return torch.linalg.vector_norm(intervention_logits - base_logits, dim=-1)
    raise ValueError(f"Unknown signature mode {mode!r}")


def collect_answer_logit_delta_signatures(
    bundle: ExperimentBundle,
    dataset,
    *,
    letters: str,
    batch_size: int = 32,
    verbose: bool = False,
) -> Dict[SiteKey, torch.Tensor]:
    """Per-site directional signatures over the answer-letter vocab.

    This is the MIB-side analog of ``mcqa_experiment``'s ``answer_logit_delta``
    mode. For each example j and each letter l in ``letters``, we record::

        S[s, j, l] = logits_cf[s, j, token(l)] - logits_base[j, token(l)]

    then flatten the (j, l) axes so each site's signature is one
    ``(N * K,)`` row directly comparable to the abstract signature
    produced by ``abstract_signatures.build_logit_delta_abstract``.
    """
    tok_ids = alphabet_token_ids(bundle.pipeline.tokenizer, letters=letters)
    base_full = collect_base_logits(bundle, dataset, batch_size=batch_size)
    base_letters = base_full[:, tok_ids]                         # (N, K)
    cf_full = collect_site_intervention_logits(
        bundle, dataset, batch_size=batch_size, verbose=verbose,
    )
    out: Dict[SiteKey, torch.Tensor] = {}
    for k, v in cf_full.items():
        delta = v[:, tok_ids] - base_letters                     # (N, K)
        out[k] = delta.reshape(-1)                               # (N*K,)
    return out


def collect_site_signatures(
    bundle: ExperimentBundle,
    dataset,
    *,
    mode: str = "whole_vocab_kl",
    batch_size: int = 32,
    verbose: bool = False,
) -> SignatureDict:
    """End-to-end: base forward + per-site intervention + collapse.

    This is the input the OT/GW/FGW/UOT solvers consume — one row of length
    ``N`` per site.
    """
    base = collect_base_logits(bundle, dataset, batch_size=batch_size)
    per_site = collect_site_intervention_logits(
        bundle, dataset, batch_size=batch_size, verbose=verbose
    )
    return {
        key: signature_from_logits(
            intervention_logits=cf, base_logits=base, mode=mode
        )
        for key, cf in per_site.items()
    }
