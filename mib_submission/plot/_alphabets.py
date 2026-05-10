"""Label alphabets for PLOT signature spaces.

PLOT's per-site effect signatures live in a low-dim "label-space": for MCQA
this was the 26-letter alphabet ``A..Z``; for ARC the same; for RAVEL the
labels are word strings (``"France"``, ``"United States"``, ``"English,Gaeli,Kymri"``).
This module abstracts that to a uniform interface so the rest of the PLOT
pipeline can treat both cases identically.

A ``LabelAlphabet`` defines:

- ``labels``: ordered tuple of label strings
- ``label_to_dim``: maps label string to its column index in signatures
- ``token_ids``: ordered tuple of LM-vocab token ids (one per dim) — shared
  with ``label_to_dim`` since multiple labels can share a first token
- ``num_dims``: signature width (== number of unique first tokens)

Collision policy: when two labels share a first token (e.g. ``"United States"``
and ``"United Kingdom"`` both first-tokenize to ``" United"`` for Gemma),
they map to the same dim. The abstract one-hot diff for either label
contributes to the same column. This is a documented loss of distinctness
that we accept rather than escalate to multi-token signatures (which would
double the signature dim for marginal gain on the small RAVEL collisions).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class LabelAlphabet:
    """Maps label strings to signature-space dims, optionally to LM token ids."""

    labels: Tuple[str, ...]                  # ordered, may include duplicates' originals
    label_to_dim: Dict[str, int]             # label string → signature column
    token_ids: Optional[Tuple[int, ...]] = None  # length = num_dims; None if unresolved
    num_dims: int = field(default=0)

    @property
    def has_tokens(self) -> bool:
        return self.token_ids is not None

    def token_ids_tensor(self) -> torch.Tensor:
        if self.token_ids is None:
            raise RuntimeError(
                "alphabet has no resolved token ids — call resolve_tokens() first"
            )
        return torch.tensor(self.token_ids, dtype=torch.long)


def from_letters(letters: str) -> LabelAlphabet:
    """Per-letter alphabet, one dim per character. Legacy MCQA/ARC path.

    The token_ids field is None — caller must call ``resolve_tokens(tokenizer)``
    once a tokenizer is available, since the LM-token mapping is tokenizer-
    specific. We keep that step lazy so configs.py can build alphabets without
    touching a model.
    """
    if not letters:
        raise ValueError("letters must be non-empty")
    if len(set(letters)) != len(letters):
        raise ValueError(f"letters has duplicates: {letters!r}")
    labs = tuple(letters)
    return LabelAlphabet(
        labels=labs,
        label_to_dim={c: i for i, c in enumerate(labs)},
        token_ids=None,
        num_dims=len(labs),
    )


def from_labels(labels: Sequence[str]) -> LabelAlphabet:
    """Multi-string label alphabet. Use for RAVEL etc.

    Strips leading/trailing whitespace from each label. Empty labels are
    dropped (RAVEL's wikipedia answer is ``""`` — irrelevant for OT).
    Duplicates across the input are collapsed but each surviving label gets
    its own dim. Token IDs are unresolved until ``resolve_tokens(tokenizer)``.
    """
    cleaned = []
    seen = set()
    for lab in labels:
        s = str(lab).strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        cleaned.append(s)
    if not cleaned:
        raise ValueError("after cleaning, no labels remain")
    return LabelAlphabet(
        labels=tuple(cleaned),
        label_to_dim={s: i for i, s in enumerate(cleaned)},
        token_ids=None,
        num_dims=len(cleaned),
    )


def resolve_tokens(alphabet: LabelAlphabet, tokenizer) -> LabelAlphabet:
    """Compute the LM-vocab first-token id for each label.

    Multiple labels can collide on a first token; in that case the alphabet
    is *compacted*: the new dim count = number of unique first tokens, and
    ``label_to_dim`` is rewritten so collided labels share a dim. Returns a
    new alphabet (alphabets are frozen).

    Tokenizer choice rules (in order of preference):

    1. ``" {lab}"`` if it encodes to a single token (e.g. Gemma " A" → 1 token,
       " France" → 1 token). Matches how the LM emits the label after a prompt
       that does not already end in whitespace.
    2. ``lab`` (no leading space) if it encodes to a single token (e.g. Gemma
       "0" → 235276). Matches how the LM emits the label after a prompt that
       *does* end in a space (arithmetic prompts trail with " ", and Gemma
       tokenises " 0" as ``[space_token, "0"_token]`` two tokens, of which the
       first is a generic leading-space token shared across all digits — using
       it would compact the entire digit alphabet to 1 dim).
    3. Otherwise, take the second token of ``" {lab}"`` (skip the generic
       leading-space token). Last resort fallback so we never raise for a
       label that *can* be tokenised.

    The historic behaviour (always pick ``" {lab}"[0]``) caused a silent
    1-dim collapse on Gemma's digit alphabet; arithmetic Stage A returned
    uniform plans and IIA = 1.0 on every site. The new rules preserve the
    MCQA / RAVEL behaviour (their labels hit rule 1) and fix arithmetic.
    """
    label_to_first_tok: Dict[str, int] = {}
    for lab in alphabet.labels:
        spaced = tokenizer.encode(f" {lab}", add_special_tokens=False)
        plain = tokenizer.encode(lab, add_special_tokens=False)
        if spaced and len(spaced) == 1:
            tid = int(spaced[0])
        elif plain and len(plain) == 1:
            tid = int(plain[0])
        elif spaced and len(spaced) >= 2:
            # Skip the leading-space token; the actual label token is index 1.
            tid = int(spaced[1])
        elif plain:
            tid = int(plain[0])
        else:
            raise ValueError(
                f"label {lab!r} did not encode to any tokens with this tokenizer"
            )
        label_to_first_tok[lab] = tid
    # Compact: new dim per UNIQUE token id, in order of first appearance.
    unique_tokens: List[int] = []
    tok_to_new_dim: Dict[int, int] = {}
    for lab in alphabet.labels:
        tid = label_to_first_tok[lab]
        if tid not in tok_to_new_dim:
            tok_to_new_dim[tid] = len(unique_tokens)
            unique_tokens.append(tid)
    new_label_to_dim = {lab: tok_to_new_dim[label_to_first_tok[lab]] for lab in alphabet.labels}
    return LabelAlphabet(
        labels=alphabet.labels,
        label_to_dim=new_label_to_dim,
        token_ids=tuple(unique_tokens),
        num_dims=len(unique_tokens),
    )


def from_causal_model_answers(causal_model) -> LabelAlphabet:
    """Build an alphabet from ``causal_model.values["answer"]``.

    Used for RAVEL where the answer set is data-derived (city attribute values).
    Empty / whitespace-only entries (e.g. RAVEL's wikipedia answer = ``""``)
    are dropped.
    """
    raw = causal_model.values.get("answer")
    if not raw:
        raise ValueError(
            f"causal_model.values['answer'] is empty; cannot build alphabet"
        )
    return from_labels(raw)
