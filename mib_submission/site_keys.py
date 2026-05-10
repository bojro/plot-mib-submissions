"""
Site-key helpers shared by ``activations.py``, ``apply_results.py``, and
the IOI attention-head pipeline.

A "site key" identifies one model_unit declared by an experiment and is
used as the dict key for caching signatures and tracking PLOT picks.

Two site shapes are supported:

- **ResidualStream** sites — `(layer, token_position.id)`. Used by MCQA,
  ARC, RAVEL, arithmetic. Filename pattern:
  ``ResidualStream(Layer-{L},Token-{T})_{featurizer,inverse_featurizer,indices}``.

- **AttentionHead** sites — `(layer, head, token_position.id)`. Used by
  IOI cells. Filename pattern:
  ``AttentionHead(Layer-{L},Head-{H},Token-{T})_{...}``. The token
  position is conventionally ``"all"`` for IOI but the slot is preserved
  to mirror the upstream `AttentionHead` model unit.

The two shapes are kept in distinct types so that mixing them by accident
fails fast at type-check time rather than producing a broken submission.
"""

from __future__ import annotations

from typing import Tuple, Union


# Legacy two-tuple shape kept for back-compat with all the residual-stream
# call sites. Treat as ``(layer, token_position_id)``.
ResidualStreamSiteKey = Tuple[int, str]
SiteKey = ResidualStreamSiteKey  # back-compat alias

# Three-tuple for attention-head sites: ``(layer, head, token_position_id)``.
AttentionHeadSiteKey = Tuple[int, int, str]

# Union for typing: existing helpers that accept either shape can use this.
AnySiteKey = Union[ResidualStreamSiteKey, AttentionHeadSiteKey]


def site_key_for_unit(unit) -> ResidualStreamSiteKey:
    """Recover ``(layer, token_position.id)`` from a ResidualStream unit."""
    return (unit.component.get_layer(), unit.token_indices.id)


def attention_head_site_key_for_unit(unit) -> AttentionHeadSiteKey:
    """Recover ``(layer, head, token_position_id)`` from an AttentionHead.

    Upstream's ``AttentionHead.__init__`` does NOT store ``token_indices``
    as an instance attribute (only ``head`` and the ``component`` are
    persisted). The token-position id is captured in the unit's ``id``
    string (uid) like ``"AttentionHead(Layer-7,Head-3,Token-all)"`` —
    we parse it from there.
    """
    layer = unit.component.get_layer()
    head = int(unit.head)
    uid = getattr(unit, "id", None)
    if uid and "Token-" in uid:
        tok_id = uid.rsplit("Token-", 1)[-1].rstrip(")")
    elif hasattr(unit, "token_indices"):
        tok_id = unit.token_indices.id  # ResidualStream-style fallback
    else:
        tok_id = "all"  # IOI default — single token position covering all
    return (layer, head, str(tok_id))


def is_attention_head_key(key: AnySiteKey) -> bool:
    """Return True if ``key`` is a 3-tuple ``(layer, head, token_id)``."""
    return isinstance(key, tuple) and len(key) == 3


def is_residual_stream_key(key: AnySiteKey) -> bool:
    """Return True if ``key`` is a 2-tuple ``(layer, token_id)``."""
    return isinstance(key, tuple) and len(key) == 2
