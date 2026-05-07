"""
Site-key helpers shared by ``activations.py`` and ``apply_results.py``.

A "site key" is the pair ``(layer, token_position.id)`` that uniquely
identifies a ResidualStream model_unit declared by a ``PatchResidualStream``
experiment. The ``ResidualStream(Layer-{L},Token-{T})`` filename string used
by upstream's serialiser is derived directly from this pair.
"""

from __future__ import annotations

from typing import Tuple


SiteKey = Tuple[int, str]


def site_key_for_unit(unit) -> SiteKey:
    """Recover ``(layer, token_position.id)`` from a ResidualStream unit."""
    return (unit.component.get_layer(), unit.token_indices.id)
