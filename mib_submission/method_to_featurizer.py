"""
Convert outputs of our alignment methods into MIB-compliant
``(Featurizer, indices)`` pairs ready for ``serialize.py``.

Every public function returns a ``MethodResult`` with two fields:

- ``featurizer`` — a ``CausalAbstraction.neural.featurizers.Featurizer`` instance
  (either an Identity featurizer or upstream's ``SubspaceFeaturizer``). Both
  encodings round-trip through upstream's ``Featurizer.load_modules``; we
  deliberately use only upstream classes so the evaluator can deserialise
  without a custom ``featurizer.py`` in the submission folder.
- ``indices`` — JSON-serialisable list of ints (or ``None`` for "all features"),
  written to the ``_indices`` file in the submission triplet.

The mapping from each method to an encoding is documented in ``CLAUDE.md`` —
in short:

| Method                                | Featurizer encoding | Indices                        |
|---------------------------------------|---------------------|--------------------------------|
| OT / GW / FGW / UOT (raw selection)   | Identity            | top-k hidden dims from plan    |
| OT+gradient (hardened mask)           | Identity            | top-k hidden dims from mask    |
| OT+DAS                                | Subspace (rotation) | range(k) (or None)             |
| OT+PCA                                | Subspace (PCA)      | top-k PCA components from plan |

These converters are pure: they take numpy/torch arrays and return Python
objects. They do not touch disk; that's ``serialize.py``'s job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch

from CausalAbstraction.neural.featurizers import (
    Featurizer,
    IdentityFeaturizerModule,
    IdentityInverseFeaturizerModule,
    SubspaceFeaturizer,
)


@dataclass
class MethodResult:
    """A single (featurizer, indices) pair ready to serialise.

    ``layer`` and ``token_position_id`` are recorded for bookkeeping only —
    the file naming in ``serialize.py`` uses them to build the ``ModelUnit``
    id (e.g. ``ResidualStream(Layer-12,Token-last_token)``).
    """

    featurizer: Featurizer
    indices: Optional[List[int]]
    layer: int
    token_position_id: str
    method: str
    variable: str


def _identity_featurizer(id: str) -> Featurizer:
    return Featurizer(
        IdentityFeaturizerModule(),
        IdentityInverseFeaturizerModule(),
        n_features=None,
        id=id,
    )


def _topk_indices_from_row(
    plan_row: np.ndarray, k: int, hidden_dim: int
) -> List[int]:
    """Pick the top-k hidden-state dims from one row of a transport plan.

    ``plan_row`` is the slice ``pi[variable_idx, sites_at_layer]``. Each "site"
    in our pipeline corresponds to a contiguous block of hidden dims, so we
    expand selected sites into the dims they cover and clip to k. If sites
    are 1-dim each (RESOLUTION=1) this is just the argsort of the row.
    """
    if plan_row.ndim != 1:
        raise ValueError("plan_row must be a 1-D array of site weights.")
    if plan_row.size != hidden_dim:
        raise ValueError(
            f"plan_row length ({plan_row.size}) != hidden_dim ({hidden_dim}); "
            "expand multi-dim sites to per-dim weights before calling."
        )
    order = np.argsort(-plan_row)
    return sorted(int(i) for i in order[:k])


# --------------------------------------------------------------------------- #
#  OT / GW / FGW / UOT (and OT+gradient, after hardening)                     #
# --------------------------------------------------------------------------- #
def from_transport_plan(
    *,
    plan_row: np.ndarray,
    hidden_dim: int,
    k: int,
    layer: int,
    token_position_id: str,
    method: str,
    variable: str,
) -> MethodResult:
    """Encode a (variable, layer) row of a transport plan as Identity + indices."""
    indices = _topk_indices_from_row(plan_row, k=k, hidden_dim=hidden_dim)
    return MethodResult(
        featurizer=_identity_featurizer(id=f"{method}_identity"),
        indices=indices,
        layer=layer,
        token_position_id=token_position_id,
        method=method,
        variable=variable,
    )


def from_gradient_mask(
    *,
    mask: np.ndarray,
    k: int,
    layer: int,
    token_position_id: str,
    variable: str,
    method: str = "ot_gradient",
) -> MethodResult:
    """Hardens a learned soft mask to its top-k entries and returns Identity + indices."""
    return from_transport_plan(
        plan_row=mask,
        hidden_dim=mask.size,
        k=k,
        layer=layer,
        token_position_id=token_position_id,
        method=method,
        variable=variable,
    )


# --------------------------------------------------------------------------- #
#  OT+DAS                                                                     #
# --------------------------------------------------------------------------- #
def from_das_rotation(
    *,
    rotation: torch.Tensor,
    layer: int,
    token_position_id: str,
    variable: str,
    method: str = "ot_das",
    indices: Optional[Sequence[int]] = None,
) -> MethodResult:
    """Wrap a trained DAS rotation matrix as a SubspaceFeaturizer.

    ``rotation`` has shape ``(d_hidden, k)``. Upstream's ``LowRankRotateLayer``
    is parametrised orthogonal — the input matrix should already satisfy
    ``Rᵀ R ≈ I``; the ``orthogonal`` parametrisation will project it onto the
    Stiefel manifold on first forward.
    """
    if rotation.ndim != 2:
        raise ValueError(f"rotation must be 2-D, got shape {tuple(rotation.shape)}")
    rotation = rotation.detach().clone()
    featurizer = SubspaceFeaturizer(
        rotation_subspace=rotation,
        trainable=False,
        id=f"{method}_subspace",
    )
    k = rotation.shape[1]
    return MethodResult(
        featurizer=featurizer,
        indices=list(range(k)) if indices is None else [int(i) for i in indices],
        layer=layer,
        token_position_id=token_position_id,
        method=method,
        variable=variable,
    )


# --------------------------------------------------------------------------- #
#  OT+PCA (optional)                                                          #
# --------------------------------------------------------------------------- #
def from_ot_pca(
    *,
    pca_basis: torch.Tensor,
    plan_row: np.ndarray,
    k: int,
    layer: int,
    token_position_id: str,
    variable: str,
    method: str = "ot_pca",
) -> MethodResult:
    """Encode an OT-over-PCA result as Subspace(PCA basis) + top-k component indices.

    ``pca_basis`` has shape ``(d_hidden, n_components)`` and orthonormal columns
    (the standard PCA components-as-columns layout). ``plan_row`` is the
    transport mass over the ``n_components`` PCA components for this variable.
    """
    if pca_basis.ndim != 2:
        raise ValueError(f"pca_basis must be 2-D, got {tuple(pca_basis.shape)}")
    if plan_row.size != pca_basis.shape[1]:
        raise ValueError(
            "plan_row length must equal pca_basis.shape[1] (n_components)."
        )
    featurizer = SubspaceFeaturizer(
        rotation_subspace=pca_basis.detach().clone(),
        trainable=False,
        id=f"{method}_subspace",
    )
    indices = _topk_indices_from_row(plan_row, k=k, hidden_dim=plan_row.size)
    return MethodResult(
        featurizer=featurizer,
        indices=indices,
        layer=layer,
        token_position_id=token_position_id,
        method=method,
        variable=variable,
    )
