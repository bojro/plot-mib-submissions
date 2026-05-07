"""Sinkhorn solvers and cost matrices for the PLOT pipeline.

Faithful port of ``experiments/binary_addition_rnn/transport.py`` from the
``codex/binary-addition-two-stage-plot`` branch — same balanced + one-sided
unbalanced solvers, same three cost metrics. Inputs are expected to already
be L2-normalised per row (see ``features.normalize_rows``); under that
assumption squared-L2 cost is bounded in ``[0, 4]`` and Sinkhorn is
well-conditioned for any reasonable epsilon.
"""

from __future__ import annotations

import torch


def cost_matrix(
    A: torch.Tensor,
    S: torch.Tensor,
    *,
    metric: str = "sq_l2",
) -> torch.Tensor:
    """Pairwise cost between rows of ``A`` (V, D) and rows of ``S`` (M, D).

    Parameters
    ----------
    metric : ``sq_l2`` | ``l1`` | ``cosine``
        Matches the ``--cost-metric`` choices on the source branch.
    """
    if A.ndim != 2 or S.ndim != 2:
        raise ValueError("A and S must be 2-D")
    if A.size(1) != S.size(1):
        raise ValueError(
            f"feature dim mismatch: A has {A.size(1)}, S has {S.size(1)}"
        )
    A = A.to(torch.float32)
    S = S.to(torch.float32)
    if metric == "sq_l2":
        diffs = A[:, None, :] - S[None, :, :]
        return torch.sum(diffs * diffs, dim=2)
    if metric == "l1":
        diffs = A[:, None, :] - S[None, :, :]
        return torch.sum(torch.abs(diffs), dim=2)
    if metric == "cosine":
        a_unit = A / A.norm(dim=1, keepdim=True).clamp_min(1e-30)
        s_unit = S / S.norm(dim=1, keepdim=True).clamp_min(1e-30)
        return 1.0 - a_unit @ s_unit.T
    raise ValueError(f"unknown cost_metric: {metric!r}")


def sinkhorn_uniform_ot(
    cost: torch.Tensor,
    *,
    epsilon: float,
    n_iter: int = 200,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Balanced entropic OT with uniform marginals.

    Verbatim port of ``transport.sinkhorn_uniform_ot`` from the source branch.
    """
    if epsilon <= 0 or n_iter <= 0 or temperature <= 0:
        raise ValueError("epsilon, n_iter, and temperature must be > 0")
    m, n = cost.shape
    a = torch.full((m,), 1.0 / m, dtype=torch.float32)
    b = torch.full((n,), 1.0 / n, dtype=torch.float32)
    kernel = torch.exp(-cost.to(torch.float32) / (epsilon * temperature)).clamp_min(1e-30)
    r = torch.ones_like(a)
    c = torch.ones_like(b)
    for _ in range(int(n_iter)):
        r = a / (kernel @ c).clamp_min(1e-30)
        c = b / (kernel.transpose(0, 1) @ r).clamp_min(1e-30)
    return r[:, None] * kernel * c[None, :]


def sinkhorn_one_sided_uot(
    cost: torch.Tensor,
    *,
    epsilon: float,
    beta_neural: float,
    n_iter: int = 200,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Unbalanced (one-sided) entropic OT.

    Relaxes the column (neural) marginal — useful when only some sites should
    receive mass and you don't want balanced Sinkhorn to spread it uniformly.
    Verbatim port of ``transport.sinkhorn_one_sided_uot``.
    """
    if epsilon <= 0 or beta_neural <= 0 or n_iter <= 0 or temperature <= 0:
        raise ValueError("epsilon, beta_neural, n_iter, and temperature must be > 0")
    m, n = cost.shape
    a = torch.full((m,), 1.0 / m, dtype=torch.float32)
    b = torch.full((n,), 1.0 / n, dtype=torch.float32)
    kernel = torch.exp(-cost.to(torch.float32) / (epsilon * temperature)).clamp_min(1e-30)
    rho_a = 1.0
    rho_b = float(beta_neural / (beta_neural + epsilon))
    r = torch.ones_like(a)
    c = torch.ones_like(b)
    for _ in range(int(n_iter)):
        r = (a / (kernel @ c).clamp_min(1e-30)).pow(rho_a)
        c = (b / (kernel.transpose(0, 1) @ r).clamp_min(1e-30)).pow(rho_b)
    return r[:, None] * kernel * c[None, :]


def row_normalize(pi: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    """Row-normalise a transport plan into per-row distributions over sites."""
    return pi / pi.sum(dim=-1, keepdim=True).clamp_min(eps)


def truncate_row(row: torch.Tensor, top_k: int) -> list[tuple[int, float]]:
    """Pick the top-k indices of a (renormalised) row, descending mass."""
    k = min(int(top_k), int(row.numel()))
    vals, idx = torch.topk(row, k=k)
    vals = vals / vals.sum().clamp_min(1e-30)
    return [(int(i.item()), float(v.item())) for i, v in zip(idx, vals)]
