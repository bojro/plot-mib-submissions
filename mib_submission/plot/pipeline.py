"""PLOT pipeline orchestrator: Stage A (layer OT) → Stage B (per-layer site OT).

Returns the surviving ``(layer, token_position)`` site set that downstream
DAS training should run on. The DAS step itself is delegated to upstream's
``experiment.train_interventions(method="DAS")`` after pruning the bundle's
``model_units_lists`` to these sites — see ``mib_submission.plot.run``.

Faithful to ``run_progressive_plot.py`` on
``codex/binary-addition-two-stage-plot``: same Stage A → Stage B pattern,
same balanced ``sinkhorn_uniform_ot`` solver, same row-truncation site
picking, and the same ``(epsilon × top_k)`` calibration sweep — each
candidate is scored by per-site IIA on the calibration set, and the
candidate maximising the score is chosen. The source's ``lambda``
(intervention scale) dimension has no MIB analog (pyvene's
``VanillaIntervention`` is full-replace) and is intentionally omitted; the
sensitivity/invariance split likewise collapses to a single IIA score
because MIB counterfactual datasets aren't pre-bucketed into positive vs
invariant banks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import torch

from ..pipeline import ExperimentBundle
from ..site_keys import site_key_for_unit
from .features import (
    NeuralOutputs,
    aggregate_mean,
    build_abstract_table,
    collect_neural_effect_signatures,
    collect_neural_outputs,
    expected_cf_letter_indices,
    normalize_rows,
    per_site_iia,
    signatures_from_outputs,
)
from .transport import (
    cost_matrix,
    row_normalize,
    sinkhorn_one_sided_uot,
    sinkhorn_uniform_ot,
    truncate_row,
)


SiteKey = Tuple[int, str]


@dataclass(frozen=True)
class PlotConfig:
    """Hyperparameters mirroring ``run_progressive_plot.py``'s defaults
    where applicable, with answer-letter alphabet adapted to MCQA."""

    variables: Tuple[str, ...] = ("answer_pointer", "answer")
    letters: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    cost_metric: str = "sq_l2"          # "sq_l2" | "l1" | "cosine"
    normalize_signatures: bool = True
    # Source PLOT (run_progressive_plot.py:_run_ot_stage) uses balanced
    # sinkhorn_uniform_ot for both stages. The "uot" branch is retained as a
    # non-default escape hatch but is not part of the original PLOT pipeline.
    stage_a_solver: str = "ot"          # "ot" (balanced) | "uot" (one-sided unbalanced)
    stage_b_solver: str = "ot"
    stage_a_epsilon: float = 0.03       # used when stage_a_epsilon_grid is empty
    stage_b_epsilon: float = 0.01
    stage_a_beta_neural: float = 0.1    # UOT only: smaller ⇒ more concentration on low-cost sites
    stage_b_beta_neural: float = 0.1
    sinkhorn_iters: int = 200
    stage_a_top_k_per_row: int = 1      # used when stage_a_top_k_grid is empty
    stage_b_top_k_per_row: int = 1
    target_row_index: int = 0           # which OT variable's pi-row drives selection
    # ---- Calibration sweep (faithful to run_progressive_plot.py) ---------- #
    # When grids are non-empty we sweep (epsilon × top_k), score each
    # candidate by mean per-site IIA over the chosen sites against
    # ``calibration_variable`` (defaults to ``variables[target_row_index]``),
    # and pick the (epsilon, top_k) with highest score. The lambda dimension
    # from the source's TransportConfig has no MIB analog (pyvene's
    # VanillaIntervention is full-replace), so it is intentionally omitted.
    stage_a_epsilon_grid: Tuple[float, ...] = (0.01, 0.03)
    stage_b_epsilon_grid: Tuple[float, ...] = (0.003, 0.01, 0.03, 0.1)
    stage_a_top_k_grid: Tuple[int, ...] = (1,)
    stage_b_top_k_grid: Tuple[int, ...] = (1, 2)
    calibration_variable: str | None = None


@dataclass
class PlotSelection:
    """Result of the two-stage selection — the sites DAS will train on."""

    selected_sites: List[SiteKey]                 # final (layer, tok_id) pairs
    stage_a_layers: List[int]                     # layers picked by Stage A
    stage_a_pi: torch.Tensor                      # (V, num_layers)
    stage_b_pi_per_layer: Dict[int, torch.Tensor] # layer -> (V, num_token_positions)
    abstract_table: torch.Tensor                  # (V, K)
    neural_table_layer: torch.Tensor              # (num_layers, K) — Stage A inputs
    config: PlotConfig
    # Calibration trail for reporting / debugging.
    stage_a_chosen: Tuple[float, int, float] = (0.0, 0, 0.0)  # (epsilon, top_k, score)
    stage_b_chosen: Tuple[float, int, float] = (0.0, 0, 0.0)
    stage_a_trials: List[Dict] = field(default_factory=list)
    stage_b_trials: List[Dict] = field(default_factory=list)


def _solve(
    cost: torch.Tensor,
    *,
    solver: str,
    epsilon: float,
    beta_neural: float,
    n_iter: int,
) -> torch.Tensor:
    """Dispatch to balanced or one-sided UOT Sinkhorn."""
    if solver == "ot":
        return sinkhorn_uniform_ot(cost, epsilon=epsilon, n_iter=n_iter)
    if solver == "uot":
        return sinkhorn_one_sided_uot(
            cost, epsilon=epsilon, beta_neural=beta_neural, n_iter=n_iter,
        )
    raise ValueError(f"unknown solver {solver!r}; expected 'ot' or 'uot'.")


def _aggregate_to_layer_table(
    site_signatures: Dict[SiteKey, torch.Tensor],
    *,
    normalize: bool,
) -> Tuple[torch.Tensor, List[int]]:
    """Mean-aggregate per-site rows into one row per layer.

    The source ``run_progressive_plot.py`` Stage A uses ``FullStateSite`` per
    timestep — one *site* per layer, no aggregation. Our MIB experiment
    declares one site per (layer, token_position) pair, so we collapse the
    token-position axis by mean. After collapse, each layer is one row in
    the same output-prob-delta space as a single site signature.
    """
    by_layer: Dict[int, List[torch.Tensor]] = {}
    for (layer, _tok_id), row in site_signatures.items():
        by_layer.setdefault(int(layer), []).append(row)
    layer_ids = sorted(by_layer)
    rows = [aggregate_mean(by_layer[L]) for L in layer_ids]
    table = torch.stack(rows, dim=0)
    if normalize:
        table = normalize_rows(table)
    return table, layer_ids


def _layer_token_table(
    site_signatures: Dict[SiteKey, torch.Tensor],
    layer: int,
) -> Tuple[torch.Tensor, List[str]]:
    """Stack the per-token-position signatures for one layer.

    Already-collected signatures are reused — Stage B does not require new
    forward passes. Returns ``(num_token_positions, K)``.
    """
    keys = [k for k in site_signatures if k[0] == layer]
    keys.sort(key=lambda k: k[1])
    if not keys:
        raise KeyError(f"no signatures cached for layer {layer}")
    rows = [site_signatures[k] for k in keys]
    return torch.stack(rows, dim=0), [k[1] for k in keys]


def _layer_iia(
    iia_per_site: Dict[SiteKey, float], layer: int,
) -> float:
    """Mean IIA across a layer's token positions — the source's Stage A score
    is the IIA of an interchange at the *full state* per timestep, which here
    we approximate by averaging IIA over all token positions at a layer."""
    vals = [v for (L, _t), v in iia_per_site.items() if int(L) == int(layer)]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _grid_or_default(grid: Tuple[float, ...] | Tuple[int, ...], default) -> tuple:
    """Use a non-empty grid, else fall back to the single-value config field."""
    return tuple(grid) if grid else (default,)


def _calibration_variable(config: PlotConfig) -> str:
    if config.calibration_variable is not None:
        return config.calibration_variable
    return config.variables[config.target_row_index]


def select_sites_via_plot(
    bundle: ExperimentBundle,
    fit_dataset,
    *,
    config: PlotConfig,
    verbose: bool = False,
) -> PlotSelection:
    """Run Stage A then Stage B with calibration sweep; return surviving sites.

    Calibration sweep mirrors ``run_progressive_plot.py:_run_ot_stage`` —
    we enumerate ``(epsilon, top_k)`` candidates, evaluate each by per-site
    IIA on the same fit_dataset (the source uses a separate calibration
    bank; for the MIB port the train split serves both roles), and pick
    the candidate maximising mean IIA over the selected sites. The
    ``lambda`` dimension from the source has no clean MIB analog and is
    omitted.
    """
    abstract_table = build_abstract_table(
        bundle.causal_model, fit_dataset,
        variables=config.variables,
        letters=config.letters,
        normalize=config.normalize_signatures,
    )                                                          # (V, K)

    outputs = collect_neural_outputs(
        bundle, fit_dataset, letters=config.letters, verbose=verbose,
    )
    site_signatures = signatures_from_outputs(
        outputs, normalize=config.normalize_signatures,
    )

    calib_var = _calibration_variable(config)
    expected_cf = expected_cf_letter_indices(
        bundle.causal_model, fit_dataset,
        variable=calib_var, letters=config.letters,
    )
    iia_by_site = per_site_iia(outputs, expected_cf)

    # ---- Stage A: per-row top-k layer picks (faithful to source) --------
    # The source's `_stage_a_timesteps` returns one timestep per OT row.
    # Each row's mass row in π is interpreted independently. We sweep
    # epsilon, evaluate the resulting per-row layer picks by mean IIA at
    # the union of layers, and keep the best epsilon.
    layer_table, layer_ids = _aggregate_to_layer_table(
        site_signatures, normalize=config.normalize_signatures,
    )
    stage_a_cost = cost_matrix(abstract_table, layer_table, metric=config.cost_metric)

    a_eps_grid = _grid_or_default(config.stage_a_epsilon_grid, config.stage_a_epsilon)
    a_topk_grid = _grid_or_default(config.stage_a_top_k_grid, config.stage_a_top_k_per_row)
    stage_a_trials: List[Dict] = []
    best_a = None  # (score, eps, top_k_per_row, pi, picks_per_row, union_layers)
    V = abstract_table.size(0)
    for eps in a_eps_grid:
        pi = _solve(
            stage_a_cost,
            solver=config.stage_a_solver,
            epsilon=float(eps),
            beta_neural=config.stage_a_beta_neural,
            n_iter=config.sinkhorn_iters,
        )
        normed = row_normalize(pi)
        for top_k in a_topk_grid:
            # Each row picks its own top-k layer(s).
            picks_per_row: Dict[int, List[int]] = {}
            for r in range(V):
                picks_per_row[r] = [
                    layer_ids[idx] for idx, _ in truncate_row(normed[r], int(top_k))
                ]
            union_layers = sorted({L for picks in picks_per_row.values() for L in picks})
            score = float(
                sum(_layer_iia(iia_by_site, L) for L in union_layers)
                / max(1, len(union_layers))
            )
            stage_a_trials.append({
                "epsilon": float(eps), "top_k_per_row": int(top_k),
                "picks_per_row": {r: list(ls) for r, ls in picks_per_row.items()},
                "union_layers": list(union_layers),
                "score": score,
            })
            if best_a is None or score > best_a[0]:
                best_a = (score, float(eps), int(top_k), pi, picks_per_row, union_layers)

    assert best_a is not None
    stage_a_score, stage_a_eps, stage_a_top_k, stage_a_pi, picks_per_row, stage_a_layer_picks = best_a

    # ---- Stage B: per-(row,layer) top-k token position picks -------------
    # For each (row, layer) pair from Stage A, run OT at that layer's 3
    # token positions and let the SAME row pick its token position. This
    # mirrors the source: row C2 picks its timestep at Stage A, then C2
    # picks its coordinate subspace within that timestep at Stage B.
    b_eps_grid = _grid_or_default(config.stage_b_epsilon_grid, config.stage_b_epsilon)
    b_topk_grid = _grid_or_default(config.stage_b_top_k_grid, config.stage_b_top_k_per_row)

    stage_b_pi_per_layer: Dict[int, torch.Tensor] = {}
    stage_b_trials: List[Dict] = []
    selected: List[SiteKey] = []
    best_b_global = None  # (eps, top_k, score)

    # For each layer in the union, find every (row, layer) pair that
    # selected it; each pair contributes a Stage B token-position choice.
    layer_to_rows: Dict[int, List[int]] = {}
    for r, layers in picks_per_row.items():
        for L in layers:
            layer_to_rows.setdefault(int(L), []).append(int(r))

    for layer in stage_a_layer_picks:
        rows_owning = layer_to_rows[int(layer)]
        token_table, token_ids = _layer_token_table(site_signatures, layer)
        if config.normalize_signatures:
            token_table = normalize_rows(token_table)
        b_cost = cost_matrix(abstract_table, token_table, metric=config.cost_metric)

        best_per_layer = None  # (score, eps, top_k, pi, picks)
        for eps in b_eps_grid:
            pi = _solve(
                b_cost,
                solver=config.stage_b_solver,
                epsilon=float(eps),
                beta_neural=config.stage_b_beta_neural,
                n_iter=config.sinkhorn_iters,
            )
            normed_b = row_normalize(pi)
            for top_k in b_topk_grid:
                k = min(int(top_k), len(token_ids))
                picks: List[SiteKey] = []
                for r in rows_owning:
                    picks.extend(
                        (int(layer), str(token_ids[idx]))
                        for idx, _ in truncate_row(normed_b[r], k)
                    )
                # Dedupe (multiple rows may agree on the same position).
                seen = set()
                dedup = []
                for p in picks:
                    if p not in seen:
                        seen.add(p)
                        dedup.append(p)
                picks = dedup
                score = float(
                    sum(iia_by_site.get(p, 0.0) for p in picks) / max(1, len(picks))
                )
                stage_b_trials.append({
                    "layer": int(layer), "epsilon": float(eps), "top_k": int(top_k),
                    "rows_owning": list(rows_owning),
                    "picks": list(picks), "score": score,
                })
                if best_per_layer is None or score > best_per_layer[0]:
                    best_per_layer = (score, float(eps), int(top_k), pi, picks)

        assert best_per_layer is not None
        score_b, eps_b, top_k_b, pi_b, picks_b = best_per_layer
        stage_b_pi_per_layer[int(layer)] = pi_b
        selected.extend(picks_b)
        if best_b_global is None or score_b > best_b_global[2]:
            best_b_global = (eps_b, top_k_b, score_b)

    # Final dedupe across layers (in case multiple rows on different layers
    # produced overlapping (layer, position) picks — rare).
    seen = set()
    dedup_selected = []
    for p in selected:
        if p not in seen:
            seen.add(p)
            dedup_selected.append(p)
    selected = dedup_selected

    assert best_b_global is not None

    return PlotSelection(
        selected_sites=selected,
        stage_a_layers=stage_a_layer_picks,
        stage_a_pi=stage_a_pi,
        stage_b_pi_per_layer=stage_b_pi_per_layer,
        abstract_table=abstract_table,
        neural_table_layer=layer_table,
        config=config,
        stage_a_chosen=(stage_a_eps, stage_a_top_k, stage_a_score),
        stage_b_chosen=best_b_global,
        stage_a_trials=stage_a_trials,
        stage_b_trials=stage_b_trials,
    )
