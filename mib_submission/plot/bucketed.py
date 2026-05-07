"""Bucketed-PLOT — break V=1 collapse by stratifying counterfactual probes.

Background. Standard PLOT requires V≥2 OT rows whose interchange patterns
are observably distinct. For MCQA, ``answer_pointer`` and ``answer``
interchanges produce identical observable letter flips → V=1 collapse →
balanced Sinkhorn returns a uniform plan. Picking ``choice_i`` rows side-
steps the collapse but probes the wrong sites: the cost-min for
``choice_i`` rows is wherever choice_i is encoded, not the pointer.

Fix. Bucket counterfactual examples by ``source.<source_variable>`` (default
``answer_pointer``). All V buckets describe the same operation
(``interchange(target_variable)``) but resolve to observably distinct letter
patterns because the destination depends on the source's pointer value.
The pointer site reproduces all V bucket-conditional patterns when patched;
non-pointer sites don't.

Same Stage A / Stage B / DAS structure as the unbucketed pipeline — only
the cost-matrix construction changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import torch

from ..pipeline import ExperimentBundle
from ..site_keys import site_key_for_unit
from .features import (
    NeuralOutputs,
    _iter_examples,
    aggregate_mean,
    collect_neural_outputs,
    expected_cf_letter_indices,
    normalize_rows,
    per_site_iia,
)
from .pipeline import PlotConfig, PlotSelection, _grid_or_default, _layer_iia, _solve
from .transport import row_normalize, truncate_row


SiteKey = Tuple[int, str]


# --------------------------------------------------------------------------- #
# Bucket assignment + bucketed feature construction                           #
# --------------------------------------------------------------------------- #


def compute_source_value_buckets(
    causal_model,
    dataset,
    *,
    source_variable: str,
) -> List[int]:
    """For each example, return ``int(source[source_variable])`` as bucket id.

    Examples whose source forward pass fails are assigned bucket -1; callers
    should drop or handle them. In MIB MCQA this never fires because the
    source input is always a valid base.
    """
    buckets: List[int] = []
    for example in _iter_examples(dataset):
        try:
            source = causal_model.run_forward(example["counterfactual_inputs"][0])
            buckets.append(int(source[source_variable]))
        except (TypeError, KeyError, IndexError):
            buckets.append(-1)
    return buckets


def build_bucketed_abstract_table(
    causal_model,
    dataset,
    *,
    target_variable: str,
    bucket_assignments: Sequence[int],
    n_buckets: int,
    letters: str,
    normalize: bool = True,
) -> torch.Tensor:
    """For each example: compute ``one_hot(source_letter) − one_hot(base_letter)``
    under ``interchange(target_variable)``, then average within each source-
    value bucket. Output shape ``(n_buckets, len(letters))``.
    """
    K = len(letters)
    letter_to_idx = {ch: i for i, ch in enumerate(letters)}
    rows_per_bucket: List[List[torch.Tensor]] = [[] for _ in range(n_buckets)]

    for i, example in enumerate(_iter_examples(dataset)):
        b = int(bucket_assignments[i])
        if b < 0 or b >= n_buckets:
            continue
        try:
            base_out = causal_model.run_forward(example["input"])
            cf_setting = causal_model.run_interchange(
                example["input"],
                {target_variable: example["counterfactual_inputs"][0]},
            )
            base_letter = str(base_out["answer"]).strip()
            source_letter = str(cf_setting["answer"]).strip()
        except (TypeError, KeyError, IndexError):
            continue
        if base_letter not in letter_to_idx or source_letter not in letter_to_idx:
            continue
        row = torch.zeros(K, dtype=torch.float32)
        row[letter_to_idx[source_letter]] += 1.0
        row[letter_to_idx[base_letter]] -= 1.0
        rows_per_bucket[b].append(row)

    rows: List[torch.Tensor] = []
    for b in range(n_buckets):
        if rows_per_bucket[b]:
            rows.append(aggregate_mean(rows_per_bucket[b]))
        else:
            rows.append(torch.zeros(K, dtype=torch.float32))
    table = torch.stack(rows, dim=0)
    if normalize:
        table = normalize_rows(table)
    return table


def bucketed_signatures_from_outputs(
    outputs: NeuralOutputs,
    bucket_assignments: Sequence[int],
    n_buckets: int,
    *,
    normalize: bool = True,
) -> Dict[SiteKey, torch.Tensor]:
    """Per-site (V, K) signature: per-bucket mean letter-prob delta.

    Output mapping: ``{(layer, tok_id): tensor of shape (n_buckets, K)}``.
    """
    out: Dict[SiteKey, torch.Tensor] = {}
    base = outputs.base_alpha_probs
    K = base.size(-1)
    bucket_t = torch.tensor(list(bucket_assignments), dtype=torch.long)
    for key, cf in outputs.cf_alpha_probs.items():
        delta = cf - base                                   # (N, K)
        rows: List[torch.Tensor] = []
        for b in range(n_buckets):
            mask = bucket_t == b
            if mask.any():
                rows.append(delta[mask].mean(dim=0))
            else:
                rows.append(torch.zeros(K, dtype=torch.float32))
        bucketed = torch.stack(rows, dim=0)                 # (V, K)
        if normalize:
            bucketed = normalize_rows(bucketed)
        out[key] = bucketed
    return out


# --------------------------------------------------------------------------- #
# Bucketed cost (different shape than the unbucketed `cost_matrix`)           #
# --------------------------------------------------------------------------- #


def bucketed_cost_matrix(
    abstract: torch.Tensor,        # (V, K)
    neural_per_unit: torch.Tensor, # (S, V, K)
    *,
    metric: str,
) -> torch.Tensor:
    """Cost ``M[v, s] = dist(abstract[v], neural_per_unit[s, v])``.

    Diagonal-style: row v of the cost is built from neural row v at each
    site. This is the key difference from the unbucketed PLOT cost, where
    every OT row is compared against the SAME (K,)-vector per site.
    """
    V, K = abstract.shape
    S = neural_per_unit.size(0)
    assert neural_per_unit.shape == (S, V, K)
    cost = torch.zeros(V, S, dtype=torch.float32)
    for v in range(V):
        a = abstract[v]                                     # (K,)
        n = neural_per_unit[:, v, :]                        # (S, K)
        if metric == "sq_l2":
            cost[v] = ((n - a.unsqueeze(0)) ** 2).sum(dim=-1)
        elif metric == "l1":
            cost[v] = (n - a.unsqueeze(0)).abs().sum(dim=-1)
        elif metric == "cosine":
            denom = (n.norm(dim=-1) * a.norm()).clamp_min(1e-12)
            cost[v] = 1.0 - (n * a.unsqueeze(0)).sum(dim=-1) / denom
        else:
            raise ValueError(f"unknown metric {metric!r}")
    return cost


def _aggregate_bucketed_to_layer_table(
    site_signatures: Dict[SiteKey, torch.Tensor],
    *,
    normalize: bool,
) -> Tuple[torch.Tensor, List[int]]:
    """Mean across token positions per layer, preserving the bucket axis.

    Output: ``(num_layers, V, K)`` and the corresponding sorted layer ids.
    """
    by_layer: Dict[int, List[torch.Tensor]] = {}
    for (layer, _tok_id), bucketed_sig in site_signatures.items():
        by_layer.setdefault(int(layer), []).append(bucketed_sig)
    layer_ids = sorted(by_layer)
    rows: List[torch.Tensor] = []
    for L in layer_ids:
        stacked = torch.stack(by_layer[L], dim=0)          # (T, V, K)
        layer_sig = stacked.mean(dim=0)                    # (V, K)
        if normalize:
            layer_sig = normalize_rows(layer_sig)
        rows.append(layer_sig)
    return torch.stack(rows, dim=0), layer_ids             # (L, V, K)


def _bucketed_layer_token_table(
    site_signatures: Dict[SiteKey, torch.Tensor],
    layer: int,
    *,
    normalize: bool,
) -> Tuple[torch.Tensor, List[str]]:
    """Stack the per-token-position bucketed signatures for one layer.

    Output: ``(num_token_positions, V, K)`` and the token id list.
    """
    keys = [k for k in site_signatures if k[0] == layer]
    keys.sort(key=lambda k: k[1])
    if not keys:
        raise KeyError(f"no signatures cached for layer {layer}")
    rows = [site_signatures[k] for k in keys]
    table = torch.stack(rows, dim=0)                       # (T, V, K)
    if normalize:
        # Normalise within each token's row of the bucketed signature.
        T, V, K = table.shape
        flat = table.reshape(T * V, K)
        flat = normalize_rows(flat)
        table = flat.reshape(T, V, K)
    return table, [k[1] for k in keys]


# --------------------------------------------------------------------------- #
# Pipeline                                                                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BucketedPlotConfig:
    """Hyperparameters for the bucketed PLOT pipeline."""

    target_variable: str = "answer_pointer"     # variable being interchanged + localized
    source_variable: str = "answer_pointer"     # bucket key (typically same as target)
    n_buckets: int = 4                          # for MCQA: 4 (one per pointer index)
    letters: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    cost_metric: str = "sq_l2"
    normalize_signatures: bool = True
    stage_a_solver: str = "ot"
    stage_b_solver: str = "ot"
    stage_a_epsilon: float = 0.03
    stage_b_epsilon: float = 0.01
    stage_a_beta_neural: float = 0.1
    stage_b_beta_neural: float = 0.1
    sinkhorn_iters: int = 200
    stage_a_top_k_per_row: int = 1
    stage_b_top_k_per_row: int = 1
    stage_a_epsilon_grid: Tuple[float, ...] = (0.01, 0.03)
    stage_b_epsilon_grid: Tuple[float, ...] = (0.003, 0.01, 0.03, 0.1)
    stage_a_top_k_grid: Tuple[int, ...] = (1,)
    stage_b_top_k_grid: Tuple[int, ...] = (1, 2)
    calibration_variable: str | None = None     # defaults to target_variable


def _calibration_variable(config: BucketedPlotConfig) -> str:
    return config.calibration_variable or config.target_variable


def select_sites_via_bucketed_plot(
    bundle: ExperimentBundle,
    fit_dataset,
    *,
    config: BucketedPlotConfig,
    verbose: bool = False,
) -> PlotSelection:
    """Stage A → Stage B with bucketed-by-source-value cost.

    Bucket assignments are computed once from the source's value of
    ``config.source_variable``. All V buckets describe ``interchange(
    config.target_variable)`` but resolve to distinct letter patterns
    because the destination depends on the bucket. Pointer-style sites
    reproduce all V patterns; non-pointer sites don't.
    """
    bucket_assignments = compute_source_value_buckets(
        bundle.causal_model, fit_dataset,
        source_variable=config.source_variable,
    )
    n_per_bucket = [bucket_assignments.count(b) for b in range(config.n_buckets)]
    n_skipped = sum(1 for b in bucket_assignments if b < 0 or b >= config.n_buckets)
    print(f"[bucketed] bucket sizes = {n_per_bucket}; skipped {n_skipped}")
    nonempty = [b for b in range(config.n_buckets) if n_per_bucket[b] > 0]
    if len(nonempty) < 2:
        raise RuntimeError(
            f"need ≥2 non-empty buckets; got {nonempty}. Pick a different "
            f"source_variable or split."
        )

    abstract = build_bucketed_abstract_table(
        bundle.causal_model, fit_dataset,
        target_variable=config.target_variable,
        bucket_assignments=bucket_assignments,
        n_buckets=config.n_buckets,
        letters=config.letters,
        normalize=config.normalize_signatures,
    )                                                       # (V, K)
    print(f"[bucketed] abstract = {tuple(abstract.shape)}")

    outputs = collect_neural_outputs(
        bundle, fit_dataset, letters=config.letters, verbose=verbose,
    )
    site_sigs = bucketed_signatures_from_outputs(
        outputs, bucket_assignments, config.n_buckets,
        normalize=config.normalize_signatures,
    )

    calib_var = _calibration_variable(config)
    expected_cf = expected_cf_letter_indices(
        bundle.causal_model, fit_dataset,
        variable=calib_var, letters=config.letters,
    )
    iia_by_site = per_site_iia(outputs, expected_cf)

    # ---- Stage A: per-bucket layer picks ---------------------------------
    layer_table, layer_ids = _aggregate_bucketed_to_layer_table(
        site_sigs, normalize=config.normalize_signatures,
    )                                                       # (L, V, K)
    stage_a_cost = bucketed_cost_matrix(
        abstract, layer_table, metric=config.cost_metric,
    )                                                       # (V, L)

    a_eps_grid = _grid_or_default(config.stage_a_epsilon_grid, config.stage_a_epsilon)
    a_topk_grid = _grid_or_default(config.stage_a_top_k_grid, config.stage_a_top_k_per_row)
    stage_a_trials: List[Dict] = []
    best_a = None
    V = config.n_buckets
    for eps in a_eps_grid:
        pi = _solve(
            stage_a_cost, solver=config.stage_a_solver,
            epsilon=float(eps), beta_neural=config.stage_a_beta_neural,
            n_iter=config.sinkhorn_iters,
        )
        normed = row_normalize(pi)
        for top_k in a_topk_grid:
            picks_per_row: Dict[int, List[int]] = {}
            for r in range(V):
                if r not in nonempty:
                    picks_per_row[r] = []
                    continue
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
                "union_layers": list(union_layers), "score": score,
            })
            if best_a is None or score > best_a[0]:
                best_a = (score, float(eps), int(top_k), pi, picks_per_row, union_layers)

    assert best_a is not None
    a_score, a_eps, a_top_k, a_pi, picks_per_row, a_layer_picks = best_a
    print(f"[bucketed] Stage A best: eps={a_eps} top_k={a_top_k} score={a_score:.4f}")
    print(f"[bucketed] Stage A picked layers: {a_layer_picks}")

    # ---- Stage B: per-(row, layer) token-position picks -----------------
    layer_to_rows: Dict[int, List[int]] = {}
    for r, layers in picks_per_row.items():
        for L in layers:
            layer_to_rows.setdefault(int(L), []).append(int(r))

    b_eps_grid = _grid_or_default(config.stage_b_epsilon_grid, config.stage_b_epsilon)
    b_topk_grid = _grid_or_default(config.stage_b_top_k_grid, config.stage_b_top_k_per_row)
    stage_b_pi_per_layer: Dict[int, torch.Tensor] = {}
    stage_b_trials: List[Dict] = []
    selected: List[SiteKey] = []
    best_b_global = None

    for layer in a_layer_picks:
        rows_owning = layer_to_rows[int(layer)]
        token_table, token_ids = _bucketed_layer_token_table(
            site_sigs, layer, normalize=config.normalize_signatures,
        )                                                   # (T, V, K)
        b_cost = bucketed_cost_matrix(
            abstract, token_table, metric=config.cost_metric,
        )                                                   # (V, T)

        best_per_layer = None
        for eps in b_eps_grid:
            pi = _solve(
                b_cost, solver=config.stage_b_solver,
                epsilon=float(eps), beta_neural=config.stage_b_beta_neural,
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
                seen, dedup = set(), []
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

    seen, dedup_selected = set(), []
    for p in selected:
        if p not in seen:
            seen.add(p)
            dedup_selected.append(p)
    selected = dedup_selected
    assert best_b_global is not None
    print(f"[bucketed] Stage B selected sites: {selected}")
    print(f"[bucketed] Stage B best: eps={best_b_global[0]} top_k={best_b_global[1]} score={best_b_global[2]:.4f}")

    # Reuse PlotSelection — repurpose `abstract_table` and `neural_table_layer`
    # to carry their bucketed shapes. Downstream consumers (run.py) read
    # selection.selected_sites and the chosen tuples, both unaffected.
    fake_config = PlotConfig(
        variables=tuple(f"bucket{b}" for b in range(config.n_buckets)),
        letters=config.letters,
        cost_metric=config.cost_metric,
        normalize_signatures=config.normalize_signatures,
        stage_a_solver=config.stage_a_solver,
        stage_b_solver=config.stage_b_solver,
        stage_a_epsilon=config.stage_a_epsilon,
        stage_b_epsilon=config.stage_b_epsilon,
        sinkhorn_iters=config.sinkhorn_iters,
        target_row_index=0,
        stage_a_epsilon_grid=config.stage_a_epsilon_grid,
        stage_b_epsilon_grid=config.stage_b_epsilon_grid,
        stage_a_top_k_grid=config.stage_a_top_k_grid,
        stage_b_top_k_grid=config.stage_b_top_k_grid,
        calibration_variable=calib_var,
    )
    return PlotSelection(
        selected_sites=selected,
        stage_a_layers=a_layer_picks,
        stage_a_pi=a_pi,
        stage_b_pi_per_layer=stage_b_pi_per_layer,
        abstract_table=abstract,
        neural_table_layer=layer_table.reshape(layer_table.size(0), -1),
        config=fake_config,
        stage_a_chosen=(a_eps, a_top_k, a_score),
        stage_b_chosen=best_b_global,
        stage_a_trials=stage_a_trials,
        stage_b_trials=stage_b_trials,
    )
