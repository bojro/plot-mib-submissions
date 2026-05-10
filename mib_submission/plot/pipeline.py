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
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ..pipeline import ExperimentBundle
from ..site_keys import site_key_for_unit
from ._alphabets import (
    LabelAlphabet,
    from_causal_model_answers,
    from_labels,
    from_letters,
    resolve_tokens,
)
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
    where applicable, with answer-label alphabet adapted to the cell's task.

    Alphabet:
        - Set ``letters`` to a single string for char-based alphabets (MCQA, ARC).
        - Set ``answer_strings`` to a tuple of multi-char labels for
          word-token alphabets (RAVEL). Only one of the two should be set.
        - Or set ``answer_alphabet_from_causal_model=True`` to derive
          ``answer_strings`` at runtime from ``causal_model.values["answer"]``.

    Per-row dataset filter:
        - Set ``per_row_filter_attribute`` to an input-dict key (e.g.
          ``"queried_attribute"`` for RAVEL) — each OT row will only see
          examples where ``input[per_row_filter_attribute] == row_variable``.
          Use when only a subset of bases is causally connected to each row's
          variable. Disabled by default.
    """

    variables: Tuple[str, ...] = ("answer_pointer", "answer")
    letters: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    answer_strings: Optional[Tuple[str, ...]] = None
    answer_alphabet_from_causal_model: bool = False
    per_row_filter_attribute: Optional[str] = None
    # IOI-style per-row dispatch: each OT row uses an entirely DIFFERENT
    # dataset (specified by train-split key, resolved against
    # ``bundle.train_data``), with all rows interchanging the same CM
    # variable (``calibration_variable``). Mutually exclusive with
    # ``per_row_filter_attribute``. Mirrors source PLOT's per-row
    # ``fit_records_for_row`` family selection on the binary GRU adder.
    # When set, ``variables`` are interpreted as labels for reporting
    # (e.g. "s1_io_flip", "s2_io_flip") not as CM-variable names.
    per_row_split_datasets: Optional[Tuple[str, ...]] = None
    on_unknown_label: str = "raise"     # "raise" | "skip"
    # Causal-model output node key. MCQA/ARC/RAVEL use ``"answer"``; arithmetic
    # exposes its output as ``"raw_output"`` (multi-digit string) — first
    # character is the alphabet member.
    output_key: str = "answer"
    # Map the causal-model output string to an alphabet key. ``None`` ⇒ use
    # the stripped string as-is (correct for MCQA single letters and RAVEL
    # multi-word labels matched verbatim against the alphabet). Arithmetic
    # sets this to take the first character of multi-digit ``raw_output``.
    label_from_output: Optional[Callable[[str], str]] = None

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
    site_signatures: Dict,
    *,
    normalize: bool,
) -> Tuple[torch.Tensor, List[int]]:
    """Mean-aggregate per-site rows into one row per layer.

    Source ``run_progressive_plot.py`` Stage A uses ``FullStateSite`` per
    timestep — one site per layer, no aggregation. Our MIB experiment
    declares one site per (layer, token_position) pair (residual stream)
    or (layer, head, token_position) (attention head), so we collapse the
    non-layer axes by mean. After collapse, each layer is one row in the
    same output-prob-delta space as a single site signature.

    Accepts both 2-tuple ``(layer, token_pos)`` and 3-tuple ``(layer,
    head, token_pos)`` keys; the layer is always at index 0.
    """
    by_layer: Dict[int, List[torch.Tensor]] = {}
    for key, row in site_signatures.items():
        layer = int(key[0])
        by_layer.setdefault(layer, []).append(row)
    layer_ids = sorted(by_layer)
    rows = [aggregate_mean(by_layer[L]) for L in layer_ids]
    table = torch.stack(rows, dim=0)
    if normalize:
        table = normalize_rows(table)
    return table, layer_ids


def _layer_token_table(
    site_signatures: Dict,
    layer: int,
) -> Tuple[torch.Tensor, List]:
    """Stack the per-subspace signatures for one layer.

    For residual-stream sites the "subspace within layer" is the token
    position — keys are 2-tuple ``(layer, token_pos)``, returned subspace
    ids are the token-pos strings.

    For attention-head sites the "subspace within layer" is the head
    index — keys are 3-tuple ``(layer, head, token_pos)``, returned ids
    are 2-tuples ``(head, token_pos)`` so Stage B can preserve both axes
    when multiple token positions exist (IOI uses a single ``"all"``
    position so this just degenerates to per-head).

    Mirrors source PLOT's "Stage B = subspace within timestep" structure.
    """
    keys = [k for k in site_signatures if int(k[0]) == int(layer)]
    if not keys:
        raise KeyError(f"no signatures cached for layer {layer}")
    # Sort by the non-layer suffix so the OT plan ordering is stable.
    keys.sort(key=lambda k: tuple(k[1:]))
    rows = [site_signatures[k] for k in keys]
    if len(keys[0]) == 2:
        ids = [k[1] for k in keys]                  # token_pos strings
    else:
        ids = [(int(k[1]), str(k[2])) for k in keys]  # (head, token_pos)
    return torch.stack(rows, dim=0), ids


def _layer_iia(
    iia_per_site: Dict, layer: int,
) -> float:
    """Mean IIA across a layer's subspaces — token positions for residual
    stream sites, ``(head, token_pos)`` pairs for attention head sites.
    Source PLOT's Stage A score is the IIA of an interchange at the
    *full state* per timestep; we approximate by averaging within-layer.

    Accepts both 2-tuple ``(layer, token_pos)`` and 3-tuple
    ``(layer, head, token_pos)`` keys.
    """
    vals = [v for k, v in iia_per_site.items() if int(k[0]) == int(layer)]
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


def _resolve_config_alphabet(config: PlotConfig, bundle: ExperimentBundle) -> LabelAlphabet:
    """Build a ``LabelAlphabet`` from the PlotConfig + bundle's tokenizer.

    Prefers ``answer_strings`` if set, else ``answer_alphabet_from_causal_model``,
    else falls back to ``letters``.

    Token IDs are resolved eagerly when a real tokenizer is available so that
    abstract and neural signatures use the *same* dim count — collision
    compaction during ``resolve_tokens`` may shrink the alphabet (e.g. RAVEL:
    928 labels → 271 dims). If the alphabet is built but resolution doesn't
    happen here (because the bundle has no tokenizer, e.g. unit tests with
    stubbed-out LM forwards), abstract and neural would disagree on K.
    """
    if config.answer_strings is not None and config.answer_alphabet_from_causal_model:
        raise ValueError(
            "PlotConfig: set either answer_strings or answer_alphabet_from_causal_model, not both"
        )
    if config.answer_strings is not None:
        alpha = from_labels(config.answer_strings)
    elif config.answer_alphabet_from_causal_model:
        alpha = from_causal_model_answers(bundle.causal_model)
    else:
        alpha = from_letters(config.letters)

    # Resolve eagerly when bundle exposes a real tokenizer. Stubs may set
    # bundle.pipeline = None or omit it; in that case stay lazy and let
    # collect_neural_outputs (which the stub replaces) handle resolution.
    pipeline = getattr(bundle, "pipeline", None)
    tokenizer = getattr(pipeline, "tokenizer", None) if pipeline is not None else None
    if tokenizer is not None:
        alpha = resolve_tokens(alpha, tokenizer)
    return alpha


def _make_attribute_filter(attr_key: str, attr_value: str):
    """Predicate ``ex["input"][attr_key] == attr_value`` (handles dict input)."""
    def _pred(example):
        inp = example.get("input", example)
        if isinstance(inp, dict):
            return inp.get(attr_key) == attr_value
        return False
    return _pred


def _per_row_cost(
    abstract_table: torch.Tensor,
    layer_tables_per_row: List[torch.Tensor],
    *,
    metric: str,
) -> torch.Tensor:
    """Per-row cost matrix for Stage A / Stage B with potentially-distinct
    neural reference tables per OT row.

    ``abstract_table`` is ``(V, K)``. ``layer_tables_per_row`` is a list of V
    tensors each of shape ``(M, K)`` (M = num layers for Stage A, num token
    positions for Stage B). Returns ``(V, M)`` where row v is the v-th row
    of ``cost_matrix(abstract_table[v:v+1], layer_tables_per_row[v])``.
    """
    V = abstract_table.size(0)
    if len(layer_tables_per_row) != V:
        raise ValueError(
            f"layer_tables_per_row has {len(layer_tables_per_row)} entries; expected {V}"
        )
    rows = [
        cost_matrix(abstract_table[v : v + 1], layer_tables_per_row[v], metric=metric)[0]
        for v in range(V)
    ]
    return torch.stack(rows, dim=0)


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
    alphabet = _resolve_config_alphabet(config, bundle)
    V = len(config.variables)

    # ---- Decide per-row datasets ---------------------------------------
    # Three modes (mutually exclusive):
    #   1. per_row_split_datasets — entirely separate datasets per row
    #      (IOI: each row uses a different counterfactual split). All rows
    #      interchange the same CM variable (``calibration_variable``).
    #   2. per_row_filter_attribute — same dataset filtered per row by an
    #      input attribute (RAVEL: each row sees only bases whose
    #      ``queried_attribute`` matches its variable name). Each row
    #      interchanges its own CM variable (``variables[i]``).
    #   3. None of the above — single shared dataset, classic per-variable
    #      interchange (MCQA, ARC, arithmetic).
    use_per_row_split = config.per_row_split_datasets is not None
    use_per_row_filter = config.per_row_filter_attribute is not None
    if use_per_row_split and use_per_row_filter:
        raise ValueError(
            "per_row_split_datasets and per_row_filter_attribute are "
            "mutually exclusive."
        )
    if use_per_row_split:
        if len(config.per_row_split_datasets) != V:
            raise ValueError(
                f"per_row_split_datasets has {len(config.per_row_split_datasets)} "
                f"entries; expected {V} (one per variable / OT row)."
            )
        if config.calibration_variable is None:
            raise ValueError(
                "per_row_split_datasets mode requires calibration_variable to "
                "be set explicitly — all rows share that variable's "
                "interchange semantics."
            )
        # Resolve named splits to dataset objects. ``fit_dataset`` is the
        # default fallback for any names not in bundle.train_data.
        train_data = getattr(bundle, "train_data", None) or {}
        per_row_datasets = []
        for split_name in config.per_row_split_datasets:
            if split_name in train_data:
                per_row_datasets.append(train_data[split_name])
            else:
                raise KeyError(
                    f"per_row_split_datasets references unknown split "
                    f"{split_name!r}; bundle.train_data has "
                    f"{sorted(train_data.keys())}"
                )
        per_row_filters = None
    elif use_per_row_filter:
        per_row_filters = [
            _make_attribute_filter(config.per_row_filter_attribute, v)
            for v in config.variables
        ]
        # Materialize per-row datasets via the same filter helper used in
        # features.py (handles HF datasets and plain iterables).
        from .features import _filter_dataset as _filter_ds  # noqa: E402
        per_row_datasets = [_filter_ds(fit_dataset, p) for p in per_row_filters]
    else:
        per_row_filters = None
        per_row_datasets = [fit_dataset] * V

    if use_per_row_split:
        # Build the abstract table row-by-row: each row interchanges the
        # same calibration variable but on its own dataset's source pairs.
        from .features import build_abstract_effect_row  # noqa: E402
        rows = []
        for ds_v in per_row_datasets:
            row = build_abstract_effect_row(
                bundle.causal_model, ds_v,
                variable=config.calibration_variable,
                alphabet=alphabet,
                normalize=config.normalize_signatures,
                on_unknown_label=config.on_unknown_label,
                output_key=config.output_key,
                label_from_output=config.label_from_output,
            )
            rows.append(row)
        abstract_table = torch.stack(rows, dim=0)
    else:
        abstract_table = build_abstract_table(
            bundle.causal_model, fit_dataset,
            variables=config.variables,
            alphabet=alphabet,
            normalize=config.normalize_signatures,
            per_row_dataset_filter=per_row_filters,
            on_unknown_label=config.on_unknown_label,
            output_key=config.output_key,
            label_from_output=config.label_from_output,
        )                                                      # (V, K)

    # ---- Neural collection: per-row when filter / split mode, else shared
    per_row_outputs: List[NeuralOutputs] = []
    per_row_sigs: List[Dict[SiteKey, torch.Tensor]] = []
    per_row_layer_tables: List[torch.Tensor] = []
    layer_ids: List[int] = []
    if use_per_row_filter or use_per_row_split:
        for v_idx, ds_v in enumerate(per_row_datasets):
            outputs_v = collect_neural_outputs(
                bundle, ds_v, alphabet=alphabet, verbose=verbose,
            )
            sigs_v = signatures_from_outputs(
                outputs_v, normalize=config.normalize_signatures,
            )
            layer_tab_v, layer_ids_v = _aggregate_to_layer_table(
                sigs_v, normalize=config.normalize_signatures,
            )
            per_row_outputs.append(outputs_v)
            per_row_sigs.append(sigs_v)
            per_row_layer_tables.append(layer_tab_v)
            if not layer_ids:
                layer_ids = layer_ids_v
            elif layer_ids != layer_ids_v:
                raise RuntimeError(
                    "Per-row datasets produced different layer-id sets — "
                    "all rows must see the same model_units_lists."
                )
    else:
        outputs = collect_neural_outputs(
            bundle, fit_dataset, alphabet=alphabet, verbose=verbose,
        )
        sigs = signatures_from_outputs(outputs, normalize=config.normalize_signatures)
        layer_table, layer_ids = _aggregate_to_layer_table(
            sigs, normalize=config.normalize_signatures,
        )
        per_row_outputs = [outputs] * V
        per_row_sigs = [sigs] * V
        per_row_layer_tables = [layer_table] * V

    # ---- IIA scoring data: based on calibration variable ---------------
    calib_var = _calibration_variable(config)
    if use_per_row_split:
        # All rows interchange the same calibration variable. Use the
        # first per-row split as the IIA evaluation dataset — picking any
        # one is fine since the variable is consistent across rows.
        # Future: union all rows for a bigger sample.
        iia_outputs = per_row_outputs[0]
        iia_dataset = per_row_datasets[0]
    elif use_per_row_filter:
        # Use the per-row outputs aligned with the calibration variable.
        if calib_var in config.variables:
            calib_idx = list(config.variables).index(calib_var)
            iia_outputs = per_row_outputs[calib_idx]
            iia_dataset = per_row_datasets[calib_idx]
        else:
            # Calibration variable is outside the OT row schema — collect on
            # its own filtered dataset.
            calib_filter = _make_attribute_filter(
                config.per_row_filter_attribute, calib_var,
            )
            from .features import _filter_dataset as _filter_ds  # noqa: E402
            iia_dataset = _filter_ds(fit_dataset, calib_filter)
            iia_outputs = collect_neural_outputs(
                bundle, iia_dataset, alphabet=alphabet, verbose=verbose,
            )
    else:
        iia_outputs = per_row_outputs[0]
        iia_dataset = fit_dataset
    expected_cf = expected_cf_letter_indices(
        bundle.causal_model, iia_dataset,
        variable=calib_var, alphabet=alphabet,
        on_unknown_label=config.on_unknown_label,
        output_key=config.output_key,
        label_from_output=config.label_from_output,
    )
    iia_by_site = per_site_iia(iia_outputs, expected_cf)

    # ---- Stage A: per-row top-k layer picks (faithful to source) --------
    # The source's `_stage_a_timesteps` returns one timestep per OT row.
    # Each row's mass row in π is interpreted independently. We sweep
    # epsilon, evaluate the resulting per-row layer picks by mean IIA at
    # the union of layers, and keep the best epsilon.
    stage_a_cost = _per_row_cost(
        abstract_table, per_row_layer_tables, metric=config.cost_metric,
    )

    a_eps_grid = _grid_or_default(config.stage_a_epsilon_grid, config.stage_a_epsilon)
    a_topk_grid = _grid_or_default(config.stage_a_top_k_grid, config.stage_a_top_k_per_row)
    stage_a_trials: List[Dict] = []
    best_a = None  # (score, eps, top_k_per_row, pi, picks_per_row, union_layers)
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
        # Build per-row token tables at this layer. When per_row_filter isn't
        # in use, all per_row_sigs are the same dict, so all per-row token
        # tables are identical — the per-row cost reduces to the legacy
        # ``cost_matrix(abstract, token_table)``.
        token_tables_per_row: List[torch.Tensor] = []
        token_ids: List[str] = []
        for r in range(V):
            tab_r, tok_ids_r = _layer_token_table(per_row_sigs[r], layer)
            if config.normalize_signatures:
                tab_r = normalize_rows(tab_r)
            token_tables_per_row.append(tab_r)
            if not token_ids:
                token_ids = tok_ids_r
            elif token_ids != tok_ids_r:
                raise RuntimeError(
                    f"Layer {layer}: per-row token-id sets differ — "
                    "all rows' filtered datasets must declare the same model_units."
                )
        b_cost = _per_row_cost(
            abstract_table, token_tables_per_row, metric=config.cost_metric,
        )

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
                picks: List = []
                for r in rows_owning:
                    for idx, _ in truncate_row(normed_b[r], k):
                        sub = token_ids[idx]
                        if isinstance(sub, tuple) and len(sub) == 2:
                            # (head, token_pos) → 3-tuple site key.
                            picks.append((int(layer), int(sub[0]), str(sub[1])))
                        else:
                            # Plain token_pos string → 2-tuple residual key.
                            picks.append((int(layer), str(sub)))
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
        neural_table_layer=per_row_layer_tables[0],
        config=config,
        stage_a_chosen=(stage_a_eps, stage_a_top_k, stage_a_score),
        stage_b_chosen=best_b_global,
        stage_a_trials=stage_a_trials,
        stage_b_trials=stage_b_trials,
    )
