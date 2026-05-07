"""Hypothesis-sweep driver for PLOT site selection on MCQA.

Runs Stage A + Stage B for several (OT row variables × signature split × cost
metric) configs and prints a one-line summary per config. Skips DAS training
to keep the sweep fast — once a winning config is identified, run the
production driver (`run.py`) with that config.

Caches per-split forward passes so we do at most one signature collection
per unique split. Bundle (LM) is loaded once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from ..pipeline import setup_residual_experiment
from ..site_keys import site_key_for_unit
from .features import (
    NeuralOutputs,
    build_abstract_table,
    collect_neural_outputs,
    expected_cf_letter_indices,
    normalize_rows,
    per_site_iia,
    signatures_from_outputs,
)
from .pipeline import (
    PlotConfig,
    _aggregate_to_layer_table,
    _grid_or_default,
    _layer_iia,
    _layer_token_table,
    _solve,
)
from .transport import cost_matrix, row_normalize, truncate_row


# --------------------------------------------------------------------------- #
# Cell config (must match run.py for the bundle to be reusable)               #
# --------------------------------------------------------------------------- #
TASK = "4_answer_MCQA"
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MODEL_CLASS_NAME = "Qwen2ForCausalLM"
VARIABLE = "answer_pointer"
DATASET_SIZE = 256  # Smaller for faster sweep — filter typically keeps 5-25%
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class Experiment:
    name: str
    variables: Tuple[str, ...]
    split: str
    cost_metric: str = "sq_l2"
    description: str = ""


EXPERIMENTS: List[Experiment] = [
    Experiment(
        name="E1_choices_apos",
        variables=("choice0", "choice1", "choice2", "choice3"),
        split="answerPosition_train",
        description="Choice rows on answerPosition split — direct probe of pointer",
    ),
    Experiment(
        name="E3_symbols_rand",
        variables=("symbol0", "symbol1", "symbol2", "symbol3"),
        split="randomLetter_train",
        description="Symbol rows on randomLetter split — should localize letter-copy site (control)",
    ),
    Experiment(
        name="E6_pointer_plus_choices",
        variables=("answer_pointer", "choice0", "choice1", "choice2"),
        split="answerPosition_train",
        description="Include the target itself as an OT row alongside choices",
    ),
    Experiment(
        name="E7_choices_cosine",
        variables=("choice0", "choice1", "choice2", "choice3"),
        split="answerPosition_train",
        cost_metric="cosine",
        description="Cost-metric ablation on E1 (cosine instead of sq_l2)",
    ),
]


# --------------------------------------------------------------------------- #
# Inline Stage A + Stage B (cached-outputs version)                           #
# --------------------------------------------------------------------------- #

def run_stages(
    *,
    bundle,
    fit_dataset,
    outputs: NeuralOutputs,
    iia_by_site: Dict[Tuple[int, str], float],
    variables: Tuple[str, ...],
    cost_metric: str,
    config: PlotConfig,
) -> dict:
    """Faithful copy of select_sites_via_plot's Stage A + Stage B logic, but
    using the pre-computed `outputs` and `iia_by_site` so we avoid
    re-running forward passes per experiment."""

    abstract_table = build_abstract_table(
        bundle.causal_model, fit_dataset,
        variables=variables, letters=LETTERS,
        normalize=config.normalize_signatures,
    )                                                          # (V, K)

    site_signatures = signatures_from_outputs(
        outputs, normalize=config.normalize_signatures,
    )

    # Stage A
    layer_table, layer_ids = _aggregate_to_layer_table(
        site_signatures, normalize=config.normalize_signatures,
    )
    cost_a = cost_matrix(abstract_table, layer_table, metric=cost_metric)
    a_eps_grid = _grid_or_default(config.stage_a_epsilon_grid, config.stage_a_epsilon)
    a_topk_grid = _grid_or_default(config.stage_a_top_k_grid, config.stage_a_top_k_per_row)
    best_a = None
    for eps in a_eps_grid:
        pi = _solve(
            cost_a, solver=config.stage_a_solver,
            epsilon=float(eps), beta_neural=config.stage_a_beta_neural,
            n_iter=config.sinkhorn_iters,
        )
        target_row = row_normalize(pi)[config.target_row_index]
        for top_k in a_topk_grid:
            picks = [layer_ids[idx] for idx, _ in truncate_row(target_row, int(top_k))]
            score = float(sum(_layer_iia(iia_by_site, L) for L in picks) / max(1, len(picks)))
            if best_a is None or score > best_a[0]:
                best_a = (score, float(eps), int(top_k), pi, picks, target_row)
    score_a, eps_a, top_k_a, pi_a, layer_picks, target_row = best_a

    # Stage B
    b_eps_grid = _grid_or_default(config.stage_b_epsilon_grid, config.stage_b_epsilon)
    b_topk_grid = _grid_or_default(config.stage_b_top_k_grid, config.stage_b_top_k_per_row)
    selected_sites: List[Tuple[int, str]] = []
    best_b_score = -1.0
    best_b_eps = None
    best_b_topk = None
    for layer in layer_picks:
        token_table, token_ids = _layer_token_table(site_signatures, layer)
        if config.normalize_signatures:
            token_table = normalize_rows(token_table)
        cost_b = cost_matrix(abstract_table, token_table, metric=cost_metric)
        best_per_layer = None
        for eps in b_eps_grid:
            pi_b = _solve(
                cost_b, solver=config.stage_b_solver,
                epsilon=float(eps), beta_neural=config.stage_b_beta_neural,
                n_iter=config.sinkhorn_iters,
            )
            target_row_b = row_normalize(pi_b)[config.target_row_index]
            for top_k in b_topk_grid:
                k = min(int(top_k), len(token_ids))
                picks = [
                    (int(layer), str(token_ids[idx]))
                    for idx, _ in truncate_row(target_row_b, k)
                ]
                score = float(sum(iia_by_site.get(p, 0.0) for p in picks) / max(1, len(picks)))
                if best_per_layer is None or score > best_per_layer[0]:
                    best_per_layer = (score, float(eps), int(top_k), picks)
        score_b, eps_b, top_k_b, picks_b = best_per_layer
        selected_sites.extend(picks_b)
        if score_b > best_b_score:
            best_b_score = score_b
            best_b_eps = eps_b
            best_b_topk = top_k_b

    # Diagnostic: how concentrated is target row?
    sorted_mass = sorted(target_row.tolist(), reverse=True)
    top1_mass = sorted_mass[0]
    nonzero_layers = sum(1 for m in sorted_mass if m > 1e-6)

    return {
        "stage_a": {
            "eps": eps_a, "top_k": top_k_a, "iia": score_a,
            "picks": layer_picks,
            "top1_mass": top1_mass,
            "nonzero_layers": nonzero_layers,
        },
        "stage_b": {
            "eps": best_b_eps, "top_k": best_b_topk, "iia": best_b_score,
            "picks": selected_sites,
        },
        "abstract_cosines_offdiag_max": _max_offdiag_cosine(abstract_table),
    }


def _max_offdiag_cosine(table: torch.Tensor) -> float:
    """Largest off-diagonal cosine similarity between OT rows.
    High value ⇒ rows are not distinct ⇒ V collapses."""
    norms = table.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    unit = table / norms
    sims = unit @ unit.T
    n = sims.size(0)
    sims = sims - torch.eye(n)        # zero out diagonal
    return float(sims.abs().max().item())


# --------------------------------------------------------------------------- #
# Main driver                                                                 #
# --------------------------------------------------------------------------- #

def main() -> None:
    print(f"[sweep] cell = {TASK} × {MODEL_CLASS_NAME} × {VARIABLE}")

    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    layers = list(range(cfg.num_hidden_layers))

    bundle = setup_residual_experiment(
        task=TASK,
        model_name=MODEL_NAME,
        layers=layers,
        target_variables=[VARIABLE],
        dtype=DTYPE,
        dataset_size=DATASET_SIZE,
        config_overrides={"output_scores": False},
        verbose=False,
    )
    print(f"[sweep] available train splits: {sorted(bundle.train_data.keys())}")

    # Cache per-split forward passes.
    splits_needed = sorted({e.split for e in EXPERIMENTS})
    cache: Dict[str, Tuple[NeuralOutputs, Dict[Tuple[int, str], float]]] = {}
    for split in splits_needed:
        if split not in bundle.train_data:
            print(f"[sweep] WARN: split {split!r} not in bundle.train_data; skipping.")
            continue
        ds = bundle.train_data[split]
        print(f"[sweep] collecting outputs for split {split!r} (n={len(ds.dataset)}) …")
        outs = collect_neural_outputs(bundle, ds, letters=LETTERS, verbose=False)
        expected_cf = expected_cf_letter_indices(
            bundle.causal_model, ds, variable=VARIABLE, letters=LETTERS,
        )
        iia = per_site_iia(outs, expected_cf)
        cache[split] = (outs, iia)
        print(f"[sweep]   sites collected = {len(outs.cf_alpha_argmax)}, "
              f"max iia at any site = {max(iia.values()):.4f}")

    # Common config (everything except variables / cost_metric).
    base_cfg = PlotConfig(
        letters=LETTERS,
        cost_metric="sq_l2",
        normalize_signatures=True,
        stage_a_solver="ot",
        stage_b_solver="ot",
        sinkhorn_iters=200,
        target_row_index=0,
        stage_a_epsilon_grid=(0.01, 0.03),
        stage_b_epsilon_grid=(0.003, 0.01, 0.03, 0.1),
        stage_a_top_k_grid=(1,),
        stage_b_top_k_grid=(1, 2),
        calibration_variable=VARIABLE,
        # variables placeholder; overridden per-experiment
        variables=("symbol0", "symbol1", "symbol2", "symbol3"),
    )

    print()
    header = (
        f"{'name':<28} {'V':>3}  {'split':<35} {'cost':<7} "
        f"{'A_pi_top1':>9} {'A_nz_L':>6} "
        f"{'A_iia':>6} {'A_layer':>7} "
        f"{'B_iia':>6} {'B_pick':<28} "
        f"{'rowsim':>6}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for e in EXPERIMENTS:
        if e.split not in cache:
            print(f"{e.name:<28}  -- skipped (split missing)")
            continue
        outs, iia = cache[e.split]
        ds = bundle.train_data[e.split]
        cfg_e = PlotConfig(
            **{**base_cfg.__dict__,
               "variables": tuple(e.variables),
               "cost_metric": e.cost_metric}
        )
        try:
            r = run_stages(
                bundle=bundle, fit_dataset=ds,
                outputs=outs, iia_by_site=iia,
                variables=tuple(e.variables),
                cost_metric=e.cost_metric,
                config=cfg_e,
            )
        except Exception as exc:
            print(f"{e.name:<28}  ERROR: {exc!r}")
            continue
        a = r["stage_a"]; b = r["stage_b"]
        b_pick = ",".join(f"({L},{t})" for L, t in b["picks"][:2])
        line = (
            f"{e.name:<28} {len(e.variables):>3}  {e.split:<35} {e.cost_metric:<7} "
            f"{a['top1_mass']:>9.4f} {a['nonzero_layers']:>6d} "
            f"{a['iia']:>6.3f} {str(a['picks']):>7} "
            f"{b['iia']:>6.3f} {b_pick:<28} "
            f"{r['abstract_cosines_offdiag_max']:>6.3f}"
        )
        print(line)
        results.append((e, r))

    print()
    print("Legend:")
    print("  V              # OT rows")
    print("  A_pi_top1      target-row Stage-A π's heaviest mass (1/24 = 0.0417 = uniform)")
    print("  A_nz_L         # layers with nonzero target-row mass (24 = uniform)")
    print("  A_iia          calibration IIA (mean per-site) at picked layer(s)")
    print("  A_layer        layer index(es) Stage A picked")
    print("  B_iia          calibration IIA at picked (layer, token_position) site(s)")
    print("  B_pick         the actual sites Stage B selected")
    print("  rowsim         max |off-diagonal cosine| between OT abstract rows")
    print("                 (high ⇒ V collapsed to ~1; we want low)")


if __name__ == "__main__":
    main()
