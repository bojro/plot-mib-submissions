"""Diagnostic: dump the per-(layer, token_position) cost matrix.

Answers the question: at the GRANULAR level — before Stage A's per-layer
mean aggregation — is (L15, last_token) the cost-min for any OT row? Or is
(L23, last_token) cost-min everywhere even at the granular level?

If L15 is granularly cost-competitive but loses at the aggregated level,
that confirms H1 (mean-aggregation dilutes L15). If L23 wins granularly
too, the bottleneck is the output-space signature itself (H2), not the
aggregation.

No DAS training, no submission write — pure diagnostic.

Usage::
    .venv-mib/bin/python -m mib_submission.plot.diagnose_costs
"""

from __future__ import annotations

import torch

from ..pipeline import setup_residual_experiment
from ..site_keys import site_key_for_unit
from .features import (
    build_abstract_table,
    collect_neural_outputs,
    expected_cf_letter_indices,
    normalize_rows,
    per_site_iia,
    signatures_from_outputs,
)
from .pipeline import _aggregate_to_layer_table
from .transport import cost_matrix


# Mirror run.py defaults
TASK = "4_answer_MCQA"
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
VARIABLE = "answer"
DATASET_SIZE = 256
FIT_SPLIT = "answerPosition_randomLetter_train"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
OT_VARIABLES = ("choice0", "choice1", "choice2", "choice3")
COST_METRIC = "sq_l2"
TOP_N = 8  # top-N (layer, pos) per OT row to print


def main() -> None:
    print(f"[diag] cell = {TASK} × Qwen2.5-0.5B × {VARIABLE}")
    from transformers import AutoConfig
    layers = list(range(AutoConfig.from_pretrained(MODEL_NAME).num_hidden_layers))
    bundle = setup_residual_experiment(
        task=TASK,
        model_name=MODEL_NAME,
        layers=layers,
        target_variables=[VARIABLE],
        dtype=torch.float32,
        dataset_size=DATASET_SIZE,
        verbose=False,
    )
    fit = bundle.train_data[FIT_SPLIT]
    print(f"[diag] split = {FIT_SPLIT}, n = {len(fit.dataset)}")

    abstract = build_abstract_table(
        bundle.causal_model, fit,
        variables=OT_VARIABLES, letters=LETTERS, normalize=True,
    )                                                              # (V, K)
    print(f"[diag] abstract_table = {tuple(abstract.shape)}; rows = {OT_VARIABLES}")

    outputs = collect_neural_outputs(bundle, fit, letters=LETTERS, verbose=False)
    site_sigs = signatures_from_outputs(outputs, normalize=True)   # {(L, t): (K,)}

    # IIA against the actual submission target (answer_pointer), per site,
    # so we can compare cost-rank against IIA-rank.
    expected = expected_cf_letter_indices(
        bundle.causal_model, fit, variable=VARIABLE, letters=LETTERS,
    )
    iia = per_site_iia(outputs, expected)

    # ---- Granular: cost matrix over (layer, position) tuples ------------
    site_keys = sorted(site_sigs.keys(), key=lambda k: (int(k[0]), str(k[1])))
    site_table = torch.stack([site_sigs[k] for k in site_keys], dim=0)  # (S, K)
    site_table = normalize_rows(site_table)
    granular = cost_matrix(abstract, site_table, metric=COST_METRIC)    # (V, S)
    print(f"[diag] granular cost = {tuple(granular.shape)} ((V, num_sites))")

    print("\n=== GRANULAR (V × site) cost ranks — top per OT row ===")
    for r, var in enumerate(OT_VARIABLES):
        order = torch.argsort(granular[r])
        print(f"\n  row[{r}] = {var}:")
        print(f"    {'rank':>4}  {'site':<24}  {'cost':>7}  {'IIA':>5}")
        for rk in range(TOP_N):
            idx = int(order[rk])
            (L, t) = site_keys[idx]
            c = float(granular[r, idx])
            print(f"    {rk+1:>4}  L{L:>2},{t:<18}  {c:>7.4f}  {iia.get(site_keys[idx], 0.0):>5.3f}")
        # Highlight L15/L20/L23 last_token regardless of rank
        for tgt in [(15, "last_token"), (20, "last_token"), (23, "last_token")]:
            if tgt in site_keys:
                idx = site_keys.index(tgt)
                rk = int((order == idx).nonzero()[0])
                c = float(granular[r, idx])
                print(f"      [tgt L{tgt[0]:>2},{tgt[1]}] rank={rk+1}/{len(site_keys)} cost={c:.4f} IIA={iia.get(tgt, 0.0):.3f}")

    # ---- Layer-aggregated (current Stage A behaviour) -------------------
    layer_table, layer_ids = _aggregate_to_layer_table(site_sigs, normalize=True)
    layer_cost = cost_matrix(abstract, layer_table, metric=COST_METRIC)  # (V, L)
    print("\n=== LAYER-AGGREGATED (V × layer) cost ranks — current Stage A ===")
    for r, var in enumerate(OT_VARIABLES):
        order = torch.argsort(layer_cost[r])
        print(f"\n  row[{r}] = {var}:")
        print(f"    {'rank':>4}  {'layer':>5}  {'cost':>7}")
        for rk in range(TOP_N):
            idx = int(order[rk])
            L = layer_ids[idx]
            c = float(layer_cost[r, idx])
            print(f"    {rk+1:>4}  L{L:>2}     {c:>7.4f}")
        for tgt_L in [15, 20, 23]:
            if tgt_L in layer_ids:
                idx = layer_ids.index(tgt_L)
                rk = int((order == idx).nonzero()[0])
                c = float(layer_cost[r, idx])
                print(f"      [tgt L{tgt_L:>2}] rank={rk+1}/{len(layer_ids)} cost={c:.4f}")


if __name__ == "__main__":
    main()
