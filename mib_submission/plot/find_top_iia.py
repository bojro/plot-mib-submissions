"""Direct IIA-max site finder on `answerPosition_randomLetter_train`.

The diagnostic showed `max IIA at any site = 0.9636` on this split. This
script loads outputs once, computes per-site IIA against `answer_pointer`,
and prints the top-10 sites — bypassing PLOT's OT step. Use the winner
as the site for a focused DAS training run.
"""

from __future__ import annotations

import torch

from ..pipeline import setup_residual_experiment
from .features import (
    collect_neural_outputs,
    expected_cf_letter_indices,
    per_site_iia,
)


TASK = "4_answer_MCQA"
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
VARIABLE = "answer_pointer"
SPLIT = "answerPosition_randomLetter_train"
DATASET_SIZE = 256
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def main() -> None:
    print(f"[topia] cell = {TASK} × {VARIABLE}, split = {SPLIT}", flush=True)

    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    layers = list(range(cfg.num_hidden_layers))

    bundle = setup_residual_experiment(
        task=TASK, model_name=MODEL_NAME,
        layers=layers, target_variables=[VARIABLE],
        dtype=DTYPE, dataset_size=DATASET_SIZE,
        config_overrides={"output_scores": False},
        verbose=False,
    )
    if SPLIT not in bundle.train_data:
        raise SystemExit(f"split {SPLIT!r} not in train_data: "
                         f"{sorted(bundle.train_data.keys())}")
    ds = bundle.train_data[SPLIT]
    print(f"[topia] split loaded, n={len(ds.dataset)}", flush=True)

    print(f"[topia] collecting per-site neural outputs ...", flush=True)
    outs = collect_neural_outputs(bundle, ds, letters=LETTERS, verbose=False)
    print(f"[topia] sites collected = {len(outs.cf_alpha_argmax)}", flush=True)

    expected = expected_cf_letter_indices(
        bundle.causal_model, ds, variable=VARIABLE, letters=LETTERS,
    )
    iia = per_site_iia(outs, expected)

    ranked = sorted(iia.items(), key=lambda kv: -kv[1])
    print()
    print(f"{'rank':>4}  {'layer':>5}  {'token':<22} {'IIA':>6}")
    print("-" * 44)
    for r, (key, score) in enumerate(ranked[:15], start=1):
        L, t = key
        print(f"{r:>4}  L{L:<4}  {t:<22} {score:>6.4f}")
    print()
    print(f"[topia] top site: {ranked[0][0]} with IIA = {ranked[0][1]:.4f}")


if __name__ == "__main__":
    main()
