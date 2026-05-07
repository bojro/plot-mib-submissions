"""End-to-end PLOT submission driver for one MIB cell.

Default cell: ``4_answer_MCQA × Qwen2.5-0.5B × answer_pointer``. Edit the
``CONFIG`` block below to retarget — same convention as the other ``*_run.py``
scripts in this repo.

Pipeline:
    1. ``setup_residual_experiment`` builds an ``ExperimentBundle`` for the
       cell with all 24 layers × 3 token positions declared.
    2. ``plot.select_sites_via_plot`` runs Stage A (layer OT) and Stage B
       (per-selected-layer site OT) on a train split, returning the
       surviving ``(layer, token_position)`` sites.
    3. Prune ``bundle.experiment.model_units_lists`` to those sites only.
    4. ``experiment.train_interventions(method="DAS")`` trains rotations on
       just those sites. Submission ships only those triplets.
    5. Run ``verify_submission.py`` for a sanity check.

Usage::

    .venv-mib/bin/python -m mib_submission.plot.run
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import torch

from ..pipeline import (
    REPO_ROOT,
    MIB_TRACK,
    add_mib_to_syspath,
    setup_residual_experiment,
)
from ..site_keys import site_key_for_unit

add_mib_to_syspath()  # serialize transitively imports CausalAbstraction

from ..serialize import cell_folder_name  # noqa: E402
from .pipeline import PlotConfig, select_sites_via_plot  # noqa: E402
from .bucketed import BucketedPlotConfig, select_sites_via_bucketed_plot  # noqa: E402


# --------------------------------------------------------------------------- #
# CONFIG                                                                       #
# --------------------------------------------------------------------------- #
TASK = "4_answer_MCQA"
MODEL_NAME = "google/gemma-2-2b"
MODEL_CLASS_NAME = "Gemma2ForCausalLM"
VARIABLE = "answer_pointer"

LAYERS: list[int] | None = None  # None ⇒ all layers of the model
N_FEATURES = 16
# Source default is `--das-epochs 12`; one epoch was a smoke-test setting.
TRAINING_EPOCHS = 12
INIT_LR = 1e-3
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 256
DATASET_SIZE: int | None = 256

PLOT_CONFIG = PlotConfig(
    # OT rows = V=8 mixed (4 choice + 4 symbol). Two complementary probes:
    #   - choice_i: "swap the i-th color word" → probes pointer mechanism.
    #     About 25% of examples per row get dropped (the ones where the
    #     answer was at position i, leaving no matching color → None pointer).
    #     The drop is systematically biased per row.
    #   - symbol_i: "swap the i-th letter label" → probes letter-copy
    #     mechanism. Never breaks the causal model — bias-free signal.
    # Multi-row Stage A picks 1 layer per row (up to 8 distinct layers).
    # Symbol rows contribute bias-free signal; choice rows still contribute
    # pointer-targeted signal. Requires answerPosition_randomLetter split
    # (only one where both probes are non-trivial).
    variables=("choice0", "choice1", "choice2", "choice3"),
    # Score each candidate by IIA on the actual submission target.
    calibration_variable=VARIABLE,
    letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
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
)

PLOT_SIGNATURE_DATASET: str | None = "answerPosition_randomLetter_train"

# Disambiguation knob: if set, skip Stage A/B and train DAS directly at these
# (layer, token_position) sites. Used to test whether off-PLOT layers beat
# PLOT's picks on the hard split. Set to None to run the normal PLOT pipeline.
BYPASS_SITES: list[tuple[int, str]] | None = None

# When True, dispatch to bucketed PLOT: V buckets indexed by source's
# value of BUCKETED_SOURCE_VARIABLE, all probing interchange(VARIABLE).
USE_BUCKETED_PLOT: bool = False
BUCKETED_PLOT_CONFIG = BucketedPlotConfig(
    target_variable=VARIABLE,
    source_variable=VARIABLE,        # bucket by source's pointer value
    n_buckets=4,                      # MCQA has 4 pointer values
    letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    cost_metric="sq_l2",
    normalize_signatures=True,
    stage_a_solver="ot",
    stage_b_solver="ot",
    sinkhorn_iters=200,
    stage_a_epsilon_grid=(0.01, 0.03),
    stage_b_epsilon_grid=(0.003, 0.01, 0.03, 0.1),
    stage_a_top_k_grid=(1,),
    stage_b_top_k_grid=(1, 2),
    calibration_variable=VARIABLE,
)
# Why this split:
#   answerPosition_train     — symbols don't vary; symbol_i interchange is a no-op
#   randomLetter_train       — answer_pointer doesn't vary; IIA against pointer is undefined
#   answerPosition_randomLetter_train — BOTH vary, so symbol_i rows are genuinely
#       distinct AND answer_pointer interchange is informative for IIA scoring.

SUBMISSION_ROOT = REPO_ROOT / "submissions" / "plot"
RUN_VERIFY = True
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


def main() -> Path:
    print(f"[plot] cell = {TASK} × {MODEL_CLASS_NAME} × {VARIABLE}")
    print(f"[plot] model = {MODEL_NAME}, dtype = {DTYPE}")

    layers = LAYERS
    if layers is None:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(MODEL_NAME)
        layers = list(range(cfg.num_hidden_layers))
    print(f"[plot] candidate layers = {layers}")

    config_overrides = {
        "training_epoch": TRAINING_EPOCHS,
        "init_lr": INIT_LR,
        "n_features": N_FEATURES,
        "batch_size": TRAIN_BATCH_SIZE,
        "evaluation_batch_size": EVAL_BATCH_SIZE,
        "output_scores": False,
    }

    bundle = setup_residual_experiment(
        task=TASK,
        model_name=MODEL_NAME,
        layers=layers,
        target_variables=[VARIABLE],
        dtype=DTYPE,
        dataset_size=DATASET_SIZE,
        config_overrides=config_overrides,
        verbose=True,
    )
    if bundle.model_class_name != MODEL_CLASS_NAME:
        raise RuntimeError(
            f"Loaded class {bundle.model_class_name!r} != expected {MODEL_CLASS_NAME!r}."
        )
    if not bundle.train_data:
        raise RuntimeError("No train splits returned by FilterExperiment.")
    print(f"[plot] train splits = {sorted(bundle.train_data.keys())}")

    # ---- PLOT site selection (Stage A + Stage B) -------------------------
    if BYPASS_SITES is not None:
        print(f"[plot] BYPASS_SITES set; skipping Stage A/B")
        print(f"[plot] hardcoded sites: {BYPASS_SITES}")
        selected_set = set(BYPASS_SITES)
    elif USE_BUCKETED_PLOT:
        fit_split = PLOT_SIGNATURE_DATASET or sorted(bundle.train_data.keys())[0]
        print(f"[plot] running BUCKETED Stage A + Stage B on split {fit_split!r}")
        selection = select_sites_via_bucketed_plot(
            bundle,
            bundle.train_data[fit_split],
            config=BUCKETED_PLOT_CONFIG,
            verbose=True,
        )
        a_eps, a_topk, a_score = selection.stage_a_chosen
        b_eps, b_topk, b_score = selection.stage_b_chosen
        print(f"[plot] Stage A best: eps={a_eps} top_k={a_topk} score={a_score:.4f}")
        print(f"[plot] Stage A picked layers: {selection.stage_a_layers}")
        print(f"[plot] Stage B best: eps={b_eps} top_k={b_topk} score={b_score:.4f}")
        print(f"[plot] Stage B selected sites: {selection.selected_sites}")
        selected_set = set(selection.selected_sites)
    else:
        fit_split = PLOT_SIGNATURE_DATASET or sorted(bundle.train_data.keys())[0]
        print(f"[plot] running Stage A + Stage B on split {fit_split!r}")
        selection = select_sites_via_plot(
            bundle,
            bundle.train_data[fit_split],
            config=PLOT_CONFIG,
            verbose=True,
        )
        a_eps, a_topk, a_score = selection.stage_a_chosen
        b_eps, b_topk, b_score = selection.stage_b_chosen
        print(f"[plot] Stage A best: eps={a_eps} top_k={a_topk} IIA={a_score:.4f}")
        print(f"[plot] Stage A picked layers: {selection.stage_a_layers}")
        print("[plot] Stage A π (target row):")
        target_row = selection.stage_a_pi[PLOT_CONFIG.target_row_index]
        for L, m in sorted(zip(range(target_row.numel()), target_row.tolist()), key=lambda x: -x[1])[:10]:
            print(f"  L{L:>2} mass={m:.4f}")
        print(f"[plot] Stage B best: eps={b_eps} top_k={b_topk} IIA={b_score:.4f}")
        print("[plot] Stage B selected sites:")
        for s in selection.selected_sites:
            print(f"  {s}")
        selected_set = set(selection.selected_sites)

    # ---- Prune the bundle to the selected sites only ---------------------
    pruned_units_lists = [
        mul for mul in bundle.experiment.model_units_lists
        if site_key_for_unit(mul[0][0]) in selected_set
    ]
    if not pruned_units_lists:
        raise RuntimeError(
            f"No bundle sites match PLOT selection {selection.selected_sites}; "
            "check that token-position ids align between PLOT and the experiment."
        )
    print(f"[plot] pruned model_units_lists: {len(pruned_units_lists)} site(s)")
    bundle.experiment.model_units_lists = pruned_units_lists

    # ---- Stage C: DAS at the surviving sites -----------------------------
    print(f"[plot] training DAS at {len(pruned_units_lists)} sites")
    bundle.experiment.train_interventions(
        bundle.train_data,
        [VARIABLE],
        method="DAS",
        verbose=True,
    )

    # ---- Write the cell --------------------------------------------------
    cell = cell_folder_name(TASK, MODEL_CLASS_NAME, VARIABLE)
    cell_dir = SUBMISSION_ROOT / cell
    if cell_dir.exists():
        print(f"[plot] removing existing {cell_dir}")
        shutil.rmtree(cell_dir)
    cell_dir.mkdir(parents=True)
    bundle.experiment.save_featurizers(None, str(cell_dir))
    n_files = sum(1 for _ in cell_dir.iterdir())
    print(f"[plot] wrote {n_files} files to {cell_dir}")

    # ---- Verify ----------------------------------------------------------
    if RUN_VERIFY:
        verify_script = MIB_TRACK / "verify_submission.py"
        cmd = [sys.executable, str(verify_script), str(SUBMISSION_ROOT)]
        print(f"[plot] $ {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        if result.returncode != 0:
            raise RuntimeError(f"verify_submission.py exited {result.returncode}.")

    return cell_dir


if __name__ == "__main__":
    main()
