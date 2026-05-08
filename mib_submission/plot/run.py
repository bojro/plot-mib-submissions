"""End-to-end PLOT submission driver for one MIB cell.

Pipeline:
    1. ``setup_residual_experiment`` builds an ``ExperimentBundle`` for the
       cell with all layers × token positions declared.
    2. ``plot.select_sites_via_plot`` runs Stage A (layer OT) and Stage B
       (per-selected-layer site OT) on a train split, returning the
       surviving ``(layer, token_position)`` sites.
    3. Prune ``bundle.experiment.model_units_lists`` to those sites only.
    4. ``experiment.train_interventions(method="DAS")`` trains rotations on
       just those sites. Submission ships only those triplets.
    5. Run ``verify_submission.py`` for a sanity check.

CLI usage::

    .venv-mib/bin/python -m mib_submission.plot.run \
        --task 4_answer_MCQA \
        --model google/gemma-2-2b \
        --variable answer_pointer

No-args usage replicates the previous behavior — defaults match cell 4
(MCQA × Gemma × answer). Edit ``DEFAULT_CELL`` below to retarget the
no-arg path; otherwise prefer CLI flags.
"""

from __future__ import annotations

import argparse
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
from .configs import RunConfig, default_config  # noqa: E402
from .pipeline import select_sites_via_plot  # noqa: E402
from .bucketed import select_sites_via_bucketed_plot  # noqa: E402


# --------------------------------------------------------------------------- #
# DEFAULT_CELL — used when run with no CLI args                                #
# --------------------------------------------------------------------------- #
DEFAULT_TASK = "4_answer_MCQA"
DEFAULT_MODEL = "google/gemma-2-2b"
DEFAULT_VARIABLE = "answer"

SUBMISSION_ROOT = REPO_ROOT / "submissions" / "plot"
RUN_VERIFY = True
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


def main(config: RunConfig) -> Path:
    print(f"[plot] cell = {config.task} × {config.model_class_name} × {config.variable}")
    print(f"[plot] model = {config.model_name}, dtype = {DTYPE}")

    layers = list(config.layers) if config.layers else None
    if layers is None:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(config.model_name)
        layers = list(range(cfg.num_hidden_layers))
    print(f"[plot] candidate layers = {layers}")

    config_overrides = {
        "training_epoch": config.training_epochs,
        "init_lr": config.init_lr,
        "n_features": config.n_features,
        "batch_size": config.train_batch_size,
        "evaluation_batch_size": config.eval_batch_size,
        "output_scores": False,
    }

    bundle = setup_residual_experiment(
        task=config.task,
        model_name=config.model_name,
        layers=layers,
        target_variables=[config.variable],
        dtype=DTYPE,
        dataset_size=config.dataset_size,
        config_overrides=config_overrides,
        checker=config.checker,
        max_new_tokens=config.max_new_tokens,
        verbose=True,
    )
    if bundle.model_class_name != config.model_class_name:
        raise RuntimeError(
            f"Loaded class {bundle.model_class_name!r} != expected "
            f"{config.model_class_name!r}."
        )
    if not bundle.train_data:
        raise RuntimeError("No train splits returned by FilterExperiment.")
    print(f"[plot] train splits = {sorted(bundle.train_data.keys())}")

    # ---- PLOT site selection (Stage A + Stage B) -------------------------
    if config.bypass_sites is not None:
        print(f"[plot] bypass_sites set; skipping Stage A/B")
        print(f"[plot] hardcoded sites: {list(config.bypass_sites)}")
        selected_set = set(config.bypass_sites)
        selection = None
    else:
        fit_split = config.signature_dataset or sorted(bundle.train_data.keys())[0]
        if fit_split not in bundle.train_data:
            raise RuntimeError(
                f"signature_dataset={fit_split!r} not in train splits "
                f"{sorted(bundle.train_data.keys())}"
            )
        if config.use_bucketed_plot:
            raise NotImplementedError(
                "use_bucketed_plot=True path needs a BucketedPlotConfig — "
                "not yet plumbed through configs.py. Restore the bucketed "
                "branch in run.py manually if needed."
            )
        print(f"[plot] running Stage A + Stage B on split {fit_split!r}")
        selection = select_sites_via_plot(
            bundle,
            bundle.train_data[fit_split],
            config=config.plot_config,
            verbose=True,
        )
        a_eps, a_topk, a_score = selection.stage_a_chosen
        b_eps, b_topk, b_score = selection.stage_b_chosen
        print(f"[plot] Stage A best: eps={a_eps} top_k={a_topk} IIA={a_score:.4f}")
        print(f"[plot] Stage A picked layers: {selection.stage_a_layers}")
        print("[plot] Stage A π (target row):")
        target_row = selection.stage_a_pi[config.plot_config.target_row_index]
        for L, m in sorted(
            zip(range(target_row.numel()), target_row.tolist()), key=lambda x: -x[1]
        )[:10]:
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
        sites_str = (
            list(selection.selected_sites) if selection is not None
            else list(config.bypass_sites or [])
        )
        raise RuntimeError(
            f"No bundle sites match selection {sites_str}; "
            "check that token-position ids align between PLOT and the experiment."
        )
    print(f"[plot] pruned model_units_lists: {len(pruned_units_lists)} site(s)")
    bundle.experiment.model_units_lists = pruned_units_lists

    # ---- Stage C: DAS at the surviving sites -----------------------------
    print(f"[plot] training DAS at {len(pruned_units_lists)} sites")
    bundle.experiment.train_interventions(
        bundle.train_data,
        [config.variable],
        method="DAS",
        verbose=True,
    )

    # ---- Write the cell --------------------------------------------------
    cell = cell_folder_name(config.task, config.model_class_name, config.variable)
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


def _parse_bypass_sites(s: str) -> tuple[tuple[int, str], ...]:
    """Parse "23:last_token,17:correct_symbol" into ((23, "last_token"), ...)."""
    out = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        layer_str, tok = chunk.split(":", 1)
        out.append((int(layer_str), tok.strip()))
    return tuple(out)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run PLOT for one MIB cell.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--variable", default=DEFAULT_VARIABLE)
    p.add_argument("--epochs", type=int, default=None,
                   help="Override training_epochs in the preset.")
    p.add_argument("--n-features", type=int, default=None)
    p.add_argument("--init-lr", type=float, default=None)
    p.add_argument("--dataset-size", type=int, default=None)
    p.add_argument("--train-batch-size", type=int, default=None)
    p.add_argument("--eval-batch-size", type=int, default=None)
    p.add_argument("--signature-dataset", default=None,
                   help="Train split key for PLOT signature collection.")
    p.add_argument("--bypass-sites", default=None,
                   help='Skip Stage A/B and train DAS at these sites. '
                        'Format: "L:tok,L:tok" e.g. "23:last_token,17:correct_symbol".')
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    overrides: dict = {}
    if args.epochs is not None: overrides["training_epochs"] = args.epochs
    if args.n_features is not None: overrides["n_features"] = args.n_features
    if args.init_lr is not None: overrides["init_lr"] = args.init_lr
    if args.dataset_size is not None: overrides["dataset_size"] = args.dataset_size
    if args.train_batch_size is not None: overrides["train_batch_size"] = args.train_batch_size
    if args.eval_batch_size is not None: overrides["eval_batch_size"] = args.eval_batch_size
    if args.signature_dataset is not None: overrides["signature_dataset"] = args.signature_dataset
    if args.bypass_sites is not None:
        overrides["bypass_sites"] = _parse_bypass_sites(args.bypass_sites)

    cfg = default_config(
        task=args.task,
        model_name=args.model,
        variable=args.variable,
        overrides=overrides,
    )
    main(cfg)
