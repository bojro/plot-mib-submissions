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
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def _set_global_seed(seed: int) -> None:
    """Seed every RNG that affects DAS rotation init, DataLoader shuffle,
    and any numpy-driven randomness. Stage A/B itself is deterministic
    (Sinkhorn is iterative but seed-free), so this primarily disambiguates
    DAS-train variance for the seed-sweep experiments.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[run] seeded all RNGs with {seed}")

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


def main_ioi(config: RunConfig, *, no_das: bool = False) -> Path | None:
    """End-to-end IOI cell — attention-head sites, MSE-on-logit-diff DAS,
    nested-folder submission, IOI-specific evaluator.

    Differences from the residual-stream ``main``:

    1. Loads linear params from ``submissions/plot/ioi_linear_params.json``
       (bootstrapped via ``mib_submission.ioi.bootstrap``).
    2. Calls ``setup_attention_head_experiment`` with all (layer, head)
       pairs auto-detected from the model config.
    3. After PLOT picks (layer, head) sites, rebuilds a joint-mode
       ``PatchAttentionHeads`` with only those heads and trains DAS.
    4. Writes nested ``ioi_task_M_V/DAS_M_V/`` layout + linear params
       JSON sibling. Skips the standard ``write_submission`` path.
    """
    from ..pipeline import setup_attention_head_experiment
    from ..ioi import (
        ensure_linear_params_json,
        load_linear_params,
        write_ioi_submission,
        LINEAR_PARAMS_FILENAME,
    )
    from ..ioi.bootstrap import model_short_name
    from ..ioi._patches import patch_lm_pipeline_load, patch_model_config_head_dim
    from ..site_keys import attention_head_site_key_for_unit

    print(f"[ioi] cell = {config.task} × {config.model_class_name} × {config.variable}")

    # Apply pipeline patches before the harness loads any LMPipeline /
    # config-dependent code. patch_lm_pipeline_load is idempotent.
    patch_lm_pipeline_load()

    # Load linear params (bootstrap them if missing).
    short = model_short_name(config.model_class_name)
    params_path = SUBMISSION_ROOT / LINEAR_PARAMS_FILENAME
    if not params_path.is_file():
        raise RuntimeError(
            f"{params_path} missing — run\n"
            f"    .venv-mib/bin/python -c "
            f"'from mib_submission.ioi import bootstrap_linear_params; "
            f"bootstrap_linear_params({short!r})'\n"
            f"first."
        )
    linear_params = load_linear_params(params_path, model_short=short)
    print(
        f"[ioi] linear params: bias={linear_params.bias:.4f} "
        f"token_coeff={linear_params.token_coeff:.4f} "
        f"position_coeff={linear_params.position_coeff:.4f} "
        f"R²={linear_params.score:.4f}"
    )

    # Build the (layer, head) candidate list.
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(config.model_name)
    n_layers = cfg.num_hidden_layers
    n_heads = (
        getattr(cfg, "num_attention_heads", None)
        or getattr(cfg, "n_head", None)
        or getattr(cfg, "num_heads", None)
    )
    if config.layers is not None:
        layer_range = list(config.layers)
    else:
        layer_range = list(range(n_layers))
    layer_head_list = [(L, H) for L in layer_range for H in range(n_heads)]
    print(f"[ioi] candidate sites: {n_layers}L × {n_heads}H = {len(layer_head_list)}")

    config_overrides = {
        "training_epoch": config.training_epochs,
        "init_lr": config.init_lr,
        "n_features": config.n_features,
        "batch_size": config.train_batch_size,
        "evaluation_batch_size": config.eval_batch_size,
    }

    bundle = setup_attention_head_experiment(
        model_name=config.model_name,
        layer_head_list=layer_head_list,
        target_variables=[config.variable],
        linear_params=linear_params.as_dict(),
        dtype=DTYPE,
        dataset_size=config.dataset_size,
        config_overrides=config_overrides,
        verbose=True,
        per_site_units=True,
    )
    # Inject head_dim onto the model config (Qwen needs it).
    patch_model_config_head_dim(bundle.pipeline.model.config)

    if not bundle.train_data:
        raise RuntimeError("No train splits returned by FilterExperiment.")
    print(f"[ioi] train splits = {sorted(bundle.train_data.keys())}")

    # ---- PLOT site selection (Stage A + Stage B) ----------------------
    if config.bypass_sites is not None:
        print(f"[ioi] bypass_sites set; skipping Stage A/B")
        print(f"[ioi] hardcoded sites: {list(config.bypass_sites)}")
        picked_heads = []
        for s in config.bypass_sites:
            if len(s) == 2:
                # Format: (layer, head) — no token position; default to all.
                L, H = s
                picked_heads.append((int(L), int(H)))
            elif len(s) == 3:
                L, H, _tok = s
                picked_heads.append((int(L), int(H)))
            else:
                raise ValueError(
                    f"IOI bypass_sites entries must be (layer, head) or "
                    f"(layer, head, token); got {s!r}"
                )
        print(f"[ioi] hardcoded picked_heads: {picked_heads}")
    else:
        fit_split = config.signature_dataset or sorted(bundle.train_data.keys())[0]
        print(f"[ioi] running Stage A + Stage B on per-row splits "
              f"{config.plot_config.per_row_split_datasets!r}")
        selection = select_sites_via_plot(
            bundle, bundle.train_data[fit_split],
            config=config.plot_config, verbose=True,
        )
        a_eps, a_topk, a_score = selection.stage_a_chosen
        b_eps, b_topk, b_score = selection.stage_b_chosen
        print(f"[ioi] Stage A best: eps={a_eps} top_k={a_topk} IIA={a_score:.4f}")
        print(f"[ioi] Stage A picked layers: {selection.stage_a_layers}")
        print(f"[ioi] Stage B best: eps={b_eps} top_k={b_topk} IIA={b_score:.4f}")
        print("[ioi] Stage B selected sites (layer, head, token):")
        picked_heads = []
        for s in selection.selected_sites:
            if len(s) == 3:
                L, H, _tok = s
                picked_heads.append((int(L), int(H)))
                print(f"  L{L}H{H}/{_tok}")
            else:
                print(f"  (unexpected residual key: {s})")
        if not picked_heads:
            raise RuntimeError("No attention-head sites selected by PLOT.")

    if no_das:
        print("[ioi] --no-das set; skipping DAS and submission write")
        return None

    # ---- Stage C: DAS at picked heads ---------------------------------
    # Re-build PatchAttentionHeads in JOINT mode (single model_units_lists
    # entry covering all picked heads) so train_interventions can patch
    # them simultaneously, matching ioi_baselines / example notebook.
    print(f"[ioi] training DAS at {len(picked_heads)} heads: {picked_heads}")
    add_mib_to_syspath()
    from experiments.attention_head_experiment import PatchAttentionHeads  # type: ignore[import-not-found]
    sys.path.insert(0, str(MIB_TRACK / "baselines" / "ioi_baselines"))
    from ioi_utils import (  # type: ignore[import-not-found]
        ioi_loss_and_metric_fn,
        checker as ioi_checker,
    )

    das_config = {
        "evaluation_batch_size": config.eval_batch_size,
        "batch_size": config.train_batch_size,
        "training_epoch": config.training_epochs,
        "check_raw": True,
        "n_features": config.n_features,
        "regularization_coefficient": 0.0,
        "output_scores": True,
        "shuffle": True,
        "temperature_schedule": (1.0, 0.01),
        "init_lr": config.init_lr,
        "loss_and_metric_fn":
            lambda pipe, intervenable, batch, units:
                ioi_loss_and_metric_fn(pipe, intervenable, batch, units),
    }
    das_experiment = PatchAttentionHeads(
        pipeline=bundle.pipeline,
        causal_model=bundle.causal_model,
        layer_head_list=picked_heads,
        token_positions=bundle.token_positions,
        checker=lambda logits, params: ioi_checker(logits, params, bundle.pipeline),
        config=das_config,
    )
    das_experiment.train_interventions(
        bundle.train_data,
        [config.variable],
        method="DAS",
        verbose=True,
    )

    # ---- Write submission ---------------------------------------------
    cell_dir = write_ioi_submission(
        SUBMISSION_ROOT,
        experiment=das_experiment,
        model_class_name=config.model_class_name,
        variable=config.variable,
        overwrite=True,
    )
    ensure_linear_params_json(
        SUBMISSION_ROOT,
        model_short=short,
        model_class_name=config.model_class_name,
        params=linear_params,
    )
    print(f"[ioi] wrote submission to {cell_dir}")
    return cell_dir


def main(config: RunConfig, *, no_das: bool = False) -> Path | None:
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
        # Map row-table indices back to actual model layer numbers.
        candidate_layers = sorted({
            site_key_for_unit(mul[0][0])[0]
            for mul in bundle.experiment.model_units_lists
        }) or sorted(layers)
        for idx, m in sorted(
            zip(range(target_row.numel()), target_row.tolist()), key=lambda x: -x[1]
        )[:10]:
            actual_L = candidate_layers[idx] if idx < len(candidate_layers) else idx
            print(f"  L{actual_L:>2} mass={m:.4f}")
        print(f"[plot] Stage B best: eps={b_eps} top_k={b_topk} IIA={b_score:.4f}")
        print("[plot] Stage B selected sites:")
        for s in selection.selected_sites:
            print(f"  {s}")
        selected_set = set(selection.selected_sites)

    # ---- Diagnostic mode: stop after Stage A/B; report and exit ----------
    if no_das:
        print("[plot] --no-das set; skipping DAS training and submission write")
        if selection is not None:
            print(f"[plot] DIAGNOSTIC SUMMARY")
            print(f"[plot]   variant rows: {config.plot_config.variables}")
            print(f"[plot]   target row index: {config.plot_config.target_row_index} "
                  f"({config.plot_config.variables[config.plot_config.target_row_index]})")
            print(f"[plot]   stage A score: {selection.stage_a_chosen[2]:.4f}")
            print(f"[plot]   stage B score: {selection.stage_b_chosen[2]:.4f}")
            print(f"[plot]   final sites ({len(selection.selected_sites)}):")
            for s in selection.selected_sites:
                print(f"[plot]     {s}")
        return None

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


def _parse_bypass_sites(s: str) -> tuple[tuple, ...]:
    """Parse bypass-sites strings, supporting both formats:

    - Residual stream: ``"23:last_token,17:correct_symbol"`` →
      ``((23, "last_token"), (17, "correct_symbol"))``
    - Attention head: ``"7:8:all,9:6:all"`` →
      ``((7, 8, "all"), (9, 6, "all"))``
    - Attention head no-token: ``"7:8,9:6"`` → ``((7, 8), (9, 6))``
      (consumer defaults to all-token position)

    The number of colon-separated fields determines the format. Used by
    ``main`` (residual) and ``main_ioi`` (attention head).
    """
    out: list[tuple] = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(":")]
        if len(parts) == 2:
            a, b = parts
            try:
                # Both numeric → (layer, head). One numeric, one not →
                # residual (layer, token_id).
                bi = int(b)
                out.append((int(a), bi))
            except ValueError:
                out.append((int(a), b))
        elif len(parts) == 3:
            L, H, tok = parts
            out.append((int(L), int(H), tok))
        else:
            raise ValueError(
                f"bypass-sites chunk {chunk!r}: expected 'layer:tokid' or "
                f"'layer:head' or 'layer:head:tok'"
            )
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
    p.add_argument("--no-das", action="store_true",
                   help="Diagnostic mode: run Stage A+B, print picks, exit "
                        "before DAS training. Does not write a submission.")
    p.add_argument("--arithmetic-variant", choices=("C", "B"), default=None,
                   help="Arithmetic OT-row schema. C = V=2 carry children "
                        "(default); B = V=4 operand digits.")
    p.add_argument("--layers", default=None,
                   help="Comma-separated subset of layer indices to consider "
                        "(diagnostic; default = all layers).")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed for torch / numpy / random / cuda RNGs. "
                        "Used for seed-variance sweeps. Stage A/B are "
                        "deterministic; this primarily affects DAS rotation "
                        "init and DataLoader shuffle.")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    if args.seed is not None:
        _set_global_seed(int(args.seed))
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
    if args.arithmetic_variant is not None:
        overrides["arithmetic_variant"] = args.arithmetic_variant
    if args.layers is not None:
        overrides["layers"] = tuple(int(x) for x in args.layers.split(",") if x.strip())
    cfg = default_config(
        task=args.task,
        model_name=args.model,
        variable=args.variable,
        overrides=overrides,
    )
    if cfg.task == "ioi_task":
        main_ioi(cfg, no_das=args.no_das)
    else:
        main(cfg, no_das=args.no_das)
