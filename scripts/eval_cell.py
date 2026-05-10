"""Patched MIB-eval driver for one cell.

Why this exists:
- ``MIB-causal-variable-track/evaluate_submission.py:147`` hardcodes
  ``LMPipeline(..., max_new_tokens=1)`` for all tasks. Arithmetic answers
  are 2-3 digits and RAVEL answers are often 2 tokens (e.g. ``"United
  States"``); both filter to ~0% under that default. We monkey-patch
  ``get_task_module_and_pipeline`` to override per-task.
- The CLI flag ``--no-private_data`` does not exist in the pinned harness
  commit; we call ``evaluate_submission_task(..., private_data=False,
  public_data=True)`` directly.

Usage:

    .venv-mib/bin/python scripts/eval_cell.py \\
        --cell ravel_task_Gemma2ForCausalLM_Country

Cell folder must already exist at ``submissions/plot/<cell>``. Eval result
JSON is left in that folder by the harness; we additionally copy it to
``mib_submission/results/<task>_<model>_<variable>.json`` for archival.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SUBMISSION_ROOT = REPO / "submissions" / "plot"
RESULTS_ROOT = REPO / "mib_submission" / "results"
TRACK = REPO / "MIB" / "MIB-causal-variable-track"

# Per-task max_new_tokens. Arithmetic and RAVEL multi-token answers fail
# under the harness's default of 1.
MAX_NEW_TOKENS = {
    "4_answer_MCQA": 1,
    "ARC_easy": 1,
    "arithmetic": 3,
    "ravel_task": 2,
    "ioi_task": 1,
}


def _patch_pipeline_max_new_tokens(task: str) -> None:
    """Override ``evaluate_submission.get_task_module_and_pipeline`` so the
    LMPipeline is created with the right ``max_new_tokens`` for this task."""
    n = MAX_NEW_TOKENS.get(task, 1)
    if n == 1:
        return  # default; nothing to do

    import evaluate_submission  # type: ignore[import-not-found]

    original = evaluate_submission.get_task_module_and_pipeline

    def patched(_task, _model):
        # Run the original to load everything, then mutate the pipeline.
        task_module, pipeline, causal_model, gcd = original(_task, _model)
        pipeline.max_new_tokens = n
        print(f"[eval] patched LMPipeline.max_new_tokens = {n} (task={_task})")
        return task_module, pipeline, causal_model, gcd

    evaluate_submission.get_task_module_and_pipeline = patched


def _parse_cell(cell: str) -> tuple[str, str, str]:
    """Reverse of ``cell_folder_name``: split ``<task>_<model>_<variable>``
    by recognising the model-class token in the middle.

    Tasks like ``4_answer_MCQA`` and ``ARC_easy`` contain underscores, so a
    naive ``split('_', 2)`` doesn't work.
    """
    known_models = (
        "GPT2LMHeadModel",
        "Qwen2ForCausalLM",
        "Gemma2ForCausalLM",
        "LlamaForCausalLM",
    )
    for m in known_models:
        token = f"_{m}_"
        if token in cell:
            i = cell.index(token)
            task = cell[:i]
            variable = cell[i + len(token):]
            return task, m, variable
    raise ValueError(f"Cannot find a known model class name in {cell!r}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell", required=True, help="Cell folder name under submissions/plot/")
    p.add_argument("--no-archive", action="store_true",
                   help="Skip copying the results JSON into mib_submission/results/.")
    args = p.parse_args()

    cell_path = SUBMISSION_ROOT / args.cell
    if not cell_path.is_dir():
        print(f"[eval] ERROR: {cell_path} does not exist", file=sys.stderr)
        return 2

    task, model, variable = _parse_cell(args.cell)
    print(f"[eval] cell = {args.cell}")
    print(f"[eval]   task = {task}")
    print(f"[eval]   model = {model}")
    print(f"[eval]   variable = {variable}")

    # Wire up the harness imports.
    sys.path.insert(0, str(TRACK))
    sys.path.insert(0, str(TRACK / "CausalAbstraction"))

    _patch_pipeline_max_new_tokens(task)

    if task == "ioi_task":
        # The harness's LMPipeline.load reads ``position_ids`` from
        # ``model.prepare_inputs_for_generation(...)``, but transformers 5.x
        # doesn't always include that key, raising KeyError. Reuse the same
        # patch the training-side runner applies.
        sys.path.insert(0, str(REPO))
        from mib_submission.ioi._patches import patch_lm_pipeline_load  # type: ignore[import-not-found]
        patch_lm_pipeline_load()
        print("[eval] applied LMPipeline.load position_ids patch (IOI)")

    if task == "ioi_task":
        # IOI uses a separate eval entry point — reads ioi_linear_params.json
        # and runs attention-head interventions instead of residual-stream.
        from ioi_evaluate_submission import evaluate_ioi_submission_task  # type: ignore[import-not-found]
        ok = evaluate_ioi_submission_task(
            task_folder_path=str(cell_path),
            submission_base_path=str(SUBMISSION_ROOT),
            private_data=False,
            public_data=True,
        )
    else:
        from evaluate_submission import evaluate_submission_task  # type: ignore[import-not-found]
        ok = evaluate_submission_task(
            task_folder_path=str(cell_path),
            submission_base_path=str(SUBMISSION_ROOT),
            private_data=False,
            public_data=True,
        )
    if not ok:
        print(f"[eval] evaluate_submission_task returned False", file=sys.stderr)
        return 1

    # Locate the results JSON (the harness writes "<...>__results.json" inside
    # the cell folder) and archive it.
    results = sorted(cell_path.glob("*results.json"))
    if not results:
        print(f"[eval] WARN: no *results.json under {cell_path}", file=sys.stderr)
        return 0
    if args.no_archive:
        return 0

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    archive = RESULTS_ROOT / f"{task}_{model}_{variable}.json"
    # Take the most-recently-modified results file in case multiple eval runs
    # left stale ones behind.
    src = max(results, key=lambda p: p.stat().st_mtime)
    shutil.copy2(src, archive)
    print(f"[eval] archived {src.name} -> {archive}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
