"""Aggregate per-cell eval JSONs into a structured results table.

Single source of truth for ``RESULTS.md``. Run as a script to regenerate
the document from raw archives::

    .venv-mib/bin/python -m mib_submission.results._aggregate > \
        mib_submission/results/RESULTS.md

Reads two sources of data per cell:

1. ``mib_submission/results/<task>_<model>_<variable>.json`` — the eval
   harness's ``evaluate_submission_task`` output, containing per-(unit,
   split) average scores.
2. ``submissions/plot/<task>_<model>_<variable>/`` — the submission folder
   whose ``ResidualStream(Layer-L,Token-T)_indices`` files identify which
   sites were *trained* (vs identity baselines the evaluator reports for
   non-trained positions of trained layers).

Cells with archives but no submission folder (e.g. ablation runs) are
listed under "Other archived runs" rather than the main results.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "mib_submission" / "results"
SUBMISSIONS_DIR = REPO_ROOT / "submissions" / "plot"


# Public test splits in their canonical reporting order. Matches CLAUDE.md /
# EVAL_LOG.md conventions: easy → hard.
MCQA_SPLITS_ORDERED = (
    "answerPosition_test",
    "randomLetter_test",
    "answerPosition_randomLetter_test",
)
MCQA_SPLIT_SHORT = {
    "answerPosition_test": "aP",
    "randomLetter_test": "rL",
    "answerPosition_randomLetter_test": "aPrL",
}

# Filename regex for cell archives: <task>_<modelclass>_<variable>.json.
# Variable can contain underscores ("answer_pointer"), so we anchor task and
# model from the front.
_TASK_NAMES = (
    "4_answer_MCQA",
    "ARC_easy",
    "arithmetic",
    "ioi_task",
    "ravel_task",
)
_MODEL_NAMES = (
    "GPT2LMHeadModel",
    "Qwen2ForCausalLM",
    "Gemma2ForCausalLM",
    "LlamaForCausalLM",
)


@dataclass(frozen=True)
class UnitResult:
    layer: int
    position: str
    per_split_iia: Dict[str, float]


@dataclass(frozen=True)
class CellResult:
    task: str
    model_class: str
    variable: str
    splits: Tuple[str, ...]
    units: Tuple[UnitResult, ...]
    picked_sites: Optional[Tuple[Tuple[int, str], ...]]  # None if no submission folder
    picked_sites_inferred: bool = False  # True ⇒ picked_sites came from IIA heuristic

    @property
    def is_shipped(self) -> bool:
        return self.picked_sites is not None and len(self.picked_sites) > 0

    @property
    def per_split_max_iia(self) -> Dict[str, float]:
        return {s: max(u.per_split_iia[s] for u in self.units) for s in self.splits}

    @property
    def mean_iia(self) -> float:
        vals = list(self.per_split_max_iia.values())
        return sum(vals) / len(vals) if vals else float("nan")

    @property
    def best_site_per_split(self) -> Dict[str, UnitResult]:
        out: Dict[str, UnitResult] = {}
        for s in self.splits:
            out[s] = max(self.units, key=lambda u: u.per_split_iia[s])
        return out

    @property
    def picked_units(self) -> Tuple[UnitResult, ...]:
        if not self.picked_sites:
            return ()
        keys = set(self.picked_sites)
        return tuple(u for u in self.units if (u.layer, u.position) in keys)


def _parse_archive_filename(stem: str) -> Optional[Tuple[str, str, str]]:
    """Parse ``{task}_{model_class}_{variable}`` from a JSON stem."""
    for task in _TASK_NAMES:
        if not stem.startswith(task + "_"):
            continue
        rest = stem[len(task) + 1 :]
        for model in _MODEL_NAMES:
            if rest.startswith(model + "_"):
                return task, model, rest[len(model) + 1 :]
    return None


def _read_picked_sites(
    task: str, model_class: str, variable: str
) -> Optional[Tuple[Tuple[int, str], ...]]:
    """Inspect the submission folder for trained sites.

    The featurizer triplet filenames embed ``ResidualStream(Layer-L,Token-T)``;
    we extract one (layer, position) per ``_featurizer`` file. Returns None
    if the submission folder doesn't exist (e.g. ablation-only archives).
    """
    cell_dir = SUBMISSIONS_DIR / f"{task}_{model_class}_{variable}"
    if not cell_dir.is_dir():
        return None
    pat = re.compile(r"ResidualStream\(Layer-(\d+),Token-([^)]+)\)_featurizer$")
    sites = []
    for p in sorted(cell_dir.iterdir()):
        m = pat.match(p.name)
        if m:
            sites.append((int(m.group(1)), m.group(2)))
    return tuple(sites) or None


def _load_archive(path: Path) -> CellResult:
    parsed = _parse_archive_filename(path.stem)
    if parsed is None:
        raise ValueError(f"unrecognised archive name: {path.name}")
    task, model_class, variable = parsed
    raw = json.loads(path.read_text())
    splits_raw = list(raw["dataset"].keys())
    # Order canonical splits first, then any others alphabetically.
    splits_canon = [s for s in MCQA_SPLITS_ORDERED if s in splits_raw]
    extras = sorted(set(splits_raw) - set(splits_canon))
    splits = tuple(splits_canon + extras)

    # Aggregate by (layer, position): a single unit may appear under each split.
    unit_iia: Dict[Tuple[int, str], Dict[str, float]] = {}
    for split in splits:
        for unit_key, unit_blob in raw["dataset"][split]["model_unit"].items():
            md = unit_blob["metadata"]
            layer = int(md["layer"])
            pos = str(md["position"])
            score = float(unit_blob[variable]["average_score"])
            unit_iia.setdefault((layer, pos), {})[split] = score

    units = tuple(
        UnitResult(
            layer=layer,
            position=pos,
            per_split_iia={s: scores.get(s, float("nan")) for s in splits},
        )
        for (layer, pos), scores in sorted(unit_iia.items())
    )

    folder_picks = _read_picked_sites(task, model_class, variable)
    inferred = False
    if folder_picks is None:
        # Heuristic fallback: identity baselines on splits where the cell
        # variable doesn't change can score perfectly (e.g. `randomLetter_test`
        # for `answer_pointer` — pointer is identical between base and source,
        # so identity is the right answer). To avoid that confound, threshold
        # IIA on the "informative" split only — the one where identity is
        # weakest. For MCQA-style splits that's `answerPosition_test`.
        informative_splits = ("answerPosition_test",)
        threshold = 0.3
        cand: List[Tuple[int, str]] = []
        for u in units:
            relevant = [u.per_split_iia[s] for s in informative_splits if s in u.per_split_iia]
            if relevant and max(relevant) >= threshold:
                cand.append((u.layer, u.position))
        if cand:
            folder_picks = tuple(cand)
            inferred = True

    return CellResult(
        task=task,
        model_class=model_class,
        variable=variable,
        splits=splits,
        units=units,
        picked_sites=folder_picks,
        picked_sites_inferred=inferred,
    )


def load_all() -> List[CellResult]:
    """Load every JSON archive whose filename matches the cell schema."""
    out = []
    skipped: list[Tuple[str, str]] = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        try:
            out.append(_load_archive(p))
        except ValueError as e:
            skipped.append((p.name, str(e)))
    if skipped:
        print(
            f"<!-- skipped non-cell archives: {[n for n, _ in skipped]} -->",
            file=sys.stderr,
        )
    return out


# --------------------------------------------------------------------------- #
# Markdown rendering                                                          #
# --------------------------------------------------------------------------- #

def _fmt_iia(x: float) -> str:
    return f"{x:.3f}" if x == x else "—"  # NaN guard


def _split_short(s: str) -> str:
    return MCQA_SPLIT_SHORT.get(s, s.replace("_test", ""))


def headline_table(results: List[CellResult]) -> str:
    """One sub-table per task (cells in different tasks have different test
    splits, e.g. MCQA's ``answerPosition_test`` vs RAVEL's ``attribute_test``,
    so a single table with shared columns isn't well-defined). Each task's
    sub-table has one row per cell shipped in that task.
    """
    shipped = [r for r in results if r.is_shipped]
    if not shipped:
        return "_(no shipped cells yet)_\n"

    # Group by task; preserve a deterministic order.
    by_task: Dict[str, List[CellResult]] = {}
    for r in shipped:
        by_task.setdefault(r.task, []).append(r)
    out_parts = []
    for task in sorted(by_task):
        rows_for_task = by_task[task]
        # All cells in a task should share splits (same dataset).
        splits = rows_for_task[0].splits
        for r in rows_for_task:
            if r.splits != splits:
                raise RuntimeError(
                    f"task {task!r}: cells have different split sets — "
                    f"{rows_for_task[0].variable!r} has {splits}, "
                    f"{r.variable!r} has {r.splits}"
                )
        headers = ["model", "variable", "sites"] + [_split_short(s) for s in splits] + ["**mean IIA**"]
        out_parts.append(f"\n#### {task}\n")
        rows = ["| " + " | ".join(headers) + " |",
                "|" + "|".join("---" for _ in headers) + "|"]
        for r in rows_for_task:
            per_split = r.per_split_max_iia
            sites_str = str(len(r.picked_sites or ()))
            if r.picked_sites_inferred:
                sites_str += "†"
            cells = [
                r.model_class,
                r.variable,
                sites_str,
            ]
            cells += [_fmt_iia(per_split[s]) for s in splits]
            cells += [f"**{_fmt_iia(r.mean_iia)}**"]
            rows.append("| " + " | ".join(cells) + " |")
        out_parts.append("\n".join(rows) + "\n")
    return "".join(out_parts)


def per_cell_appendix(r: CellResult) -> str:
    """Per-cell breakdown: picked sites, per-site IIA across splits."""
    splits = r.splits
    out = [f"### {r.task} × {r.model_class} × {r.variable}\n"]
    out.append(f"**Mean IIA: {_fmt_iia(r.mean_iia)}** (sites trained: {len(r.picked_sites or ())})\n")
    if r.picked_sites:
        out.append("Picked sites: " + ", ".join(f"L{L}/{t}" for L, t in r.picked_sites) + "\n")
    out.append("")
    # Best-per-split
    best = r.best_site_per_split
    out.append("Best site per split:")
    for s in splits:
        u = best[s]
        out.append(f"- `{s}` → L{u.layer}/{u.position} = **{_fmt_iia(u.per_split_iia[s])}**")
    out.append("")
    # Picked-sites detail
    if r.picked_sites:
        out.append("Picked-site IIA:\n")
        out.append("| site | " + " | ".join(_split_short(s) for s in splits) + " |")
        out.append("|" + "|".join("---" for _ in range(1 + len(splits))) + "|")
        picked_keys = set(r.picked_sites)
        for u in r.units:
            if (u.layer, u.position) in picked_keys:
                row = [f"L{u.layer}/{u.position}"]
                row += [_fmt_iia(u.per_split_iia[s]) for s in splits]
                out.append("| " + " | ".join(row) + " |")
        out.append("")
    return "\n".join(out) + "\n"


# Manually-curated metadata. The eval JSON does not store DAS hyperparams or
# wall-clock — we record them here so RESULTS.md is self-contained.
RUN_META = {
    ("4_answer_MCQA", "Qwen2ForCausalLM", "answer_pointer"): {
        "fit_split": "answerPosition_randomLetter_train",
        "ot_rows": "V=4 (choice0..3)",
        "n_features": 16, "epochs": 12, "lr": 1e-3,
        "wall_min": None,  # not measured
        "leaderboard_das_best": 0.96, "leaderboard_das_avg": 0.76,
    },
    ("4_answer_MCQA", "Qwen2ForCausalLM", "answer"): {
        "fit_split": "answerPosition_randomLetter_train",
        "ot_rows": "V=4 (choice0..3)",
        "n_features": 16, "epochs": 12, "lr": 1e-3,
        "wall_min": None,
        "leaderboard_das_best": 0.97, "leaderboard_das_avg": 0.86,
    },
    ("4_answer_MCQA", "Gemma2ForCausalLM", "answer_pointer"): {
        "fit_split": "answerPosition_randomLetter_train",
        "ot_rows": "V=4 (choice0..3)",
        "n_features": 16, "epochs": 12, "lr": 1e-3,
        "wall_min": 50,
        "leaderboard_das_best": None, "leaderboard_das_avg": None,
    },
    ("4_answer_MCQA", "Gemma2ForCausalLM", "answer"): {
        "fit_split": "answerPosition_randomLetter_train",
        "ot_rows": "V=4 (choice0..3)",
        "n_features": 16, "epochs": 12, "lr": 1e-3,
        "wall_min": 90,  # ~1.5 h (site 1 throttled on battery)
        "leaderboard_das_best": None, "leaderboard_das_avg": None,
    },
}


def methods_section() -> str:
    return """## Methods

### PLOT pipeline

For each cell:

1. **Stage A (layer OT)**: per-layer mean-aggregated effect signatures
   are matched against per-OT-row abstract-effect signatures via balanced
   entropic Sinkhorn. Each OT row picks its top-1 layer; the union enters
   Stage B.
2. **Stage B (per-(row, layer) token-position OT)**: within each Stage-A
   layer, an OT cost matrix between abstract rows and per-token-position
   neural rows determines which (layer, token_position) sites enter Stage C.
3. **Stage C (DAS)**: orthogonal-rotation featurizers are trained at the
   selected sites only, using the harness's `train_interventions(method="DAS")`.
4. **Submission**: trained featurizers + identity-mapped indices ship as
   the cell folder. The harness evaluator (`evaluate_submission_task`) then
   computes per-split IIA on the public test sets.

Hyperparameters used so far (uniform across shipped cells unless noted):

- **PLOT**: cost = sq_l2 on L2-normalized rows, balanced Sinkhorn, ε grid
  Stage A {0.01, 0.03} × Stage B {0.003, 0.01, 0.03, 0.1}, top_k_per_row
  Stage A 1, Stage B {1, 2}; calibration sweep selects (ε, top_k) by per-site
  IIA on the calibration variable.
- **DAS**: `n_features=16`, 12 epochs, AdamW lr 1e-3, batch_size=32,
  `dataset_size=256` HF examples per train split (after correctness filter).
- **Eval**: `evaluate_submission_task(public_data=True, private_data=False)`.
  Mean IIA = unweighted mean of per-split max IIA (the "best" leaderboard
  convention).

### Hardware

RTX 4060 Laptop (8 GB VRAM, 125 W max power), Gemma-2-2b at fp16,
Qwen-2.5-0.5B at fp16. Cell 4's wall-clock was inflated to ~1.5 h by
site-1 training while the laptop was on battery; sites 2–4 trained at
expected speed once plugged in.

### Reproducibility

- MIB harness commit: `b69dabe9899251d4a8fe90789afa4d655afc84c7`
- CausalAbstraction commit: `f9ed6777ea5d88bfd88a1488f0903daa50402cc7`
- Pinned package versions: `mib_submission/ENV.md`
- Per-cell driver: `python -m mib_submission.plot.run --task <T> --model <M> --variable <V>`
- This document: regenerated via
  `python -m mib_submission.results._aggregate > mib_submission/results/RESULTS.md`
"""


def emit_results_md(results: List[CellResult]) -> str:
    out = ["# PLOT MIB submission — Results\n"]
    out.append(
        "Per-cell IIA on the public MIB test splits, with picked sites and "
        "site-level breakdowns. Numbers in this document are computed from "
        "raw eval JSON archives by `_aggregate.py`; do not edit by hand.\n"
    )

    shipped = [r for r in results if r.is_shipped]
    n_total = 26  # constraint from VALID_TASK_MODELS × TASK_VARIABLES
    out.append(f"**Status: {len(shipped)} / {n_total} cells shipped.**\n")

    out.append("## Headline\n")
    out.append(headline_table(shipped))
    if any(r.picked_sites_inferred for r in shipped):
        out.append(
            "\n† Picked sites for this cell were inferred from the eval JSON's "
            "per-site IIA pattern (no submission folder present locally). The "
            "count is the number of (layer, token-position) units whose IIA "
            "exceeded an identity-baseline threshold on at least one split.\n"
        )

    out.append(
        "\n‡ **Cell 8 ARC × Gemma × answer (0.999)** is driven by the "
        "harness's automatic identity-fallback at L25 last_token — a position "
        "PLOT did not pick to train. PLOT's actually-trained DAS rotations "
        "scored 0.04–0.79 at this cell. The win is methodologically valid "
        "per the eval's scoring rules (it scores every position at picked "
        "layers, defaulting to identity at unselected positions) but is not "
        "a direct PLOT-rotation result. See "
        "`mib_submission/PLOT_SHORTCOMINGS.md` §15 for the full mechanism.\n"
    )

    # Compare-to-baseline rows where we have leaderboard DAS numbers.
    out.append("### Comparison to baseline DAS (leaderboard)\n")
    out.append("Baseline DAS scores from the public MIB leaderboard. PLOT trains DAS at "
               "≤6 sites; baseline trains at all 72.\n")
    rows = ["| cell | PLOT mean | baseline DAS best | baseline DAS avg | Δ (PLOT − baseline best) |",
            "|---|---|---|---|---|"]
    for r in shipped:
        meta = RUN_META.get((r.task, r.model_class, r.variable))
        if not meta:
            continue
        b_best = meta.get("leaderboard_das_best")
        b_avg = meta.get("leaderboard_das_avg")
        if b_best is None:
            continue
        delta = r.mean_iia - b_best
        rows.append(
            f"| {r.task} × {r.model_class} × {r.variable} | "
            f"{_fmt_iia(r.mean_iia)} | {_fmt_iia(b_best)} | {_fmt_iia(b_avg) if b_avg else '—'} | "
            f"{'+' if delta >= 0 else ''}{delta:.3f} |"
        )
    if len(rows) > 2:
        out.append("\n".join(rows) + "\n")
    else:
        out.append("_(no leaderboard baselines wired yet)_\n")

    out.append(methods_section())

    out.append("\n## Per-cell breakdowns\n")
    for r in shipped:
        out.append(per_cell_appendix(r))
        meta = RUN_META.get((r.task, r.model_class, r.variable))
        if meta:
            extras = []
            extras.append(f"OT rows: {meta['ot_rows']}")
            extras.append(f"fit split: `{meta['fit_split']}`")
            extras.append(f"DAS: n_features={meta['n_features']}, "
                          f"epochs={meta['epochs']}, lr={meta['lr']}")
            if meta.get("wall_min"):
                extras.append(f"wall-clock: ~{meta['wall_min']} min")
            out.append("Run metadata: " + " · ".join(extras) + "\n")

    other = [r for r in results if not r.is_shipped]
    if other:
        out.append("\n## Other archived runs (no current submission folder)\n")
        out.append(
            "These archives exist from earlier diagnostic / ablation runs and "
            "are not part of the official submission state.\n"
        )
        for r in other:
            out.append(f"- `{r.task} × {r.model_class} × {r.variable}` "
                       f"(mean IIA in archive = {_fmt_iia(r.mean_iia)})")
        out.append("")

    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write", metavar="PATH",
        help="Write to PATH instead of stdout (useful for in-place regen).",
    )
    args = parser.parse_args()

    results = load_all()
    md = emit_results_md(results)
    if args.write:
        Path(args.write).write_text(md)
        print(f"wrote {len(md)} bytes to {args.write}", file=sys.stderr)
    else:
        sys.stdout.write(md)


if __name__ == "__main__":
    main()
