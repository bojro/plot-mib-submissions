# PLOT MIB submission

PLOT (**Progressive Localization via Optimal Transport**) submissions to the [MIB Causal Variable Localization Track](https://github.com/aaronmueller/MIB). PLOT picks `(layer, token-position)` sites via two-stage Sinkhorn OT, then trains DAS rotations only at the picked sites — targeting baseline-DAS-comparable accuracy at ≤10× fewer rotations trained.

What this repo ships is the **PLOT-DAS** variant from the source paper (Stage A + Stage B + DAS at picked sites). Other PLOT variants in the paper — `PLOT` (localization only), `PLOT-native` / `PLOT-PCA` (Stage B handles in native or PCA coords), `Full DAS` — aren't implemented here. Source repo for the paper: <https://github.com/jchang153/causal-abstractions-ot>.

## Headline status

**12 of 26 cells with submissions** (46.2%). 11 at full quality, 1 (arithmetic) at smoke.

Of the 12 shipped:

| status | cells | mechanism |
|---|---|---|
| 🏆 win/tie vs DAS leaderboard | 1 (Qwen pointer), 7 (ARC pointer), 8 (ARC answer)\*, 22 (RAVEL Continent) | PLOT picks well |
| 📏 small structural gap (~5–7%) | 3 (Gemma pointer), 4 (Gemma answer) | confirmed outside seed band |
| ❌ documented structural gap | 2 (Qwen answer), 13/14 (IOI), 21/23 (RAVEL Country/Language) | each diagnosed in `mib_submission/PLOT_SHORTCOMINGS.md` |
| ⚠ smoke quality | 11 (arithmetic) | scale-up regressed; reverted to smoke |

\* Cell 8's 0.999 score includes a non-obvious mechanism — see `mib_submission/PLOT_SHORTCOMINGS.md` §15.

The other 14 of 26 cells require ≥16 GB GPU (Qwen/Gemma IOI + 10 Llama cells); deferred to cloud.

## Where to look

- **`CLAUDE.md`** — project context, status table, leaderboard comparison, rollout plan. Dense; engineer-oriented.
- **`mib_submission/PLOT_SHORTCOMINGS.md`** — 15-section catalog of diagnosed limitations. Read this for a calibrated view of where PLOT works vs doesn't, and *why*.
- **`mib_submission/results/RESULTS.md`** — auto-generated per-cell IIA tables.
- **`mib_submission/results/CELLS.md`** — per-cell status tracker.
- **`mib_submission/results/JOURNAL.md`** — methodological narrative, append-only by date. The full engineering record.
- **`HYPOTHESES.md`** — experimental hypotheses and outcomes from the diagnostic sessions.

## What's the value proposition

PLOT trains DAS rotations at **2–7 picked sites per cell** vs the baseline's **72 sites** (every layer × token position). On cells where PLOT's site selection is well-matched to the task, scores are competitive at 10–25× fewer trained rotations. On cells where PLOT's signature design picks the wrong sites, the gap to baseline DAS is structural and documented.

A surprise finding from the diagnostic sessions: PLOT's value-add is concentrated in **layer selection** (Stage A). Stage B (position selection) and DAS training can be subtractive on some cells — see `mib_submission/PLOT_SHORTCOMINGS.md` §15. A leaner "Stage A only" PLOT remains an open follow-up.

## Reproducing

```bash
# Verify the shipped submissions
.venv-mib/bin/python MIB/MIB-causal-variable-track/verify_submission.py submissions/plot

# Run a single cell end-to-end (defaults to MCQA × Gemma × answer)
.venv-mib/bin/python -m mib_submission.plot.run \
    --task <TASK> --model <MODEL> --variable <VARIABLE>

# Patched eval driver (handles harness gotchas: per-task max_new_tokens, IOI position_ids fix)
.venv-mib/bin/python scripts/eval_cell.py --cell <cell_folder_name>

# Tests (126 currently passing)
.venv-mib/bin/python -m pytest tests/
```

The `MIB/` submodule, `submissions/`, `logs/`, `models/`, and `.venv-mib/` are gitignored and need to be created on a fresh clone — see `CLAUDE.md` for the WSL-specific setup notes (chmod issues on `/mnt/c` mean MIB and the venv must be cloned to `~/` and symlinked).

## Hardware

Developed on an 8 GB RTX 4060 Laptop. 12 of 26 cells fit at this scale. The other 14 (4 Qwen/Gemma IOI cells via pyvene's `IntervenableModel` + 10 Llama-8B cells) need ≥16 GB VRAM — cloud GPU work, deferred.

## Caveats for a careful reader

- **Cell 8's 0.999 leaderboard-relative win comes from an interaction with the eval harness's identity-fallback** at unselected positions, not from PLOT-trained DAS rotations. Methodologically valid per the harness's scoring rules. Full mechanism documented in `mib_submission/PLOT_SHORTCOMINGS.md` §15.
- **5 of 12 reachable cells have real structural gaps to DAS baseline.** Each is diagnosed in `PLOT_SHORTCOMINGS.md` (§2 cell 2, §13 cells 13/14, §14 cells 21/23). Closing them is out of scope for this submission.
- **Cell 11 arithmetic ds=1024 scale-up regressed.** Reverted to the smoke result; the failed scale-up's submission is preserved at `submissions/_plot_backups/arithmetic_*_pre_c6_*` for reference.
