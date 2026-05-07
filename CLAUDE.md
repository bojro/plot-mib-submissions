# CLAUDE.md — PLOT MIB Submissions

Project guidance for Claude Code (claude.ai/code) when working in this repository.

## Repository purpose

Standalone repo for PLOT (Progressive Localized Optimal Transport) submissions to the MIB Causal Variable Localization Track. PLOT picks (layer, token-position) sites via two-stage Sinkhorn OT, then trains DAS rotations only at the picked sites — targeting baseline-DAS-comparable accuracy at ~10× fewer rotations trained. The full method narrative is in `mib_submission/JOURNEY.md`; structural limits are in `mib_submission/PLOT_SHORTCOMINGS.md`.

Source-of-truth PLOT (binary addition GRU origin) is preserved offline at `reference/source_plot/` for reference.

## MIB Causal Variable Track — submission plan

Goal: benchmark **PLOT only** (Progressive Localized Optimal Transport — OT site selection + DAS rotation training at picked sites) on the MIB Causal Variable Localization Track. Earlier scope included raw OT / GW / FGW / UOT / OT+gradient / OT+PCA; those are dropped. Circuit track is out of scope.

Total target cells = **26** (constrained by `verify_submission.py:VALID_TASK_MODELS`):
- ioi_task: 2 vars × 4 models = 8
- 4_answer_MCQA: 2 vars × 3 models = 6
- ARC_easy: 2 vars × 2 models = 4
- arithmetic: 1 var × 2 models = 2
- ravel_task: 3 vars × 2 models = 6

Documented PLOT shortcomings live in `mib_submission/PLOT_SHORTCOMINGS.md` — read before interpreting per-cell results.

Reference points:
- Harness: `https://github.com/aaronmueller/MIB`, submodule `MIB-causal-variable-track/CausalAbstraction` (`https://github.com/atticusg/CausalAbstraction`).
- `Featurizer` interface lives in `causalab/neural/featurizers.py`: paired `featurizer (x → (features, error))` + `inverse_featurizer ((features, error) → x̂)`, plus `n_features` and string `id`. `to_dict/from_dict` already round-trips `SubspaceFeaturizerModule` (rotation matrix). Custom adapters serialize via the same pattern.
- Submission unit: a folder per `{TASK}_{MODEL}_{VARIABLE}` containing `{ModelUnit}_featurizer`, `{ModelUnit}_inverse_featurizer`, `{ModelUnit}_indices`. Optional top-level `featurizer.py` / `token_position.py` register custom classes. Verify with `verify_submission.py`; private eval runs `evaluate_submission.py`.
- Tasks: IOI, simple_MCQA, ARC, two_digit_addition, RAVEL. Models: GPT-2 Small, Qwen-2.5-0.5B, Gemma-2-2B, Llama-3.1-8B.
- Average leaderboard requires every layer; "best" leaderboard accepts a single layer.

All MIB harness code lives under `MIB/` (gitignored). Our submission-side code lives under a new `mib_submission/` package in this repo. Do not reuse the existing `*_experiment/` pair banks for submissions — only the alignment math.

### Step 1 — Stand up the harness (done)
- Clone `aaronmueller/MIB` into `MIB/` with submodules; install `MIB-causal-variable-track/requirements.txt` into a Python 3.12 venv (`sae_lens` requires it).
- Run `example_submission.ipynb` end-to-end against the provided baseline DAS to confirm we can produce, verify, and locally evaluate a submission folder.
- Record exact versions of `pyvene`, `transformers`, `torch`, `CausalAbstraction` commit SHA in `mib_submission/ENV.md`.
- Exit criterion: `verify_submission.py` passes on the baseline mock submission **(done)**, harness imports + HF dataset load both work **(done)**. Full `evaluate_submission.py` IIA smoke run is deferred to Step 5 — it requires a real `{TASK}_{MODEL}_{VARIABLE}` triplet folder with trained weights (the stock `mock_submission/` has only `.py` stubs and the evaluator no-ops on it).
- Setup notes: venv lives at `./.venv-mib/` (PEP 668 forced this — homebrew Python 3.12 won't `pip install` system-wide). All MIB commands must use `.venv-mib/bin/python`. The `MIB-circuit-track` submodule fails to clone (its EAP-IG sub-submodule uses SSH); harmless because we ignore the circuit track.
- Pinned commit of `CausalAbstraction`: `f9ed6777ea5d88bfd88a1488f0903daa50402cc7`. At this commit the package layout is `CausalAbstraction/neural/featurizers.py` (no `causalab/` wrapper that the GitHub `main` tree shows).

### Step 2 — Pick the initial cells
- **Valid (task, model) pairs are constrained by MIB** (see `MIB/MIB-causal-variable-track/verify_submission.py:VALID_TASK_MODELS`). GPT-2-Small is only valid for `ioi_task` — NOT for MCQA or arithmetic. Do not waste effort on GPT-2-Small × MCQA/arithmetic.
- Primary: `4_answer_MCQA × Qwen2ForCausalLM` (Qwen-2.5-0.5B, smallest valid LM for MCQA, ~1 GB; closest analog to our `mcqa_experiment/`).
- Secondary: `arithmetic × Gemma2ForCausalLM` (Gemma-2-2B is the smallest valid LM for arithmetic).
- Tertiary stretch: `ioi_task × GPT2LMHeadModel` if we want the cheapest possible cell.
- Defer Llama-3.1-8B until step 6.

### Step 3 — Map our methods onto upstream's two Featurizer classes
**Important**: `Featurizer.load_modules` upstream only handles `SubspaceFeaturizerModule` and `IdentityFeaturizerModule`; new module classes raise `ValueError`. The submission interface allows custom classes via `featurizer.py`, but the evaluator's load path goes through `Featurizer.load_modules`, so anything we ship must reduce to one of those two.

`{ModelUnit}_indices` is a JSON list of ints (or `null`) selecting feature-space dims to interchange. That gives us one extra knob without inventing new module classes.

Encoding (PLOT only):
- **PLOT** → `SubspaceFeaturizerModule` with the DAS-trained rotation R at each surviving site, `_indices = list(range(k))`.

Final layout of `mib_submission/`:

Submission-side primitives (task-agnostic, used by every method):
- `featurizers.py` — re-exports of upstream's `Featurizer`, `Identity*Module`, `Subspace*` classes. We deliberately do NOT ship a custom `featurizer.py` in submissions: every method reduces to upstream's two module classes, so `Featurizer.load_modules` deserialises natively. Shipping a `.py` would also force shipping a valid `token_position.py` (verify_submission errors otherwise).
- `site_keys.py` — single `site_key_for_unit(unit) -> (layer, tok_id)` helper, shared by `activations.py`, `apply_results.py`, and `plot/`.
- `method_to_featurizer.py` — `MethodResult` dataclass + converters: `from_transport_plan`, `from_gradient_mask` (both → Identity + top-k indices), `from_das_rotation` (→ SubspaceFeaturizer), `from_ot_pca` (→ SubspaceFeaturizer with PCA basis).
- `serialize.py` — `write_submission(...)`: `MethodResult`s → on-disk triplet, no LM required. Validates against `VALID_TASK_MODELS` / `TASK_VARIABLES`.
- `pipeline.py` — `setup_residual_experiment(...)` builds an `ExperimentBundle` (LMPipeline + PatchResidualStream + filtered datasets) for a (task, model) pair. `_TASK_MODULES` maps MIB task ids to their upstream module paths.
- `apply_results.py` — alternative save path via `experiment.save_featurizers`; byte-identical to `serialize.write_submission` (regression-guarded by `tests/test_mib_submission_cross_equiv.py`).
- `activations.py` — `collect_base_activations(bundle, dataset)` for PCA fits / non-causal OT cost matrices.
- `signatures.py` — `alphabet_token_ids(tokenizer, letters)` for per-letter token id lookup; `signature_from_logits(...)` for KL / L2 collapses (kept for future non-PLOT methods, not used by the PLOT pipeline).

PLOT method package (`mib_submission/plot/`) — the production submission method, ported faithfully from `codex/binary-addition-two-stage-plot:experiments/binary_addition_rnn`:
- `features.py` — output-prob-delta effect signatures, L2-normalised. Abstract row: aggregated one-hot diff over `causal_model.run_interchange`. Neural row: aggregated `softmax(intervened_logits)[alphabet] - softmax(base_logits)[alphabet]` per site.
- `transport.py` — verbatim port of the source's `sinkhorn_uniform_ot`, `sinkhorn_one_sided_uot`, `cost_matrix(metric ∈ {sq_l2, l1, cosine})`, `truncate_row`.
- `pipeline.py` — `select_sites_via_plot(bundle, fit_dataset, config) -> PlotSelection`. Stage A (layer OT over per-layer-aggregated signatures) → Stage B (per-Stage-A-layer site OT over the 3 token positions there). Uses V=2 abstract rows by default (`answer_pointer`, `answer`) so balanced Sinkhorn discriminates. The sensitivity/invariance calibration sweep on top_k/lambda is documented as a seam but not yet wired in — for the DAS-guided submission `lambda` is irrelevant and `top_k` is fixed by config.
- `run.py` — driver: setup bundle → `select_sites_via_plot` → prune `experiment.model_units_lists` to the surviving sites → upstream `train_interventions(method="DAS")` → `experiment.save_featurizers` → `verify_submission.py`. Submission lands at `submissions/plot/`.

Why "PLOT" and not "OT-DAS": the source-of-truth implementation is named "Progressive Localized Optimal Transport" by the upstream branch. We adopt that name.

### Critical implementation choices (do not regress)
1. **Output-space signatures, not feature-space.** `S` and `A` are length-K (alphabet size) per row, aggregated across examples — not `(N · K)` flattened. Earlier attempts on this work used the flattened form and produced uniform Sinkhorn plans because squared-L2 over thousands of dims with magnitudes O(1) gives costs O(10³), and `exp(-10³ / ε)` underflows for any ε ≪ 100.
2. **L2-normalise rows of `A` and `S` before cost.** Matches `--normalize-signatures=True` in the source. Brings squared-L2 cost into `[0, 4]`, well-conditioned for any ε.
3. **V ≥ 2 OT variables** (`answer_pointer`, `answer`). Balanced Sinkhorn with V=1 forces uniform plan mathematically (column marginals = 1/M means π[0,j]=1/M regardless of cost). The source pipeline always uses V≥2 for this reason.
4. **Stage B uses Stage-A-cached signatures.** No new forward passes between stages — same signature dict is reused, just sliced per layer.
5. **DAS only on selected sites.** Prune `experiment.model_units_lists` rather than rebuild the bundle; cleaner and avoids re-loading the LM.

### Step 5 — End-to-end PLOT submission for one cell (done; tuning open)
Target cell: `4_answer_MCQA × Qwen2ForCausalLM × answer_pointer`.
- Driver: `python -m mib_submission.plot.run`. Submission lands in `submissions/plot/`.
- Reuses MIB train/dev splits via `setup_residual_experiment`. Do NOT reuse our `*_experiment/` pair banks for submissions.

#### Status (as of 2026-05-06)

The pipeline runs end-to-end and produces a verified submission. Best result so far is **mean IIA = 0.944** vs baseline DAS's **1.000** on the public test sets — a 5.6% gap closed from 14.3% earlier in the session. Detailed eval log is in `mib_submission/results/EVAL_LOG.md` (one row per run; raw JSON of the latest run is also archived there).

Faithfulness fixes that landed (vs the source `experiments/binary_addition_rnn/`):
- Stage A and B both use balanced `sinkhorn_uniform_ot` (the source's choice; UOT branch retained as a non-default escape hatch).
- Stage A is **per-row top-1 layer pick** (not top-k from one row). Each of the V OT rows picks its own best layer; the union goes to Stage B. Faithful to the source's `_stage_a_timesteps` which returns one timestep per OT row.
- Stage B per-(row, layer) top-k token-position pick using the same row that picked the layer at Stage A.
- `(epsilon × top_k)` calibration sweep wired in, scored by per-site IIA on the calibration variable. Source's `lambda` dimension is intentionally dropped (no MIB analog under `VanillaIntervention`).
- `_causal_letter_pairs` skips examples where the chosen variable's interchange yields an undefined downstream value (e.g. `choice_i` swap that removes the question's color from the choices → pointer = None). Skip rate ~25% per choice row; symbol rows have skip rate 0.

Investigations that were run and what we learned:
- **Effective V=1 collapse on `answer_pointer + answer`** (both interchanges produce the same observable letter change in this dataset's fixed `[A,B,C,D]` ordering). Resolved by switching OT row variables to either `choice_i` (probes pointer mechanism) or `symbol_i` (probes letter-copy).
- **Split selection matters.** `choice_i` rows are non-trivial only in `answerPosition_*` splits; `symbol_i` rows only in `randomLetter_*`. The `answerPosition_randomLetter_train` split makes both non-trivial and exposes both `answer_pointer` and `answer` as varying.
- **V=4 (choices) on `answerPosition_randomLetter_train`** picked `[L0, L2, L8, L23]`. Discovered `(L23, last_token)` as the strongest site for `answer_pointer` — perfect IIA on the easy split, 0.833 on the hard split.
- **V=8 mixed (choices + symbols)** on the same split picked `[L2, L4, L7, L8, L13, L23]` (six unique layers, seven sites after Stage B). **Same final IIA as V=4** — confirmed `L23, last_token` is the bottleneck site, not site selection breadth. Bias-fixing the choice rows did not change the picked site or the score.

#### Open: closing the 5.6% gap

The remaining gap is concentrated on `answerPosition_randomLetter_test` (5 / 30 examples wrong at `L23, last_token`). Two competing explanations, not yet disambiguated:
1. PLOT's residual-stream candidate space might miss a better site — e.g., `(L15..L22, last_token)` — that we've never trained DAS at.
2. DAS hyperparameters at `L23, last_token` leave headroom — `n_features=16`, `training_epochs=12`, `init_lr=1e-3`, training data capped at filter output.

Cheapest disambiguation: train DAS at one or two off-PLOT layers (e.g., L15, L20 at `last_token`) and eval. If IIA exceeds 0.833 there, gap is site selection; if not, gap is DAS quality (try `n_features=32`, `epochs=24`).

Exit criterion for "done with this cell": mean IIA within 0.02 of baseline DAS, OR a documented argument that the remaining gap is the cost of PLOT's compute saving (we use 4-7 sites vs DAS's 72 — roughly an order of magnitude fewer DAS rotations).

### Step 6 — Multi-cell PLOT rollout (active)
**MCQA × Qwen2.5-0.5B × answer_pointer is shipped at 0.944.** The 0.056 residual gap to baseline DAS is documented (site-selection limitation, see `mib_submission/JOURNEY.md` and `mib_submission/PLOT_SHORTCOMINGS.md`). Not pursuing further on this cell.

Rollout order (cheap → expensive by reused infrastructure, cell budget = 26 total):
1. **MCQA × Qwen × answer_pointer** — done (cell #1, 0.956, ties DAS best).
2. **MCQA × Qwen × answer** — done (cell #2, 0.801; PLOT picked correctly per diagnostic, gap is DAS@n_features=16 generalization).
3. **MCQA × Gemma × {answer_pointer, answer}** — same task, model-scale validation. One-line config change. Requires `HF_TOKEN` (Gemma is gated). **NEXT.**
4. **MCQA × Llama × {answer_pointer, answer}** — same task, larger model. Cost-heavy.
5. **ARC_easy × Gemma × {answer_pointer, answer}** — structurally similar to MCQA, reuses most config.
6. **IOI × GPT-2-Small × {output_token, output_position}** — smallest model overall but new task plumbing: V=3 natural rows (`name_A/B/C`), name-token alphabet, single TokenPosition per layer. ~2-3h to wire.
7. **IOI × {Qwen, Gemma, Llama}** — reuses IOI plumbing.
8. **RAVEL × Gemma × {Country, Continent, Language}** — V≥3 distinct vars, best PLOT candidate. New task plumbing.
9. **Arithmetic × Gemma × ones_carry** — single var, V=1 collapse risk. Likely needs bucketing or adjacent-variable workaround.
10. **ARC_easy × Llama, RAVEL × Llama, Arithmetic × Llama** — Llama 8B stretch.

Per-cell deliverable: a row in `mib_submission/results/EVAL_LOG.md` with mean IIA, per-split IIA, picked sites, and a one-line note on any per-cell config tweaks. No JOURNEY.md per cell — the MCQA one is the reference.

**Tracker**: `mib_submission/results/CELLS.md` — single-source-of-truth status table for all 26 cells. Update it when a cell ships.

### Hard constraints
- The official MIB `Featurizer` must be invertible. Any "selection" featurizer routes the unselected dims through `error` rather than discarding them.
- Use upstream `evaluate_submission.py` / `verify_submission.py` / `aggregate_results.py` verbatim. Do not reimplement IIA.
- `reference/source_plot/` is read-only. Don't import from or modify it; it exists as a snapshot of the original PLOT implementation we ported, preserved for offline reference.
- All run outputs (`logs/`, `submissions/`, `models/`, `signatures/`, `MIB/`, `.venv-mib/`) are gitignored. Curated results land in `mib_submission/results/`.
