# Porting PLOT to the MIB Causal Variable Track

A process log of converting PLOT (Progressive Localized Optimal Transport) from its binary-addition origin to a MIB submission for `4_answer_MCQA × Qwen2.5-0.5B × answer_pointer`. Documents what worked, what didn't, what we learned about the method's structural limits, and where the final submission lands relative to baseline DAS.

## Starting point

PLOT was developed on a synthetic binary-addition task (3-bit, fixed `C1` carry benchmark) using a small GRU. The source pipeline (`origin/codex/binary-addition-two-stage-plot:experiments/binary_addition_rnn/`) localizes carry variables in two stages:

- **Stage A (layer OT)**: cost matrix between abstract per-variable signatures and per-layer (timestep) neural signatures; balanced Sinkhorn picks layers.
- **Stage B (subspace OT)**: within picked layers, OT over coordinate subspaces picks a feature subspace.
- **Stage C (DAS)**: train rotations only at surviving sites.

For binary addition, V=3 came naturally (three carry variables produce observably distinct interchange patterns). The method runs end-to-end, finds the right sites, and is much faster than full DAS.

The MIB target was structurally similar — pick sites in a transformer's residual stream — but the surface differences turned out to matter more than expected.

## Pipeline port

**MIB harness setup** (`mib_submission/pipeline.py`). Wraps `setup_residual_experiment` to build an `ExperimentBundle` containing the LM pipeline, causal model, and filtered counterfactual datasets across train/dev/test splits. The 24 layers × 3 token positions enumeration gives us 72 candidate sites.

**Featurizer mapping** (`mib_submission/method_to_featurizer.py`, `serialize.py`). MIB's `Featurizer.load_modules` accepts only `IdentityFeaturizerModule` and `SubspaceFeaturizerModule`. Custom subclasses raise `ValueError`. So:
- OT/GW/FGW/UOT raw selection → Identity + top-k indices
- OT+DAS (PLOT) → Subspace with rotation R, indices `range(k)`

This constraint is hard. We can't ship custom featurizers; the submission's expressivity is bounded by what those two classes can encode.

**Output-space signatures** (`plot/features.py`). Adapted from source's effect signatures: per-site `softmax(intervened_logits) − softmax(base_logits)` over the alphabet, aggregated across examples. The abstract row is `one_hot(source_letter) − one_hot(base_letter)` averaged across the same examples. Both rows live in length-K alphabet space with bounded magnitudes.

Faithful to source: balanced Sinkhorn both stages, sq_l2 cost on L2-normalized rows, multi-row Stage A (one layer per OT row, faithful to source's `_stage_a_timesteps`), per-(row, layer) Stage B token-position picks, calibration sweep over `(epsilon × top_k)` scored by per-site IIA on the calibration variable. The source's `lambda` (intervention strength) dimension was dropped — pyvene's `VanillaIntervention` is full-replace, no analog.

## Faithfulness fixes (the easy gains)

Several early attempts produced very low IIA. Each was a faithfulness gap from source:

- **Flattened signatures (N·K)**: produced uniform Sinkhorn plans because sq_l2 over thousands of dims with magnitudes O(1) gives costs O(10³), which underflow at any reasonable ε. Fix: aggregate across examples to (K,) per row.
- **Logit space, not probability space**: amplified noise. Fix: signatures on `softmax`.
- **No row normalization**: caused later layers (which produce sharper output deltas) to dominate by sheer magnitude. Fix: L2-normalise rows before cost.
- **One Stage A layer pick, not multi-row**: collapsed all V rows into one layer. Source returns one timestep *per OT row*. Fix: per-row top-k.

These got us to **mean IIA 0.857** (single-layer) → **0.944** (V=4 multi-row choice rows). At this point baseline DAS sits at **1.000**.

## The V=1 collapse and what it forced

PLOT requires V≥2 rows whose interchanges produce observably distinct patterns. For MCQA, the natural variables — `answer_pointer` and `answer` — produce identical observable letter changes in this dataset (because the symbol mapping is fixed across base/source within a sample). Balanced Sinkhorn with V=1 forces a uniform plan regardless of cost.

So we had to use *adjacent* variables: `choice_i` (color-word interchanges, V=4) and optionally `symbol_i` (V=8 mixed). These produce distinct patterns. PLOT picked sites and IIA settled at 0.944.

The 0.056 residual gap concentrated entirely on `answerPosition_randomLetter_test` (5/30 wrong at the picked site `(L23, last_token)`). The other two splits saturated at 1.000.

## The disambiguation experiment

We didn't know whether the gap was a site-selection problem (PLOT picked the wrong site) or a DAS-quality problem (right site, undertrained rotation). Cheapest test: hardcode two off-PLOT sites, train DAS there directly.

Picked `(L15, last_token)` and `(L20, last_token)` — both *missed* by PLOT. Result:

| Site | answerPosition | randomLetter | answerPosition_randomLetter | mean |
|---|---|---|---|---|
| (L23, last_token) | 1.000 | 1.000 | 0.833 | 0.944 |
| **(L15, last_token)** | **1.000** | **1.000** | **1.000** | **1.000** |
| (L20, last_token) | 1.000 | 1.000 | 0.900 | 0.967 |

L15 closes the entire gap. L23 was the wrong site.

This wasn't subtle in retrospect: `(L15, last_token)` IS the abstract `answer_pointer` site (residual encodes "the pointer is at index *i*"). `(L23, last_token)` is the answer-letter site (residual encodes "the letter is *Y*"). They diverge precisely on the hard split where source and base use different letter mappings — patching L23 forces source's letter; patching L15 forces source's pointer with base's symbols, giving the expected counterfactual letter.

## Why PLOT picked L23

We dumped the full cost matrix at (layer, token_position) granularity. Three findings:

1. `(L15, last_token)` had IIA = 0.964 against `answer_pointer` but ranked **29–67 out of 72** in cost across all four `choice_i` OT rows.
2. The cost-min sites for `choice_i` rows were mid-network sites with IIA ≈ 0 (e.g., `(L8, correct_symbol_period)`, `(L7, last_token)`).
3. Aggregation across token positions per layer was *not* the bottleneck — granular ranks confirmed L15 was uncompetitive at the (layer, position) level too.

The diagnosis: PLOT was doing exactly what we asked. It found sites whose patch effects mimic `interchange(choice_i)`. Patching L15/last_token doesn't reproduce a choice swap — it reproduces a pointer swap. Different operation, high cost against `choice_i` abstract rows. PLOT could never find L15 with these OT rows.

## The bucketing attempt

To probe for the pointer site directly, we needed V≥2 rows that all describe `interchange(answer_pointer)` but produce observably distinct patterns. The proposal: bucket counterfactual examples by `source.answer_pointer` value (V=4 buckets), construct V abstract rows where each row is the average letter flip *within* its bucket. The pointer site reproduces all V bucket-conditional patterns when patched; non-pointer sites don't.

Implementation in `mib_submission/plot/bucketed.py`. Reuses Stage A / Stage B / DAS structure; only the cost-matrix construction changes (M[v, s] = dist(abstract[v], neural[s, v]) — a bucket-diagonal cost rather than the standard PLOT cost).

Result: **failed**. Stage A picked layers `[0, 1, 2, 18]` — three of four buckets favored embedding-adjacent early layers, which produce vanishingly small output deltas.

Diagnosis: when patching the residual at L0/L1/L2, the output delta is tiny — those layers are barely past the embedding. After L2-row-normalization, *any* small direction gets amplified to unit length. A noise-level alignment with a bucket's destination letter reads as a perfect cosine match to OT cost, beating real-signal sites that point in approximately the same direction but didn't get spuriously amplified.

The pre-existing V=4 choice-row PLOT didn't suffer from this because its abstract rows were less specific (averaged across all examples), so spurious noise alignment was less likely. Bucketing made the abstract row sharper (good for L15) but also made it easier for noise at low-magnitude layers to spuriously align (bad).

A secondary issue: in the random-letter split we used for fitting, base symbols vary per example, so bucket k's destination letter varies across examples within the bucket. The mean abstract row is smeared rather than sharp. Bucketing would likely work better on `answerPosition_train` where symbols are fixed and bucket k → letter k cleanly. We didn't pursue that — the early-layer pathology suggested deeper signature issues than bucketing alone could fix.

We reverted. The 0.944 V=4 choice-row result remains the best PLOT we have.

## Where we landed

| Method | mean IIA | answerPosition | randomLetter | answerPosition_randomLetter | sites trained |
|---|---|---|---|---|---|
| Baseline DAS (leaderboard) | **1.000** | — | — | — | 72 |
| Off-PLOT oracle (L15+L20) | 1.000 | 1.000 | 1.000 | 1.000 | 2 |
| **PLOT V=4 (shippable)** | **0.944** | 1.00 | 1.00 | 0.83 | **5** |
| PLOT V=8 mixed | 0.944 | 1.00 | 1.00 | 0.83 | 7 |
| Bucketed PLOT (failed) | (not evaluated) | — | — | — | 5 |

PLOT trains DAS at 5 sites instead of DAS's 72. That's ~14× fewer rotations to train, with the corresponding wall-clock and compute reduction roughly linear: a baseline DAS run at ~3 minutes per site is ~3.5 hours; PLOT lands the same submission shape in under 30 minutes including signature collection. On larger models (Gemma-2-2B, Llama-3.1-8B) where each DAS rotation cost scales with hidden size and dataset, the savings get larger in absolute terms.

The 0.056 IIA gap to baseline DAS is real and concentrates on the hardest split. The disambiguation showed it's a site-selection issue, not a DAS-quality issue. Closing it requires PLOT to find `(L15, last_token)` instead of `(L23, last_token)`, which the OT-row construction we explored cannot do.

## What we learned about the method

**PLOT's load-bearing assumption is V≥2 observably-distinct interchange patterns whose causally-correct sites also produce sharp letter flips.** When variables collapse observationally, you need to construct distinct patterns another way. When the causally-correct site produces *attenuated* output deltas (because downstream layers can re-process), output-space signatures structurally bias toward late layers regardless of what's happening inside.

For MCQA, both conditions broke: `answer_pointer` collapsed observationally into `answer`, and the pointer site (mid-layer, encoding an abstract index) produces less sharp output deltas than the answer site (late, encoding a letter). PLOT picked the late site. We saw the gap.

For other MIB tasks the picture varies. RAVEL has many natural variables (entity, attribute, queried_relation) that should give V≥2 distinct patterns directly. Two-digit addition has multiple intermediate digits and carries. Both should be more PLOT-friendly than MCQA's single-pointer-and-letter setup.

## What we'd try next

If the goal is closing the 0.056 gap on this cell:

- **Magnitude-floor cost**: drop sites whose raw signature L2 norm is below threshold *before* normalization. Removes the early-layer noise-amplification entry point that broke bucketing.
- **Try bucketing on `answerPosition_train`** (fixed symbols) — bucket k → letter k cleanly. May land closer to L15.
- **Hybrid signature**: blend output-space delta with the residual-stream delta at the patched site. Late layers no longer dominate because the residual change is the same magnitude order across layers.

If the goal is moving to other MIB cells:

- RAVEL and arithmetic likely don't suffer the same V=1 collapse. The infrastructure (`pipeline.py`, `serialize.py`, `bucketed.py` if needed) is reusable; only the per-cell config changes.

## Files of record

- `mib_submission/plot/run.py` — CLI driver (argparse: `--task --model --variable` + overrides).
- `mib_submission/plot/configs.py` — per-task `PlotConfig` presets.
- `mib_submission/plot/pipeline.py` — `select_sites_via_plot` (Stage A + B with calibration sweep).
- `mib_submission/plot/features.py` — signatures + abstract table; per-row dataset filter; alphabet support.
- `mib_submission/plot/_alphabets.py` — `LabelAlphabet` (letters / multi-string / causal-model).
- `mib_submission/plot/transport.py` — Sinkhorn solvers (verbatim port).
- `mib_submission/plot/diagnose_costs.py` — granular cost-matrix dump.
- `mib_submission/plot/bucketed.py` — bucketed variant (parked, not used).
- `mib_submission/results/RESULTS.md` — auto-generated per-cell IIA table.
- `mib_submission/results/JOURNAL.md` — methodological narrative across sessions.
- `mib_submission/results/v8_mixed_results.json`, `offplot_L15_L20.json` — archived eval JSONs from cell-1 disambiguation.

---

## Postscript (2026-05-08)

This document is the cell-1 port story. Subsequent cells (3, 4, 7, 8, 22)
shipped in later sessions; their narrative lives in
`mib_submission/results/JOURNAL.md`. The pipeline structure described
above carried through unchanged for MCQA × Gemma (cells 3, 4) and ARC ×
Gemma (cells 7, 8). RAVEL needed a non-trivial extension — token-set
alphabets, per-row dataset filtering, multi-token labels, custom checker
— landed in the cell-22 work. See `JOURNAL.md` for the RAVEL design
rationale and `CLAUDE.md` for the current overall plan.
