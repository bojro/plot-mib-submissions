# PLOT shortcomings — a reasonable assessment

Notes on the structural limits of PLOT (Progressive Localized Optimal Transport) as a site-selection method, drawn from the MCQA × Qwen-2.5-0.5B porting effort. Documented as a reference when interpreting PLOT's results on other cells and when comparing against baseline DAS.

PLOT's core value proposition stands: it trains DAS at a small handful of OT-selected sites instead of all 72, giving roughly an order-of-magnitude reduction in DAS rotation training. None of the shortcomings below invalidate that. They explain where the gap to baseline DAS comes from on harder cells.

## 1. V=1 collapse when variables are observationally indistinguishable

Balanced Sinkhorn requires V≥2 OT rows. If the V variables you supply produce *identical observable outputs* under interchange (e.g., MCQA's `answer_pointer` and `answer` both flip the output letter the same way when the symbol mapping is fixed), the V abstract rows are linearly dependent, and the cost matrix has degenerate columns. The plan returns to a uniform marginal regardless of cost.

**When it bites**: tasks where the causal model has multiple variables that read out through the same observable (a single output letter / digit / token). MCQA is the canonical example. Arithmetic (with one carry variable per stage and a 200-class sum output) is at risk for any single-variable cell. RAVEL (multiple distinct attributes per entity) likely is not.

**Workarounds**: pick "adjacent" variables that are observationally distinct (`choice_i` instead of `answer_pointer` for MCQA — V=4 from interchanges of separate choice positions). This sidesteps the collapse but introduces shortcoming #2.

## 2. The OT picks sites whose patch effect mimics the OT-row variable, not sites that "represent" the variable you actually want

This is the deepest limitation. PLOT's neural row at site (L, t) measures "what does the output do when I patch (L, t) with source's residual?" The abstract row for variable V says "what does the output do when I run interchange(V)?" The OT cost rewards sites whose patch effect matches `interchange(V)`.

If V's site is the *causal* implementation of V, the two will match. But if you're forced to use adjacent variables (because of #1), the OT picks sites that implement *those* adjacent variables, not the variable you ultimately want to localize.

**MCQA case study**: we used `choice_i` rows (V=4). PLOT picked `(L23, last_token)` — the answer-letter site. The pointer site `(L15, last_token)` had IIA = 0.964 against `answer_pointer` but ranked 29–67 out of 72 in cost across all four `choice_i` rows. PLOT was doing exactly what we asked; we asked the wrong question.

**Fundamental implication**: PLOT-by-OT can localize a variable V *only if* V is observationally distinct enough to use directly as an OT row. When you have to pivot to adjacent variables, you lose the connection to V.

## 3. Output-space signatures are structurally biased toward late layers

PLOT's signature is `softmax(intervened_logits) − softmax(base_logits)` over the alphabet. Late layers, when patched, produce *sharp* letter swaps because the residual is one transformation from unembedding. Mid layers produce *attenuated* swaps because downstream layers can partially absorb or rewrite the patch. After L2-row-normalization (needed to put cost into a well-conditioned range), sq_l2 cost is monotone in cosine, which rewards directional sharpness — and late layers always win on directional sharpness.

When the causally-correct site happens to be late (e.g., the answer-letter site), PLOT works well. When it's mid-layer (e.g., the pointer site), output-space signatures push the OT toward the late confound.

**Workarounds attempted, none yet successful**:
- Bucketing the abstract rows by source variable value: failed because sharper abstract rows make small-magnitude noise at *very* early layers (post-embedding) spuriously align after L2-normalization. Picked layers 0, 1, 2.
- Magnitude-floor cost: untested. Would drop sites whose raw signature norm is below threshold before normalization. Plausible fix for the early-layer pathology but doesn't address late-layer dominance.
- Hybrid output-plus-residual signatures: untested. Would require collecting residual deltas at the patched site itself, doubling forward-pass cost.

## 4. The MIB Featurizer interface is a hard upper bound on submission expressivity

`Featurizer.load_modules` upstream accepts only `IdentityFeaturizerModule` and `SubspaceFeaturizerModule`. Custom classes raise `ValueError`. This means any submission must reduce to (a) identity featurizer with a top-k index list, or (b) orthogonal-rotation subspace featurizer.

For PLOT this is fine — Stage C is DAS, which is naturally a subspace featurizer with a learned rotation. But it eliminates richer Featurizer designs (e.g., learned non-orthogonal projections, soft-mixing, intervention-strength-weighted features) that might better exploit OT's site rankings. The interface bounds what we can ship, regardless of what we discover internally.

## 5. The IIA scoring on the calibration set can be noisy

Per-site IIA is computed over a train split (~50 examples after the harness's correctness filter). For variables with skip-on-fail behavior (e.g., `choice_i` interchanges that yield `None` pointer when the swap removes the answer's color), the effective sample size shrinks further. The calibration sweep over `(epsilon × top_k)` is then choosing among hyperparameter configs whose IIA estimates have meaningful variance.

This isn't unique to PLOT — DAS faces the same data scarcity — but PLOT's calibration sweep multiplies the issue (multiple configs evaluated on the same small set). Consequence: occasionally-suboptimal hyperparameter picks. Hard to quantify without a held-out calibration split, which the harness doesn't separately provide.

## 6. The two-stage structure may be redundant for transformer residual streams

Source PLOT (binary addition GRU) does Stage A over timesteps and Stage B over coordinate subspaces within a timestep. For a transformer with three token positions per layer, "Stage A picks layer, Stage B picks token position" reduces to selecting from 24 layers then from 3 positions — only 72 candidates total. Single-stage OT over (layer, position) tuples directly would lose nothing in expressivity and avoid the "Stage A's per-layer aggregation" question entirely.

We preserved the two-stage structure for faithfulness, but the empirical case for it on transformer residual streams is weaker than on the source's setting. Any future PLOT user on similar architectures should consider this.

## 7. Compute-savings claim is real but the numerator depends on how aggressive PLOT is

PLOT trains DAS at the surviving sites. On MCQA we picked 5–7 sites versus DAS's 72 — roughly 10–14× fewer rotations. But `top_k_per_row` is a knob: with V=8 rows × top_k=2 on Stage A AND Stage B, you can balloon to 32 sites. The compute savings are entirely controlled by `top_k` × V, and the optimal value depends on the cell. There's no automatic guarantee that PLOT picks "few" sites — only that it picks whichever number you configure it to pick.

Practical implication: report the actual site count alongside IIA when comparing PLOT runs. Don't quote "10× speedup" without specifying the configuration.

## 8. Stage B `top_k_grid=(1,2)` degenerates on tasks with only 2 token positions

Discovered with ARC (`get_token_positions` returns 2 positions: `correct_symbol`, `last_token`). Stage B's calibration is the *sum* of per-site IIA over the picks; adding a marginally-positive site never lowers the sum, so the grid prefers `top_k=2` and effectively keeps every (Stage-A-layer, token-position) combination. Cell 8 picked 7 sites where 4 would have sufficed; the extras failed to converge during DAS but were saved in the submission anyway. Best-per-split scoring rescued the IIA but DAS time was wasted.

**Mitigation**: for any task where `len(token_positions) ≤ 2`, set `stage_b_top_k_grid=(1,)`. Force Stage B to actually select.

## 9. `correct_symbol` doesn't carry the variable in late ARC layers

Empirical from cell 8 (ARC × Gemma × answer): every `last_token` site converged cleanly (DAS train accuracy → 1.0); only 1 of 4 `correct_symbol` sites did. Loss at the failing sites *climbed* during training (5.95 → 8.7) — classic noise-fitting signature when the underlying activation has no useful causal signal.

Diagnosis: by late layers (L17–L25 in Gemma), the residual stream at the `correct_symbol` position has moved on to processing downstream context. The answer letter is generated at `last_token`, not at the symbol position. PLOT can pick those sites because Stage B gives them mass, but no orthogonal-rotation rescue exists — the activation simply doesn't carry the signal.

Combined with #8, this means cells where the natural variable lives at one specific token position (e.g., `last_token` for ARC `answer`/`answer_pointer`) get bloat from Stage B's lax `top_k`. Future ARC config: layer-aware Stage B prior (prefer `last_token` at L≥mid_layer).

## 10. Single-seed runs are samples, not results

Every shipped cell is one DAS training run. DAS is stochastic: orthogonal init of the rotation, DataLoader shuffle, and (potentially) CUDA non-determinism in matmul backward all introduce variance. We have no estimate of that variance.

Differences between shipped cells need to be read with this caveat:
- Cell 4 (Gemma MCQA answer) = 0.908 vs cell 8 (Gemma ARC answer) = 0.923. 0.015 — could be within seed noise.
- Cell 1 (Qwen MCQA pointer) = 0.956 vs cell 3 (Gemma MCQA pointer) = 0.955. Within 0.001.
- Cell 7 (ARC pointer) = 0.884 vs cell 8 (ARC answer) = 0.923. 0.04 — big enough to be a real effect *if* seed variance is small.

**To validate any cell's IIA as a "result," run it 3× with distinct seeds and report mean ± std.** Not done so far. Adding `--seed` to the CLI is straightforward but hasn't landed.

## What this means for the multi-cell rollout (updated 2026-05-08)

Per-cell expectations, refined by data from cells 1–8 + RAVEL Continent smoke:

- **MCQA × Gemma**: validated. Cells 3 (`answer_pointer` = 0.955) and 4 (`answer` = 0.908) shipped. Both within 0.05 of cell 1's Qwen result.
- **ARC × Gemma**: validated, with caveat. Cell 7 = 0.884, cell 8 = 0.923. The `top_k=2` Stage B and `correct_symbol`-doesn't-carry-answer issues (#8, #9) are now characterized but not yet fixed in the preset.
- **RAVEL × Gemma × Continent**: smoke ran at 0.845 with tiny config (n_features=64, dataset_size=128, 1 epoch). Full-config rerun expected ≥0.90. **The cleanest PLOT result so far** — V=3 distinct attributes + per-row filter eliminates the no-op-base SNR drag that hurt MCQA-style runs.
- **RAVEL × Gemma × {Country, Language}**: not yet run. Country has 160 distinct values (28 within-attribute first-token collisions); Language has 174 distinct values with **63% multi-token answers** — heaviest collision noise of the three RAVEL variables.
- **Arithmetic** (`ones_carry`): single variable, V=1 collapse risk. Deferred; needs adjacent-variable workaround or bucketing.
- **IOI** (`output_token`, `output_position`): two distinct outputs, V=2+ should work. **Bootstrap blocked** on per-model linear-parameter learning (`baselines/ioi_baselines/ioi_learn_linear_params.py`).

The honest expectation is that PLOT lands within ~0.05 of baseline DAS on RAVEL and most MCQA cells, with larger gaps on harder splits or where the variable is mid-network. The compute savings (PLOT's 4-7 sites vs baseline DAS's 72) remain real across all shipped cells.
