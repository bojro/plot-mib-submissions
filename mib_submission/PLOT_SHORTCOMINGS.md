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

## 11. Tokenizer-specific alphabet gotchas (added 2026-05-08, arithmetic)

The signature alphabet maps each label to its LM-vocab first-token id. For
some tokenizer × label-set combinations, the obvious "encode `' ' + label`
and take token 0" rule collapses the alphabet because the leading-space
token is shared across labels. Concrete example: Gemma's tokenizer
encodes ` A`..` Z` as single tokens (vocab merged) but ` 0`..` 9` as
**two tokens each** — `[space_token, digit_token]`, with a single shared
space_token (235248). The historic always-take-`encode(' '+lab)[0]` rule
collapsed all 10 digit labels onto one dim, making the cost matrix
uniform, the OT plan uniform, and IIA trivially 1.0 on every site.

**This is generic — any tokenizer that doesn't have merged vocab for
your alphabet will hit this.** Llama-3 vs Gemma-2 vs Qwen-2 all
tokenise digit-vs-letter-vs-word vocabularies differently. The fix in
`_alphabets.py:resolve_tokens` is multi-rule: prefer single-token
encoding (with or without leading space), else skip the leading-space
token of the spaced encoding. Regression test in
`test_alphabets_and_ravel.py::ResolveTokens::test_multi_token_spaced_label_skips_leading_space`.

**Smoke check before any new (task × model) cell**: run
`resolve_tokens(alphabet, tokenizer)` and assert `num_dims` matches the
intended size. A 1-dim collapse is the canonical "you'll get IIA=1.0
on everything" failure mode.

## 12. Causal-model output node naming is task-specific (added 2026-05-08, arithmetic)

MCQA / ARC / RAVEL expose their output as `causal_model.run_*()["answer"]`.
Arithmetic exposes it as `["raw_output"]` (and the value is a multi-char
string like "68" / "168", not a single label). The historic
`_causal_letter_pairs` hardcoded `["answer"]` and was caught by a generic
`KeyError` handler that silently skipped every example — producing an
all-zero abstract table. New `output_key` and `label_from_output` config
fields let each task declare its output node and any string→alphabet-key
projection. Arithmetic uses `output_key="raw_output"` and
`label_from_output=lambda s: s.strip()[:1]` to project multi-digit
output strings to the first digit of the alphabet.

Lesson: don't trust silent skips. Adding `print(f"skipped {n}/{total}")`
inside the existing handler immediately surfaced the bug — without it,
the run would have silently completed with degenerate signatures.

## 13. PLOT signature picks loud direct-effect heads, misses indirect-effect heads (added 2026-05-09, IOI)

**Affects: IOI cells 13, 14.** Confirmed by E-I-2 + A.

PLOT's per-site signature aggregates *direct logit-diff effects* (first
forward pass with the patched site, measure change in output logit
diff). For attention heads, this rewards sites whose hidden-state has
a large *direct* projection on the unembedding — Name Movers in IOI's
GPT-2 (L9 H1, H2, H3, H5, H6, H9, H10, H11). Heads that contribute
*indirectly* — by routing information into Name Movers, e.g.
S-Inhibition heads at L7 H3, H9 and L8 H6, H10 — show small direct
effects and are systematically passed over.

Evidence: PLOT picked L9 Name Movers for both `output_token` and
`output_position`. For `output_position` specifically, the L9 heads
carry token info, not position info, so position-flip splits land at
22+ MSE. Bypassing to S-Inhibition heads cuts mean MSE from 16.0 to
**4.12** — within +1.92 of DAS baseline. The right heads exist; PLOT's
signature can't see them.

The cross-cutting "V-row coupling" hypothesis was tested via per-row
independent OT (decoupling row marginal constraints in the Sinkhorn
solver). It changed which heads PLOT picks (to L1 H1, L1 H2, L4 H0)
but didn't help MSE (16.22 ≈ 16.0). The *solver* isn't the bug; the
*signature* is. Closing this requires replacing the logit-diff-effect
signature with one that captures cascading effects (e.g. ablation that
breaks downstream heads), which is substantially more code than
swapping solvers.

**Decision (2026-05-09)**: ship pure PLOT for cells 13, 14 with the
honest scores. The bypass diagnostic stays in `submissions/_plot_backups/`
for reference. Future work tracked as H-IOI-NEW-1 in `HYPOTHESES.md`.

## 14. PLOT site-selection ceilings on high-cardinality outputs (added 2026-05-09, RAVEL)

**Affects: RAVEL Country (cell 21), RAVEL Language (cell 23).** Confirmed by E-R-4 + D + C-split.

When the output variable has many classes (Country: 160 values,
Language: 174 values), the per-site DAS rotation at any single late
layer hits a hard ceiling around 0.6 IIA. E-R-4 tested 4 alternative
(layer, position) pairs for Country; none beat PLOT's L25
entity_last_token at 0.615. Even an *identity featurizer* (no rotation,
full residual swap) at L25 gives the same 0.615 — DAS isn't adding
value over identity at that site, suggesting the Country information
is diffusely distributed across the L25 residual rather than isolated
in any low-rank subspace a single rotation can capture.

Adding more sites doesn't break the ceiling either: D ran with
`stage_a_top_k=2` (4 picked sites: L5+L6+L24+L25 entity_last_token)
and got 0.6148 highest-view — same as 2-site PLOT. The extra sites
only drag the average-view down (0.563 vs 0.582 with 2 sites).

The structural reading: PLOT's value proposition is "comparable IIA
at far fewer sites than DAS." For high-cardinality outputs that
distribute information broadly, "far fewer sites" pays for itself in
expressivity. DAS baseline ships 72 sites; PLOT ships 2-4. The
information that's spread across the remaining 68 sites isn't
recoverable at our site count. Closing the gap would require either
(a) PLOT trained as densely as DAS (defeats its compute savings) or
(b) a different DAS architecture (more `n_features`, different loss),
which is also out-of-scope for this submission.

**Decision (2026-05-09)**: accept the gap and document. Cells 21
(0.615) and 23 (0.629) ship as the honest outcome of PLOT's site
selection on high-cardinality variables. Future work tracked as
H-RAVEL-NEW-1 in `HYPOTHESES.md`.

## 15. DAS rotation at a chosen site can score *worse* than the harness's identity featurizer at the same site (added 2026-05-10, ARC)

**Affects: ARC × Gemma × answer (cell 8). Possibly other late-layer
last-token sites.** Surfaced by D.7's `stage_b_top_k_grid=(1,)` rerun.

The MIB harness scores each `(layer, token_position)` site at every
picked layer. Where a featurizer is shipped, eval uses the trained
featurizer; where it isn't, eval falls back to a default
`Featurizer(n_features=hidden_size)` — an **identity**, full-residual
swap. See
`MIB-causal-variable-track/CausalAbstraction/experiments/residual_stream_experiment.py:227`.

**Empirical evidence — cell 8 ARC × Gemma × answer at L25 last_token:**

| config | what was at L25 last_token | IIA |
|---|---|---|
| Original (`top_k_grid=(1,2)`) | trained DAS rotation (Stage B picked both positions per layer) | **0.764** |
| D.7 (`top_k_grid=(1,)`) | identity featurizer (Stage B only picks 1 position; eval fills in identity at the other) | **0.999** |

The trained rotation was **0.235 IIA worse than identity** at the same
site. DAS at this site is *subtractive*: the orthogonal-rotation
parameterisation actively hurts the swap.

**Why this can happen:** when the residual at a late-layer last-token
position already cleanly encodes the answer (i.e. the unembedding maps
that residual direction to the correct logit with high probability),
a full residual swap from base→source flips the answer near-perfectly
under identity. DAS-trained rotations restrict the swap to a learned
subspace; on training data with limited sample size, the rotation
over-fits to a noisy direction and the held-out swap is less effective
than just swapping the whole residual.

**Implication for `top_k_grid`:** the previous SHORTCOMINGS §8 read
"Stage B `top_k_grid=(1,2)` degenerates on tasks with only 2 token
positions" because both positions get picked and weak DAS dilutes the
joint result. We tightened ARC to `(1,)` for D.7 to avoid that. The
side effect — discovered after the rerun — is that the harness then
scores identity at the un-trained position, and identity sometimes
beats the DAS rotation. Tightening `top_k` was net-positive on cell 8
(0.849 → 0.999 highest-view) precisely *because* it left more positions
to identity-fallback.

**Implication for cells generally:** PLOT's value-add is layer
selection (Stage A); position selection (Stage B) and DAS training
both add risk on top. Where the residual already works under identity,
the layer pick alone is enough — Stage B's correct/incorrect choice
matters less than expected, and DAS can be a regression. A "leaner
PLOT" that ships only Stage A picks (with no Stage B narrowing) and
relies entirely on the harness's automatic identity-fallback at every
position would be functionally equivalent in the best-position case
and avoid the DAS-subtractive-failure mode.

**Decision (2026-05-10)**: ship the D.7-tweaked submissions (cells 7,
8). Keep `top_k_grid=(1,)` for ARC. Future work: an ablation that
trains DAS at all picked positions vs only the chosen one would
quantify how often DAS is subtractive — a candidate for the next
diagnostic session.

## What this means for the multi-cell rollout (updated 2026-05-10)

Per-cell expectations, post-overnight + post-diagnostic-session:

- **MCQA × Qwen × answer_pointer (cell 1)**: shipped. Seed sweep (3 seeds) gave 1.000 ± 0.000 highest-view (vs original 0.8915, vs DAS LB 1.000). The original gap was seed noise; now matching DAS LB.
- **MCQA × Qwen × answer (cell 2)**: -0.125 gap. Per §2, structural. **Decision: accept and document.**
- **MCQA × Gemma × answer_pointer (cell 3)**: shipped. Seed sweep gave 0.923 ± 0.006 (vs DAS LB 0.974, gap -0.051 outside seed band). Real ~5% structural gap.
- **MCQA × Gemma × answer (cell 4)**: shipped. Seed sweep gave 0.904 ± 0.010 (vs DAS LB 0.974, gap -0.070 outside seed band). Real ~7% structural gap.
- **ARC × Gemma × answer_pointer (cell 7)**: shipped at 0.827 (D.7 config; identical to pre-tweak). The `correct_symbol`-doesn't-carry-answer issue (#9) is the residual gap to DAS LB 0.836.
- **ARC × Gemma × answer (cell 8)**: shipped at **0.999 highest-view** with D.7 config (top_k=1). Up from 0.849 pre-D.7. The improvement is real per §15: tightening top_k let identity fall through at L25 last_token where DAS was actively subtractive.
- **RAVEL × Gemma × Continent (cell 22)**: shipped at 0.856 (full settings). Tied with DAS baseline (+0.008). PLOT's strongest result.
- **RAVEL × Gemma × Country (cell 21)**: shipped at 0.615. **Decision: accept; structural per §14.**
- **RAVEL × Gemma × Language (cell 23)**: shipped at 0.629. **Decision: accept; structural per §14.**
- **Arithmetic × Gemma (cell 11)**: shipped at smoke 0.440. ds=1024 scale-up REGRESSED to 0.265 (different layers picked, ones_carry_test ~0); reverted to smoke. Future work: rerun with `--signature-dataset ones_carry_train`.
- **IOI** (cells 13, 14): shipped at 5.16 / 16.0 MSE. **Decision: accept pure-PLOT scores; structural per §13.**

The honest expectation is that PLOT lands within ~0.05 of baseline DAS on the cells where its V-row + signature design is well-matched (cells 1, 7, 8, 22). On structural-mismatch cells (2, 13, 14, 21, 23) the gap is larger and **closing it is out-of-scope for this submission** — see §2, §13, §14, §15 + `HYPOTHESES.md`.

The compute savings (PLOT's 2-7 sites vs baseline DAS's 72) remain real across all shipped cells.
