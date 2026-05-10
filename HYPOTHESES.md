# Where PLOT failed: hypotheses + results

After 2026-05-09 overnight + diagnostic session, we have 12/26 cells shipped and 4 cells with **structural** gaps to the DAS baseline that exceed seed-variance bounds:

| cell | task × var | PLOT | DAS LB | gap |
|---|---|---|---|---|
| 13 | IOI × output_token | 5.16 MSE | 2.08 | +3.08 (2.5× worse) |
| 14 | IOI × output_position | 16.0 MSE | 2.20 | +13.8 (7.3× worse) |
| 21 | RAVEL × Country | 0.615 | 0.957 | -0.342 |
| 23 | RAVEL × Language | 0.629 | 0.812 | -0.183 |

(Cell 2 MCQA Qwen answer is also structural — documented in `mib_submission/PLOT_SHORTCOMINGS.md` §2.)

This doc records the hypotheses we tested, the experimental results, and the refined hypotheses that survived. The candidate-fix CLI flags added during testing have been reverted; the experimental scripts are deleted. This is now a results record, not a runnable plan.

---

## §RAVEL — Country (cell 21) and Language (cell 23)

### Observation that motivated the hypotheses

All 3 RAVEL fulls picked **the same 2 sites**: `(L6, entity_last_token)` + `(L25, entity_last_token)`. Yet:

| variable | values | per-value support (ds=256) | Stage B IIA | eval highest | gap to DAS |
|---|---|---|---|---|---|
| Continent | 6 | 43 | 0.757 | 0.856 | +0.008 ✓ |
| Country | 160 | 1.6 | 0.655 | 0.615 | -0.342 ✗ |
| Language | 174 | 1.5 | 0.787 | 0.629 | -0.183 ✗ |

### Hypotheses tested

| ID | Hypothesis | Status | Evidence |
|---|---|---|---|
| H-RAVEL-1 | High-cardinality starvation of OT signatures | **PARTIALLY SUPPORTED** | L25 is a ceiling site; identity featurizer at L25 also gives ~0.615, suggesting the residual carries diffuse Country info no rotation can isolate. But scaling ds is untested and would be costly. |
| H-RAVEL-2 | First-token compaction collapses many countries | **REFUTED for Country** (only 2 collision groups, max 3 labels). **CONFIRMED for Language** (13 collision groups, max 9 labels). | CPU diagnostic on causal_model values + Gemma tokenizer. |
| H-RAVEL-3 | Per-row dataset filter dilutes Country signal | **NOT TESTED** (E-R-2 not run). |
| H-RAVEL-4 | Wrong picks for Country | **REFUTED** | E-R-4 tested 4 alternative bypass configs; none beat PLOT's 0.6147. L25 entity_last_token is the ceiling at 0.615 regardless of which other layers are paired with it. |
| H-RAVEL-5 | DAS rotation under-parameterized | **NOT TESTED** (n_features=512 not run). |
| H-RAVEL-6 | Multi-token answer DAS loss penalty | **PARTIALLY REFUTED for Country** (85% single-token labels, so multi-token can explain at most 15% of the gap). Untested for Language directly. |

### Experimental results

**E-R-1: alphabet tokenization analysis (CPU, no GPU).**
- Country: 136 labels, 85.3% single-token, 2 collision groups, max group of 3.
- Continent: 6 labels, 66.7% single-token, no collisions.
- Language: 129 labels, 45.7% single-token, **13 collision groups, largest group 9** (e.g. `" Arabic"` covers 9 distinct multi-language labels: `"Arabic"`, `"Arabic, Berberi"`, `"Arabic, French, English"`, …).

**E-R-4: alternative bypass-sites for Country (4 GPU runs, ~1.7 hr).**
| Run | Trained sites | Per-layer trained avgs | highest-view |
|---|---|---|---|
| **PLOT pick** | L6 + L25 entity_last_token | L6=0.549, L25=0.615 | **0.6147** |
| R1 | L10 + L15 entity_last_token | L10=0.539, L15=0.552 | 0.5518 |
| R2 | L20 + L25 entity_last_token | L20=0.591, L25=0.615 | **0.6148** ≈ tied |
| R3 | L6 + L25 last_token | L6=0.597, L25=0.296 | 0.5969 |
| R4 | L12 + L22 entity_last_token | L12=0.530, L22=0.609 | 0.6090 |

**D: Country with stage_a_top_k=2 (4 sites instead of 2).**
- Picked L5+L6+L24+L25 entity_last_token. Highest-view 0.6148, average-view 0.5632 (worse than 2-site original 0.582). Adding sites doesn't break the L25 ceiling and drags the average down.

**C-split: Language with comma-split alphabet.**
- PLOT picked the *same* sites (L6+L25 entity_last_token) with or without splitting. Stage B internal IIA went from 0.787 → 0.797 (richer signature) but eval IIA was identical (0.6287 in both runs). Compaction was real but didn't affect site selection or eval.

### Distilled conclusion for RAVEL

PLOT's site selection is **reasonable** for Country and Language. The structural gap is fundamentally about what a small set of late-layer sites can express on a 160-174-class problem. Closing it would require either:
- More sites (we tested 2 and 4; would need ~30+ to approach baseline's 72)
- More aggressive DAS training (we use 1 epoch with `n_features=288` and `dataset_size=256`; baseline likely uses larger settings)
- Or accepting the gap as the cost of using ≤4 sites

**Refined hypothesis for follow-up:**
- **H-RAVEL-NEW-1**: Country/Language IIA scales monotonically with site count up to ~30, where it asymptotes near baseline DAS. Test cost: ~3-4 hr GPU per run × 3 runs (top_k 4, 8, 16). Falsification: a plateau before 30 sites argues for a different bottleneck (DAS hyperparams).

---

## §IOI — output_token (cell 13) and output_position (cell 14)

### Observation that motivated the hypotheses

PLOT picked at L9 (correct per IOI literature: Name Movers live there). But position-flip splits in cell 14 collapse to 22+ MSE because the picked heads carry token info, not position info.

| cell | picks | s1_io_flip | s2_io_flip | s1_ioi_flip_s2_ioi_flip | mean |
|---|---|---|---|---|---|
| 13 token | L9H11, L9H2, L9H3 | 2.68 | 6.11 | 6.67 | 5.16 |
| 14 position | L9H10, L9H5, L9H1 | **22.95** | **22.28** | 2.79 | **16.0** |

### Hypotheses tested

| ID | Hypothesis | Status | Evidence |
|---|---|---|---|
| H-IOI-7 | V=splits doesn't separate token vs position info | **NOT DIRECTLY TESTED** (V=4 explicit token/position splits not implemented). |
| H-IOI-8 | PLOT picks loud Name Movers, misses Position Movers | **CONFIRMED** | E-I-2: bypass to L7-L8 S-Inhibition heads gave MSE 4.12 vs PLOT's 16.0. The right heads exist; PLOT didn't pick them. |
| H-IOI-9 | Linear-params bootstrap mismatch | **NOT TESTED**. |
| H-IOI-10 | Joint DAS over 3 picks dilutes a single good head | **NOT TESTED** (eval JSON doesn't expose per-head MSE). |
| H-IOI-11 | `Token-all` over-regularizes | **NOT TESTED**. |
| H-IOI-12 | ds=512 still too small | **WEAK SUPPORT** (smoke→full ds=128→512 didn't move MSE). |

### Experimental results

**E-I-1: per-head MSE breakdown (CPU).**
- The IOI eval JSON only contains the joint score for all picked heads, not per-head metrics. Hypothesis untestable without code change or per-head DAS rerun.

**E-I-2: bypass cell 14 to literature S-Inhibition heads `(7,3) (7,9) (8,6) (8,10)` (~50 min GPU).**
| split | PLOT pick (L9 Name Movers) | S-Inhibition heads | improvement |
|---|---|---|---|
| s1_io_flip_test | 22.95 | **6.08** | 3.8× |
| s2_io_flip_test | 22.28 | **3.93** | 5.7× |
| s1_ioi_flip_s2_ioi_flip_test | 2.79 | 2.36 | slight |
| **mean** | **16.0** | **4.12** | **3.9× better** |

vs DAS baseline 2.20 → gap closed from +13.8 to +1.92.

**A: cell 14 with per-row independent OT (cross-cutting candidate fix, ~35 min GPU).**
- Picked `L1H1, L1H2, L4H0` (early layers, very different from PLOT's L9 picks).
- MSE 16.22 — basically identical to PLOT's 16.0.
- Per-row decoupling did change picks but didn't auto-discover the right heads.

### Distilled conclusion for IOI

The **picks are wrong** (E-I-2 proves there are heads that work much better at the same cell). But **the OT solver isn't the issue** (A proves decoupling rows still produces wrong picks). The signature itself — logit-diff effect aggregated per (layer, head) — favors heads with large *direct* effects (Name Movers) over heads with quiet *indirect* effects (S-Inhibition heads route position info into Name Movers). PLOT can't see indirect contributions.

**Refined hypothesis for follow-up:**
- **H-IOI-NEW-1**: Replacing the logit-diff-effect signature with an **ablation-cascade signature** (drop a head, re-run downstream, measure cascading effect on output) would surface S-Inhibition heads. Test cost: substantial — requires reimplementing signature pipeline. ~1 day code, then re-run cells 13 and 14.

---

## Cross-cutting hypothesis (V-row coupling) — REFUTED

The hypothesis was: PLOT's V×M balanced Sinkhorn lets loud rows monopolize columns and force quiet rows onto orthogonal-information sites. The proposed fix was per-row independent OT (V independent 1×M softmax plans, no column-marginal coupling).

**Result of A + D + C-split (combined ~2 hr GPU):**
- A (per-row independent on IOI 14): different picks, identical MSE → V-row coupling wasn't the bug.
- D (more sites for Country): no improvement → site count wasn't the bug.
- C-split (alphabet split for Language): no eval change → alphabet compaction wasn't the bug.

The synthetic test on a hand-crafted cost matrix did show per-row independent recovers the correct row's preferred site when balanced Sinkhorn distorts it — so the *mechanism* exists, but it's not actually the dominant failure mode on real PLOT signatures for these cells.

**The candidate-fix code (`--per-row-independent-ot`, `--stage-a-top-k`, `--ravel-split-alternatives`) has been reverted from the codebase.**

---

## Estimated combined experiment cost (already-spent)

| experiment | time | discrimination |
|---|---|---|
| E-R-1 single-token labels | 10 min CPU | H-RAVEL-2/6 ✓ |
| E-I-1 per-head MSE | 10 min CPU | inconclusive |
| E-I-2 bypass to L7-L8 | 50 min GPU | H-IOI-8 ✓ |
| E-R-4 bypass-sites grid (×4) | ~1.7 hr GPU | H-RAVEL-4 ✓ |
| A per-row OT | ~35 min GPU | cross-cutting ✗ |
| D top_k=2 | ~50 min GPU | H-RAVEL site-count ✗ |
| C-split alphabet | ~50 min GPU | H-RAVEL-2 effect ✗ |
| **total** | **~5 hr GPU + 20 min CPU** | |

## What survived

Two refined hypotheses for follow-up work, both more expensive to test than this session's experiments:
1. **H-RAVEL-NEW-1** (Country/Language ⇒ asymptote near baseline at top_k≈30): ~3-4 hr GPU each
2. **H-IOI-NEW-1** (replace logit-diff signature with ablation-cascade signature): ~1 day code + re-run

Cell 2 (MCQA Qwen answer) was not investigated this session; its hypothesis is in `mib_submission/PLOT_SHORTCOMINGS.md` §2.
