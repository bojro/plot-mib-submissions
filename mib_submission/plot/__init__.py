"""PLOT — Progressive Localization via Optimal Transport for MIB submissions.

Two-stage hierarchical OT site selection followed by DAS-trained rotations:

    Stage A: layer-level OT  (sites = full residual stream per layer)
    Stage B: per-layer site OT (sites = (selected_layer, token_position))
    Stage C: DAS rotations trained only at the surviving (layer, token_pos) sites

This mirrors the algorithm in
``codex/binary-addition-two-stage-plot:experiments/binary_addition_rnn`` —
specifically ``transport.py``, ``features.py``, and ``run_progressive_plot.py``
on that branch — adapted to MIB's data layer (``CounterfactualDataset`` /
``ExperimentBundle``) instead of the binary-addition exhaustive bank API.

Key implementation choices preserved from the source:
    1. Effect signatures live in *output space* (probability deltas over an
       answer-letter vocab), not feature/activation space. This keeps the
       cost matrix bounded and well-conditioned.
    2. Each row of A and S is L2-normalised before cost computation, so
       squared-Euclidean cost is in [0, 4] regardless of dimension and
       Sinkhorn doesn't underflow.
    3. Stage B's Sinkhorn is constrained to Stage A's selected layers.
    4. Top-K sites per row are picked directly from transport mass; full
       sensitivity/invariance calibration (per ``transport.py``) is not yet
       wired in — see ``calibrate_transport_rows`` for the seam.
"""

from .features import (  # noqa: F401
    aggregate_mean,
    build_abstract_effect_row,
    collect_neural_effect_signatures,
    normalize_rows,
)
from .transport import (  # noqa: F401
    cost_matrix,
    sinkhorn_one_sided_uot,
    sinkhorn_uniform_ot,
)
from .pipeline import (  # noqa: F401
    PlotConfig,
    PlotSelection,
    select_sites_via_plot,
)
