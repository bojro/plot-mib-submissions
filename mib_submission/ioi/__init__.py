"""IOI-specific helpers for the PLOT submission pipeline.

The IOI cell of the MIB Causal Variable Localization Track has structural
differences from MCQA / ARC / RAVEL that can't share the residual-stream
pipeline directly:

- The causal model's ``logit_diff`` mechanism requires per-model linear
  parameters ``{bias, token_coeff, position_coeff}`` learned by the harness
  bootstrap (`baselines/ioi_baselines/ioi_learn_linear_params.py`). Without
  these, ``get_causal_model(parameters)`` is degenerate.
- Submissions ship attention-head featurizers in a nested folder layout:
  ``ioi_task_<MODEL>_<VARIABLE>/DAS_<MODEL>_<VARIABLE>/AttentionHead(...)``.
- Evaluation goes through ``ioi_evaluate_submission.evaluate_ioi_submission_task``
  with an MSE-on-logit-diff scoring function — different from the standard
  ``evaluate_submission_task``.

This module collects the IOI-only pieces. The PLOT machinery itself
(Stage A timestep OT, Stage B subspace OT, signature collection, DAS
training, ...) is shared with the residual-stream cells via
``mib_submission.plot``.
"""

from __future__ import annotations

from .bootstrap import (
    LINEAR_PARAMS_FILENAME,
    bootstrap_linear_params,
    load_linear_params,
)
from .submission import (
    cell_dir,
    ensure_linear_params_json,
    write_ioi_submission,
)

__all__ = [
    "LINEAR_PARAMS_FILENAME",
    "bootstrap_linear_params",
    "load_linear_params",
    "cell_dir",
    "ensure_linear_params_json",
    "write_ioi_submission",
]
