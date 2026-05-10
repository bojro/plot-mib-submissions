"""Runtime monkey-patches for harness/transformers/pyvene incompatibilities
that block the IOI bootstrap on our environment.

Three issues we patch:

1. **`LMPipeline.load`** (`CausalAbstraction/neural/pipeline.py:165`) — when
   `self.position_ids=True`, the harness reads
   ``model.prepare_inputs_for_generation(...)["position_ids"]``. In
   transformers 5.x, this dict does not always contain `position_ids` for
   GPT-2, raising `KeyError`. Patch: compute position_ids from
   `attention_mask.cumsum(-1) - 1` when the key is missing.

2. **`Qwen2Config.head_dim`** — pyvene 0.1.8's
   `get_dimension_by_component` does `getattr(config, "head_dim")`, which
   doesn't exist on `Qwen2Config` in transformers 5.x. Patch: inject
   `head_dim = hidden_size // num_attention_heads` onto the loaded model's
   config object before pyvene reads it.

3. **`AttentionHead.head_dim` for the same reason** — `PatchAttentionHeads`
   reads `model.config.head_dim` directly to compute the unit's `shape`
   (line 60). Same fix as above; the config patch resolves both.

These patches are idempotent — applying them multiple times is a no-op.
"""

from __future__ import annotations

import torch


def patch_lm_pipeline_load() -> None:
    """Make `LMPipeline.load` compute position_ids manually when the
    harness's preferred path returns no `position_ids` key."""
    from neural.pipeline import LMPipeline  # type: ignore[import-not-found]

    if getattr(LMPipeline, "_plot_position_ids_patched", False):
        return

    original_load = LMPipeline.load

    def patched_load(self, input, *args, **kwargs):
        # Save original flag, run with position_ids=False to skip the
        # harness's broken dict-lookup, then attach manually.
        wants_position_ids = bool(getattr(self, "position_ids", False))
        prev = self.position_ids
        try:
            self.position_ids = False
            enc = original_load(self, input, *args, **kwargs)
        finally:
            self.position_ids = prev
        if wants_position_ids and "position_ids" not in enc:
            attention_mask = enc["attention_mask"].long()
            position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
            enc["position_ids"] = position_ids.to(self.model.device)
        return enc

    LMPipeline.load = patched_load
    LMPipeline._plot_position_ids_patched = True


def patch_model_config_head_dim(model_config) -> None:
    """Inject `head_dim` onto a model config object when missing.

    Idempotent — if the attribute already exists (e.g. Gemma2Config) we
    don't touch it.
    """
    if hasattr(model_config, "head_dim") and model_config.head_dim is not None:
        return
    n_heads = (
        getattr(model_config, "num_attention_heads", None)
        or getattr(model_config, "n_head", None)
        or getattr(model_config, "num_heads", None)
    )
    hidden = getattr(model_config, "hidden_size", None) or getattr(model_config, "n_embd", None)
    if n_heads is None or hidden is None:
        raise RuntimeError(
            "Cannot infer head_dim: config has neither (num_attention_heads, hidden_size) "
            "nor any equivalent."
        )
    model_config.head_dim = int(hidden // n_heads)


def apply_all(pipeline=None) -> None:
    """Apply every patch needed for the IOI bootstrap to run on our
    environment. Pass `pipeline` (an LMPipeline already loaded) to also
    patch its model config in place.
    """
    patch_lm_pipeline_load()
    if pipeline is not None:
        patch_model_config_head_dim(pipeline.model.config)
