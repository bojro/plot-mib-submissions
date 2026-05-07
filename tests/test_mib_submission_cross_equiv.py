"""Cross-equivalence: ``serialize.write_submission`` and
``apply_results.apply_method_results`` produce byte-identical triplet files
for the same ``MethodResult``.

The two save paths share their primitives (``Featurizer.save_modules`` plus a
JSON dump of the indices) so byte equality is essentially guaranteed by
construction. This test is a regression guard against future divergence —
e.g. if we ever change indices encoding on one side without the other.

We avoid loading a real LM by stubbing the bits of the upstream experiment
API that ``apply_method_results`` actually touches: ``model_units_lists``,
``save_featurizers``, plus per-unit ``id`` / ``featurizer`` /
``set_featurizer`` / ``set_feature_indices`` / ``get_feature_indices``.
"""

from __future__ import annotations

import filecmp
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MIB_TRACK = REPO_ROOT / "MIB" / "MIB-causal-variable-track"

sys.path.insert(0, str(MIB_TRACK))
sys.path.insert(0, str(MIB_TRACK / "CausalAbstraction"))
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from mib_submission import apply_results, method_to_featurizer as m2f, serialize  # noqa: E402
from mib_submission.pipeline import ExperimentBundle  # noqa: E402


HIDDEN_DIM = 16
SUBSPACE_K = 4
TASK = "4_answer_MCQA"
MODEL = "Qwen2ForCausalLM"
VARIABLE = "answer_pointer"


class _FakeUnit:
    """Minimal stand-in for ``AtomicModelUnit`` covering the surface that
    upstream's ``save_featurizers`` and our ``apply_method_results`` touch."""

    def __init__(self, layer: int, tok_id: str):
        self.id = serialize.model_unit_id(layer, tok_id)
        self._layer = layer
        self._tok_id = tok_id
        self.featurizer = None
        self.feature_indices = None

        # Minimal stubs of nested attributes used by site_keys.site_key_for_unit.
        outer = self

        class _Component:
            def get_layer(self_inner):
                return outer._layer

        class _TokenIndices:
            id = tok_id

        self.component = _Component()
        self.token_indices = _TokenIndices()

    def set_featurizer(self, featurizer):
        self.featurizer = featurizer

    def set_feature_indices(self, indices):
        self.feature_indices = indices

    def get_feature_indices(self):
        return self.feature_indices


class _FakeExperiment:
    """Stand-in that replays only what ``apply_method_results`` calls."""

    def __init__(self, units):
        # Match the [experiments][groups][units] triple-nesting of
        # PatchResidualStream.model_units_lists.
        self.model_units_lists = [[[u]] for u in units]

    def save_featurizers(self, model_units, model_dir):
        # Mirror upstream intervention_experiment.save_featurizers.
        import os
        if model_units is None or len(model_units) == 0:
            model_units = [u for outer in self.model_units_lists for grp in outer for u in grp]
        os.makedirs(model_dir, exist_ok=True)
        for unit in model_units:
            base = os.path.join(model_dir, unit.id)
            unit.featurizer.save_modules(base)
            with open(base + "_indices", "w") as f:
                indices = unit.get_feature_indices()
                if indices is not None:
                    json.dump([int(i) for i in indices], f)
                else:
                    json.dump(None, f)


def _make_results():
    """Two MethodResults — one Identity, one Subspace — at distinct layers."""
    import numpy as np

    plan = np.zeros(HIDDEN_DIM)
    plan[[1, 4, 9]] = [0.5, 0.3, 0.1]
    identity = m2f.from_transport_plan(
        plan_row=plan,
        hidden_dim=HIDDEN_DIM,
        k=3,
        layer=2,
        token_position_id="last_token",
        method="ot",
        variable=VARIABLE,
    )

    torch.manual_seed(0)
    rotation, _ = torch.linalg.qr(torch.randn(HIDDEN_DIM, SUBSPACE_K))
    subspace = m2f.from_das_rotation(
        rotation=rotation,
        layer=5,
        token_position_id="last_token",
        variable=VARIABLE,
    )
    return [identity, subspace]


def _make_bundle(results) -> ExperimentBundle:
    units = [_FakeUnit(r.layer, r.token_position_id) for r in results]
    experiment = _FakeExperiment(units)
    return ExperimentBundle(
        task=TASK,
        model_name="fake",
        model_class_name=MODEL,
        target_variables=[VARIABLE],
        layers=[r.layer for r in results],
        causal_model=None,
        pipeline=None,
        token_positions=[],
        filtered_datasets={},
        train_data={},
        test_data={},
        experiment=experiment,
    )


class CrossEquivalence(unittest.TestCase):
    def test_byte_identical_output(self):
        results = _make_results()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            dir_a = tmp_root / "via_serialize"
            dir_b = tmp_root / "via_apply"

            serialize.write_submission(
                dir_a,
                task=TASK,
                model_class_name=MODEL,
                variable=VARIABLE,
                results=results,
            )

            bundle = _make_bundle(results)
            apply_results.apply_method_results(
                bundle,
                results,
                submission_root=dir_b,
                variable=VARIABLE,
            )

            cell = serialize.cell_folder_name(TASK, MODEL, VARIABLE)
            cell_a = dir_a / cell
            cell_b = dir_b / cell
            self.assertTrue(cell_a.is_dir())
            self.assertTrue(cell_b.is_dir())

            files_a = sorted(p.name for p in cell_a.iterdir())
            files_b = sorted(p.name for p in cell_b.iterdir())
            self.assertEqual(files_a, files_b)

            for name in files_a:
                self.assertTrue(
                    filecmp.cmp(cell_a / name, cell_b / name, shallow=False),
                    f"{name} differs between save paths",
                )


if __name__ == "__main__":
    unittest.main()
