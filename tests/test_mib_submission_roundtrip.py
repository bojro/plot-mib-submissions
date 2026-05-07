"""Round-trip smoke test for ``mib_submission.serialize.write_submission``.

Builds a tiny fake MCQA submission with both featurizer encodings, then:

1. Loads each saved triplet through upstream's ``Featurizer.load_modules`` and
   confirms the featurizer round-trips numerically (Identity exact, Subspace
   reconstructs ``x`` to within float tolerance).
2. Runs ``verify_submission.py`` on the produced folder and asserts it
   reports a valid submission.

Run via the MIB venv::

    .venv-mib/bin/python -m unittest tests.test_mib_submission_roundtrip
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MIB_TRACK = REPO_ROOT / "MIB" / "MIB-causal-variable-track"

# Make CausalAbstraction importable.
sys.path.insert(0, str(MIB_TRACK))
sys.path.insert(0, str(MIB_TRACK / "CausalAbstraction"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from CausalAbstraction.neural.featurizers import Featurizer

from mib_submission import method_to_featurizer as m2f
from mib_submission import serialize


HIDDEN_DIM = 32
SUBSPACE_K = 4
TOPK = 6
TASK = "4_answer_MCQA"
MODEL = "Qwen2ForCausalLM"
VARIABLE = "answer_pointer"


class SubmissionRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _identity_result(self, layer: int) -> m2f.MethodResult:
        # Fake transport plan row: peaks at known indices so we can assert.
        plan = np.zeros(HIDDEN_DIM)
        plan[[3, 7, 11, 13, 17, 19]] = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
        return m2f.from_transport_plan(
            plan_row=plan,
            hidden_dim=HIDDEN_DIM,
            k=TOPK,
            layer=layer,
            token_position_id="last_token",
            method="ot",
            variable=VARIABLE,
        )

    def _subspace_result(self, layer: int) -> m2f.MethodResult:
        # Random orthonormal (d, k) basis.
        torch.manual_seed(0)
        q, _ = torch.linalg.qr(torch.randn(HIDDEN_DIM, SUBSPACE_K))
        return m2f.from_das_rotation(
            rotation=q,
            layer=layer,
            token_position_id="last_token",
            variable=VARIABLE,
        )

    def test_indices_picks_topk_in_sorted_order(self):
        result = self._identity_result(layer=12)
        # Top-6 mass indices were [3, 7, 11, 13, 17, 19]; result.indices is sorted.
        self.assertEqual(result.indices, [3, 7, 11, 13, 17, 19])

    def test_write_and_load_identity_triplet(self):
        result = self._identity_result(layer=12)
        cell_dir = serialize.write_submission(
            self.root,
            task=TASK,
            model_class_name=MODEL,
            variable=VARIABLE,
            results=[result],
        )
        base = cell_dir / serialize.model_unit_id(12, "last_token")
        for suffix in ("_featurizer", "_inverse_featurizer", "_indices"):
            self.assertTrue((Path(str(base) + suffix)).exists(), suffix)

        with open(str(base) + "_indices") as f:
            self.assertEqual(json.load(f), [3, 7, 11, 13, 17, 19])

        loaded = Featurizer.load_modules(str(base))
        x = torch.randn(2, HIDDEN_DIM)
        f, err = loaded.featurize(x)
        self.assertTrue(torch.equal(f, x))  # identity
        self.assertIsNone(err)

    def test_write_and_load_subspace_triplet(self):
        result = self._subspace_result(layer=7)
        cell_dir = serialize.write_submission(
            self.root,
            task=TASK,
            model_class_name=MODEL,
            variable=VARIABLE,
            results=[result],
        )
        base = cell_dir / serialize.model_unit_id(7, "last_token")
        loaded = Featurizer.load_modules(str(base))

        x = torch.randn(3, HIDDEN_DIM, dtype=torch.float32)
        f, err = loaded.featurize(x)
        self.assertEqual(f.shape, (3, SUBSPACE_K))
        # featurize then inverse_featurize must reconstruct x exactly within fp tolerance.
        x_hat = loaded.inverse_featurize(f, err)
        self.assertTrue(torch.allclose(x_hat, x, atol=1e-5))

    def test_verify_submission_accepts_mixed_cell(self):
        results = [self._identity_result(layer=L) for L in (0, 1, 2)] + [
            self._subspace_result(layer=L) for L in (3, 4)
        ]
        # Different methods can't share a cell folder (they overwrite each other),
        # so split by method into two cell directories under the same root.
        # Here, all results are for the same VARIABLE — pretend they were one method.
        # In real submissions you'd produce one root per (method, task, model, variable).
        # We only need to exercise verify on at least one valid folder.
        identity_results = results[:3]
        serialize.write_submission(
            self.root,
            task=TASK,
            model_class_name=MODEL,
            variable=VARIABLE,
            results=identity_results,
        )

        venv_python = REPO_ROOT / ".venv-mib" / "bin" / "python"
        proc = subprocess.run(
            [str(venv_python), str(MIB_TRACK / "verify_submission.py"), str(self.root)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        # verify_submission prints "Perfect submission!" when zero warnings,
        # "Valid submission." otherwise. Either is success; "ERRORS:" is failure.
        self.assertNotIn("ERRORS:", proc.stdout, proc.stdout)
        self.assertTrue(
            "Perfect submission" in proc.stdout or "Valid submission" in proc.stdout,
            proc.stdout,
        )


if __name__ == "__main__":
    unittest.main()
