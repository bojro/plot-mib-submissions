"""Tests for ``mib_submission.ioi.submission`` — folder layout helpers
and the linear-params JSON writer. Doesn't exercise the actual
``write_ioi_submission`` (which needs a real PatchAttentionHeads + GPU);
that's covered indirectly by the end-to-end smoke run."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mib_submission.ioi.bootstrap import LinearParams  # noqa: E402
from mib_submission.ioi.submission import (  # noqa: E402
    cell_dir,
    ensure_linear_params_json,
)


class CellDir(unittest.TestCase):
    def test_flat_layout(self):
        """Files live at the top of ``ioi_task_M_V/`` because the harness
        eval's listdir scan is non-recursive."""
        d = cell_dir(Path("/tmp/sub"), "Gemma2ForCausalLM", "output_token")
        self.assertEqual(
            d, Path("/tmp/sub/ioi_task_Gemma2ForCausalLM_output_token"),
        )

    def test_invalid_variable_rejected(self):
        with self.assertRaises(ValueError):
            cell_dir(Path("/tmp/sub"), "Gemma2ForCausalLM", "Country")


class EnsureLinearParamsJson(unittest.TestCase):
    def test_writes_required_keys(self):
        with TemporaryDirectory() as d:
            params = LinearParams(
                bias=0.05, token_coeff=0.77, position_coeff=2.00,
                score=0.93, model_name="google/gemma-2-2b",
            )
            path = ensure_linear_params_json(
                Path(d), model_short="gemma",
                model_class_name="Gemma2ForCausalLM", params=params,
            )
            self.assertTrue(path.exists())
            blob = json.loads(path.read_text())
            self.assertEqual(blob["model_class"], "Gemma2ForCausalLM")
            self.assertIn("gemma", blob)
            self.assertAlmostEqual(blob["gemma"]["bias"], 0.05)
            self.assertAlmostEqual(blob["gemma"]["token_coeff"], 0.77)
            self.assertAlmostEqual(blob["gemma"]["position_coeff"], 2.00)
            self.assertAlmostEqual(blob["gemma"]["score"], 0.93)
            self.assertEqual(blob["gemma"]["model_name"], "google/gemma-2-2b")

    def test_optional_fields_omitted_when_none(self):
        with TemporaryDirectory() as d:
            params = LinearParams(
                bias=0.0, token_coeff=0.5, position_coeff=1.5,
                score=None, model_name=None,
            )
            path = ensure_linear_params_json(
                Path(d), model_short="qwen",
                model_class_name="Qwen2ForCausalLM", params=params,
            )
            blob = json.loads(path.read_text())
            self.assertNotIn("score", blob["qwen"])
            self.assertNotIn("model_name", blob["qwen"])

    def test_creates_parent_dir(self):
        with TemporaryDirectory() as d:
            root = Path(d) / "deep" / "nest"
            params = LinearParams(0.0, 0.0, 1.0)
            path = ensure_linear_params_json(
                root, model_short="gemma",
                model_class_name="Gemma2ForCausalLM", params=params,
            )
            self.assertTrue(path.exists())
            self.assertEqual(path.parent, root)


if __name__ == "__main__":
    unittest.main()
