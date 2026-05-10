"""Unit tests for ``mib_submission.ioi.bootstrap``.

Covers:
- ``load_linear_params`` reading a bootstrapped JSON
- ``model_short_name`` / ``model_class_name`` round-trip
- ``LinearParams.as_dict`` returning the 3-key dict the IOI causal model
  expects
- Error paths: missing file, missing model entry, missing required keys
- The bootstrap harness invocation is NOT exercised here (it requires the
  MIB submodule + a GPU + ~15 min). See ``logs/ioi_bootstrap_*.log`` for
  smoke evidence after running it once per model.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mib_submission.ioi.bootstrap import (  # noqa: E402
    LINEAR_PARAMS_FILENAME,
    LinearParams,
    load_linear_params,
    model_class_name,
    model_short_name,
)


class ShortNameMapping(unittest.TestCase):
    def test_round_trip(self):
        for short in ("gpt2", "qwen", "gemma", "llama"):
            self.assertEqual(model_short_name(model_class_name(short)), short)

    def test_known_classes(self):
        self.assertEqual(model_short_name("GPT2LMHeadModel"), "gpt2")
        self.assertEqual(model_short_name("Qwen2ForCausalLM"), "qwen")
        self.assertEqual(model_short_name("Gemma2ForCausalLM"), "gemma")
        self.assertEqual(model_short_name("LlamaForCausalLM"), "llama")

    def test_unknown_class_rejected(self):
        with self.assertRaises(ValueError):
            model_short_name("MysteryForCausalLM")

    def test_unknown_short_rejected(self):
        with self.assertRaises(ValueError):
            model_class_name("mistral")


class LinearParamsAsDict(unittest.TestCase):
    def test_includes_only_three_keys(self):
        p = LinearParams(
            bias=0.05, token_coeff=0.77, position_coeff=2.00,
            score=0.93, model_name="google/gemma-2-2b",
        )
        self.assertEqual(
            p.as_dict(),
            {"bias": 0.05, "token_coeff": 0.77, "position_coeff": 2.00},
        )


class LoadLinearParams(unittest.TestCase):
    """Verifies the JSON loader on synthetic blobs that mirror what the
    harness script (``ioi_learn_linear_params.py``) writes."""

    def _write(self, blob: dict) -> Path:
        # New tempdir per test so they don't stomp on each other.
        d = TemporaryDirectory()
        self.addCleanup(d.cleanup)
        p = Path(d.name) / LINEAR_PARAMS_FILENAME
        p.write_text(json.dumps(blob))
        return p

    def test_basic_load_by_short_name(self):
        path = self._write({
            "gemma": {
                "bias": 0.05,
                "token_coeff": 0.77,
                "position_coeff": 2.00,
                "score": 0.93,
                "model_name": "google/gemma-2-2b",
            }
        })
        p = load_linear_params(path, model_short="gemma")
        self.assertAlmostEqual(p.bias, 0.05)
        self.assertAlmostEqual(p.token_coeff, 0.77)
        self.assertAlmostEqual(p.position_coeff, 2.00)
        self.assertAlmostEqual(p.score, 0.93)
        self.assertEqual(p.model_name, "google/gemma-2-2b")

    def test_basic_load_by_class_name(self):
        path = self._write({
            "qwen": {"bias": 0.1, "token_coeff": 0.5, "position_coeff": 1.5}
        })
        p = load_linear_params(path, model_class_name_filter="Qwen2ForCausalLM")
        self.assertAlmostEqual(p.bias, 0.1)
        self.assertIsNone(p.score)
        self.assertIsNone(p.model_name)

    def test_tolerates_top_level_model_class_field(self):
        """Example notebook adds ``model_class`` at the top level. Loader
        should ignore it."""
        path = self._write({
            "gemma": {"bias": 0.0, "token_coeff": 0.0, "position_coeff": 1.0},
            "model_class": "Gemma2ForCausalLM",
        })
        p = load_linear_params(path, model_short="gemma")
        self.assertEqual(p.position_coeff, 1.0)

    def test_missing_file_raises(self):
        with TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                load_linear_params(
                    Path(d) / "nope.json", model_short="gpt2",
                )

    def test_unknown_model_in_file_raises(self):
        path = self._write({
            "gpt2": {"bias": 0, "token_coeff": 0, "position_coeff": 0},
        })
        with self.assertRaises(KeyError) as ctx:
            load_linear_params(path, model_short="qwen")
        self.assertIn("qwen", str(ctx.exception))

    def test_missing_required_keys_raises(self):
        path = self._write({
            "gemma": {"bias": 0.0, "token_coeff": 0.0},  # no position_coeff
        })
        with self.assertRaises(KeyError) as ctx:
            load_linear_params(path, model_short="gemma")
        self.assertIn("position_coeff", str(ctx.exception))

    def test_must_provide_exactly_one_filter(self):
        path = self._write({"gpt2": {"bias": 0, "token_coeff": 0, "position_coeff": 0}})
        with self.assertRaises(ValueError):
            load_linear_params(path)
        with self.assertRaises(ValueError):
            load_linear_params(
                path, model_short="gpt2",
                model_class_name_filter="GPT2LMHeadModel",
            )

    def test_returns_dict_compatible_with_causal_model(self):
        """``get_causal_model(linear_params.as_dict())`` is the canonical
        downstream usage. Verify the dict has exactly those 3 keys."""
        path = self._write({
            "gpt2": {"bias": 0.3, "token_coeff": 0.6, "position_coeff": 2.2}
        })
        p = load_linear_params(path, model_short="gpt2")
        d = p.as_dict()
        self.assertEqual(set(d.keys()), {"bias", "token_coeff", "position_coeff"})


if __name__ == "__main__":
    unittest.main()
