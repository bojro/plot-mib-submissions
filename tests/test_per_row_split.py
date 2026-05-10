"""Tests for the ``per_row_split_datasets`` mode in ``select_sites_via_plot``.

Coverage:
- Validation: mutual exclusivity with ``per_row_filter_attribute``;
  ``calibration_variable`` required; length match.
- Resolution: split names are looked up in ``bundle.train_data``;
  unknown split → KeyError.
- (End-to-end behaviour with real model is deferred to the IOI smoke run;
  here we assert the validation paths fail loudly with clear messages.)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mib_submission.plot.pipeline import PlotConfig, select_sites_via_plot  # noqa: E402


def _stub_bundle(train_data: dict | None = None):
    """Minimal ExperimentBundle stand-in. The validation branches we test
    don't reach the model, so most fields can be None."""
    return SimpleNamespace(
        causal_model=None,
        pipeline=None,
        experiment=None,
        train_data=train_data or {},
    )


class MutualExclusivity(unittest.TestCase):
    def test_split_and_filter_modes_rejected(self):
        config = PlotConfig(
            variables=("a", "b"),
            per_row_filter_attribute="queried_attribute",
            per_row_split_datasets=("split_a_train", "split_b_train"),
            calibration_variable="x",
        )
        with self.assertRaises(ValueError) as ctx:
            select_sites_via_plot(_stub_bundle(), object(), config=config)
        self.assertIn("mutually exclusive", str(ctx.exception))


class LengthMismatch(unittest.TestCase):
    def test_split_count_must_equal_variable_count(self):
        config = PlotConfig(
            variables=("a", "b", "c"),
            per_row_split_datasets=("only_one",),
            calibration_variable="x",
        )
        with self.assertRaises(ValueError) as ctx:
            select_sites_via_plot(
                _stub_bundle({"only_one": object()}), object(), config=config,
            )
        self.assertIn("3", str(ctx.exception))


class CalibrationVariableRequired(unittest.TestCase):
    def test_split_mode_requires_calibration_variable(self):
        config = PlotConfig(
            variables=("a", "b"),
            per_row_split_datasets=("a_train", "b_train"),
            calibration_variable=None,
        )
        with self.assertRaises(ValueError) as ctx:
            select_sites_via_plot(
                _stub_bundle({"a_train": object(), "b_train": object()}),
                object(), config=config,
            )
        self.assertIn("calibration_variable", str(ctx.exception))


class UnknownSplit(unittest.TestCase):
    def test_unknown_split_name_raises_keyerror(self):
        config = PlotConfig(
            variables=("a", "b"),
            per_row_split_datasets=("real_split", "missing_split"),
            calibration_variable="x",
        )
        with self.assertRaises(KeyError) as ctx:
            select_sites_via_plot(
                _stub_bundle({"real_split": object()}),
                object(), config=config,
            )
        self.assertIn("missing_split", str(ctx.exception))


class PreservesExistingModes(unittest.TestCase):
    """Sanity: PlotConfig with neither flag works (legacy / MCQA path is
    untouched). We can't run select_sites_via_plot fully without a bundle,
    but we can verify the dataclass accepts the existing args."""

    def test_neither_flag_set(self):
        config = PlotConfig(variables=("a", "b"))
        self.assertIsNone(config.per_row_split_datasets)
        self.assertIsNone(config.per_row_filter_attribute)

    def test_split_default_none(self):
        config = PlotConfig()
        self.assertIsNone(config.per_row_split_datasets)


if __name__ == "__main__":
    unittest.main()
