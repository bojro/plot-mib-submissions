"""Unit tests for ``mib_submission.results._aggregate``.

The aggregator is the single source of truth for ``RESULTS.md`` — its
filename parser, picked-site heuristic, and markdown renderer all need
to be correct or the published numbers drift from raw eval data. Tests
exercise each in isolation against synthetic JSON archives in tmpdirs,
plus one end-to-end smoke check against the live archives in this repo.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mib_submission.results import _aggregate as agg  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers — build synthetic archives                                          #
# --------------------------------------------------------------------------- #

def _make_archive_dict(
    *,
    task: str = "4_answer_MCQA",
    model_class: str = "Qwen2ForCausalLM",
    variable: str = "answer_pointer",
    splits=("answerPosition_test", "randomLetter_test", "answerPosition_randomLetter_test"),
    units=(),
):
    """Build a JSON dict matching the harness's eval output schema.

    ``units`` is a list of dicts each with keys
    ``layer, position, scores: dict[split, float]``.
    """
    out = {
        "method_name": "submission",
        "model_name": model_class,
        "task_name": task,
        "dataset": {},
    }
    for split in splits:
        out["dataset"][split] = {"model_unit": {}}
        for u in units:
            unit_id = (
                f"[[AtomicModelUnit(id='ResidualStream(Layer-{u['layer']},"
                f"Token-{u['position']})')]]"
            )
            out["dataset"][split]["model_unit"][unit_id] = {
                "metadata": {"layer": u["layer"], "position": u["position"]},
                variable: {"average_score": u["scores"][split]},
            }
    return out


def _write_archive(dirpath: Path, name: str, payload: dict) -> Path:
    p = dirpath / name
    p.write_text(json.dumps(payload))
    return p


# --------------------------------------------------------------------------- #
# Filename parser                                                             #
# --------------------------------------------------------------------------- #

class FilenameParser(unittest.TestCase):
    def test_simple_variable(self):
        self.assertEqual(
            agg._parse_archive_filename("4_answer_MCQA_Qwen2ForCausalLM_answer"),
            ("4_answer_MCQA", "Qwen2ForCausalLM", "answer"),
        )

    def test_variable_with_underscore(self):
        # `answer_pointer` is the canonical variable name with underscore;
        # the parser must not mistake it for an extra task or model segment.
        self.assertEqual(
            agg._parse_archive_filename("4_answer_MCQA_Gemma2ForCausalLM_answer_pointer"),
            ("4_answer_MCQA", "Gemma2ForCausalLM", "answer_pointer"),
        )

    def test_ravel_country(self):
        self.assertEqual(
            agg._parse_archive_filename("ravel_task_LlamaForCausalLM_Country"),
            ("ravel_task", "LlamaForCausalLM", "Country"),
        )

    def test_unknown_task_returns_none(self):
        self.assertIsNone(agg._parse_archive_filename("UnknownTask_Qwen2ForCausalLM_x"))

    def test_unknown_model_returns_none(self):
        self.assertIsNone(agg._parse_archive_filename("4_answer_MCQA_BadModel_answer"))

    def test_ablation_archive_returns_none(self):
        # Cell-1 ablation archives shouldn't accidentally be parsed as cells.
        self.assertIsNone(agg._parse_archive_filename("offplot_L15_L20"))
        self.assertIsNone(agg._parse_archive_filename("v8_mixed_results"))


# --------------------------------------------------------------------------- #
# CellResult basics                                                           #
# --------------------------------------------------------------------------- #

class CellResultProperties(unittest.TestCase):
    def _make_cell(self, units=None, picked=None, inferred=False):
        if units is None:
            units = (
                agg.UnitResult(layer=23, position="last_token",
                               per_split_iia={"aP": 1.0, "rL": 0.9, "aPrL": 0.8}),
                agg.UnitResult(layer=17, position="correct_symbol",
                               per_split_iia={"aP": 0.5, "rL": 0.6, "aPrL": 0.7}),
            )
        return agg.CellResult(
            task="4_answer_MCQA",
            model_class="Qwen2ForCausalLM",
            variable="answer_pointer",
            splits=("aP", "rL", "aPrL"),
            units=units,
            picked_sites=picked,
            picked_sites_inferred=inferred,
        )

    def test_per_split_max_iia_picks_max(self):
        c = self._make_cell()
        self.assertEqual(c.per_split_max_iia, {"aP": 1.0, "rL": 0.9, "aPrL": 0.8})

    def test_mean_iia_unweighted_mean_of_split_max(self):
        c = self._make_cell()
        # max per split: 1.0, 0.9, 0.8 → mean 0.9
        self.assertAlmostEqual(c.mean_iia, 0.9, places=6)

    def test_best_site_per_split_picks_argmax(self):
        c = self._make_cell()
        best = c.best_site_per_split
        # Easy split: L23/last_token wins outright
        self.assertEqual((best["aP"].layer, best["aP"].position), (23, "last_token"))
        self.assertEqual((best["rL"].layer, best["rL"].position), (23, "last_token"))
        self.assertEqual((best["aPrL"].layer, best["aPrL"].position), (23, "last_token"))

    def test_is_shipped_requires_picked_sites(self):
        self.assertFalse(self._make_cell(picked=None).is_shipped)
        self.assertFalse(self._make_cell(picked=()).is_shipped)
        self.assertTrue(self._make_cell(picked=((23, "last_token"),)).is_shipped)

    def test_picked_units_filter(self):
        c = self._make_cell(picked=((23, "last_token"),))
        units = c.picked_units
        self.assertEqual(len(units), 1)
        self.assertEqual((units[0].layer, units[0].position), (23, "last_token"))

    def test_picked_units_empty_when_no_picks(self):
        self.assertEqual(self._make_cell(picked=None).picked_units, ())


# --------------------------------------------------------------------------- #
# _load_archive                                                                #
# --------------------------------------------------------------------------- #

class LoadArchive(unittest.TestCase):
    def test_canonical_split_order(self):
        # Author the archive with splits in REVERSE order; loader should
        # canonicalise back to easy → hard.
        units = [
            {"layer": 23, "position": "last_token",
             "scores": {"answerPosition_test": 1.0, "randomLetter_test": 1.0,
                        "answerPosition_randomLetter_test": 0.85}},
        ]
        with tempfile.TemporaryDirectory() as td:
            p = _write_archive(
                Path(td), "4_answer_MCQA_Qwen2ForCausalLM_answer.json",
                _make_archive_dict(
                    splits=("answerPosition_randomLetter_test", "randomLetter_test", "answerPosition_test"),
                    variable="answer", units=units,
                ),
            )
            with mock.patch.object(agg, "SUBMISSIONS_DIR", Path(td) / "no-such"):
                cell = agg._load_archive(p)
        self.assertEqual(
            cell.splits,
            ("answerPosition_test", "randomLetter_test", "answerPosition_randomLetter_test"),
        )
        # Mean IIA = max-per-split average = (1.0 + 1.0 + 0.85)/3
        self.assertAlmostEqual(cell.mean_iia, (1.0 + 1.0 + 0.85) / 3, places=6)

    def test_aggregates_per_unit_across_splits(self):
        units = [
            {"layer": 23, "position": "last_token",
             "scores": {"answerPosition_test": 1.0, "randomLetter_test": 0.9,
                        "answerPosition_randomLetter_test": 0.8}},
            {"layer": 23, "position": "correct_symbol",
             "scores": {"answerPosition_test": 0.0, "randomLetter_test": 0.029,
                        "answerPosition_randomLetter_test": 0.081}},
        ]
        with tempfile.TemporaryDirectory() as td:
            p = _write_archive(
                Path(td), "4_answer_MCQA_Qwen2ForCausalLM_answer_pointer.json",
                _make_archive_dict(units=units),
            )
            with mock.patch.object(agg, "SUBMISSIONS_DIR", Path(td) / "no-such"):
                cell = agg._load_archive(p)
        self.assertEqual(len(cell.units), 2)
        # Sorted by (layer, position)
        u_correct, u_last = sorted(cell.units, key=lambda u: u.position)
        self.assertEqual(u_last.position, "last_token")
        self.assertAlmostEqual(u_last.per_split_iia["answerPosition_test"], 1.0)
        self.assertAlmostEqual(u_correct.per_split_iia["answerPosition_test"], 0.0)


# --------------------------------------------------------------------------- #
# Picked-site heuristic                                                        #
# --------------------------------------------------------------------------- #

class PickedSiteHeuristic(unittest.TestCase):
    """The heuristic should:
    - ship when there's no submission folder
    - flag inferred=True
    - threshold on `answerPosition_test` only (the informative split)
    - exclude identity baselines (which can score 1.0 on randomLetter_test
      where the variable doesn't change)
    """

    def _load_with_units(self, units, variable="answer_pointer"):
        with tempfile.TemporaryDirectory() as td:
            p = _write_archive(
                Path(td), f"4_answer_MCQA_Qwen2ForCausalLM_{variable}.json",
                _make_archive_dict(variable=variable, units=units),
            )
            # Force the submission folder to NOT exist so heuristic kicks in.
            with mock.patch.object(agg, "SUBMISSIONS_DIR", Path(td) / "no-such"):
                return agg._load_archive(p)

    def test_includes_high_iia_units(self):
        units = [
            {"layer": 23, "position": "last_token",
             "scores": {"answerPosition_test": 0.95, "randomLetter_test": 1.0,
                        "answerPosition_randomLetter_test": 0.8}},
        ]
        cell = self._load_with_units(units)
        self.assertTrue(cell.picked_sites_inferred)
        self.assertEqual(cell.picked_sites, ((23, "last_token"),))

    def test_excludes_identity_baselines_with_high_rL(self):
        # Identity baselines for `answer_pointer` score perfectly on
        # randomLetter_test (where pointer is identical between base and
        # source). The heuristic must NOT pick these.
        units = [
            {"layer": 23, "position": "last_token",
             "scores": {"answerPosition_test": 0.95, "randomLetter_test": 1.0,
                        "answerPosition_randomLetter_test": 0.8}},
            # Identity-baseline pattern: aP=0, rL≈1, aPrL=small
            {"layer": 23, "position": "correct_symbol",
             "scores": {"answerPosition_test": 0.0, "randomLetter_test": 1.0,
                        "answerPosition_randomLetter_test": 0.081}},
            # Another identity baseline
            {"layer": 17, "position": "correct_symbol",
             "scores": {"answerPosition_test": 0.0, "randomLetter_test": 0.95,
                        "answerPosition_randomLetter_test": 0.05}},
        ]
        cell = self._load_with_units(units)
        self.assertEqual(cell.picked_sites, ((23, "last_token"),))

    def test_no_inference_when_folder_present(self):
        # If the submission folder *does* exist, picked_sites comes from
        # the featurizer files, not the heuristic.
        units = [
            {"layer": 23, "position": "last_token",
             "scores": {"answerPosition_test": 0.95, "randomLetter_test": 1.0,
                        "answerPosition_randomLetter_test": 0.8}},
        ]
        with tempfile.TemporaryDirectory() as td:
            p = _write_archive(
                Path(td), "4_answer_MCQA_Qwen2ForCausalLM_answer_pointer.json",
                _make_archive_dict(units=units),
            )
            # Build a fake submission folder with one *different* picked site
            # so we can distinguish folder-from-heuristic.
            cell_dir = Path(td) / "4_answer_MCQA_Qwen2ForCausalLM_answer_pointer"
            cell_dir.mkdir()
            (cell_dir / "ResidualStream(Layer-15,Token-last_token)_featurizer").touch()
            (cell_dir / "ResidualStream(Layer-15,Token-last_token)_inverse_featurizer").touch()
            (cell_dir / "ResidualStream(Layer-15,Token-last_token)_indices").touch()
            with mock.patch.object(agg, "SUBMISSIONS_DIR", Path(td)):
                cell = agg._load_archive(p)
        self.assertFalse(cell.picked_sites_inferred)
        # Folder picks were L15, not the L23 the heuristic would have chosen.
        self.assertEqual(cell.picked_sites, ((15, "last_token"),))

    def test_returns_none_when_no_units_qualify(self):
        # Archives whose units are all identity-grade. Heuristic should not
        # invent picks; cell remains "not shipped".
        units = [
            {"layer": 0, "position": "last_token",
             "scores": {"answerPosition_test": 0.0, "randomLetter_test": 0.029,
                        "answerPosition_randomLetter_test": 0.081}},
        ]
        cell = self._load_with_units(units)
        self.assertIsNone(cell.picked_sites)
        self.assertFalse(cell.is_shipped)


# --------------------------------------------------------------------------- #
# _read_picked_sites                                                          #
# --------------------------------------------------------------------------- #

class ReadPickedSites(unittest.TestCase):
    def test_returns_none_when_folder_missing(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(agg, "SUBMISSIONS_DIR", Path(td)):
                self.assertIsNone(agg._read_picked_sites(
                    "4_answer_MCQA", "Qwen2ForCausalLM", "answer_pointer"))

    def test_extracts_from_featurizer_filenames(self):
        with tempfile.TemporaryDirectory() as td:
            cell_dir = Path(td) / "4_answer_MCQA_Qwen2ForCausalLM_answer_pointer"
            cell_dir.mkdir()
            for L, t in [(7, "correct_symbol"), (17, "last_token"), (17, "correct_symbol")]:
                base = cell_dir / f"ResidualStream(Layer-{L},Token-{t})"
                (Path(str(base) + "_featurizer")).touch()
                (Path(str(base) + "_inverse_featurizer")).touch()
                (Path(str(base) + "_indices")).touch()
            with mock.patch.object(agg, "SUBMISSIONS_DIR", Path(td)):
                picks = agg._read_picked_sites(
                    "4_answer_MCQA", "Qwen2ForCausalLM", "answer_pointer")
        # Sorted alphabetically by filename → (L17, correct_symbol),
        # (L17, last_token), (L7, correct_symbol).
        self.assertEqual(set(picks), {(7, "correct_symbol"),
                                       (17, "last_token"),
                                       (17, "correct_symbol")})
        self.assertEqual(len(picks), 3)

    def test_ignores_non_featurizer_files(self):
        with tempfile.TemporaryDirectory() as td:
            cell_dir = Path(td) / "4_answer_MCQA_Qwen2ForCausalLM_answer_pointer"
            cell_dir.mkdir()
            (cell_dir / "ResidualStream(Layer-23,Token-last_token)_featurizer").touch()
            (cell_dir / "ResidualStream(Layer-23,Token-last_token)_inverse_featurizer").touch()
            (cell_dir / "ResidualStream(Layer-23,Token-last_token)_indices").touch()
            (cell_dir / "submission_results.json").touch()  # noise file
            with mock.patch.object(agg, "SUBMISSIONS_DIR", Path(td)):
                picks = agg._read_picked_sites(
                    "4_answer_MCQA", "Qwen2ForCausalLM", "answer_pointer")
        self.assertEqual(picks, ((23, "last_token"),))


# --------------------------------------------------------------------------- #
# Markdown rendering                                                          #
# --------------------------------------------------------------------------- #

class HeadlineTable(unittest.TestCase):
    def _make_shipped_cell(self, mean_target, inferred=False):
        # Build a cell whose mean IIA equals ``mean_target`` exactly.
        # 3 splits with max IIA mean_target each.
        units = (
            agg.UnitResult(
                layer=23, position="last_token",
                per_split_iia={
                    "answerPosition_test": mean_target,
                    "randomLetter_test": mean_target,
                    "answerPosition_randomLetter_test": mean_target,
                },
            ),
        )
        return agg.CellResult(
            task="4_answer_MCQA", model_class="Qwen2ForCausalLM",
            variable="answer_pointer",
            splits=("answerPosition_test", "randomLetter_test", "answerPosition_randomLetter_test"),
            units=units,
            picked_sites=((23, "last_token"),),
            picked_sites_inferred=inferred,
        )

    def test_table_has_required_columns(self):
        # Per-task sub-tables: task name appears as a heading; "task" is no
        # longer a per-row column.
        out = agg.headline_table([self._make_shipped_cell(0.9)])
        self.assertIn("4_answer_MCQA", out)  # task as heading
        for col in ["model", "variable", "sites",
                    "aP", "rL", "aPrL", "**mean IIA**"]:
            self.assertIn(col, out)

    def test_mean_iia_bolded(self):
        out = agg.headline_table([self._make_shipped_cell(0.9)])
        self.assertIn("**0.900**", out)

    def test_dagger_only_when_inferred(self):
        out_inf = agg.headline_table([self._make_shipped_cell(0.9, inferred=True)])
        out_real = agg.headline_table([self._make_shipped_cell(0.9, inferred=False)])
        self.assertIn("4†", out_inf.replace("1†", "4†"))  # dagger present
        self.assertNotIn("†", out_real)

    def test_empty_input(self):
        out = agg.headline_table([])
        self.assertIn("no shipped cells", out.lower())


class PerCellAppendix(unittest.TestCase):
    def test_includes_mean_iia_and_picks(self):
        units = (
            agg.UnitResult(
                layer=23, position="last_token",
                per_split_iia={
                    "answerPosition_test": 1.0,
                    "randomLetter_test": 0.9,
                    "answerPosition_randomLetter_test": 0.85,
                },
            ),
        )
        cell = agg.CellResult(
            task="4_answer_MCQA", model_class="Qwen2ForCausalLM",
            variable="answer_pointer",
            splits=("answerPosition_test", "randomLetter_test", "answerPosition_randomLetter_test"),
            units=units, picked_sites=((23, "last_token"),),
        )
        out = agg.per_cell_appendix(cell)
        self.assertIn("Mean IIA: 0.917", out)  # (1.0 + 0.9 + 0.85)/3
        self.assertIn("L23/last_token", out)
        # Best per split section should reference each split
        for s in ["answerPosition_test", "randomLetter_test", "answerPosition_randomLetter_test"]:
            self.assertIn(s, out)


# --------------------------------------------------------------------------- #
# End-to-end                                                                   #
# --------------------------------------------------------------------------- #

class EndToEnd(unittest.TestCase):
    def test_emit_results_md_against_live_archives(self):
        """Smoke check: the live archives in this repo render without error
        and produce a doc with the expected sections."""
        results = agg.load_all()
        md = agg.emit_results_md(results)
        for marker in [
            "# PLOT MIB submission — Results",
            "## Headline",
            "## Methods",
            "## Per-cell breakdowns",
            "**mean IIA**",
        ]:
            self.assertIn(marker, md)

    def test_load_all_skips_ablation_archives(self):
        # ``offplot_L15_L20.json`` and ``v8_mixed_results.json`` exist in
        # the live archives; their filenames don't match the cell schema.
        results = agg.load_all()
        names = {(r.task, r.model_class, r.variable) for r in results}
        # Make sure none of the parsed cells have a "weird" prefix/model.
        for t, m, v in names:
            self.assertIn(t, agg._TASK_NAMES)
            self.assertIn(m, agg._MODEL_NAMES)


if __name__ == "__main__":
    unittest.main()
