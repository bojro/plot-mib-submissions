"""Unit tests for the RAVEL extension.

Covers:
- ``_alphabets.py`` LabelAlphabet construction (letters / labels / causal-model)
- ``_alphabets.py`` token resolution + collision compaction
- ``features.py`` alphabet kwarg (alongside legacy ``letters``)
- ``features.py`` per-row dataset filter
- ``features.py`` ``on_unknown_label="skip"`` behaviour
- ``configs.py`` RAVEL preset structure + custom checker
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from mib_submission.plot._alphabets import (  # noqa: E402
    LabelAlphabet,
    from_causal_model_answers,
    from_labels,
    from_letters,
    resolve_tokens,
)
from mib_submission.plot.configs import (  # noqa: E402
    _ravel_checker,
    _ravel_v3_attributes,
    default_config,
)
from mib_submission.plot.features import (  # noqa: E402
    build_abstract_effect_row,
    build_abstract_table,
    expected_cf_letter_indices,
    per_site_iia,
    NeuralOutputs,
)


# --------------------------------------------------------------------------- #
# Alphabet construction                                                       #
# --------------------------------------------------------------------------- #

class FromLetters(unittest.TestCase):
    def test_basic(self):
        a = from_letters("ABCD")
        self.assertEqual(a.labels, ("A", "B", "C", "D"))
        self.assertEqual(a.label_to_dim, {"A": 0, "B": 1, "C": 2, "D": 3})
        self.assertEqual(a.num_dims, 4)
        self.assertFalse(a.has_tokens)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            from_letters("")

    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError):
            from_letters("AABC")


class FromLabels(unittest.TestCase):
    def test_strips_and_dedupes(self):
        a = from_labels(["France", "  France  ", "Germany", "", " ", "Italy"])
        self.assertEqual(a.labels, ("France", "Germany", "Italy"))
        self.assertEqual(a.num_dims, 3)

    def test_all_empty_rejected(self):
        with self.assertRaises(ValueError):
            from_labels(["", " ", "  "])


class FromCausalModel(unittest.TestCase):
    def test_reads_answer_values(self):
        class CM:
            values = {"answer": ["X", "Y", "", "Z"]}
        a = from_causal_model_answers(CM())
        self.assertEqual(set(a.labels), {"X", "Y", "Z"})


class ResolveTokens(unittest.TestCase):
    def test_compacts_collisions(self):
        # Stub tokenizer that maps:
        #   " A" -> [10], " B" -> [10] (collision), " C" -> [11]
        class StubTokenizer:
            def encode(self, s, add_special_tokens=False):
                first = s.lstrip()[0] if s.lstrip() else ""
                return {"A": [10], "B": [10], "C": [11]}.get(first, [0])
        a = from_letters("ABC")
        r = resolve_tokens(a, StubTokenizer())
        # 3 labels collapsed to 2 dims (A and B share token 10)
        self.assertEqual(r.num_dims, 2)
        self.assertEqual(r.token_ids, (10, 11))
        # A and B point to the same dim
        self.assertEqual(r.label_to_dim["A"], r.label_to_dim["B"])
        self.assertNotEqual(r.label_to_dim["A"], r.label_to_dim["C"])

    def test_no_collision(self):
        class StubTokenizer:
            def encode(self, s, add_special_tokens=False):
                return {"A": [10], "B": [11], "C": [12]}.get(s.lstrip()[0], [0])
        a = from_letters("ABC")
        r = resolve_tokens(a, StubTokenizer())
        self.assertEqual(r.num_dims, 3)
        self.assertEqual(set(r.token_ids), {10, 11, 12})

    def test_multi_token_spaced_label_skips_leading_space(self):
        """Regression for arithmetic-on-Gemma. Gemma encodes ' 0', ' 1', ...
        as ``[space_token, digit_token]`` — two tokens, of which the first is
        a generic leading-space token shared by all digits. The historic
        ``resolve_tokens`` always took ``encode(' '+lab)[0]`` which collided
        every digit on the same token id, compacting the alphabet to 1 dim and
        making Stage A's plan uniform / IIA trivially 1.0.

        Fix: when ``" {lab}"`` produces multiple tokens, prefer the no-space
        encoding if it's a single token (digit case), otherwise skip the
        leading-space token and use the second token of the spaced encoding.
        """
        class GemmaLikeTokenizer:
            """Mimics Gemma: ' 0'..' 9' encode as 2 tokens; '0'..'9' encode
            as 1 token each. ' A'..' D' encode as 1 token each (vocab merged
            "_A" etc.). ' France' as 1 token."""
            SPACE_TOK = 100
            DIGIT_TOKS = {str(d): 200 + d for d in range(10)}
            LETTER_TOKS = {chr(ord("A") + i): 300 + i for i in range(26)}
            WORD_TOKS = {"France": 400, "Germany": 401}

            def encode(self, s, add_special_tokens=False):
                if s.startswith(" "):
                    rest = s[1:]
                    if rest in self.DIGIT_TOKS:
                        return [self.SPACE_TOK, self.DIGIT_TOKS[rest]]
                    if rest in self.LETTER_TOKS:
                        return [self.LETTER_TOKS[rest]]
                    if rest in self.WORD_TOKS:
                        return [self.WORD_TOKS[rest]]
                    return []
                if s in self.DIGIT_TOKS:
                    return [self.DIGIT_TOKS[s]]
                if s in self.LETTER_TOKS:
                    return [self.LETTER_TOKS[s]]
                if s in self.WORD_TOKS:
                    return [self.WORD_TOKS[s]]
                return []

        tok = GemmaLikeTokenizer()
        # Digits: with the bug, all collide on SPACE_TOK=100 → num_dims=1.
        # With the fix, each maps to its own digit token → num_dims=10.
        digits = resolve_tokens(from_letters("0123456789"), tok)
        self.assertEqual(digits.num_dims, 10)
        self.assertNotIn(tok.SPACE_TOK, digits.token_ids)
        for d in range(10):
            self.assertEqual(digits.label_to_dim[str(d)], d)

        # Letters: " A".." D" each encode as a single token; that should
        # remain the picked id (no regression vs MCQA / RAVEL behaviour).
        letters = resolve_tokens(from_letters("ABCD"), tok)
        self.assertEqual(letters.num_dims, 4)
        self.assertEqual(letters.token_ids, (300, 301, 302, 303))


# --------------------------------------------------------------------------- #
# Stub causal model + dataset for features.py tests                           #
# --------------------------------------------------------------------------- #

class _StubCausalModel:
    """Deterministic causal model for testing per-row filtering.

    Variables: queried_attr (str), answer (one of A-Z), and a "swap variable"
    that the test specifies via ``run_interchange``.
    """
    values = {"answer": list("ABCDEFGHIJ")}

    def run_forward(self, inp):
        # Base: answer is whatever's in inp under "base_answer"
        return {"answer": inp["base_answer"]}

    def run_interchange(self, inp, swap):
        # If the swap variable matches inp["queried_attr"], answer changes to
        # source. Otherwise, answer stays = base.
        var = next(iter(swap.keys()))
        cf_input = swap[var]
        if inp.get("queried_attr") == var:
            return {"answer": cf_input["base_answer"]}
        return {"answer": inp["base_answer"]}


def _make_stub_dataset(rows):
    """Return a list-of-dicts dataset compatible with features.py iteration."""
    return [{"input": r["input"], "counterfactual_inputs": [r["cf"]]} for r in rows]


# --------------------------------------------------------------------------- #
# build_abstract_effect_row / build_abstract_table extensions                  #
# --------------------------------------------------------------------------- #

class AlphabetKwarg(unittest.TestCase):
    def test_alphabet_arg_supported(self):
        cm = _StubCausalModel()
        ds = _make_stub_dataset([
            {"input": {"queried_attr": "X", "base_answer": "A"},
             "cf": {"queried_attr": "X", "base_answer": "C"}},
        ])
        alpha = from_letters("ABCDEFGHIJ")
        row = build_abstract_effect_row(
            cm, ds, variable="X", alphabet=alpha, normalize=False,
        )
        # Only example: base=A (idx 0), source=C (idx 2). Row mean = +1@C, -1@A.
        self.assertAlmostEqual(row[0].item(), -1.0)
        self.assertAlmostEqual(row[2].item(), 1.0)

    def test_letters_legacy_path_still_works(self):
        cm = _StubCausalModel()
        ds = _make_stub_dataset([
            {"input": {"queried_attr": "X", "base_answer": "A"},
             "cf": {"queried_attr": "X", "base_answer": "B"}},
        ])
        row = build_abstract_effect_row(
            cm, ds, variable="X", letters="ABC", normalize=False,
        )
        self.assertAlmostEqual(row[0].item(), -1.0)
        self.assertAlmostEqual(row[1].item(), 1.0)

    def test_must_provide_one(self):
        cm = _StubCausalModel()
        ds = _make_stub_dataset([])
        with self.assertRaises(ValueError):
            build_abstract_effect_row(cm, ds, variable="X")

    def test_cant_provide_both(self):
        cm = _StubCausalModel()
        ds = _make_stub_dataset([])
        alpha = from_letters("AB")
        with self.assertRaises(ValueError):
            build_abstract_effect_row(
                cm, ds, variable="X",
                alphabet=alpha, letters="AB",
            )


class PerRowDatasetFilter(unittest.TestCase):
    """The bread-and-butter RAVEL feature: each OT row sees its own subset."""

    def test_filter_partitions_dataset(self):
        cm = _StubCausalModel()
        # 4 rows: 2 queried for X, 2 queried for Y.
        # X rows: interchange(X) flips A→B and A→C
        # Y rows: interchange(Y) flips A→D and A→E
        ds = _make_stub_dataset([
            {"input": {"queried_attr": "X", "base_answer": "A"},
             "cf": {"queried_attr": "X", "base_answer": "B"}},
            {"input": {"queried_attr": "X", "base_answer": "A"},
             "cf": {"queried_attr": "X", "base_answer": "C"}},
            {"input": {"queried_attr": "Y", "base_answer": "A"},
             "cf": {"queried_attr": "Y", "base_answer": "D"}},
            {"input": {"queried_attr": "Y", "base_answer": "A"},
             "cf": {"queried_attr": "Y", "base_answer": "E"}},
        ])
        alpha = from_letters("ABCDEFGHIJ")
        filt_X = lambda ex: ex["input"]["queried_attr"] == "X"
        filt_Y = lambda ex: ex["input"]["queried_attr"] == "Y"
        table = build_abstract_table(
            cm, ds, variables=("X", "Y"),
            alphabet=alpha, normalize=False,
            per_row_dataset_filter=[filt_X, filt_Y],
        )
        # Row X: averaged over its 2 examples — base always A, source ∈ {B,C}.
        # Row vector: -1 at A, +0.5 at B, +0.5 at C.
        self.assertAlmostEqual(table[0, 0].item(), -1.0)  # A
        self.assertAlmostEqual(table[0, 1].item(), 0.5)   # B
        self.assertAlmostEqual(table[0, 2].item(), 0.5)   # C
        self.assertAlmostEqual(table[0, 3].item(), 0.0)   # D — not in X's pool
        # Row Y: -1 at A, +0.5 at D, +0.5 at E.
        self.assertAlmostEqual(table[1, 3].item(), 0.5)   # D
        self.assertAlmostEqual(table[1, 4].item(), 0.5)   # E
        self.assertAlmostEqual(table[1, 1].item(), 0.0)   # B — not in Y's pool

    def test_filter_length_must_match_variables(self):
        cm = _StubCausalModel()
        ds = _make_stub_dataset([])
        alpha = from_letters("AB")
        with self.assertRaises(ValueError):
            build_abstract_table(
                cm, ds, variables=("X", "Y"),
                alphabet=alpha,
                per_row_dataset_filter=[lambda ex: True],  # length 1, not 2
            )

    def test_no_filter_uses_all_examples(self):
        """Default behaviour (per_row_dataset_filter=None) is unchanged."""
        cm = _StubCausalModel()
        ds = _make_stub_dataset([
            {"input": {"queried_attr": "X", "base_answer": "A"},
             "cf": {"queried_attr": "X", "base_answer": "B"}},
        ])
        alpha = from_letters("AB")
        table = build_abstract_table(
            cm, ds, variables=("X",),
            alphabet=alpha, normalize=False,
        )
        self.assertAlmostEqual(table[0, 0].item(), -1.0)
        self.assertAlmostEqual(table[0, 1].item(), 1.0)


class OnUnknownLabel(unittest.TestCase):
    def test_skip_drops_unknown(self):
        cm = _StubCausalModel()
        ds = _make_stub_dataset([
            {"input": {"queried_attr": "X", "base_answer": "A"},
             "cf": {"queried_attr": "X", "base_answer": "B"}},
            {"input": {"queried_attr": "X", "base_answer": "A"},
             "cf": {"queried_attr": "X", "base_answer": "Z"}},  # Z not in alphabet
        ])
        alpha = from_letters("ABC")
        # raise mode: error
        with self.assertRaises(ValueError):
            build_abstract_effect_row(
                cm, ds, variable="X",
                alphabet=alpha, normalize=False, on_unknown_label="raise",
            )
        # skip mode: row computed from the surviving example only
        row = build_abstract_effect_row(
            cm, ds, variable="X",
            alphabet=alpha, normalize=False, on_unknown_label="skip",
        )
        self.assertAlmostEqual(row[0].item(), -1.0)  # A
        self.assertAlmostEqual(row[1].item(), 1.0)   # B
        # Z's example was skipped — only the A→B row contributed.

    def test_expected_cf_skip_returns_minus1(self):
        cm = _StubCausalModel()
        ds = _make_stub_dataset([
            {"input": {"queried_attr": "X", "base_answer": "A"},
             "cf": {"queried_attr": "X", "base_answer": "B"}},
            {"input": {"queried_attr": "X", "base_answer": "A"},
             "cf": {"queried_attr": "X", "base_answer": "Z"}},  # unknown
        ])
        alpha = from_letters("ABC")
        idx = expected_cf_letter_indices(
            cm, ds, variable="X", alphabet=alpha, on_unknown_label="skip",
        )
        self.assertEqual(idx.tolist(), [1, -1])


class PerSiteIIAMasking(unittest.TestCase):
    def test_minus1_indices_excluded_from_iia(self):
        # Build a fake NeuralOutputs with 4 examples and 1 site.
        # Site argmax: [0, 1, 2, 3]. Expected: [0, 1, -1, 5]
        # → matches at j=0, j=1; j=2 masked; j=3 mismatch.
        # Effective: 2 of 3 valid examples correct → IIA = 2/3.
        alpha = from_letters("ABCDEF")
        outs = NeuralOutputs(
            base_alpha_probs=torch.zeros(4, 6),
            base_alpha_argmax=torch.zeros(4, dtype=torch.long),
            cf_alpha_probs={(0, "x"): torch.zeros(4, 6)},
            cf_alpha_argmax={(0, "x"): torch.tensor([0, 1, 2, 3])},
            alphabet=alpha,
        )
        expected = torch.tensor([0, 1, -1, 5])
        iia = per_site_iia(outs, expected)
        self.assertAlmostEqual(iia[(0, "x")], 2 / 3, places=6)


# --------------------------------------------------------------------------- #
# RAVEL config + checker                                                      #
# --------------------------------------------------------------------------- #

class RavelConfig(unittest.TestCase):
    def test_preset_for_each_attribute(self):
        for v in ("Country", "Continent", "Language"):
            pc = _ravel_v3_attributes(v)
            self.assertEqual(pc.variables, ("Country", "Continent", "Language"))
            self.assertEqual(pc.calibration_variable, v)
            self.assertEqual(pc.per_row_filter_attribute, "queried_attribute")
            self.assertTrue(pc.answer_alphabet_from_causal_model)
            self.assertEqual(pc.on_unknown_label, "skip")
            self.assertEqual(pc.target_row_index,
                             ("Country", "Continent", "Language").index(v))

    def test_invalid_variable_rejected(self):
        with self.assertRaises(ValueError):
            _ravel_v3_attributes("Latitude")

    def test_default_config_for_ravel_sets_baseline_overrides(self):
        c = default_config("ravel_task", "google/gemma-2-2b", "Country")
        self.assertEqual(c.n_features, 288)
        self.assertEqual(c.training_epochs, 1)
        self.assertEqual(c.max_new_tokens, 2)
        self.assertEqual(c.signature_dataset, "attribute_train")
        self.assertIsNotNone(c.checker)
        # checker is callable
        self.assertTrue(callable(c.checker))


class RavelChecker(unittest.TestCase):
    def test_simple_match(self):
        self.assertTrue(_ravel_checker("Italy", "Italy"))
        self.assertTrue(_ravel_checker("italy.", "Italy"))
        self.assertFalse(_ravel_checker("France", "Italy"))

    def test_none_output(self):
        self.assertFalse(_ravel_checker(None, "Italy"))

    def test_comma_list_expected(self):
        self.assertTrue(_ravel_checker("French", "French,Wolof"))
        self.assertTrue(_ravel_checker("Wolof", "French,Wolof"))
        self.assertFalse(_ravel_checker("English", "French,Wolof"))

    def test_united_states_edge_case(self):
        # Documenting the upstream baseline's actual behaviour (ported from
        # ravel_baselines.py, kept verbatim).
        self.assertTrue(_ravel_checker("United States", "United States"))
        self.assertTrue(_ravel_checker("us", "United States"))
        # Stripped lowercased: "the united kingdom" - "the " = "united kingdom"
        self.assertTrue(_ravel_checker("The United Kingdom", "United Kingdom"))
        # Note: "USA" is *not* matched by the upstream checker — it doesn't
        # know that abbreviation. Documented here so the behaviour change
        # would be flagged in tests.
        self.assertFalse(_ravel_checker("USA", "United States"))

    def test_south_korea(self):
        self.assertTrue(_ravel_checker("Korea", "South Korea"))
        self.assertTrue(_ravel_checker("South Korea", "South Korea"))

    def test_north_america(self):
        self.assertTrue(_ravel_checker("America", "North America"))
        self.assertTrue(_ravel_checker("NA", "North America"))
        self.assertTrue(_ravel_checker("North America", "North America"))

    def test_chinese_mandarin(self):
        self.assertTrue(_ravel_checker("Chinese", "Mandarin"))
        self.assertTrue(_ravel_checker("Mandarin", "Chinese"))


class TokenResolutionEagerInPipeline(unittest.TestCase):
    """Regression: when ``select_sites_via_plot`` builds an alphabet via
    ``answer_alphabet_from_causal_model``, abstract and neural tables must
    use the SAME post-resolution dim count. Earlier bug: 928 labels →
    abstract used 928 dims while neural's lazy ``resolve_tokens`` shrank
    to 271 dims, causing ``cost_matrix`` to fail with feature-dim mismatch.
    """

    def test_resolved_alphabet_dim_matches_token_ids_count(self):
        from mib_submission.plot.pipeline import _resolve_config_alphabet, PlotConfig

        # Causal model with 6 answers but pretend tokenizer collapses 2 pairs
        # to the same first token (so 6 → 4 dims after resolution).
        class CM:
            values = {"answer": ["United States", "United Kingdom",
                                 "South Africa", "South Korea",
                                 "France", "Germany"]}

        class StubTokenizer:
            def encode(self, s, add_special_tokens=False):
                # First token = first word's first letter, with hard-coded
                # collisions on " United" and " South".
                stripped = s.lstrip()
                if stripped.startswith("United"): return [10]
                if stripped.startswith("South"):  return [20]
                if stripped.startswith("France"): return [30]
                if stripped.startswith("Germany"): return [40]
                return [99]

        class StubPipeline:
            tokenizer = StubTokenizer()

        class StubBundle:
            causal_model = CM()
            pipeline = StubPipeline()

        config = PlotConfig(
            variables=("x",),
            answer_alphabet_from_causal_model=True,
            letters="",  # not used
        )
        alpha = _resolve_config_alphabet(config, StubBundle())
        self.assertTrue(alpha.has_tokens, "alphabet must be resolved when tokenizer is available")
        self.assertEqual(alpha.num_dims, 4)
        self.assertEqual(set(alpha.token_ids), {10, 20, 30, 40})
        # Crucially: every label maps to a dim within [0, num_dims).
        for lab, d in alpha.label_to_dim.items():
            self.assertLess(d, alpha.num_dims, f"label {lab!r} dim={d} >= num_dims")

    def test_no_tokenizer_keeps_alphabet_unresolved(self):
        """Stub-bundle path (no pipeline.tokenizer) should leave alphabet
        unresolved so unit tests that stub ``collect_neural_outputs`` still work."""
        from mib_submission.plot.pipeline import _resolve_config_alphabet, PlotConfig

        class StubBundle:
            pass  # no pipeline, no tokenizer

        config = PlotConfig(letters="ABCD")
        alpha = _resolve_config_alphabet(config, StubBundle())
        # letters path → from_letters; no tokenizer means no resolve.
        self.assertFalse(alpha.has_tokens)
        self.assertEqual(alpha.num_dims, 4)


class ArithmeticConfig(unittest.TestCase):
    """Arithmetic V=2 carry-children + V=4 operand presets and the
    ``label_from_output`` thread-through."""

    def test_variant_C_default_uses_carry_children(self):
        cfg = default_config(
            "arithmetic", "google/gemma-2-2b", "ones_carry",
        )
        self.assertEqual(cfg.plot_config.variables, ("tens_out", "hundreds_out"))
        self.assertEqual(cfg.plot_config.calibration_variable, "ones_carry")
        self.assertEqual(cfg.plot_config.output_key, "raw_output")
        self.assertEqual(cfg.plot_config.label_from_output("68"), "6")
        self.assertEqual(cfg.plot_config.label_from_output("168"), "1")
        self.assertEqual(cfg.plot_config.label_from_output(""), "")
        # Two-position task: stage_b grid must be (1,) per shortcoming §8.
        self.assertEqual(cfg.plot_config.stage_b_top_k_grid, (1,))
        # Baseline-aligned hyperparameters.
        self.assertEqual(cfg.n_features, 16)
        self.assertEqual(cfg.training_epochs, 1)
        self.assertEqual(cfg.max_new_tokens, 3)
        self.assertEqual(cfg.signature_dataset, "random_train")

    def test_variant_B_uses_operand_digits(self):
        cfg = default_config(
            "arithmetic", "google/gemma-2-2b", "ones_carry",
            overrides={"arithmetic_variant": "B"},
        )
        self.assertEqual(
            cfg.plot_config.variables,
            ("op1_ones", "op2_ones", "op1_tens", "op2_tens"),
        )

    def test_invalid_variant_rejected(self):
        with self.assertRaises(ValueError):
            default_config(
                "arithmetic", "google/gemma-2-2b", "ones_carry",
                overrides={"arithmetic_variant": "Q"},
            )

    def test_invalid_variable_rejected(self):
        with self.assertRaises(ValueError):
            default_config(
                "arithmetic", "google/gemma-2-2b", "tens_out",
            )

    def test_arithmetic_variant_rejected_on_other_tasks(self):
        with self.assertRaises(ValueError):
            default_config(
                "4_answer_MCQA", "google/gemma-2-2b", "answer",
                overrides={"arithmetic_variant": "C"},
            )

    def test_label_from_output_threads_through_abstract_table(self):
        """Regression: build_abstract_table must use label_from_output to
        project arithmetic ``raw_output`` strings ("68") down to alphabet
        keys ("6"). Without this, every example silently skipped."""
        from mib_submission.plot._alphabets import from_letters
        from mib_submission.plot.configs import _arithmetic_first_digit

        # Stub causal model that returns raw_output for run_forward
        # and the source's raw_output under interchange.
        class StubCM:
            def run_forward(self, inp):
                return {"raw_output": "68"}

            def run_interchange(self, inp, swap):
                # tens_out swapped to source's tens_out (whatever value); we
                # return a fake "38" to force first-char shift 6 -> 3.
                return {"raw_output": "38"}

        ds = [
            {"input": {}, "counterfactual_inputs": [{}]},
            {"input": {}, "counterfactual_inputs": [{}]},
        ]
        alpha = from_letters("0123456789")
        row = build_abstract_effect_row(
            StubCM(), ds, variable="tens_out",
            alphabet=alpha,
            output_key="raw_output",
            label_from_output=_arithmetic_first_digit,
            normalize=False,
        )
        # Two examples, each contributing +1 at idx=3 ("3") and -1 at idx=6 ("6").
        # After mean-aggregation: row[3] = 1.0, row[6] = -1.0, rest = 0.
        self.assertAlmostEqual(row[3].item(), 1.0, places=5)
        self.assertAlmostEqual(row[6].item(), -1.0, places=5)
        self.assertAlmostEqual(row.sum().item(), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
