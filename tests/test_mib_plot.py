"""Pure tests for the PLOT pipeline (no LM, no MIB harness needed).

We exercise the math and shape contracts:
  - L2 row normalisation
  - cost_matrix on the three metrics
  - balanced + one-sided UOT solvers
  - build_abstract_effect_row sign / sparsity
  - aggregate_to_layer + select_sites_via_plot end-to-end on a stub
    ExperimentBundle, planted-signal layer/position recovery
"""

from __future__ import annotations

import math
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from mib_submission.plot.features import (  # noqa: E402
    NeuralOutputs,
    aggregate_mean,
    build_abstract_effect_row,
    build_abstract_table,
    normalize_rows,
)
from mib_submission.plot.transport import (  # noqa: E402
    cost_matrix,
    row_normalize,
    sinkhorn_one_sided_uot,
    sinkhorn_uniform_ot,
    truncate_row,
)
from mib_submission.plot.pipeline import (  # noqa: E402
    PlotConfig,
    _aggregate_to_layer_table,
    _layer_token_table,
    select_sites_via_plot,
)


# --------------------------------------------------------------------------- #
# Stubs used by the end-to-end test                                           #
# --------------------------------------------------------------------------- #

class _StubCausalModel:
    """Causal model whose interchange always sets the output to a fixed letter
    derived from the variable name and the counterfactual input."""

    def run_forward(self, input_dict):
        return {"answer": " " + str(input_dict["base_letter"])}

    def run_interchange(self, input_dict, intervention):
        # variable -> cf_letter map keyed on variable name to give distinct rows
        var, cf = next(iter(intervention.items()))
        # for `answer_pointer`: source letter mirrors cf input
        # for `answer`: source letter is shifted (cyclic) so the two abstract
        # rows differ — V=2 gives Sinkhorn discrimination
        cf_letter = str(cf["source_letter"])
        if var == "answer":
            cf_letter = chr(((ord(cf_letter) - ord("A") + 1) % 26) + ord("A"))
        return {"answer": " " + cf_letter}


@dataclass
class _StubDatasetExample:
    base_letter: str
    source_letter: str

    def __getitem__(self, key):
        return {
            "input": {"base_letter": self.base_letter},
            "counterfactual_inputs": [{"source_letter": self.source_letter}],
        }[key]


class _StubDataset:
    """Yields dict-like examples with `input` and `counterfactual_inputs`."""

    def __init__(self, examples):
        self._examples = examples

    def __iter__(self):
        return iter(self._examples.__iter__.__call__() if False else (
            {
                "input": {"base_letter": ex.base_letter},
                "counterfactual_inputs": [{"source_letter": ex.source_letter}],
            }
            for ex in self._examples
        ))


# --------------------------------------------------------------------------- #
# Pure-math tests                                                             #
# --------------------------------------------------------------------------- #

class NormalizeRows(unittest.TestCase):
    def test_unit_norm(self):
        M = torch.tensor([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
        out = normalize_rows(M)
        self.assertAlmostEqual(out[0].norm().item(), 1.0, places=6)
        self.assertEqual(out[1].norm().item(), 0.0)         # zero row preserved
        self.assertAlmostEqual(out[2].norm().item(), 1.0, places=6)

    def test_aggregate_mean(self):
        rows = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 0.0])]
        out = aggregate_mean(rows)
        self.assertTrue(torch.allclose(out, torch.tensor([2.0, 1.0])))


class CostMatrix(unittest.TestCase):
    def test_sq_l2_known(self):
        A = torch.tensor([[1.0, 0.0]])
        S = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        out = cost_matrix(A, S, metric="sq_l2")
        self.assertTrue(torch.allclose(out, torch.tensor([[0.0, 2.0]])))

    def test_l1_known(self):
        A = torch.tensor([[1.0, 0.0]])
        S = torch.tensor([[1.0, 1.0]])
        out = cost_matrix(A, S, metric="l1")
        self.assertEqual(out.item(), 1.0)

    def test_cosine_unit(self):
        A = torch.tensor([[1.0, 0.0]])
        S = torch.tensor([[0.0, 1.0]])
        out = cost_matrix(A, S, metric="cosine")
        self.assertAlmostEqual(out.item(), 1.0, places=5)    # orthogonal
        S2 = torch.tensor([[1.0, 0.0]])
        out2 = cost_matrix(A, S2, metric="cosine")
        self.assertAlmostEqual(out2.item(), 0.0, places=5)   # parallel

    def test_dim_mismatch(self):
        with self.assertRaises(ValueError):
            cost_matrix(torch.zeros(1, 3), torch.zeros(2, 4))


class Sinkhorn(unittest.TestCase):
    def test_uniform_ot_marginals(self):
        cost = torch.tensor([[0.0, 2.0, 2.0], [2.0, 0.0, 2.0]])
        pi = sinkhorn_uniform_ot(cost, epsilon=0.1, n_iter=300)
        # row marginals 1/V, col marginals 1/M
        self.assertTrue(torch.allclose(pi.sum(dim=1), torch.full((2,), 0.5), atol=1e-4))
        self.assertTrue(torch.allclose(pi.sum(dim=0), torch.full((3,), 1.0/3), atol=1e-4))

    def test_uniform_ot_argmax_with_v2(self):
        # 2 rows, 3 cols. Row 0 matches col 0; row 1 matches col 2.
        cost = torch.tensor([
            [0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0],
        ])
        pi = sinkhorn_uniform_ot(cost, epsilon=0.05, n_iter=500)
        self.assertEqual(int(torch.argmax(pi[0]).item()), 0)
        self.assertEqual(int(torch.argmax(pi[1]).item()), 2)

    def test_one_sided_uot_relaxes_columns(self):
        # One-sided UOT relaxes the column marginal; mass concentrates more.
        cost = torch.tensor([[0.0, 5.0, 5.0]])
        pi_bal = sinkhorn_uniform_ot(cost, epsilon=0.1, n_iter=300)
        pi_uot = sinkhorn_one_sided_uot(cost, epsilon=0.1, beta_neural=0.1, n_iter=300)
        # UOT should be more peaked at col 0 (lowest cost).
        self.assertGreater(pi_uot[0, 0] / pi_uot[0].sum(), pi_bal[0, 0] / pi_bal[0].sum())

    def test_truncate_row_normalises(self):
        row = torch.tensor([0.4, 0.1, 0.3, 0.2])
        out = truncate_row(row, top_k=2)
        self.assertEqual({i for i, _ in out}, {0, 2})
        self.assertAlmostEqual(sum(w for _, w in out), 1.0, places=6)


# --------------------------------------------------------------------------- #
# Abstract effect signature                                                   #
# --------------------------------------------------------------------------- #

class AbstractEffectRow(unittest.TestCase):
    def test_unnormalized_signs(self):
        cm = _StubCausalModel()
        ds = list(_StubDataset([
            _StubDatasetExample("A", "C"),
            _StubDatasetExample("B", "A"),
        ]))
        # Build abstract row for variable "answer_pointer" (source = cf_letter).
        row = build_abstract_effect_row(
            cm, ds, variable="answer_pointer",
            letters="ABCD", normalize=False,
        )
        # Per-example one-hot diffs:
        #   ex0: +1 at C, -1 at A
        #   ex1: +1 at A, -1 at B
        # mean: A=0, B=-0.5, C=+0.5, D=0
        expected = torch.tensor([0.0, -0.5, 0.5, 0.0])
        self.assertTrue(torch.allclose(row, expected, atol=1e-6))

    def test_normalized_unit_norm(self):
        cm = _StubCausalModel()
        ds = list(_StubDataset([_StubDatasetExample("A", "C")]))
        row = build_abstract_effect_row(
            cm, ds, variable="answer_pointer", letters="ABCD", normalize=True,
        )
        self.assertAlmostEqual(row.norm().item(), 1.0, places=5)

    def test_two_variables_distinct_rows(self):
        cm = _StubCausalModel()
        ds = list(_StubDataset([_StubDatasetExample("A", "C")]))
        table = build_abstract_table(
            cm, ds, variables=("answer_pointer", "answer"),
            letters="ABCD", normalize=False,
        )
        # answer_pointer: +C -A; answer: +D (cyclic shift of C) -A → distinct
        self.assertFalse(torch.allclose(table[0], table[1]))


# --------------------------------------------------------------------------- #
# Pipeline helpers + end-to-end                                               #
# --------------------------------------------------------------------------- #

class LayerAggregation(unittest.TestCase):
    def test_aggregate_to_layer_table(self):
        sigs = {
            (0, "a"): torch.tensor([1.0, 0.0]),
            (0, "b"): torch.tensor([3.0, 0.0]),
            (1, "a"): torch.tensor([0.0, 2.0]),
            (1, "b"): torch.tensor([0.0, 4.0]),
        }
        table, ids = _aggregate_to_layer_table(sigs, normalize=False)
        self.assertEqual(ids, [0, 1])
        self.assertTrue(torch.allclose(table[0], torch.tensor([2.0, 0.0])))
        self.assertTrue(torch.allclose(table[1], torch.tensor([0.0, 3.0])))

    def test_layer_token_table(self):
        sigs = {
            (5, "alpha"): torch.tensor([1.0, 2.0]),
            (5, "beta"): torch.tensor([3.0, 4.0]),
            (3, "alpha"): torch.tensor([7.0, 8.0]),
        }
        table, names = _layer_token_table(sigs, layer=5)
        self.assertEqual(names, ["alpha", "beta"])
        self.assertTrue(torch.allclose(table[0], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.allclose(table[1], torch.tensor([3.0, 4.0])))


class _StubExperiment:
    def __init__(self, model_units_lists):
        self.model_units_lists = model_units_lists


@dataclass
class _StubBundle:
    causal_model: Any
    experiment: Any


class EndToEndStub(unittest.TestCase):
    """Drive ``select_sites_via_plot`` end-to-end with a monkeypatched
    ``collect_neural_effect_signatures`` so we don't need an LM. We plant a
    clear signal at (layer=2, token='last_token') and verify recovery."""

    def test_planted_signal_recovered(self):
        from mib_submission.plot import pipeline as plot_pipeline

        cm = _StubCausalModel()
        ds = _StubDataset([
            _StubDatasetExample("A", "C"),
            _StubDatasetExample("B", "D"),
            _StubDatasetExample("C", "A"),
            _StubDatasetExample("D", "B"),
        ])

        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        target_row = build_abstract_effect_row(
            cm, list(ds), variable="answer_pointer", letters=letters,
            normalize=False,
        )
        # Per-example expected counterfactual letter under "answer_pointer"
        # interchange is the source letter (matching _StubCausalModel).
        expected_letters = ["C", "D", "A", "B"]

        rng = torch.Generator().manual_seed(0)
        N = len(expected_letters)
        K = len(letters)

        # Plant matching signature at L2/last_token only. To make IIA peak
        # there too, plant per-example argmax = source letter at that site;
        # everywhere else, argmax noise.
        from mib_submission.plot._alphabets import from_letters as _from_letters

        def _stub_collect_outputs(_bundle, _dataset, *, alphabet=None, letters=None, batch_size=32, verbose=False):
            # Match the new signature; either alphabet or letters may arrive.
            if alphabet is None:
                alphabet = _from_letters(letters)
            letters_str = "".join(alphabet.labels)
            base_probs = torch.zeros(N, K)
            base_probs[:, 0] = 1.0  # base picks "A"
            base_argmax = torch.zeros(N, dtype=torch.long)
            cf_probs: dict = {}
            cf_argmax: dict = {}
            for L in range(4):
                for tok in ("first_token", "mid_token", "last_token"):
                    if (L, tok) == (2, "last_token"):
                        # Probability mass concentrated on expected_cf letter.
                        probs = torch.zeros(N, K)
                        argm = torch.zeros(N, dtype=torch.long)
                        for i, ch in enumerate(expected_letters):
                            j = letters_str.index(ch)
                            probs[i, j] = 1.0
                            argm[i] = j
                        cf_probs[(L, tok)] = probs
                        cf_argmax[(L, tok)] = argm
                    else:
                        # Random softmaxes; argmax is noise.
                        logits = torch.randn(N, K, generator=rng)
                        probs = torch.softmax(logits, dim=-1)
                        cf_probs[(L, tok)] = probs
                        cf_argmax[(L, tok)] = torch.argmax(probs, dim=-1)
            return NeuralOutputs(
                base_alpha_probs=base_probs,
                base_alpha_argmax=base_argmax,
                cf_alpha_probs=cf_probs,
                cf_alpha_argmax=cf_argmax,
                alphabet=alphabet,
            )

        plot_pipeline.collect_neural_outputs = _stub_collect_outputs

        bundle = _StubBundle(causal_model=cm, experiment=_StubExperiment([]))
        config = PlotConfig(
            variables=("answer_pointer", "answer"),
            letters=letters,
            stage_a_top_k_per_row=1,
            stage_b_top_k_per_row=1,
            stage_a_epsilon_grid=(0.01, 0.03),
            stage_b_epsilon_grid=(0.01, 0.03),
            stage_a_top_k_grid=(1,),
            stage_b_top_k_grid=(1,),
        )
        sel = select_sites_via_plot(bundle, ds, config=config)

        # Multi-row Stage A: each row picks independently. The planted
        # signal at L2 should be among the picks (the row whose abstract
        # signature matches "answer_pointer" picks L2). Other rows may pick
        # other layers based on noise.
        self.assertIn(2, sel.stage_a_layers)
        self.assertIn((2, "last_token"), sel.selected_sites)
        # The planted L2/last_token site should achieve ~1.0 IIA.
        self.assertGreater(sel.stage_b_chosen[2], 0.99)


if __name__ == "__main__":
    unittest.main()
