"""Pure unit tests for ``mib_submission.signatures.signature_from_logits``.

The LM-touching paths (``collect_base_logits`` /
``collect_site_intervention_logits``) require a real model + dataset; those
are exercised end-to-end by the per-method drivers, not here.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from mib_submission.signatures import signature_from_logits  # noqa: E402


class SignatureFromLogits(unittest.TestCase):
    def test_kl_zero_when_identical(self):
        logits = torch.randn(5, 7)
        out = signature_from_logits(
            intervention_logits=logits, base_logits=logits, mode="whole_vocab_kl"
        )
        self.assertEqual(tuple(out.shape), (5,))
        self.assertTrue(torch.allclose(out, torch.zeros(5), atol=1e-6))

    def test_kl_matches_manual(self):
        # Two-class case with a known closed form.
        cf = torch.tensor([[0.0, math.log(3.0)]])  # softmax = [0.25, 0.75]
        base = torch.tensor([[0.0, 0.0]])  # softmax = [0.5, 0.5]
        out = signature_from_logits(
            intervention_logits=cf, base_logits=base, mode="whole_vocab_kl"
        )
        # KL([0.25,0.75] || [0.5,0.5]) = 0.25*log(0.5) + 0.75*log(1.5)
        expected = 0.25 * math.log(0.5) + 0.75 * math.log(1.5)
        self.assertAlmostEqual(out.item(), expected, places=5)

    def test_logit_l2(self):
        cf = torch.tensor([[1.0, 2.0, 3.0]])
        base = torch.tensor([[0.0, 0.0, 0.0]])
        out = signature_from_logits(
            intervention_logits=cf, base_logits=base, mode="logit_l2"
        )
        self.assertAlmostEqual(out.item(), math.sqrt(14.0), places=5)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            signature_from_logits(
                intervention_logits=torch.zeros(2, 3),
                base_logits=torch.zeros(2, 4),
            )

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            signature_from_logits(
                intervention_logits=torch.zeros(1, 2),
                base_logits=torch.zeros(1, 2),
                mode="bogus",
            )


if __name__ == "__main__":
    unittest.main()
