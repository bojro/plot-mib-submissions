"""Tests for ``mib_submission.site_keys`` — the two site shapes and their
detection helpers. Stub units mock the upstream ResidualStream / AttentionHead
classes so the tests don't need pyvene or the MIB harness loaded."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mib_submission.site_keys import (  # noqa: E402
    AttentionHeadSiteKey,
    ResidualStreamSiteKey,
    attention_head_site_key_for_unit,
    is_attention_head_key,
    is_residual_stream_key,
    site_key_for_unit,
)


def _stub_residual_unit(layer: int, token_id: str):
    """Match the upstream ResidualStream unit shape: a unit with
    ``unit.component.get_layer()`` and ``unit.token_indices.id``."""
    return SimpleNamespace(
        component=SimpleNamespace(get_layer=lambda: layer),
        token_indices=SimpleNamespace(id=token_id),
    )


def _stub_attention_unit(layer: int, head: int, token_id: str):
    """Match the upstream AttentionHead unit shape: a unit with
    ``unit.component.get_layer()``, ``unit.head``, ``unit.token_indices.id``."""
    return SimpleNamespace(
        component=SimpleNamespace(get_layer=lambda: layer),
        head=head,
        token_indices=SimpleNamespace(id=token_id),
    )


class ResidualStreamKeys(unittest.TestCase):
    def test_extracts_layer_and_position(self):
        u = _stub_residual_unit(layer=12, token_id="last_token")
        self.assertEqual(site_key_for_unit(u), (12, "last_token"))

    def test_returns_tuple_type(self):
        u = _stub_residual_unit(layer=0, token_id="x")
        k = site_key_for_unit(u)
        self.assertIsInstance(k, tuple)
        self.assertEqual(len(k), 2)


class AttentionHeadKeys(unittest.TestCase):
    def test_extracts_layer_head_and_position(self):
        u = _stub_attention_unit(layer=7, head=3, token_id="all")
        self.assertEqual(attention_head_site_key_for_unit(u), (7, 3, "all"))

    def test_head_is_coerced_to_int(self):
        u = _stub_attention_unit(layer=0, head=5, token_id="all")
        # Some pyvene versions return the head as a tensor scalar; the
        # helper should coerce to int.
        self.assertIsInstance(attention_head_site_key_for_unit(u)[1], int)

    def test_parses_token_id_from_uid_when_token_indices_missing(self):
        """Regression: upstream ``AttentionHead.__init__`` doesn't store
        ``token_indices`` as an attribute (only ``head`` + ``component``).
        The token-pos id lives in the unit's ``id`` string. Verify the
        helper parses it correctly."""
        u = SimpleNamespace(
            component=SimpleNamespace(get_layer=lambda: 8),
            head=1,
            id="AttentionHead(Layer-8,Head-1,Token-all)",
            # NOTE: deliberately NO ``token_indices`` attribute.
        )
        self.assertEqual(attention_head_site_key_for_unit(u), (8, 1, "all"))

    def test_token_id_with_underscore_or_colon(self):
        u = SimpleNamespace(
            component=SimpleNamespace(get_layer=lambda: 14),
            head=2,
            id="AttentionHead(Layer-14,Head-2,Token-last_token)",
        )
        self.assertEqual(attention_head_site_key_for_unit(u), (14, 2, "last_token"))


class ShapeDiscriminators(unittest.TestCase):
    def test_residual_key_recognised(self):
        self.assertTrue(is_residual_stream_key((10, "last_token")))
        self.assertFalse(is_attention_head_key((10, "last_token")))

    def test_attention_key_recognised(self):
        self.assertTrue(is_attention_head_key((10, 3, "all")))
        self.assertFalse(is_residual_stream_key((10, 3, "all")))

    def test_non_tuples_rejected(self):
        for x in [None, "abc", 42, [10, "foo"], {"layer": 10}]:
            self.assertFalse(is_residual_stream_key(x))  # type: ignore[arg-type]
            self.assertFalse(is_attention_head_key(x))   # type: ignore[arg-type]

    def test_wrong_length_tuples_rejected(self):
        # A four-tuple isn't either shape.
        self.assertFalse(is_residual_stream_key((1, 2, 3, 4)))   # type: ignore[arg-type]
        self.assertFalse(is_attention_head_key((1, 2, 3, 4)))    # type: ignore[arg-type]


class TypeAliasBackCompat(unittest.TestCase):
    """Pre-existing callers import ``SiteKey`` and use it for both runtime
    annotations and dict keys. The alias must keep the legacy 2-tuple shape."""

    def test_residual_alias_round_trips(self):
        from mib_submission.site_keys import SiteKey
        k: SiteKey = (1, "last")
        d: dict[SiteKey, int] = {k: 42}
        self.assertEqual(d[(1, "last")], 42)


if __name__ == "__main__":
    unittest.main()
