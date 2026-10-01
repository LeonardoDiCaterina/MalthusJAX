"""Tests for BackendRegistry infrastructure."""

from __future__ import annotations

import pytest

from malthusjax.composer.backend_registry import (
    _BACKEND_REGISTRY,
    get_backends,
    list_backends,
    register_backend,
)
from malthusjax.composer.strategies.base import BaseStrategy


class DummyProvider:
    @property
    def name(self) -> str:
        return "dummy"

    def default_strategy(self, **kwargs):
        return BaseStrategy()

    def resolve_evaluator(self, fitness_spec, **kwargs):
        return None

    def build_engine(self, strategy, evaluator, **kwargs):
        return None

    def generate_initial_population(self, config, pop_seed):
        return None


def test_register_and_retrieve():
    """Basic register + get round-trip."""
    dummy = DummyProvider()
    register_backend("_test_roundtrip", dummy, override=True)
    try:
        backends = get_backends()
        assert "_test_roundtrip" in backends
        assert backends["_test_roundtrip"][0] is dummy
    finally:
        _BACKEND_REGISTRY.pop("_test_roundtrip", None)


def test_register_duplicate_raises():
    """Duplicate registration without override must raise KeyError."""
    dummy = DummyProvider()
    register_backend("_test_dup", dummy, override=True)
    try:
        with pytest.raises(KeyError, match="already registered"):
            register_backend("_test_dup", dummy, override=False)
    finally:
        _BACKEND_REGISTRY.pop("_test_dup", None)


def test_register_with_override():
    """Duplicate registration with override=True replaces the provider."""
    d1 = DummyProvider()
    d2 = DummyProvider()
    register_backend("_test_override", d1, override=True)
    try:
        register_backend("_test_override", d2, override=True)
        assert get_backends()["_test_override"][0] is d2
    finally:
        _BACKEND_REGISTRY.pop("_test_override", None)


def test_register_with_defaults():
    """Provider registration supports attaching default parameters."""
    dummy = DummyProvider()
    register_backend("_test_defaults", dummy, defaults={"composable": True}, override=True)
    try:
        backends = get_backends()
        assert backends["_test_defaults"][1] == {"composable": True}
    finally:
        _BACKEND_REGISTRY.pop("_test_defaults", None)


def test_list_backends_sorted():
    """list_backends() returns sorted list of registered backend names."""
    names = list_backends()
    assert names == sorted(names)
    assert "evosax" in names
    assert "malthusjax" in names


def test_get_registry_returns_copy():
    """get_backends() returns a shallow copy to prevent accidental mutations."""
    backends = get_backends()
    backends["_malicious_mutation"] = (DummyProvider(), {}, {})
    assert "_malicious_mutation" not in get_backends()


def test_aliases_route_to_same_provider():
    """Variant keys (composable_evosax) share the underlying provider object."""
    backends = get_backends()
    assert "evosax" in backends
    assert "composable_evosax" in backends
    assert backends["evosax"][0] is backends["composable_evosax"][0]
    assert backends["composable_evosax"][1] == {"composable": True}
