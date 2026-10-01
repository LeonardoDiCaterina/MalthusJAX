"""Tests for BackendProvider protocol compliance across all registered providers."""

from __future__ import annotations

import pytest

from malthusjax.composer.backend_provider import BackendProvider
from malthusjax.composer.backend_registry import get_backends
from malthusjax.composer.strategies.base import BaseStrategy


@pytest.fixture(
    params=[
        "malthusjax",
        "evosax",
        "composable_evosax",
        "qdax",
        "tensorneat",
        "composable_tensorneat",
        "map_elites",
        "stub",
    ]
)
def provider_entry(request):
    """Parametrize over all registered backend providers."""
    backends = get_backends()
    name = request.param
    if name not in backends:
        pytest.skip(f"Backend '{name}' not registered")
    provider, defaults, metadata = backends[name]
    return provider, defaults, metadata


def test_provider_satisfies_protocol(provider_entry):
    """Every provider must satisfy the runtime-checkable BackendProvider protocol."""
    provider, _, _ = provider_entry
    assert isinstance(provider, BackendProvider)


def test_provider_has_name(provider_entry):
    """Every provider must have a non-empty name property."""
    provider, _, _ = provider_entry
    assert hasattr(provider, "name")
    assert isinstance(provider.name, str)
    assert len(provider.name) > 0


def test_provider_has_build_engine(provider_entry):
    """Every provider must implement a callable build_engine()."""
    provider, _, _ = provider_entry
    assert hasattr(provider, "build_engine")
    assert callable(provider.build_engine)


def test_provider_has_default_strategy(provider_entry):
    """Every provider must implement a callable default_strategy()."""
    provider, _, _ = provider_entry
    assert hasattr(provider, "default_strategy")
    assert callable(provider.default_strategy)


def test_provider_has_resolve_evaluator(provider_entry):
    """Every provider must implement a callable resolve_evaluator()."""
    provider, _, _ = provider_entry
    assert hasattr(provider, "resolve_evaluator")
    assert callable(provider.resolve_evaluator)


def test_default_strategy_returns_base_strategy(provider_entry):
    """default_strategy() must return a BaseStrategy instance."""
    provider, defaults, _ = provider_entry
    strategy = provider.default_strategy(**defaults)
    assert isinstance(strategy, BaseStrategy)
