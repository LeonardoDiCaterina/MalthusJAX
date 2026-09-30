"""Tests for BackendProvider registration-time validation."""

import pytest

from malthusjax.composer.backend_registry import (
    _BACKEND_REGISTRY,
    get_backends,
    register_backend,
)
from malthusjax.composer.strategies.base import BaseStrategy


class CompleteProvider:
    @property
    def name(self) -> str:
        return "valid_provider"

    def default_strategy(self, **kwargs):
        return BaseStrategy()

    def resolve_evaluator(self, fitness_spec, **kwargs):
        return None

    def build_engine(self, strategy, evaluator, **kwargs):
        return None

    def generate_initial_population(self, config, pop_seed):
        return None


class MissingBuildEngine:
    @property
    def name(self) -> str:
        return "no_build_engine"

    def default_strategy(self, **kwargs):
        return BaseStrategy()

    def resolve_evaluator(self, fitness_spec, **kwargs):
        return None

    def generate_initial_population(self, config, pop_seed):
        return None


class MissingResolveEvaluator:
    @property
    def name(self) -> str:
        return "no_resolve_evaluator"

    def default_strategy(self, **kwargs):
        return BaseStrategy()

    def build_engine(self, strategy, evaluator, **kwargs):
        return None

    def generate_initial_population(self, config, pop_seed):
        return None


class NonCallableBuildEngine:
    @property
    def name(self) -> str:
        return "non_callable"

    build_engine = "not_callable"

    def default_strategy(self, **kwargs):
        return BaseStrategy()

    def resolve_evaluator(self, fitness_spec, **kwargs):
        return None

    def generate_initial_population(self, config, pop_seed):
        return None


def test_valid_provider_registers():
    """Valid provider passes registration check and is registered."""
    provider = CompleteProvider()
    register_backend("_test_valid", provider, override=True)
    try:
        assert "_test_valid" in get_backends()
    finally:
        _BACKEND_REGISTRY.pop("_test_valid", None)


def test_missing_build_engine_raises():
    """Provider missing build_engine raises TypeError."""
    provider = MissingBuildEngine()
    with pytest.raises(TypeError, match="Missing or non-callable methods:.*build_engine"):
        register_backend("_test_missing_build", provider, override=True)


def test_missing_resolve_evaluator_raises():
    """Provider missing resolve_evaluator raises TypeError."""
    provider = MissingResolveEvaluator()
    with pytest.raises(TypeError, match="Missing or non-callable methods:.*resolve_evaluator"):
        register_backend("_test_missing_resolve", provider, override=True)


def test_non_callable_method_raises():
    """Provider with non-callable method raises TypeError."""
    provider = NonCallableBuildEngine()
    with pytest.raises(TypeError, match="Missing or non-callable methods"):
        register_backend("_test_non_callable", provider, override=True)
