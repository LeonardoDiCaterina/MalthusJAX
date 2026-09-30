"""Tests verifying third-party extensibility and error handling in backend registration."""
from __future__ import annotations

import pytest

from malthusjax.benchmarking.runner import StubEngine
from malthusjax.composer.backend_registry import _BACKEND_REGISTRY, register_backend
from malthusjax.composer.catalog import OperatorCatalog
from malthusjax.composer.composer import Composer
from malthusjax.composer.strategies.base import BaseStrategy


class CustomPluginProvider:
    @property
    def name(self) -> str:
        return "custom_plugin"

    def default_strategy(self, **kwargs):
        return BaseStrategy()

    def resolve_evaluator(self, fitness_spec, **kwargs):
        return OperatorCatalog().get("sphere:dim=2")

    def build_engine(self, strategy, evaluator, **kwargs):
        return StubEngine(generations=kwargs.get("generations", 1))

    def generate_initial_population(self, config, pop_seed):
        return None


def test_external_plugin_registers_backend():
    """A third-party module can register a new backend provider and quick_run will use it."""
    register_backend("custom_plugin", CustomPluginProvider(), override=True)
    try:
        composer = Composer()
        result = composer.quick_run(
            backend="custom_plugin",
            pop_size=4,
            generations=1,
            seeds=(1,),
        )
        assert len(result.runs) == 1
        assert result.runs[0].status == "success"
    finally:
        _BACKEND_REGISTRY.pop("custom_plugin", None)


def test_unknown_backend_raises_helpful_error():
    """Requesting an unknown backend raises ValueError with list of available backends."""
    composer = Composer()
    with pytest.raises(ValueError, match="Unknown backend 'nonexistent_xyz'.*Available:"):
        composer.quick_run(backend="nonexistent_xyz")
