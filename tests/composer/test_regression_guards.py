"""Regression guard tests preserving behavioral invariants across composer refactoring."""
from __future__ import annotations

import pytest

from malthusjax.benchmarking.runner import StubEngine
from malthusjax.composer.backend_registry import get_backends
from malthusjax.composer.composer import Composer
from malthusjax.composer.strategies.core import EvoSAXStrategy


def test_stub_engine_fallback():
    """When no backend matches and no operators are provided, system falls back to StubEngine."""
    composer = Composer()
    result = composer.quick_run(seeds=(1,), generations=2)
    assert len(result.runs) == 1
    assert result.runs[0].status == "success"


def test_strategy_kwarg_bypasses_auto_creation():
    """Passing strategy= explicitly uses the provided strategy object."""
    strategy = EvoSAXStrategy(algorithm_name="LM_MA_ES")
    composer = Composer()
    result = composer.quick_run(
        strategy=strategy,
        pop_size=4,
        generations=1,
        seeds=(1,),
    )
    assert len(result.runs) == 1


def test_engine_kwarg_bypasses_all_dispatch():
    """Passing engine= explicitly skips both cascades."""
    engine = StubEngine(generations=2)
    composer = Composer()
    result = composer.quick_run(engine=engine, seeds=(1,))
    assert len(result.runs) == 1


def test_maximize_sign_convention_preserved():
    """maximize=True must produce positive best_fitness values in summary for Sphere."""
    composer = Composer()
    result = composer.quick_run(
        fitness="sphere:dim=2",
        maximize=True,
        pop_size=4,
        generations=3,
        seeds=(1,),
    )
    assert result.runs[0].metrics["best_fitness"] <= 0
    assert result.runs[0].status == "success"


def test_evosax_default_strategy_name():
    """EvosaxProvider.default_strategy() defaults to 'SimpleGA'."""
    provider = get_backends()["evosax"][0]
    strategy = provider.default_strategy()
    assert strategy.algorithm_name == "SimpleGA"
