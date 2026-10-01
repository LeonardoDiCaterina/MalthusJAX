"""Tests verifying composable backend consolidation via provider flags."""

from __future__ import annotations

import jax.random as jr

from malthusjax.composer.backend_registry import get_backends


def test_composable_evosax_flag():
    """provider.build_engine(composable=True) produces a composable evosax adapter."""
    provider = get_backends()["evosax"][0]
    strategy = provider.default_strategy(evosax_strategy="SimpleGA")
    evaluator = provider.resolve_evaluator("sphere:dim=2")

    engine_std = provider.build_engine(
        strategy, evaluator, pop_size=4, generations=1, num_dims=2, composable=False
    )
    engine_comp = provider.build_engine(
        strategy, evaluator, pop_size=4, generations=1, num_dims=2, composable=True
    )

    assert hasattr(engine_std, "run_once")
    assert hasattr(engine_comp, "run_once")

    res_std = engine_std.run_once(jr.PRNGKey(42))
    res_comp = engine_comp.run_once(jr.PRNGKey(42))

    assert "history" in res_std
    assert "history" in res_comp


def test_composable_tensorneat_flag():
    """provider.build_engine(composable=True) produces a composable tensorneat adapter."""
    provider = get_backends()["tensorneat"][0]
    strategy = provider.default_strategy()
    evaluator = provider.resolve_evaluator("xor", composable=True)

    engine_std = provider.build_engine(
        strategy, evaluator, pop_size=4, generations=1, composable=False
    )
    engine_comp = provider.build_engine(
        strategy, evaluator, pop_size=4, generations=1, composable=True
    )

    assert hasattr(engine_std, "run_once")
    assert hasattr(engine_comp, "run_once")
