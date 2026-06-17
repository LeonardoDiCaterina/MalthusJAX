"""
BENCHMARK GROUP 12 — Adapter Overhead
=====================================

Measures pure execution wall-clock time overhead of the `EvosaxEngineAdapter`
facade compared to a raw `evosax` scan loop.
"""

import jax
import jax.random as jr
import pytest

from malthusjax.composer.evosax_adapter import build_evosax_engine
from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator
from tests.benchmarks.conftest_benchmarks import (
    DIMENSIONS,
    NUM_GENERATIONS_LONG,
    SEED,
    EvosaxBenchEngine,
)


class TestAdapterOverhead:
    """Proves the adapter facade adds zero statistical execution overhead.
    
    Both the adapter and the raw evosax implementation lower the ask/eval/tell
    loop into a single JAX scan block. The adapter should merely be a Python
    configuration overlay that disappears entirely inside the XLA compiled graph.
    """

    @pytest.mark.parametrize("pop_size", [100, 500])
    @pytest.mark.parametrize("dims", DIMENSIONS)
    def test_adapter_overhead_execution(self, benchmark, pop_size: int, dims: int):
        """Adapter: pure execution time after warmup."""
        num_gens = NUM_GENERATIONS_LONG

        evalr = BBOBEvaluator.create(
            BBOBConfig(fn_name="sphere", num_dims=dims, seed=SEED, maximize=False)
        )
        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            evaluator=evalr,
            pop_size=pop_size,
            generations=num_gens,
            bounds=(-5.0, 5.0),
            maximize=False,
        )

        # Warmup and caching
        _ = adapter.run_once(jr.PRNGKey(0))

        def _run():
            result = adapter.run_once(jr.PRNGKey(SEED))
            assert result["summary"]["best_fitness"] is not None

        benchmark.group = f"adapter_overhead/pop{pop_size}_d{dims}"
        benchmark.name = "adapter_facade"
        benchmark.pedantic(_run, rounds=20, warmup_rounds=2)

    @pytest.mark.parametrize("pop_size", [100, 500])
    @pytest.mark.parametrize("dims", DIMENSIONS)
    def test_raw_evosax_execution(self, benchmark, pop_size: int, dims: int):
        """Raw Evosax: pure execution time after warmup."""
        num_gens = NUM_GENERATIONS_LONG

        raw_engine = EvosaxBenchEngine(
            pop_size=pop_size,
            dims=dims,
            problem="sphere",
            num_generations=num_gens,
        )

        # Warmup
        _ = raw_engine.run_once(jr.PRNGKey(0))

        def _run():
            result = raw_engine.run_once(jr.PRNGKey(SEED))
            assert result["summary"]["best_fitness"] is not None

        benchmark.group = f"adapter_overhead/pop{pop_size}_d{dims}"
        benchmark.name = "raw_evosax"
        benchmark.pedantic(_run, rounds=20, warmup_rounds=2)
