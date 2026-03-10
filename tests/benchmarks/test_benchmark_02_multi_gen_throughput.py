"""
BENCHMARK GROUP 2 — Multi-Generation Throughput (scan loop)
===========================================================

Full evolution loop: N generations via jax.lax.scan.
"""

import jax.random as jr
import pytest

from tests.benchmarks.conftest_benchmarks import (
    DIMENSIONS,
    NUM_GENERATIONS_SHORT,
    POP_SIZES,
    SEED,
    EvosaxBenchEngine,
    MalthusJAXBenchEngine,
)


class TestMultiGenThroughput:
    """Full evolution loop: N generations via jax.lax.scan.

    Uses the ``Engine``-protocol adapters
    (:class:`MalthusJAXBenchEngine` / :class:`EvosaxBenchEngine`) so the
    benchmarked code path is identical to what :class:`BenchmarkRunner`
    executes.  pytest-benchmark still handles the wall-clock timing.
    """

    @pytest.mark.parametrize("pop_size", POP_SIZES)
    @pytest.mark.parametrize("dims", DIMENSIONS)
    @pytest.mark.parametrize("use_evosax_ops", [False, True])
    def test_malthusjax_scan(self, benchmark, pop_size: int, dims: int, use_evosax_ops: bool):
        """MalthusJAX: full scan loop for {NUM_GENERATIONS_SHORT} gens."""
        bench_engine = MalthusJAXBenchEngine(
            pop_size=pop_size,
            dims=dims,
            num_generations=NUM_GENERATIONS_SHORT,
            use_evosax_ops=use_evosax_ops,
        )
        # Warm-up (compile)
        bench_engine.run_once(jr.PRNGKey(0))

        def _run():
            result = bench_engine.run_once(jr.PRNGKey(SEED))
            assert result["summary"]["best_fitness"] is not None

        benchmark.group = f"scan_{NUM_GENERATIONS_SHORT}gen/pop{pop_size}_d{dims}"
        benchmark.name = "malthusjax_evosaxops" if use_evosax_ops else "malthusjax"
        benchmark(_run)

    @pytest.mark.parametrize("pop_size", POP_SIZES)
    @pytest.mark.parametrize("dims", DIMENSIONS)
    def test_evosax_scan(self, benchmark, pop_size: int, dims: int):
        """Evosax SimpleGA: full scan loop for {NUM_GENERATIONS_SHORT} gens."""
        bench_engine = EvosaxBenchEngine(
            pop_size=pop_size,
            dims=dims,
            num_generations=NUM_GENERATIONS_SHORT,
        )
        # Warm-up (compile)
        bench_engine.run_once(jr.PRNGKey(0))

        def _run():
            result = bench_engine.run_once(jr.PRNGKey(SEED))
            assert result["summary"]["best_fitness"] is not None

        benchmark.group = f"scan_{NUM_GENERATIONS_SHORT}gen/pop{pop_size}_d{dims}"
        benchmark.name = "evosax"
        benchmark(_run)
