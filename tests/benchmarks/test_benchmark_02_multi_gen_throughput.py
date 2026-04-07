"""
BENCHMARK GROUP 2 — Multi-Generation Throughput (scan loop)
===========================================================

Full evolution loop: N generations via jax.lax.scan.
"""

import jax.random as jr
import pytest

from tests.benchmarks.conftest_benchmarks import (
    DIMENSIONS,
    SEED,
    EvosaxBenchEngine,
    MalthusJAXBenchEngine,
)

def pytest_generate_tests(metafunc):
    """Dynamically parameterize pop_size and num_gens from the CLI options."""
    if "pop_size" in metafunc.fixturenames:
        pop_sizes_str = metafunc.config.getoption("--pop-sizes")
        pop_sizes = [int(p.strip()) for p in pop_sizes_str.split(",")]
        metafunc.parametrize("pop_size", pop_sizes)
        
    if "num_gens" in metafunc.fixturenames:
        num_gens_str = metafunc.config.getoption("--num-gens")
        num_gens = [int(n.strip()) for n in num_gens_str.split(",")]
        metafunc.parametrize("num_gens", num_gens)


class TestMultiGenThroughput:
    """Full evolution loop: N generations via jax.lax.scan.

    Uses the ``Engine``-protocol adapters
    (:class:`MalthusJAXBenchEngine` / :class:`EvosaxBenchEngine`) so the
    benchmarked code path is identical to what :class:`BenchmarkRunner`
    executes.  pytest-benchmark still handles the wall-clock timing.
    """

    @pytest.mark.parametrize("dims", DIMENSIONS)
    @pytest.mark.parametrize("use_evosax_ops", [False, True])
    def test_malthusjax_scan(self, benchmark, pop_size: int, num_gens: int, dims: int, use_evosax_ops: bool):
        """MalthusJAX: full scan loop for {num_gens} gens."""
        bench_engine = MalthusJAXBenchEngine(
            pop_size=pop_size,
            dims=dims,
            num_generations=num_gens,
            use_evosax_ops=use_evosax_ops,
        )
        # Warm-up (compile)
        bench_engine.run_once(jr.PRNGKey(0))

        def _run():
            result = bench_engine.run_once(jr.PRNGKey(SEED))
            assert result["summary"]["best_fitness"] is not None

        benchmark.group = f"scan_{num_gens}gen/pop{pop_size}_d{dims}"
        benchmark.name = "malthusjax_evosaxops" if use_evosax_ops else "malthusjax"
        benchmark.pedantic(_run, iterations=1, rounds=100, warmup_rounds=2)

    @pytest.mark.parametrize("dims", DIMENSIONS)
    def test_malthusjax_scan_roulette(self, benchmark, pop_size: int, num_gens: int, dims: int):
        """MalthusJAX: full scan loop using roulette selection and Evosax ops."""
        bench_engine = MalthusJAXBenchEngine(
            pop_size=pop_size,
            dims=dims,
            num_generations=num_gens,
            selection_type="roulette",
            crossover_type="uniform",
            mutation_type="gaussian",
            use_evosax_ops=True,
        )
        bench_engine.run_once(jr.PRNGKey(0))

        def _run():
            result = bench_engine.run_once(jr.PRNGKey(SEED))
            assert result["summary"]["best_fitness"] is not None

        benchmark.group = f"scan_{num_gens}gen/pop{pop_size}_d{dims}"
        benchmark.name = "malthusjax_roulette_evosaxops"
        benchmark.pedantic(_run, iterations=1, rounds=100, warmup_rounds=2)

    @pytest.mark.parametrize("dims", DIMENSIONS)
    def test_malthusjax_scan_tournament(self, benchmark, pop_size: int, num_gens: int, dims: int):
        """MalthusJAX: full scan loop using tournament selection and Evosax ops."""
        bench_engine = MalthusJAXBenchEngine(
            pop_size=pop_size,
            dims=dims,
            num_generations=num_gens,
            selection_type="tournament",
            crossover_type="uniform",
            mutation_type="gaussian",
            use_evosax_ops=True,
        )
        bench_engine.run_once(jr.PRNGKey(0))

        def _run():
            result = bench_engine.run_once(jr.PRNGKey(SEED))
            assert result["summary"]["best_fitness"] is not None

        benchmark.group = f"scan_{num_gens}gen/pop{pop_size}_d{dims}"
        benchmark.name = "malthusjax_tournament_evosaxops"
        benchmark.pedantic(_run, iterations=1, rounds=100, warmup_rounds=2)

    @pytest.mark.parametrize("dims", DIMENSIONS)
    def test_evosax_scan(self, benchmark, pop_size: int, num_gens: int, dims: int):
        """Evosax SimpleGA: full scan loop for {num_gens} gens."""
        bench_engine = EvosaxBenchEngine(
            pop_size=pop_size,
            dims=dims,
            num_generations=num_gens,
        )
        # Warm-up (compile)
        bench_engine.run_once(jr.PRNGKey(0))

        def _run():
            result = bench_engine.run_once(jr.PRNGKey(SEED))
            assert result["summary"]["best_fitness"] is not None

        benchmark.group = f"scan_{num_gens}gen/pop{pop_size}_d{dims}"
        benchmark.name = "evosax"
        benchmark.pedantic(_run, iterations=1, rounds=100, warmup_rounds=2)
