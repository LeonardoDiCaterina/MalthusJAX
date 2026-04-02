"""
Performance benchmarks for the genetic engine.

These tests measure throughput and execution speed. They use pytest's
benchmark plugin and are marked with @pytest.mark.performance to allow
exclusion from standard unit test runs (they may fail on noisy CI/CD runners).

Run with:
    pytest benchmarks/performance_tests.py -m performance
    pytest benchmarks/performance_tests.py -m performance -v --benchmark-only
"""

import time

import jax.random as jar
import pytest

from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.operators.crossover.real import SimulatedBinaryCrossover
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.operators.selection.elite_pool import ElitePoolSelection


@pytest.fixture
def performance_engine():
    """Fully configured engine for performance benchmarking."""
    key = jar.PRNGKey(42)
    pop_size = 100
    genome_shape = (10,)
    bounds = (-5.0, 5.0)

    genome_config = RealGenomeConfig(shape=genome_shape, bounds=bounds)

    engine_params = GeneticEngineParams(
        pop_size=pop_size,
        elitism=2,
        num_generations=500,
    )

    bbob_config = BBOBConfig(fn_name="sphere", num_dims=genome_shape[0], maximize=False)
    evaluator = BBOBEvaluator.create(bbob_config)

    selection = ElitePoolSelection(num_selections=pop_size, elite_k=10)
    crossover = SimulatedBinaryCrossover(num_offspring=2, eta=15.0)
    mutation = GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5)

    engine = GeneticEngine(
        engine_params=engine_params,
        genome_config=genome_config,
        evaluator=evaluator,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        enable_progress_bar=False,
    )

    return engine, key


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmarks for the genetic engine.

    These tests measure throughput and are marked with @pytest.mark.performance
    to allow exclusion from standard CI/CD runs via:
        pytest -m "not performance"
    """

    def test_throughput_benchmark(self, performance_engine):
        """
        Measure engine throughput in generations per second.

        This test performs a longer run (500 generations) to measure
        sustained performance. It includes a warmup run for JIT compilation
        and then measures the actual execution speed.

        Expected behavior:
            - GPU: >1000 gens/sec
            - CPU: >100 gens/sec

        This test is marked @pytest.mark.performance because it:
        - Requires stable execution environment (may be noisy on shared CI/CD)
        - Takes longer than typical unit tests
        - Should not block PR merges if occasionally slow
        """
        engine, key = performance_engine

        state = engine.init_state(key)

        # Warmup (Compile)
        print("  Compiling...", end="", flush=True)
        t0 = time.time()
        final_state, _, _ = engine.run(state, compile=True)
        _ = final_state.best_fitness.block_until_ready()
        print(f" Done ({time.time() - t0:.2f}s)")

        # Real Run
        state = engine.init_state(key)

        t0 = time.time()
        final_state, _, _ = engine.run(state, compile=True)
        _ = final_state.best_fitness.block_until_ready()
        duration = time.time() - t0

        NUM_GENS = engine.engine_params.num_generations
        gens_per_sec = NUM_GENS / duration
        print(f"  Speed: {gens_per_sec:,.2f} gens/sec ({NUM_GENS} gens in {duration:.2f}s)")

        # Conservative threshold for CI/CD compatibility
        # (tune based on your target hardware)
        assert gens_per_sec > 50, (
            f"Engine throughput degraded: {gens_per_sec:.2f} gens/sec "
            f"(threshold: >50). Check for Python fallback or optimization issues."
        )
