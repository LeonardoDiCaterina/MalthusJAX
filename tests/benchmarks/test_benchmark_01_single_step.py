"""
BENCHMARK GROUP 1 — Single-Step Latency (warm dispatch)
=======================================================

Warm single-step latency for both frameworks at different scales.
"""

import jax
import pytest

from tests.benchmarks.conftest_benchmarks import (
    DIMENSIONS,
    POP_SIZES,
    _build_malthusjax_engine,
    _evosax_init_and_warmup,
    _malthusjax_init_and_warmup,
)


class TestSingleStepLatency:
    """Warm single-step latency for both frameworks at different scales."""

    # --- MalthusJAX ---

    @pytest.mark.parametrize("pop_size", POP_SIZES)
    @pytest.mark.parametrize("dims", DIMENSIONS)
    @pytest.mark.parametrize("use_evosax_ops", [False, True])
    def test_malthusjax_step(self, benchmark, pop_size: int, dims: int, use_evosax_ops: bool):
        """MalthusJAX: single jit-compiled step (warm)."""
        engine = _build_malthusjax_engine(pop_size, dims, use_evosax_ops=use_evosax_ops)
        state, jit_step = _malthusjax_init_and_warmup(engine)

        def _run():
            s, _ = jit_step(state)
            s.best_fitness.block_until_ready()

        benchmark.group = f"single_step/pop{pop_size}_d{dims}"
        benchmark.name = "malthusjax_evosaxops" if use_evosax_ops else "malthusjax"
        benchmark.pedantic(_run, iterations=1, rounds=5, warmup_rounds=2)

    @pytest.mark.parametrize("pop_size", POP_SIZES)
    @pytest.mark.parametrize("dims", DIMENSIONS)
    def test_malthusjax_step_roulette(self, benchmark, pop_size: int, dims: int):
        """MalthusJAX: single jit-compiled step using roulette selection and Evosax ops."""
        engine = _build_malthusjax_engine(
            pop_size,
            dims,
            selection_type="roulette",
            crossover_type="uniform",
            mutation_type="gaussian",
            use_evosax_ops=True,
        )
        state, jit_step = _malthusjax_init_and_warmup(engine)

        def _run():
            s, _ = jit_step(state)
            s.best_fitness.block_until_ready()

        benchmark.group = f"single_step/pop{pop_size}_d{dims}"
        benchmark.name = "malthusjax_roulette_evosaxops"
        benchmark.pedantic(_run, iterations=1, rounds=5, warmup_rounds=2)

    @pytest.mark.parametrize("pop_size", POP_SIZES)
    @pytest.mark.parametrize("dims", DIMENSIONS)
    def test_malthusjax_step_tournament(self, benchmark, pop_size: int, dims: int):
        """MalthusJAX: single jit-compiled step using tournament selection and Evosax ops."""
        engine = _build_malthusjax_engine(
            pop_size,
            dims,
            selection_type="tournament",
            crossover_type="uniform",
            mutation_type="gaussian",
            use_evosax_ops=True,
        )
        state, jit_step = _malthusjax_init_and_warmup(engine)

        def _run():
            s, _ = jit_step(state)
            s.best_fitness.block_until_ready()

        benchmark.group = f"single_step/pop{pop_size}_d{dims}"
        benchmark.name = "malthusjax_tournament_evosaxops"
        benchmark.pedantic(_run, iterations=1, rounds=5, warmup_rounds=2)

    # --- Evosax ---

    @pytest.mark.parametrize("pop_size", POP_SIZES)
    @pytest.mark.parametrize("dims", DIMENSIONS)
    def test_evosax_step(self, benchmark, pop_size: int, dims: int):
        """Evosax SimpleGA: single jit-compiled step (warm)."""
        carry, jit_step = _evosax_init_and_warmup(pop_size, dims)

        def _run():
            c, _ = jit_step(carry)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), c)

        benchmark.group = f"single_step/pop{pop_size}_d{dims}"
        benchmark.name = "evosax"
        benchmark.pedantic(_run, iterations=1, rounds=5, warmup_rounds=2)
