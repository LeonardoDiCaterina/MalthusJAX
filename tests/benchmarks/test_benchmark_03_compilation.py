"""
BENCHMARK GROUP 3 — JIT Compilation Time
========================================

Measures cold JIT compilation time (first call overhead).
"""

import jax
import jax.random as jr
import pytest

from tests.benchmarks.conftest_benchmarks import (
    DIMENSIONS,
    POP_SIZES,
    SEED,
    _build_evosax_ga,
    _build_malthusjax_engine,
    _evosax_step_fn,
)


class TestCompilationTime:
    """Measures cold JIT compilation time (first call overhead)."""

    @pytest.mark.parametrize("pop_size", POP_SIZES)
    @pytest.mark.parametrize("dims", DIMENSIONS)
    def test_malthusjax_compile(self, benchmark, pop_size: int, dims: int):
        """MalthusJAX: time to JIT-compile the step function (cold)."""

        def _compile():
            # Clear JAX caches by creating a fresh engine each time
            engine = _build_malthusjax_engine(pop_size, dims)
            key = jr.PRNGKey(SEED)
            state = engine.init_state(key)
            jit_step = jax.jit(engine.step)
            s, _ = jit_step(state)
            s.best_fitness.block_until_ready()

        benchmark.group = f"compile/pop{pop_size}_d{dims}"
        benchmark.name = "malthusjax"
        # Use 1 round for compilation benchmarks — they're expensive
        benchmark.pedantic(_compile, rounds=3, warmup_rounds=0)

    @pytest.mark.parametrize("pop_size", POP_SIZES)
    @pytest.mark.parametrize("dims", DIMENSIONS)
    def test_evosax_compile(self, benchmark, pop_size: int, dims: int):
        """Evosax SimpleGA: time to JIT-compile the step function (cold)."""

        def _compile():
            strategy, params, es_problem, carry = _build_evosax_ga(pop_size, dims)
            step = _evosax_step_fn(strategy, params, es_problem)
            jit_step = jax.jit(step)
            c, _ = jit_step(carry)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), c)

        benchmark.group = f"compile/pop{pop_size}_d{dims}"
        benchmark.name = "evosax"
        benchmark.pedantic(_compile, rounds=3, warmup_rounds=0)
