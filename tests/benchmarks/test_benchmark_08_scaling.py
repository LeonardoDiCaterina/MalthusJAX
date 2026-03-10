"""
BENCHMARK GROUP 8 — Scaling Sweep
=================================

Measure how throughput scales with population size.
"""

import jax
import pytest

from tests.benchmarks.conftest_benchmarks import (
    _build_malthusjax_engine,
    _evosax_init_and_warmup,
    _malthusjax_init_and_warmup,
    size_sweep_pop_sizes,
)


def test_malthusjax_scaling(benchmark, pop_size: int):
    dims = 10
    engine = _build_malthusjax_engine(pop_size, dims)
    state, jit_step = _malthusjax_init_and_warmup(engine)

    def _run():
        s, _ = jit_step(state)
        s.best_fitness.block_until_ready()

    benchmark.group = "scaling_d10"
    benchmark.name = f"malthusjax_pop{pop_size}"
    benchmark(_run)


def test_evosax_scaling(benchmark, pop_size: int):
    dims = 10
    carry, jit_step = _evosax_init_and_warmup(pop_size, dims)

    def _run():
        c, _ = jit_step(carry)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), c)

    benchmark.group = "scaling_d10"
    benchmark.name = f"evosax_pop{pop_size}"
    benchmark(_run)


@pytest.mark.parametrize("pop_size", size_sweep_pop_sizes)
def test_malthusjax_scaling_param(benchmark, pop_size: int):
    test_malthusjax_scaling(benchmark, pop_size)


@pytest.mark.parametrize("pop_size", size_sweep_pop_sizes)
def test_evosax_scaling_param(benchmark, pop_size: int):
    test_evosax_scaling(benchmark, pop_size)
