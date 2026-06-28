"""
BENCHMARK GROUP 6 — Unroll Factor Sweep
=======================================

Measure how lax.scan unroll_num affects per-generation throughput.
"""

import jax.random as jr
import pytest

from tests.benchmarks.conftest_benchmarks import (
    SEED,
    UNROLL_FACTORS,
    MalthusJAXBenchEngine,
)


class TestUnrollSweep:
    """Measure how lax.scan unroll_num affects per-generation throughput.

    Higher unroll values allow XLA to fuse more steps into one HLO program,
    reducing dispatch overhead at the cost of compile time and peak memory.
    Run at fixed pop=100, d=10 to isolate the unroll effect cleanly.
    """

    _POP = 100
    _DIMS = 10
    _GENS = 50

    @pytest.mark.parametrize("unroll", UNROLL_FACTORS)
    def test_malthusjax_unroll_scan(self, benchmark, unroll: int):
        """MalthusJAX: {_GENS}-gen scan at unroll={unroll}."""
        bench_engine = MalthusJAXBenchEngine(
            pop_size=self._POP,
            dims=self._DIMS,
            num_generations=self._GENS,
            unroll_num=unroll,
        )
        bench_engine.run_once(jr.PRNGKey(0))

        def _run():
            result = bench_engine.run_once(jr.PRNGKey(SEED))
            assert result["summary"]["best_fitness"] is not None

        benchmark.group = f"unroll_scan_{self._GENS}gen/pop{self._POP}_d{self._DIMS}"
        benchmark.name = f"unroll_{unroll}"
        benchmark.pedantic(_run, iterations=1, rounds=5, warmup_rounds=2)
