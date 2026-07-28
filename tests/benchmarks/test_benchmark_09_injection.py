"""
BENCHMARK GROUP 9 — Injection Operator Performance
===================================================

Compare injection-mode operators against their standard counterparts.
"""

import jax.random as jr
import pytest

from tests.benchmarks.conftest_benchmarks import (
    _INJECTION_CROSSOVER_TYPES,
    _INJECTION_MUTATION_TYPES,
    NUM_GENERATIONS_SHORT,
    SEED,
    MalthusJAXBenchEngine,
    _malthusjax_init_and_warmup,
)


def _run_bench_engine(loop_engine, seed=SEED):
    result = loop_engine.run_once(jr.PRNGKey(seed))
    assert result["summary"]["best_fitness"] is not None


class TestInjectionOperators:
    """Compare injection-mode operators against their standard counterparts.

    Fixed configuration (pop=100, d=10) isolates operator cost from scaling
    effects.  Each crossover and mutation type is benchmarked in both standard
    mode (per-pair key allocation) and injection mode (single key, full
    materialisation), with the same selection operator and fitness function.

    These benchmarks quantify the materialisation overhead of injection mode
    and serve as regression baselines for its cache/memory trade-offs.
    """

    _POP = 100
    _DIMS = 10

    @pytest.mark.parametrize("crossover_type", _INJECTION_CROSSOVER_TYPES)
    @pytest.mark.parametrize("use_injection", [False, True])
    def test_crossover_step(self, benchmark, crossover_type: str, use_injection: bool):
        engine = MalthusJAXBenchEngine(
            pop_size=self._POP,
            dims=self._DIMS,
            crossover_type=crossover_type,
            use_injection_ops=use_injection,
        )
        state, jit_step = _malthusjax_init_and_warmup(engine)

        def _run():
            s, _ = jit_step(state)
            s.best_fitness.block_until_ready()

        mode = "injection" if use_injection else "standard"
        benchmark.group = f"injection_crossover/pop{self._POP}_d{self._DIMS}"
        benchmark.name = f"{crossover_type}_{mode}"
        benchmark.pedantic(_run, iterations=1, rounds=5, warmup_rounds=2)

    @pytest.mark.parametrize("mutation_type", _INJECTION_MUTATION_TYPES)
    @pytest.mark.parametrize("use_injection", [False, True])
    def test_mutation_step(self, benchmark, mutation_type: str, use_injection: bool):
        engine = MalthusJAXBenchEngine(
            pop_size=self._POP,
            dims=self._DIMS,
            mutation_type=mutation_type,
            use_injection_ops=use_injection,
        )
        state, jit_step = _malthusjax_init_and_warmup(engine)

        def _run():
            s, _ = jit_step(state)
            s.best_fitness.block_until_ready()

        mode = "injection" if use_injection else "standard"
        benchmark.group = f"injection_mutation/pop{self._POP}_d{self._DIMS}"
        benchmark.name = f"{mutation_type}_{mode}"
        benchmark.pedantic(_run, iterations=1, rounds=5, warmup_rounds=2)

    @pytest.mark.parametrize("crossover_type", _INJECTION_CROSSOVER_TYPES)
    @pytest.mark.parametrize("use_injection", [False, True])
    def test_crossover_scan(self, benchmark, crossover_type: str, use_injection: bool):
        bench_engine = MalthusJAXBenchEngine(
            pop_size=self._POP,
            dims=self._DIMS,
            num_generations=NUM_GENERATIONS_SHORT,
            crossover_type=crossover_type,
            use_injection_ops=use_injection,
        )
        bench_engine.run_once(jr.PRNGKey(0))  # warm-up / compile

        def _run():
            _run_bench_engine(bench_engine)

        mode = "injection" if use_injection else "standard"
        benchmark.group = (
            f"injection_crossover_scan_{NUM_GENERATIONS_SHORT}gen/pop{self._POP}_d{self._DIMS}"
        )
        benchmark.name = f"{crossover_type}_{mode}"
        benchmark.pedantic(_run, iterations=1, rounds=5, warmup_rounds=2)

    @pytest.mark.parametrize("mutation_type", _INJECTION_MUTATION_TYPES)
    @pytest.mark.parametrize("use_injection", [False, True])
    def test_mutation_scan(self, benchmark, mutation_type: str, use_injection: bool):
        bench_engine = MalthusJAXBenchEngine(
            pop_size=self._POP,
            dims=self._DIMS,
            num_generations=NUM_GENERATIONS_SHORT,
            mutation_type=mutation_type,
            use_injection_ops=use_injection,
        )
        bench_engine.run_once(jr.PRNGKey(0))  # warm-up / compile

        def _run():
            _run_bench_engine(bench_engine)

        mode = "injection" if use_injection else "standard"
        benchmark.group = (
            f"injection_mutation_scan_{NUM_GENERATIONS_SHORT}gen/pop{self._POP}_d{self._DIMS}"
        )
        benchmark.name = f"{mutation_type}_{mode}"
        benchmark.pedantic(_run, iterations=1, rounds=5, warmup_rounds=2)
