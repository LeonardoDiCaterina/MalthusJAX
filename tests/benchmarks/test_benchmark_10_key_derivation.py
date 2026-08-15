"""
BENCHMARK GROUP 10 — Key Derivation Strategy (SPLIT vs FOLD)
============================================================

Compare ``KeyDerivationStrategy.SPLIT`` vs ``FOLD`` entropy strategies.
"""

import jax.random as jr
import pytest

from tests.benchmarks.conftest_benchmarks import (
    NUM_GENERATIONS_SHORT,
    SEED,
    KeyDerivationStrategy,
    MalthusJAXBenchEngine,
    _build_malthusjax_engine,
    _malthusjax_init_and_warmup,
)


class TestKeyDerivationStrategy:
    """Compare ``KeyDerivationStrategy.SPLIT`` vs ``FOLD`` entropy strategies.

    ``SPLIT`` uses sequential ``jax.random.split`` to generate uncorrelated
    sub-keys, whereas ``FOLD`` uses ``jax.random.fold_in`` which is
    parallelisable at the cost of weaker independence guarantees.

    Tests run at pop=500, d=10 where the per-step key-allocation cost is
    most visible compared with operator latency; both single-step and full
    scan timings are recorded.
    """

    _POP = 500
    _DIMS = 10

    @pytest.mark.parametrize(
        "key_derivation",
        [KeyDerivationStrategy.SPLIT, KeyDerivationStrategy.FOLD],
        ids=["split", "fold"],
    )
    def test_single_step(self, benchmark, key_derivation: KeyDerivationStrategy):
        """Single warm step under each key-derivation strategy."""
        engine = _build_malthusjax_engine(
            self._POP,
            self._DIMS,
            key_derivation=key_derivation,
        )
        state, jit_step = _malthusjax_init_and_warmup(engine)

        def _run():
            s, _ = jit_step(state)
            s.best_fitness.block_until_ready()

        benchmark.group = f"key_derivation/pop{self._POP}_d{self._DIMS}"
        benchmark.name = key_derivation.value
        benchmark.pedantic(_run, iterations=1, rounds=5, warmup_rounds=2)

    @pytest.mark.parametrize(
        "key_derivation",
        [KeyDerivationStrategy.SPLIT, KeyDerivationStrategy.FOLD],
        ids=["split", "fold"],
    )
    def test_scan(self, benchmark, key_derivation: KeyDerivationStrategy):
        """50-generation scan under each key-derivation strategy."""
        bench_engine = MalthusJAXBenchEngine(
            pop_size=self._POP,
            dims=self._DIMS,
            num_generations=NUM_GENERATIONS_SHORT,
            key_derivation=key_derivation,
        )
        bench_engine.run_once(jr.PRNGKey(0))  # warm-up / compile

        def _run():
            result = bench_engine.run_once(jr.PRNGKey(SEED))
            assert result["summary"]["best_fitness"] is not None

        benchmark.group = (
            f"key_derivation_scan_{NUM_GENERATIONS_SHORT}gen/pop{self._POP}_d{self._DIMS}"
        )
        benchmark.name = key_derivation.value
        benchmark.pedantic(_run, iterations=1, rounds=5, warmup_rounds=2)

    @pytest.mark.parametrize("pop_size", [100, 500, 1024])
    @pytest.mark.parametrize(
        "key_derivation",
        [KeyDerivationStrategy.SPLIT, KeyDerivationStrategy.FOLD],
        ids=["split", "fold"],
    )
    def test_scaling(self, benchmark, pop_size: int, key_derivation: KeyDerivationStrategy):
        """Key derivation strategy scaling sweep (d=10, varying pop_size)."""
        dims = 10
        engine = _build_malthusjax_engine(pop_size, dims, key_derivation=key_derivation)
        state, jit_step = _malthusjax_init_and_warmup(engine)

        def _run():
            s, _ = jit_step(state)
            s.best_fitness.block_until_ready()

        benchmark.group = "key_derivation_scaling_d10"
        benchmark.name = f"{key_derivation.value}_pop{pop_size}"
        benchmark.pedantic(_run, iterations=1, rounds=5, warmup_rounds=2)
