import time

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from malthusjax.composer.adapters import EvalMode
from malthusjax.composer.evosax_adapter import EvosaxEngineAdapter
from malthusjax.composer.qdax_adapter import QDaxEngineAdapter
from malthusjax.composer.tensorneat_adapter import TensorNEATEngineAdapter
from malthusjax.engine.qd.map_elites import MapElitesEngine
from tests.mock_backends.mock_engine import MockEvaluator, MockUniversalEngine

# Mock components that bypass framework init
POP_SIZE = 128
GENOME_LENGTH = 10
GENERATIONS = 100
MAXIMIZE = True


@pytest.fixture
def mock_strategy():
    return MockUniversalEngine(pop_size=POP_SIZE, genome_length=GENOME_LENGTH, maximize=MAXIMIZE)


@pytest.fixture
def mock_evaluator():
    return MockEvaluator(pop_size=POP_SIZE, genome_length=GENOME_LENGTH)


def _run_and_measure(adapter, name: str):
    key = jr.PRNGKey(42)

    # Run once to compile
    print(f"\n[{name}] Compiling (Warmup)...")
    adapter.run_once(key)

    # Run again for pure execution time
    t0 = time.perf_counter()
    res = adapter.run_once(key)
    t1 = time.perf_counter()

    exec_time = t1 - t0
    print(f"[{name}] Pure Execution Time: {exec_time:.5f}s")

    return res, exec_time


def test_adapter_overhead_parity(mock_strategy, mock_evaluator):
    results = {}
    timings = {}

    # 1. Evosax
    evosax_adapter = EvosaxEngineAdapter(
        strategy=mock_strategy,
        params={},
        pop_size=POP_SIZE,
        num_generations=GENERATIONS,
        maximize=MAXIMIZE,
        eval_mode=EvalMode.MALTHUSJAX,
        evaluator=mock_evaluator,
        num_dims=GENOME_LENGTH,
    )
    results["evosax"], timings["evosax"] = _run_and_measure(evosax_adapter, "EvoSAX")

    # 2. QDAX
    # For QDAX, we must supply history_metrics explicitly since it doesn't default to all available metrics
    qdax_adapter = QDaxEngineAdapter(
        strategy=mock_strategy,
        params={},
        pop_size=POP_SIZE,
        num_generations=GENERATIONS,
        maximize=MAXIMIZE,
        eval_mode=EvalMode.MALTHUSJAX,
        evaluator=mock_evaluator,
        history_metrics=["qd_score", "coverage", "max_fitness"],
    )
    results["qdax"], timings["qdax"] = _run_and_measure(qdax_adapter, "QDAX")

    # 3. TensorNEAT
    tn_adapter = TensorNEATEngineAdapter(
        strategy=mock_strategy,
        params={},
        pop_size=POP_SIZE,
        num_generations=GENERATIONS,
        maximize=MAXIMIZE,
        eval_mode=EvalMode.NATIVE,
        problem=mock_evaluator,
        problem_state=None,
    )
    results["tensorneat"], timings["tensorneat"] = _run_and_measure(tn_adapter, "TensorNEAT")

    # 4. MalthusJAX
    # MapElitesEngine doesn't use UniversalAdapterEngine decorator, it inherits from BaseEngine natively
    class MockMapElitesEngine(MapElitesEngine):
        num_generations: int = GENERATIONS

        def _build_jit_loop(self):
            # Bypass the real step logic and use the mock strategy step
            def scan_step(carry, _):
                rng, state = carry
                rng, key_step = jax.random.split(rng)
                state, metrics = mock_strategy.step(key_step, state)
                return (rng, state), metrics

            def run_loop(rng, state_init):
                carry = (rng, state_init)
                carry, metrics = jax.lax.scan(
                    scan_step, carry, None, length=self.num_generations, unroll=1
                )
                return carry[1], metrics

            return jax.jit(run_loop)

        def run_once(self, key):
            state_init = (jnp.zeros(1), jnp.zeros(1), 0)  # Mock state
            state, metrics = self._build_jit_loop()(key, state_init)
            summary = {
                "best_fitness": float(metrics["best_fitness"][-1]),
                "qd_score": float(metrics["qd_score"][-1]),
                "mean_fitness": float(metrics["mean_fitness"][-1]),
                "std_fitness": float(metrics["std_fitness"][-1]),
            }
            return {"state": state, "summary": summary}

    mjax_adapter = MockMapElitesEngine(
        emitter=None,  # Mock uses strategy directly
        evaluator=mock_evaluator,
        engine_params=mock_strategy.engine_params,
    )
    # MapElites expects maximize properly set inside engine_params, which we did
    results["malthusjax"], timings["malthusjax"] = _run_and_measure(mjax_adapter, "MalthusJAX")

    # Assert Result Parity
    ev_best = results["evosax"]["summary"]["best_fitness"]
    qd_best = results["qdax"]["summary"]["best_fitness"]
    tn_best = results["tensorneat"]["summary"]["best_fitness"]
    mj_best = results["malthusjax"]["summary"]["best_fitness"]

    assert ev_best == qd_best == tn_best == mj_best == float(POP_SIZE - 1), (
        f"Result parity failed for best_fitness: {ev_best}, {qd_best}, {tn_best}, {mj_best}"
    )

    # For QD Score, only QDAX and MalthusJAX track it natively in this mock setup
    qd_qdscore = results["qdax"]["summary"]["qd_score"]
    mj_qdscore = results["malthusjax"]["summary"]["qd_score"]

    assert qd_qdscore == mj_qdscore, (
        f"Result parity failed for qd_score: {qd_qdscore} != {mj_qdscore}"
    )

    # Assert Timing Parity
    t_vals = list(timings.values())
    t_mean = jnp.mean(jnp.array(t_vals))
    t_std = jnp.std(jnp.array(t_vals))

    print("\nTiming Parity Analysis:")
    print(f"Mean execution time: {t_mean:.5f}s ± {t_std:.5f}s")
    for name, t in timings.items():
        print(f"  {name}: {t:.5f}s")

    # We tolerate some variance, but the standard deviation should be small
    assert t_std < 0.05, (
        f"Timing standard deviation too high: {t_std:.5f}s (Threshold: 0.05s). Timings: {timings}"
    )
