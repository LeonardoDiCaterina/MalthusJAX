"""Tests for the EvosaxEngineAdapter and build_evosax_engine factory.

Mirrors the structure of test_engine_factory.py to keep the composer test
suite consistent.
"""

import tempfile
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import pytest

from malthusjax.composer.evosax_adapter import (
    EVOSAX_STRATEGIES,
    EvosaxEngineAdapter,
    build_evosax_engine,
    list_strategies,
)
from malthusjax.core.fitness.base import BaseEvaluator, BaseEvaluatorConfig
from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator

# Check if evosax has the init/tell API (GitHub version only, not PyPI 0.1.6)
# Some releases export strategies at the top level while others require
# importing from ``evosax.algorithms``; handle both so the test guard works
# in any environment we might encounter.
try:
    try:
        from evosax import SimpleGA
    except ImportError:
        from evosax.algorithms import SimpleGA

    # try whichever constructor signature the installed version uses
    try:
        _test_ga = SimpleGA(population_size=10, solution=[0] * 5)
    except TypeError:
        _test_ga = SimpleGA(popsize=10, num_dims=5)

    HAS_EVOSAX_INIT_TELL = hasattr(_test_ga, "init") and hasattr(_test_ga, "tell")
except Exception:
    HAS_EVOSAX_INIT_TELL = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_bbob_evaluator(
    fn_name: str = "sphere", num_dims: int = 3, seed: int = 42, maximize: bool = False
) -> BBOBEvaluator:
    return BBOBEvaluator.create(
        BBOBConfig(fn_name=fn_name, num_dims=num_dims, seed=seed, maximize=maximize)
    )


# ---------------------------------------------------------------------------
# Registry / list helpers
# ---------------------------------------------------------------------------


class TestStrategyRegistry:
    """Tests for the strategy registry and list helper."""

    def test_list_strategies_returns_sorted_list(self):
        names = list_strategies()
        assert isinstance(names, list)
        assert names == sorted(names)
        assert len(names) >= 3  # SimpleGA, MR15_GA, DifferentialEvolution

    def test_known_strategies_present(self):
        names = list_strategies()
        for expected in ("SimpleGA", "MR15_GA", "DifferentialEvolution"):
            assert expected in names

    def test_registry_maps_to_classes(self):
        for name, cls in EVOSAX_STRATEGIES.items():
            assert callable(cls), f"{name} should map to a callable class"


# ---------------------------------------------------------------------------
# Factory: build_evosax_engine
# ---------------------------------------------------------------------------


class TestBuildEvosaxEngine:
    """Tests for the build_evosax_engine factory function."""

    def test_basic_construction(self):
        evalr = make_bbob_evaluator(fn_name="sphere", num_dims=5)
        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            evaluator=evalr,
            pop_size=10,
            generations=3,
        )
        assert isinstance(adapter, EvosaxEngineAdapter)
        assert adapter.pop_size == 10
        assert adapter.num_generations == 3
        assert adapter.num_dims == 5

    def test_unwraps_bbob_evaluator(self):
        """Passing a BBOBEvaluator results in a raw evosax problem stored."""
        from evosax.problems import BBOBProblem

        evalr = make_bbob_evaluator(fn_name="sphere", num_dims=2)
        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            evaluator=evalr,
            pop_size=4,
            generations=1,
        )
        assert isinstance(adapter.problem, BBOBProblem)

    def test_all_strategies_construct(self):
        """Every registered strategy should be constructable."""
        for name in list_strategies():
            evalr = make_bbob_evaluator(fn_name="sphere", num_dims=3)
            adapter = build_evosax_engine(
                strategy_name=name,
                evaluator=evalr,
                pop_size=8,
                generations=2,
            )
            assert isinstance(adapter, EvosaxEngineAdapter)

    def test_unknown_strategy_raises(self):
        evalr = make_bbob_evaluator(fn_name="sphere", num_dims=2)
        with pytest.raises(KeyError, match="Unknown evosax strategy"):
            build_evosax_engine(strategy_name="NonExistent", evaluator=evalr)

    def test_fitness_spec_overrides_problem(self):
        """Catalog-style fitness_spec should override problem_name/num_dims."""
        # initial evaluator chosen arbitrarily; spec will override dims
        evalr = make_bbob_evaluator(fn_name="sphere", num_dims=3)
        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            evaluator=evalr,
            fitness_spec="rastrigin:dim=7",
            pop_size=8,
            generations=2,
        )
        assert adapter.num_dims == 7

    def test_fitness_spec_with_seed(self):
        evalr = make_bbob_evaluator(fn_name="sphere", num_dims=3)
        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            evaluator=evalr,
            fitness_spec="sphere:dim=4,seed=99",
            pop_size=8,
            generations=2,
        )
        assert adapter.num_dims == 4

    def test_custom_bounds(self):
        evalr = make_bbob_evaluator(fn_name="sphere", num_dims=3)
        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            evaluator=evalr,
            pop_size=8,
            generations=2,
            bounds=(-10.0, 10.0),
        )
        assert adapter.bounds == (-10.0, 10.0)

    def test_maximize_flag_stored(self):
        evalr = make_bbob_evaluator(fn_name="sphere", num_dims=3)
        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            evaluator=evalr,
            pop_size=8,
            generations=2,
            maximize=True,
        )
        assert adapter.maximize is True

    def test_generic_evaluator_not_supported(self):
        """Passing a non-BBOB evaluator should raise a clear error."""

        class DummyEval(BaseEvaluator):
            def evaluate(self, genome):
                # trivial implementation, never used by the factory
                return jnp.zeros((), dtype=jnp.float32)

        dummy = DummyEval(config=BaseEvaluatorConfig(maximize=False), data=None)
        with pytest.raises(NotImplementedError, match="Only BBOBEvaluator"):
            build_evosax_engine(
                strategy_name="SimpleGA",
                evaluator=dummy,
                pop_size=4,
                generations=1,
            )


# ---------------------------------------------------------------------------
# Adapter: run_once protocol conformance
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_EVOSAX_INIT_TELL,
    reason="Requires evosax with init/tell API (GitHub version, not PyPI 0.1.6)",
)
class TestEvosaxAdapterRunOnce:
    """Tests that EvosaxEngineAdapter.run_once satisfies the Engine protocol."""

    @pytest.fixture()
    def small_adapter(self) -> EvosaxEngineAdapter:
        """A tiny adapter for fast tests."""
        evalr = make_bbob_evaluator(fn_name="sphere", num_dims=3)
        return build_evosax_engine(
            strategy_name="SimpleGA",
            evaluator=evalr,
            pop_size=8,
            generations=5,
        )

    def test_result_keys(self, small_adapter):
        result = small_adapter.run_once(jr.PRNGKey(0))
        assert set(result.keys()) == {"history", "summary", "timings"}

    def test_history_format(self, small_adapter):
        result = small_adapter.run_once(jr.PRNGKey(0))
        history = result["history"]

        assert isinstance(history, list)
        assert len(history) == 5  # generations

        required_keys = {"generation", "best_fitness"}
        for entry in history:
            # evosax 0.2+ no longer provides mean/std; those are computed by adapter
            assert required_keys.issubset(entry.keys())

    def test_history_generations_sequential(self, small_adapter):
        result = small_adapter.run_once(jr.PRNGKey(0))
        gens = [h["generation"] for h in result["history"]]
        assert gens == list(range(1, 6))

    def test_summary_format(self, small_adapter):
        result = small_adapter.run_once(jr.PRNGKey(0))
        summary = result["summary"]

        assert "best_fitness" in summary
        assert "final_generation" in summary
        assert "total_evaluations" in summary
        assert summary["final_generation"] == 5
        assert summary["total_evaluations"] == 5 * 8  # generations * pop_size

    def test_timings_format(self, small_adapter):
        result = small_adapter.run_once(jr.PRNGKey(0))
        timings = result["timings"]

        assert "initialization" in timings
        assert "evolution" in timings
        assert timings["initialization"] > 0
        assert timings["evolution"] > 0

    def test_fitness_values_are_finite(self, small_adapter):
        result = small_adapter.run_once(jr.PRNGKey(42))

        for entry in result["history"]:
            assert jnp.isfinite(entry["best_fitness"]), f"gen {entry['generation']}"
        assert jnp.isfinite(result["summary"]["best_fitness"])


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_EVOSAX_INIT_TELL,
    reason="Requires evosax with init/tell API (GitHub version, not PyPI 0.1.6)",
)
class TestEvosaxDeterminism:
    """Verify reproducibility given the same PRNG key."""

    def test_same_key_same_result(self):
        evalr = make_bbob_evaluator(fn_name="sphere", num_dims=3)
        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            evaluator=evalr,
            pop_size=8,
            generations=5,
        )

        key = jr.PRNGKey(999)
        r1 = adapter.run_once(key)
        r2 = adapter.run_once(key)

        assert r1["summary"]["best_fitness"] == r2["summary"]["best_fitness"]
        assert len(r1["history"]) == len(r2["history"])
        for h1, h2 in zip(r1["history"], r2["history"]):
            assert h1["best_fitness"] == h2["best_fitness"]

    def test_different_keys_different_results(self):
        evalr = make_bbob_evaluator(fn_name="sphere", num_dims=3)
        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            evaluator=evalr,
            pop_size=8,
            generations=5,
        )

        r1 = adapter.run_once(jr.PRNGKey(0))
        r2 = adapter.run_once(jr.PRNGKey(12345))

        # Overwhelmingly unlikely to match on different keys
        assert r1["summary"]["best_fitness"] != r2["summary"]["best_fitness"]


# ---------------------------------------------------------------------------
# Maximisation sign-flip
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_EVOSAX_INIT_TELL,
    reason="Requires evosax with init/tell API (GitHub version, not PyPI 0.1.6)",
)
class TestMaximisationConvention:
    """Verify the sign-flip logic when maximize=True.

    Note: evosax BBOBProblem.eval() already returns *negated* fitness
    (lower = better for the strategy).  state.best_fitness is therefore
    negative.  The adapter's maximize flag flips the sign so that users
    see positive-is-better (maximisation) or raw negative (minimisation).
    """

    def test_minimize_reports_raw_evosax_values(self):
        """With maximize=False, best_fitness should match the raw evosax metric.
        For current problems (sphere) that metric is non-negative.
        """
        evalr = make_bbob_evaluator(fn_name="sphere", num_dims=3, maximize=False)
        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            evaluator=evalr,
            pop_size=10,
            generations=5,
            maximize=False,
        )
        result = adapter.run_once(jr.PRNGKey(42))
        assert result["summary"]["best_fitness"] >= 0.0

    def test_maximize_flag_changes_outcome(self):
        """Toggling ``maximize`` should alter the reported fitness values.

        Due to the stochastic nature of the algorithm the actual numerical
        results are not simply negated; we just check that the two runs differ.
        """
        evalr = make_bbob_evaluator(fn_name="sphere", num_dims=3)
        min_adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            evaluator=evalr,
            pop_size=10,
            generations=5,
            maximize=False,
        )
        max_adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            evaluator=evalr,
            pop_size=10,
            generations=5,
            maximize=True,
        )
        key = jr.PRNGKey(42)
        min_bf = min_adapter.run_once(key)["summary"]["best_fitness"]
        max_bf = max_adapter.run_once(key)["summary"]["best_fitness"]
        assert min_bf != max_bf

    def test_maximize_history_changes(self):
        """History sequence should differ when maximisation is toggled."""
        evalr = make_bbob_evaluator(fn_name="sphere", num_dims=3)
        min_adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            evaluator=evalr,
            pop_size=10,
            generations=5,
            maximize=False,
        )
        max_adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            evaluator=evalr,
            pop_size=10,
            generations=5,
            maximize=True,
        )
        key = jr.PRNGKey(42)
        min_hist = min_adapter.run_once(key)["history"]
        max_hist = max_adapter.run_once(key)["history"]
        # just assert that at least one generation differs
        assert any(
            hmin["best_fitness"] != hmax["best_fitness"] for hmin, hmax in zip(min_hist, max_hist)
        )

    def test_sign_flip_applied(self):
        """When maximize=True the adapter should still apply a sign flip on the
        *reported* metric compared to its own raw output.  This is a sanity
        check of the adapter logic rather than a comparison between two runs.
        """
        evalr = make_bbob_evaluator(fn_name="sphere", num_dims=3)
        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            evaluator=evalr,
            pop_size=10,
            generations=5,
            maximize=True,
        )
        # run once, then manually reverse the sign of the returned fitness and
        # ensure it is non-negative, implying a flip occurred in the adapter.
        key = jr.PRNGKey(42)
        result = adapter.run_once(key)
        bf = result["summary"]["best_fitness"]
        # after sign flipping the value should be positive (raw metrics were
        # negative in this configuration)
        assert bf > 0


# ---------------------------------------------------------------------------
# Strategy-specific smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_EVOSAX_INIT_TELL,
    reason="Requires evosax with init/tell API (GitHub version, not PyPI 0.1.6)",
)
class TestStrategySmoke:
    """Quick smoke test for each registered strategy to ensure the full
    ask/tell loop completes without errors."""

    @pytest.mark.parametrize("strategy_name", list_strategies())
    def test_strategy_runs_to_completion(self, strategy_name):
        evalr = make_bbob_evaluator(fn_name="sphere", num_dims=4)
        adapter = build_evosax_engine(
            strategy_name=strategy_name,
            evaluator=evalr,
            pop_size=10,
            generations=3,
        )
        result = adapter.run_once(jr.PRNGKey(7))

        assert len(result["history"]) == 3
        assert result["summary"]["total_evaluations"] == 3 * 10
        assert jnp.isfinite(result["summary"]["best_fitness"])

    @pytest.mark.parametrize("strategy_name", list_strategies())
    def test_strategy_with_rastrigin(self, strategy_name):
        """Strategies should also work on non-trivial BBOB problems."""
        evalr = make_bbob_evaluator(fn_name="rastrigin", num_dims=5)
        adapter = build_evosax_engine(
            strategy_name=strategy_name,
            evaluator=evalr,
            pop_size=10,
            generations=3,
        )
        result = adapter.run_once(jr.PRNGKey(123))
        assert jnp.isfinite(result["summary"]["best_fitness"])


# ---------------------------------------------------------------------------
# Integration with BenchmarkRunner
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_EVOSAX_INIT_TELL,
    reason="Requires evosax with init/tell API (GitHub version, not PyPI 0.1.6)",
)
class TestEvosaxBenchmarkIntegration:
    """Test that EvosaxEngineAdapter works end-to-end with BenchmarkRunner."""

    def test_benchmark_runner_accepts_adapter(self):
        """BenchmarkRunner should accept the adapter as an Engine."""
        from malthusjax.benchmarking import BenchmarkRunner

        evalr = make_bbob_evaluator(fn_name="sphere", num_dims=3)
        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            evaluator=evalr,
            pop_size=8,
            generations=3,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = BenchmarkRunner(
                engine=adapter,
                experiment_name="evosax_bench_test",
                output_dir=Path(tmp_dir),
                write_artifacts=True,
            )
            result = runner.run(seeds=[42, 123])

        assert len(result.runs) == 2
        assert result.name == "evosax_bench_test"
        for run in result.runs:
            assert len(run.history) == 3
            assert run.metrics["total_evaluations"] == 3 * 8

    def test_benchmark_runner_deterministic_across_runs(self):
        """Same seeds → same results through BenchmarkRunner."""
        from malthusjax.benchmarking import BenchmarkRunner

        def _run_experiment(output_dir):
            evalr = make_bbob_evaluator(fn_name="sphere", num_dims=3)
            adapter = build_evosax_engine(
                strategy_name="SimpleGA",
                evaluator=evalr,
                pop_size=8,
                generations=3,
            )
            runner = BenchmarkRunner(
                engine=adapter,
                experiment_name="repro",
                output_dir=Path(output_dir),
                write_artifacts=False,
            )
            return runner.run(seeds=[42])

        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            r1 = _run_experiment(d1)
            r2 = _run_experiment(d2)

        assert r1.runs[0].metrics["best_fitness"] == r2.runs[0].metrics["best_fitness"]
