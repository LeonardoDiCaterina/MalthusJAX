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
        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            problem_name="sphere",
            num_dims=5,
            pop_size=10,
            generations=3,
        )
        assert isinstance(adapter, EvosaxEngineAdapter)
        assert adapter.pop_size == 10
        assert adapter.num_generations == 3
        assert adapter.num_dims == 5

    def test_all_strategies_construct(self):
        """Every registered strategy should be constructable."""
        for name in list_strategies():
            adapter = build_evosax_engine(
                strategy_name=name,
                problem_name="sphere",
                num_dims=3,
                pop_size=8,
                generations=2,
            )
            assert isinstance(adapter, EvosaxEngineAdapter)

    def test_unknown_strategy_raises(self):
        with pytest.raises(KeyError, match="Unknown evosax strategy"):
            build_evosax_engine(strategy_name="NonExistent")

    def test_fitness_spec_overrides_problem(self):
        """Catalog-style fitness_spec should override problem_name/num_dims."""
        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            fitness_spec="rastrigin:dim=7",
            pop_size=8,
            generations=2,
        )
        assert adapter.num_dims == 7

    def test_fitness_spec_with_seed(self):
        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            fitness_spec="sphere:dim=4,seed=99",
            pop_size=8,
            generations=2,
        )
        assert adapter.num_dims == 4

    def test_custom_bounds(self):
        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            problem_name="sphere",
            num_dims=3,
            pop_size=8,
            generations=2,
            bounds=(-10.0, 10.0),
        )
        assert adapter.bounds == (-10.0, 10.0)

    def test_maximize_flag_stored(self):
        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            problem_name="sphere",
            num_dims=3,
            pop_size=8,
            generations=2,
            maximize=True,
        )
        assert adapter.maximize is True


# ---------------------------------------------------------------------------
# Adapter: run_once protocol conformance
# ---------------------------------------------------------------------------


class TestEvosaxAdapterRunOnce:
    """Tests that EvosaxEngineAdapter.run_once satisfies the Engine protocol."""

    @pytest.fixture()
    def small_adapter(self) -> EvosaxEngineAdapter:
        """A tiny adapter for fast tests."""
        return build_evosax_engine(
            strategy_name="SimpleGA",
            problem_name="sphere",
            num_dims=3,
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

        required_keys = {"generation", "best_fitness", "mean_fitness", "std_fitness"}
        for entry in history:
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
            assert jnp.isfinite(entry["mean_fitness"]), f"gen {entry['generation']}"

        assert jnp.isfinite(result["summary"]["best_fitness"])


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestEvosaxDeterminism:
    """Verify reproducibility given the same PRNG key."""

    def test_same_key_same_result(self):
        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            problem_name="sphere",
            num_dims=3,
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
            assert h1["mean_fitness"] == h2["mean_fitness"]

    def test_different_keys_different_results(self):
        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            problem_name="sphere",
            num_dims=3,
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


class TestMaximisationConvention:
    """Verify the sign-flip logic when maximize=True.

    Note: evosax BBOBProblem.eval() already returns *negated* fitness
    (lower = better for the strategy).  state.best_fitness is therefore
    negative.  The adapter's maximize flag flips the sign so that users
    see positive-is-better (maximisation) or raw negative (minimisation).
    """

    def test_minimize_reports_raw_evosax_values(self):
        """With maximize=False, best_fitness is the raw evosax value (negative for sphere)."""
        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            problem_name="sphere",
            num_dims=3,
            pop_size=10,
            generations=5,
            maximize=False,
        )
        result = adapter.run_once(jr.PRNGKey(42))
        # evosax BBOBProblem returns negative fitness for sphere
        assert result["summary"]["best_fitness"] < 0.0

    def test_maximize_flips_to_positive(self):
        """With maximize=True, best_fitness should be positive (−(negative) > 0)."""
        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            problem_name="sphere",
            num_dims=3,
            pop_size=10,
            generations=5,
            maximize=True,
        )
        result = adapter.run_once(jr.PRNGKey(42))
        # Flipped: −(negative evosax value) > 0
        assert result["summary"]["best_fitness"] > 0.0

    def test_maximize_history_consistent(self):
        """History entries should also be sign-flipped to positive."""
        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            problem_name="sphere",
            num_dims=3,
            pop_size=10,
            generations=5,
            maximize=True,
        )
        result = adapter.run_once(jr.PRNGKey(42))
        for entry in result["history"]:
            assert entry["best_fitness"] > 0.0

    def test_sign_flip_is_symmetric(self):
        """maximize=True and maximize=False should produce opposite-sign fitness."""
        min_adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            problem_name="sphere",
            num_dims=3,
            pop_size=10,
            generations=5,
            maximize=False,
        )
        max_adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            problem_name="sphere",
            num_dims=3,
            pop_size=10,
            generations=5,
            maximize=True,
        )
        key = jr.PRNGKey(42)
        min_bf = min_adapter.run_once(key)["summary"]["best_fitness"]
        max_bf = max_adapter.run_once(key)["summary"]["best_fitness"]
        assert jnp.isclose(min_bf, -max_bf)


# ---------------------------------------------------------------------------
# Strategy-specific smoke tests
# ---------------------------------------------------------------------------


class TestStrategySmoke:
    """Quick smoke test for each registered strategy to ensure the full
    ask/tell loop completes without errors."""

    @pytest.mark.parametrize("strategy_name", list_strategies())
    def test_strategy_runs_to_completion(self, strategy_name):
        adapter = build_evosax_engine(
            strategy_name=strategy_name,
            problem_name="sphere",
            num_dims=4,
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
        adapter = build_evosax_engine(
            strategy_name=strategy_name,
            problem_name="rastrigin",
            num_dims=5,
            pop_size=10,
            generations=3,
        )
        result = adapter.run_once(jr.PRNGKey(123))
        assert jnp.isfinite(result["summary"]["best_fitness"])


# ---------------------------------------------------------------------------
# Integration with BenchmarkRunner
# ---------------------------------------------------------------------------


class TestEvosaxBenchmarkIntegration:
    """Test that EvosaxEngineAdapter works end-to-end with BenchmarkRunner."""

    def test_benchmark_runner_accepts_adapter(self):
        """BenchmarkRunner should accept the adapter as an Engine."""
        from malthusjax.benchmarking import BenchmarkRunner

        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            problem_name="sphere",
            num_dims=3,
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
            adapter = build_evosax_engine(
                strategy_name="SimpleGA",
                problem_name="sphere",
                num_dims=3,
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
