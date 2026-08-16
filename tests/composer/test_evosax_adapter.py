"""Tests for the EvosaxEngineAdapter and build_evosax_engine factory.

Mirrors the structure of test_engine_factory.py to keep the composer test
suite consistent.
"""

import tempfile
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import pytest
pytest.importorskip('evosax')

from malthusjax.composer.evosax_adapter import (
    EVOSAX_STRATEGIES,
    EvosaxEngineAdapter,
    build_evosax_engine,
    list_strategies,
)
from malthusjax.core.fitness.base import BaseEvaluator
from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator

from .base_adapter_suite import BaseAdapterTestSuite

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
        """Every registered strategy should be constructable.

        Note: LGA is skipped due to JAX version compatibility issues with
        evosax's pickled parameter loading.
        """
        # Skip strategies due to external library compatibility issue with JAX pickling
        skip_strategies = {"LGA", "EvoTF_ES", "LES", "LM_MA_ES", "SV_CMA_ES", "SV_Open_ES", "DES"}

        for name in list_strategies():
            if name in skip_strategies:
                continue
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

    def test_strategy_params_updates_strategy_attributes(self):
        evalr = make_bbob_evaluator(fn_name="sphere", num_dims=3)
        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            evaluator=evalr,
            pop_size=8,
            generations=2,
            strategy_params={
                "crossover_rate": 0.5,
                "elite_ratio": 0.1,
                "std_schedule": lambda step: 0.1,
            },
        )
        assert adapter.params.crossover_rate == 0.5
        assert adapter.strategy.elite_ratio == 0.1
        assert adapter.strategy.std_schedule(0) == 0.1

    def test_unknown_strategy_param_raises(self):
        evalr = make_bbob_evaluator(fn_name="sphere", num_dims=3)
        with pytest.raises(KeyError, match="Unknown evosax strategy parameter"):
            build_evosax_engine(
                strategy_name="SimpleGA",
                evaluator=evalr,
                pop_size=8,
                generations=2,
                strategy_params={"nope": 123},
            )

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

    def test_generic_evaluator_is_supported(self):
        """Passing a generic evaluator should configure MALTHUSJAX eval_mode."""

        class DummyEval(BaseEvaluator):
            def evaluate_population(self, population):
                # trivial implementation
                return population

            def evaluate(self, genome):
                return jnp.zeros((), dtype=jnp.float32)

        from flax import struct

        from malthusjax.core.genome.real_genome import RealGenomeConfig

        @struct.dataclass
        class MockConfig:
            genome_config: RealGenomeConfig = struct.field(pytree_node=False)
            maximize: bool = struct.field(pytree_node=False, default=False)

        dummy = DummyEval(config=MockConfig(genome_config=RealGenomeConfig(shape=(4,))), data=None)

        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            evaluator=dummy,
            pop_size=4,
            generations=1,
        )
        assert adapter.engine.eval_mode == "malthusjax"


# ---------------------------------------------------------------------------
# General Adapter Harness conformance
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_EVOSAX_INIT_TELL,
    reason="Requires evosax with init/tell API (GitHub version, not PyPI 0.1.6)",
)
class TestEvosaxAdapter(BaseAdapterTestSuite):
    """General MalthusJAX test harness conformance for Evosax adapter."""

    def make_adapter(self, maximize: bool = False, eval_mode: str = "native", seed: int = 0):
        if eval_mode == "native":
            evalr = make_bbob_evaluator(fn_name="sphere", num_dims=3, maximize=maximize, seed=seed)
        else:

            class DummyEval(BaseEvaluator):
                def evaluate_population(self, population):
                    return population

                def evaluate(self, genome):
                    return jnp.zeros((), dtype=jnp.float32)

            from flax import struct

            from malthusjax.core.genome.real_genome import RealGenomeConfig

            _maximize = maximize

            @struct.dataclass
            class MockConfig:
                genome_config: RealGenomeConfig = struct.field(pytree_node=False)
                maximize: bool = struct.field(pytree_node=False, default=_maximize)

            evalr = DummyEval(
                config=MockConfig(genome_config=RealGenomeConfig(shape=(3,))), data=None
            )

        return build_evosax_engine(
            strategy_name="SimpleGA",
            evaluator=evalr,
            pop_size=8,
            generations=5,
            maximize=maximize,
        )


# ---------------------------------------------------------------------------
# Maximisation sign-flip (Evosax Specifics)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_EVOSAX_INIT_TELL,
    reason="Requires evosax with init/tell API (GitHub version, not PyPI 0.1.6)",
)
class TestMaximisationConvention:
    """Verify the sign-flip logic when maximize=True for evosax."""

    def test_minimize_reports_raw_evosax_values(self):
        """With maximize=False, best_fitness should match the raw evosax metric.
        If we force seed=0, the sphere optimum is positive.
        """
        evalr = make_bbob_evaluator(fn_name="sphere", num_dims=3, maximize=False, seed=0)
        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            evaluator=evalr,
            pop_size=10,
            generations=5,
            maximize=False,
        )
        result = adapter.run_once(jr.PRNGKey(42))
        assert result["summary"]["best_fitness"] >= 0.0

    def test_sign_flip_applied(self):
        """When maximize=True the adapter should still apply a sign flip on the
        *reported* metric compared to its own raw output.
        """
        evalr = make_bbob_evaluator(fn_name="sphere", num_dims=3)
        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            evaluator=evalr,
            pop_size=10,
            generations=5,
            maximize=True,
        )
        key = jr.PRNGKey(42)
        result = adapter.run_once(key)
        bf = result["summary"]["best_fitness"]
        # after sign flipping the value should be positive (raw metrics were
        # negative in this configuration)
        assert bf > 0

    def test_maximize_preserves_objective_space_mean(self):
        """mean_fitness should be emitted as a finite objective-space signal."""
        evalr = make_bbob_evaluator(fn_name="sphere", num_dims=3)
        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            evaluator=evalr,
            pop_size=10,
            generations=5,
            maximize=True,
        )

        result = adapter.run_once(jr.PRNGKey(42))
        # Keep this robust across evosax/BBOB variants: require finite outputs
        # and presence in every history row.
        for row in result["history"]:
            assert "mean_fitness" in row
            assert jnp.isfinite(row["mean_fitness"]), row


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
        # Skip due to JAX version compatibility issue with evosax pickling
        if strategy_name in {
            "LGA",
            "EvoTF_ES",
            "LES",
            "LM_MA_ES",
            "SV_CMA_ES",
            "SV_Open_ES",
            "DES",
        }:
            pytest.skip(f"{strategy_name} skipped due to evosax JAX compatibility issue")

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
        # Skip due to JAX version compatibility issue with evosax pickling
        if strategy_name in {
            "LGA",
            "EvoTF_ES",
            "LES",
            "LM_MA_ES",
            "SV_CMA_ES",
            "SV_Open_ES",
            "DES",
        }:
            pytest.skip(f"{strategy_name} skipped due to evosax JAX compatibility issue")

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
