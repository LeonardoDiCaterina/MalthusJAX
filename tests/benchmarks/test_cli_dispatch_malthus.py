"""Tests for the MalthusJAX Adapter Profiler CLI."""

import pytest
import tempfile
from pathlib import Path
import jax
import jax.random as jr
import numpy as np

from benchmarks.cli_dispatch_malthus import (
    get_available_engines,
    build_adapter,
    measure_single_run,
    profile_configuration,
    compute_statistics,
    RunResult,
    StatsSummary,
)
from benchmarks.framework.registry import ComparisonRegistry


class TestGetAvailableEngines:
    """Test engine discovery from registry."""

    def test_returns_list(self):
        """Test that get_available_engines returns a list."""
        engines = get_available_engines()
        assert isinstance(engines, list)

    def test_contains_standard_ga(self):
        """Test that Standard_GA is in the available engines."""
        engines = get_available_engines()
        assert "Standard_GA" in engines

    @pytest.mark.skip(reason="MR15_GA depends on OneFifthGeneticEngine which is not yet implemented")
    def test_contains_mr15_ga(self):
        """Test that MR15_GA is in the available engines."""
        engines = get_available_engines()
        assert "MR15_GA" in engines

    def test_all_engines_have_malthus_factory(self):
        """Test that all returned engines have a malthus_factory."""
        engines = get_available_engines()
        for engine_name in engines:
            spec = ComparisonRegistry.get(engine_name)
            assert spec.malthus_factory is not None


class TestBuildAdapter:
    """Test adapter construction from registry."""

    def test_build_standard_ga(self):
        """Test building a Standard_GA adapter."""
        adapter = build_adapter(
            engine_name="Standard_GA",
            task="sphere",
            dim=10,
            pop_size=32,
            seed=42,
        )
        assert adapter.__class__.__name__ == "MalthusAdapter"
        assert adapter.engine is not None

    @pytest.mark.skip(reason="MR15_GA depends on OneFifthGeneticEngine which is not yet implemented")
    def test_build_mr15_ga(self):
        """Test building a MR15_GA adapter."""
        adapter = build_adapter(
            engine_name="MR15_GA",
            task="sphere",
            dim=10,
            pop_size=32,
            seed=42,
        )
        assert adapter.__class__.__name__ == "MalthusAdapter"

    def test_build_different_tasks(self):
        """Test building adapters for different BBOB tasks."""
        for task in ["sphere", "rosenbrock"]:
            adapter = build_adapter(
                engine_name="Standard_GA",
                task=task,
                dim=10,
                pop_size=32,
                seed=42,
            )
            assert adapter.__class__.__name__ == "MalthusAdapter"

    def test_build_invalid_engine_raises(self):
        """Test that building an invalid engine raises an error."""
        with pytest.raises(ValueError):
            build_adapter(
                engine_name="NonExistentEngine",
                task="sphere",
                dim=10,
                pop_size=32,
                seed=42,
            )


class TestMeasureSingleRun:
    """Test the timing measurement function."""

    def test_measure_simple_function(self):
        """Test measuring timing for a simple JAX function."""
        def simple_fn(x):
            return x * 2

        x = jax.numpy.ones((100,))
        cold_ms, warm_ms, compile_ms = measure_single_run(
            simple_fn, x, warmup_runs=1, timed_runs=3
        )

        assert cold_ms > 0
        assert warm_ms >= 0
        assert compile_ms >= 0

    def test_warm_faster_than_cold(self):
        """Test that warm execution is faster than cold (includes compilation)."""
        def fn(x):
            return jax.numpy.sum(x ** 2)

        x = jax.numpy.ones((1000,))
        cold_ms, warm_ms, compile_ms = measure_single_run(
            fn, x, warmup_runs=2, timed_runs=5
        )

        # Warm should be faster than cold (cold includes compilation)
        assert warm_ms < cold_ms

    def test_compile_time_positive(self):
        """Test that compile time estimate is positive for new functions."""
        def unique_fn(x):
            # Unique function to force new compilation
            return jax.numpy.sin(x) * jax.numpy.cos(x) + x

        x = jax.numpy.ones((500,))
        cold_ms, warm_ms, compile_ms = measure_single_run(
            unique_fn, x, warmup_runs=1, timed_runs=3
        )

        assert compile_ms >= 0


class TestProfileConfiguration:
    """Test multi-run profiling."""

    def test_profile_single_run(self):
        """Test profiling with a single run."""
        results = profile_configuration(
            engine_name="Standard_GA",
            task="sphere",
            dim=10,
            pop_size=32,
            num_gens=5,
            unroll=1,
            n_runs=1,
            seed=42,
            warmup_runs=1,
            timed_runs=2,
        )

        assert len(results) == 1
        assert isinstance(results[0], RunResult)
        assert results[0].engine == "Standard_GA"
        assert results[0].unroll == 1
        assert results[0].warm_ms > 0

    def test_profile_multiple_runs(self):
        """Test profiling with multiple runs."""
        results = profile_configuration(
            engine_name="Standard_GA",
            task="sphere",
            dim=10,
            pop_size=32,
            num_gens=5,
            unroll=1,
            n_runs=3,
            seed=42,
            warmup_runs=1,
            timed_runs=2,
        )

        assert len(results) == 3
        for i, r in enumerate(results):
            assert r.run_id == i
            assert r.engine == "Standard_GA"

    def test_profile_different_unroll(self):
        """Test profiling with different unroll factors."""
        results_u1 = profile_configuration(
            engine_name="Standard_GA",
            task="sphere",
            dim=10,
            pop_size=32,
            num_gens=10,
            unroll=1,
            n_runs=1,
            seed=42,
            warmup_runs=1,
            timed_runs=2,
        )

        results_u4 = profile_configuration(
            engine_name="Standard_GA",
            task="sphere",
            dim=10,
            pop_size=32,
            num_gens=10,
            unroll=4,
            n_runs=1,
            seed=42,
            warmup_runs=1,
            timed_runs=2,
        )

        assert results_u1[0].unroll == 1
        assert results_u4[0].unroll == 4


class TestComputeStatistics:
    """Test statistical computation."""

    def test_compute_stats_single_run(self):
        """Test computing statistics from a single run."""
        results = [
            RunResult(
                run_id=0,
                engine="TestEngine",
                unroll=1,
                task="sphere",
                dim=10,
                pop_size=32,
                num_gens=10,
                cold_ms=100.0,
                warm_ms=10.0,
                compile_ms=90.0,
                ms_per_step=1.0,
            )
        ]

        stats = compute_statistics(results)

        assert isinstance(stats, StatsSummary)
        assert stats.engine == "TestEngine"
        assert stats.n_runs == 1
        assert stats.warm_mean == 10.0
        assert stats.warm_std == 0  # Single run has no std

    def test_compute_stats_multiple_runs(self):
        """Test computing statistics from multiple runs."""
        results = [
            RunResult(
                run_id=i,
                engine="TestEngine",
                unroll=1,
                task="sphere",
                dim=10,
                pop_size=32,
                num_gens=10,
                cold_ms=100.0 + i,
                warm_ms=10.0 + i * 0.1,
                compile_ms=90.0,
                ms_per_step=1.0 + i * 0.01,
            )
            for i in range(5)
        ]

        stats = compute_statistics(results)

        assert stats.n_runs == 5
        assert stats.warm_mean == pytest.approx(10.2, rel=0.01)  # Mean of 10.0, 10.1, 10.2, 10.3, 10.4
        assert stats.warm_std > 0
        assert stats.warm_min == 10.0
        assert stats.warm_max == 10.4
        assert stats.warm_cv > 0  # Coefficient of variation

    def test_compute_stats_confidence_interval(self):
        """Test that confidence interval is computed."""
        results = [
            RunResult(
                run_id=i,
                engine="TestEngine",
                unroll=1,
                task="sphere",
                dim=10,
                pop_size=32,
                num_gens=10,
                cold_ms=100.0,
                warm_ms=10.0 + np.random.randn() * 0.5,
                compile_ms=90.0,
                ms_per_step=1.0,
            )
            for i in range(30)
        ]

        stats = compute_statistics(results)

        # CI should bracket the mean
        assert stats.warm_ci_low <= stats.warm_mean <= stats.warm_ci_high
        # CI width should be reasonable (not too wide)
        ci_width = stats.warm_ci_high - stats.warm_ci_low
        assert ci_width < stats.warm_std * 3  # Should be tighter than 3 std devs

    def test_compute_stats_empty_raises(self):
        """Test that empty results raises an error."""
        with pytest.raises(ValueError):
            compute_statistics([])


class TestAdapterExecution:
    """Test that adapters can actually execute."""

    def test_adapter_init_state(self):
        """Test that adapters can initialize state."""
        adapter = build_adapter(
            engine_name="Standard_GA",
            task="sphere",
            dim=10,
            pop_size=32,
            seed=42,
        )

        key = jr.PRNGKey(42)
        state = adapter.init(key)

        assert state is not None
        assert hasattr(state, 'population')
        assert hasattr(state, 'generation')

    def test_adapter_step(self):
        """Test that adapters can execute a step."""
        adapter = build_adapter(
            engine_name="Standard_GA",
            task="sphere",
            dim=10,
            pop_size=32,
            seed=42,
        )

        key = jr.PRNGKey(42)
        state = adapter.init(key)
        step_fn = adapter.make_step_fn()

        new_state, _ = step_fn(state, None)

        assert new_state is not None
        assert new_state.generation == state.generation + 1

    def test_adapter_evolution_loop(self):
        """Test that adapters can run an evolution loop."""
        adapter = build_adapter(
            engine_name="Standard_GA",
            task="sphere",
            dim=10,
            pop_size=32,
            seed=42,
        )

        key = jr.PRNGKey(42)
        state = adapter.init(key)
        step_fn = adapter.make_step_fn()

        def evolution_loop(state):
            final, _ = jax.lax.scan(step_fn, state, None, length=10)
            return final

        jit_loop = jax.jit(evolution_loop)
        final_state = jit_loop(state)

        assert final_state.generation == 10


class TestRunResult:
    """Test RunResult dataclass."""

    def test_run_result_creation(self):
        """Test creating a RunResult."""
        result = RunResult(
            run_id=0,
            engine="Standard_GA",
            unroll=1,
            task="sphere",
            dim=10,
            pop_size=32,
            num_gens=50,
            cold_ms=100.0,
            warm_ms=10.0,
            compile_ms=90.0,
            ms_per_step=0.2,
        )

        assert result.run_id == 0
        assert result.engine == "Standard_GA"
        assert result.ms_per_step == 0.2


class TestStatsSummary:
    """Test StatsSummary dataclass."""

    def test_stats_summary_creation(self):
        """Test creating a StatsSummary."""
        stats = StatsSummary(
            engine="Standard_GA",
            unroll=1,
            task="sphere",
            dim=10,
            pop_size=32,
            num_gens=50,
            n_runs=30,
            warm_mean=10.0,
            warm_std=0.5,
            warm_min=9.0,
            warm_max=11.0,
            warm_p5=9.2,
            warm_p50=10.0,
            warm_p95=10.8,
            warm_ci_low=9.8,
            warm_ci_high=10.2,
            warm_cv=5.0,
            compile_mean=90.0,
            compile_std=5.0,
        )

        assert stats.n_runs == 30
        assert stats.warm_cv == 5.0


class TestIntegration:
    """Integration tests for the profiler."""

    def test_full_profiling_workflow(self, tmp_path):
        """Test the complete profiling workflow."""
        from benchmarks.cli_dispatch_malthus import (
            save_raw_results,
            save_statistics,
        )

        # Profile
        results = profile_configuration(
            engine_name="Standard_GA",
            task="sphere",
            dim=10,
            pop_size=32,
            num_gens=5,
            unroll=1,
            n_runs=3,
            seed=42,
            warmup_runs=1,
            timed_runs=2,
        )

        # Compute stats
        stats = compute_statistics(results)

        # Save results
        raw_path = tmp_path / "raw_results.csv"
        stats_path = tmp_path / "statistics.csv"

        save_raw_results(results, raw_path)
        save_statistics([stats], stats_path)

        # Verify files exist
        assert raw_path.exists()
        assert stats_path.exists()

        # Verify content
        with open(raw_path) as f:
            lines = f.readlines()
            assert len(lines) == 4  # Header + 3 runs

        with open(stats_path) as f:
            lines = f.readlines()
            assert len(lines) == 2  # Header + 1 stats row
