import jax.random as jr

from malthusjax.benchmarking.runner import BenchmarkRunner, StubEngine


def test_benchmark_runner_basic():
    """Test basic runner functionality with stub engine."""
    engine = StubEngine(generations=2)
    runner = BenchmarkRunner(
        engine=engine,
        experiment_name="test_run",
        write_artifacts=False,  # Don't write files in this test
    )

    seeds = [1, 2]
    result = runner.run(seeds)

    assert result.name == "test_run"
    assert len(result.runs) == 2
    assert all(run.status == "success" for run in result.runs)
    assert all(run.seed in seeds for run in result.runs)

    # Check aggregated summary works
    agg = result.aggregated_summary()
    assert "best_fitness" in agg
    assert agg["best_fitness"]["mean"] is not None


def test_runner_with_artifacts(tmp_path):
    """Test runner writes artifacts correctly."""
    engine = StubEngine(generations=1)
    runner = BenchmarkRunner(
        engine=engine,
        experiment_name="artifact_test",
        output_dir=tmp_path,
        write_artifacts=True,
    )

    result = runner.run([42])

    # Check files were written
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "histories_combined.csv").exists()
    assert (tmp_path / "seed_0042").is_dir()

    # Check metadata includes paths
    assert "artifact_paths" in result.metadata


def test_stub_engine_deterministic():
    """Test that stub engine produces deterministic results."""
    engine = StubEngine(generations=2)

    key = jr.PRNGKey(123)
    result1 = engine.run_once(key)
    result2 = engine.run_once(key)

    # Should be identical
    assert result1["summary"]["best_fitness"] == result2["summary"]["best_fitness"]
    assert len(result1["history"]) == len(result2["history"])


def test_runner_handles_engine_errors():
    """Test runner handles engine failures gracefully."""

    class FailingEngine:
        def run_once(self, key):
            raise ValueError("Simulated engine failure")

    runner = BenchmarkRunner(
        engine=FailingEngine(),
        write_artifacts=False,
    )

    result = runner.run([1])
    assert len(result.runs) == 1
    assert result.runs[0].status == "error"
    assert "Simulated engine failure" in result.runs[0].error


def test_runner_serialize_history():
    """Test that serialize_history=False drops the history array."""
    engine = StubEngine(generations=2)
    runner = BenchmarkRunner(
        engine=engine,
        write_artifacts=False,
        serialize_history=False,
    )

    result = runner.run([1])
    assert len(result.runs) == 1
    
    run = result.runs[0]
    # The history should be empty
    assert len(run.history) == 0
    import pytest
    assert run.metrics["best_fitness"] == pytest.approx(0.85)
    # The summary from the engine should still have total_evaluations
    assert run.metrics["total_evaluations"] == 2 * 50
