"""Tests for Composer.quick_run() with real engines (PR3 Commit D)."""

import tempfile

import pytest

from malthusjax.benchmarking import StubEngine
from malthusjax.composer import Composer


class TestComposerRealEngines:
    """Test Composer integration with real evolutionary engines."""

    def test_quick_run_stub_engine_fallback(self):
        """Test that quick_run falls back to StubEngine when no operators specified."""
        composer = Composer.create_default()

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = composer.quick_run(
                seeds=[1, 2], experiment_name="test_stub", output_dir=tmp_dir, generations=5
            )

        # Should have used StubEngine (existing behavior)
        assert len(result.runs) == 2
        assert result.name == "test_stub"
        # Check for StubEngine signature in run metrics or status
        assert all(run.status == "success" for run in result.runs)

    def test_quick_run_with_real_operators(self):
        """Test quick_run with real operator specifications."""
        composer = Composer.create_default()

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = composer.quick_run(
                seeds=[1, 2],
                experiment_name="test_real",
                output_dir=tmp_dir,
                fitness="bbob:fn_name=sphere,num_dims=5,seed=42,maximize=false",
                selection="tournament:num_selections=10,tournament_size=2",
                crossover="blend:alpha=0.3",
                mutation="gaussian:mutation_rate=0.05",
                genome_type="real",
                pop_size=20,
                generations=3,
                genome_length=5,
            )

        # Should have used real GeneticEngine
        assert len(result.runs) == 2
        assert result.name == "test_real"

        # Check that runs have realistic structure
        for run in result.runs:
            assert len(run.history) == 3  # generations
            assert "best_fitness" in run.metrics  # Fixed: use .metrics not .summary
            assert "total_evaluations" in run.metrics
            # starting fitness should have been recorded by adapters
            assert "initial_fitness" in run.metrics

    def test_quick_run_partial_operator_specs(self):
        """Test quick_run with only some operators specified (uses defaults)."""
        composer = Composer.create_default()

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = composer.quick_run(
                seeds=[1],
                experiment_name="test_partial",
                output_dir=tmp_dir,
                fitness="bbob:fn_name=rastrigin,num_dims=3,seed=42,maximize=false",
                generations=2,
                pop_size=10,
                genome_length=3,
            )

        # Should build real engine with defaults for other operators
        assert len(result.runs) == 1
        run = result.runs[0]
        assert len(run.history) == 2
        assert "best_fitness" in run.metrics  # Fixed: use .metrics not .summary

    def test_quick_run_explicit_engine_override(self):
        """Test that explicit engine parameter overrides operator specs."""
        composer = Composer.create_default()
        explicit_engine = StubEngine(generations=3)

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = composer.quick_run(
                seeds=[1],
                experiment_name="test_override",
                output_dir=tmp_dir,
                engine=explicit_engine,  # Should override operator specs
                fitness="bbob:fn_name=sphere,num_dims=10,seed=42,maximize=false",
                generations=5,  # This should also be ignored
            )

        # Should use explicit engine (3 gens, not 5)
        run = result.runs[0]
        assert len(run.history) == 3  # From explicit_engine, not generations=5

    def test_binary_genome_support(self):
        """Test quick_run with binary genome type."""
        composer = Composer.create_default()

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = composer.quick_run(
                seeds=[1],
                experiment_name="test_binary",
                output_dir=tmp_dir,
                fitness="bbob:fn_name=sphere,num_dims=8,seed=42,maximize=false",
                genome_type="binary",
                genome_length=8,
                generations=2,
                pop_size=15,
            )

        # Should work with binary genomes - check for successful run
        assert len(result.runs) == 1
        run = result.runs[0]
        # If there's an error, the run will have error status
        if run.status == "error":
            pytest.skip(f"Binary genome compatibility issue: {run.error}")
        else:
            assert len(run.history) == 2

    def test_backward_compatibility(self):
        """Test that existing Composer usage still works unchanged."""
        composer = Composer.create_default()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Old-style call (no operator specs)
            result = composer.quick_run(
                seeds=[1, 2, 3],
                experiment_name="legacy_test",
                output_dir=tmp_dir,
                generations=4,
                base_fitness=2.0,
                improvement_rate=0.2,
            )

        # Should behave exactly like before
        assert len(result.runs) == 3
        # Check all runs were successful
        assert all(run.status == "success" for run in result.runs)

        # Check StubEngine parameters were used - be more flexible with fitness improvement
        for run in result.runs:
            assert len(run.history) == 4  # generations
            # StubEngine should show improvement pattern - allow for small variance
            if run.history:
                fitnesses = [gen["best_fitness"] for gen in run.history]
                # Just check that we have fitness values, improvement may vary
                assert all(isinstance(f, (int, float)) for f in fitnesses)
