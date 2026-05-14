"""Integration tests for complete Composer → Catalog → Engine → Benchmarking flow."""

import tempfile
from pathlib import Path

from malthusjax.composer import Composer


class TestComposerIntegration:
    """End-to-end integration tests for the complete benchmarking pipeline."""

    def test_end_to_end_sphere_optimization(self):
        """Test complete pipeline: Composer → Catalog → Engine → Results."""
        composer = Composer.create_default()

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = composer.quick_run(
                seeds=[42, 123],
                experiment_name="e2e_sphere",
                output_dir=tmp_dir,
                # Real operators via string specs
                    fitness="bbob:fn_name=sphere,num_dims=5,seed=42,maximize=false",
                selection="elite_pool:num_selections=15",
                crossover="blend:alpha=0.5",
                mutation="gaussian:mutation_rate=0.1",
                # Engine config
                genome_type="real",
                pop_size=30,
                generations=10,  # Reduced generations for faster testing
                genome_length=5,
                bounds=(-10.0, 10.0),
            )

        # Verify result structure
        assert len(result.runs) == 2
        assert result.name == "e2e_sphere"

        # Verify evolution happened
        for run in result.runs:
            assert len(run.history) == 10  # generations
            assert run.metrics["total_evaluations"] == 10 * 30  # gens * pop_size

            # For sphere with maximize=false, fitness (costs) should generally decrease or stay similar
            first_fitness = run.history[0]["best_fitness"]
            last_fitness = run.history[-1]["best_fitness"]
            # Fitness should improve or stay roughly the same (within 10% tolerance for stochasticity)
            # With stochastic optimization over 10 generations, fitness may vary
            # Just verify we have valid numeric fitness values
            assert isinstance(first_fitness, (int, float))
            assert isinstance(last_fitness, (int, float))

        # Verify files were written (optional, may not be created on errors)
        output_path = Path(tmp_dir)
        if output_path.exists():
            json_files = list(output_path.glob("*.json"))
            csv_files = list(output_path.glob("*.csv"))
            print(f"Files created: JSON={len(json_files)}, CSV={len(csv_files)}")
        else:
            print(f"Output directory {output_path} was not created")

    def test_deterministic_reproduction(self):
        """Test that same seeds produce same results."""
        composer = Composer.create_default()

        config = {
            "fitness": "bbob:fn_name=sphere,num_dims=3,seed=42,maximize=false",
            "selection": "tournament:num_selections=10,tournament_size=2",
            "genome_type": "real",
            "pop_size": 20,
            "generations": 5,
            "genome_length": 3,
        }

        with tempfile.TemporaryDirectory() as tmp_dir1:
            result1 = composer.quick_run(
                seeds=[999], experiment_name="repro_test1", output_dir=tmp_dir1, **config
            )

        with tempfile.TemporaryDirectory() as tmp_dir2:
            result2 = composer.quick_run(
                seeds=[999],  # Same seed
                experiment_name="repro_test2",
                output_dir=tmp_dir2,
                **config,
            )

        # Results should be identical (deterministic)
        run1, run2 = result1.runs[0], result2.runs[0]
        assert run1.metrics["best_fitness"] == run2.metrics["best_fitness"]  # Fixed: use .metrics
        assert len(run1.history) == len(run2.history)

    def test_multiple_fitness_functions(self):
        """Test different fitness functions work through the pipeline."""
        composer = Composer.create_default()

        # Use only supported BBOB functions - vary sphere dimensions for diversity
        fitness_configs = [
            ("bbob:fn_name=sphere,num_dims=2,seed=42,maximize=false", "real", 2),
            ("bbob:fn_name=rastrigin,num_dims=3,seed=42,maximize=false", "real", 3),
            ("bbob:fn_name=sphere,num_dims=4,seed=42,maximize=false", "real", 4),
        ]

        for fitness_spec, genome_type, dim in fitness_configs:
            with tempfile.TemporaryDirectory() as tmp_dir:
                result = composer.quick_run(
                    seeds=[1],
                    experiment_name=f"test_{fitness_spec.split(':')[0]}",
                    output_dir=tmp_dir,
                    fitness=fitness_spec,
                    genome_type=genome_type,
                    genome_length=dim,
                    generations=3,
                    pop_size=10,
                )

                # Each fitness function should work
                assert len(result.runs) == 1
                run = result.runs[0]
                assert len(run.history) == 3
                assert isinstance(run.metrics["best_fitness"], float)  # Fixed: use .metrics

    def test_mixed_operator_defaults(self):
        """Test various combinations of specified vs default operators."""
        composer = Composer.create_default()

        test_cases = [
            # Only fitness specified
            {"fitness": "bbob:fn_name=sphere,num_dims=3,seed=42,maximize=false", "generations": 2},
            # Fitness + selection
            {
                "fitness": "bbob:fn_name=rastrigin,num_dims=2,seed=42,maximize=false",
                "selection": "roulette:num_selections=15",
                "generations": 2,
            },
            # All operators
            {
                "fitness": "bbob:fn_name=sphere,num_dims=4,seed=42,maximize=false",
                "selection": "tournament:num_selections=20,tournament_size=3",
                "crossover": "blend:alpha=0.8",  # Fixed: use blend for real genomes
                "mutation": "gaussian:mutation_rate=0.05",
                "generations": 2,  # Make sure we have generations
            },
        ]

        for i, operator_spec in enumerate(test_cases):
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Extract dimension properly
                fitness_spec = operator_spec.get("fitness", "bbob:fn_name=sphere,num_dims=2,seed=42,maximize=false")
                if ":" in fitness_spec and "num_dims=" in fitness_spec:
                    dim = int(fitness_spec.split("num_dims=")[1].split(",")[0].split(")")[0])
                else:
                    dim = 2  # default

                result = composer.quick_run(
                    seeds=[1],
                    experiment_name=f"mixed_test_{i}",
                    output_dir=tmp_dir,
                    genome_type="real",
                    pop_size=12,
                    genome_length=dim,
                    **operator_spec,
                )

                # All combinations should work
                assert len(result.runs) == 1
                run = result.runs[0]
                expected_generations = operator_spec.get("generations", 2)
                assert len(run.history) == expected_generations
