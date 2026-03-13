"""Test sign normalization in ComparisonResult for cross-backend comparisons."""

import pytest

from malthusjax.composer import Composer


class TestSignNormalization:
    """Test automatic sign normalization between MalthusJAX and evosax backends."""

    def test_negate_map_auto_built(self):
        """Test that negate_map is automatically built based on backend type."""
        composer = Composer.create_default()

        pipelines = {
            "malthusjax_ga": {"backend": "malthusjax", "crossover": "blend:alpha=0.5"},
            "evosax_ga": {"backend": "evosax", "evosax_strategy": "SimpleGA"},
        }

        result = composer.compare(
            pipelines=pipelines,
            seeds=(42,),  # Single seed for faster testing
            fitness="sphere:dim=5",
            pop_size=20,
            generations=5,
            shared_initial_population=True,
        )

        # Evosax should be negated (its raw fitness is positive), MalthusJAX already reports
        # negative(?) fitness values (lower is better) so it should not be flipped.
        assert result.negate_map["malthusjax_ga"] is False
        assert result.negate_map["evosax_ga"] is True

    def test_summary_table_sign_normalization(self):
        """Test that summary_table applies sign normalization to fitness values."""
        composer = Composer.create_default()

        pipelines = {
            "malthusjax_ga": {
                "backend": "malthusjax",
                "crossover": "blend:alpha=0.5",
                "selection": "tournament:num_selections=10,tournament_size=3",
                "mutation": "gaussian:mutation_rate=0.1",
            },
            "evosax_ga": {"backend": "evosax", "evosax_strategy": "SimpleGA"},
        }

        result = composer.compare(
            pipelines=pipelines,
            seeds=(42,),
            fitness="sphere:dim=5",
            pop_size=20,
            generations=10,
            genome_length=5,
            shared_initial_population=True,
        )

        summary = result.summary_table()

        # Check the negate_map is correct
        assert result.negate_map["malthusjax_ga"] is False
        assert result.negate_map["evosax_ga"] is True

        # Both backends should now have fitness in "lower is better" convention (negative values)
        malthusjax_fitness = summary["malthusjax_ga"]["best_fitness"]
        evosax_fitness = summary["evosax_ga"]["best_fitness"]

        # Both should be negative (normalised to 'lower is better')
        assert malthusjax_fitness < 0, f"malthusjax best_fitness >= 0: {malthusjax_fitness}"
        assert evosax_fitness < 0, f"evosax best_fitness >= 0: {evosax_fitness}"

    def test_convergence_data_sign_normalization(self):
        """Test that convergence_data applies sign normalization to history."""
        composer = Composer.create_default()

        pipelines = {
            "malthusjax_ga": {
                "backend": "malthusjax",
                "crossover": "blend:alpha=0.5",
                "selection": "tournament:num_selections=10,tournament_size=3",
                "mutation": "gaussian:mutation_rate=0.1",
            },
            "evosax_ga": {"backend": "evosax", "evosax_strategy": "SimpleGA"},
        }

        result = composer.compare(
            pipelines=pipelines,
            seeds=(42,),
            fitness="sphere:dim=5",
            pop_size=20,
            generations=5,
            genome_length=5,
            shared_initial_population=True,
        )

        conv_data = result.convergence_data(seed_index=0)

        # Check that both histories are present
        malthusjax_history = conv_data["malthusjax_ga"]
        evosax_history = conv_data["evosax_ga"]

        assert len(malthusjax_history) > 0
        assert len(evosax_history) > 0

        # All fitness values should be negative (lower is better convention)
        # Evosax reports a positive minimisation objective, so it is negated during normalization.
        for entry in malthusjax_history:
            assert entry["best_fitness"] < 0, f"MalthusJAX fitness should be negative: {entry}"

        for entry in evosax_history:
            assert entry["best_fitness"] < 0, f"Evosax fitness should be negative: {entry}"

    def test_plot_convergence_uses_normalized_data(self):
        """Test that plot_convergence uses the sign-normalized data."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        composer = Composer.create_default()

        pipelines = {
            "malthusjax_ga": {
                "backend": "malthusjax",
                "crossover": "blend:alpha=0.5",
                "selection": "tournament:num_selections=10,tournament_size=3",
                "mutation": "gaussian:mutation_rate=0.1",
            },
            "evosax_ga": {"backend": "evosax", "evosax_strategy": "SimpleGA"},
        }

        result = composer.compare(
            pipelines=pipelines,
            seeds=(42,),
            fitness="sphere:dim=5",
            pop_size=20,
            generations=5,
            genome_length=5,
            shared_initial_population=True,
        )

        fig, ax = plt.subplots()
        result.plot_convergence(seed_index=0, ax=ax)

        # The plot should have been created without errors
        # and should use the normalized data (tested implicitly via convergence_data)
        assert len(ax.get_lines()) == 2  # Two pipelines = two lines
        plt.close(fig)

    def test_from_toml_sign_normalization(self):
        """Test that from_toml also builds negate_map correctly."""
        # Create a minimal TOML content for testing
        toml_content = """
[experiment]
name = "sign_test"

[experiment.shared]
fitness = "sphere:dim=5"
pop_size = 20
generations = 3
genome_length = 5
seeds = [42]

[pipelines.malthusjax_blend]
backend = "malthusjax"
crossover = "blend:alpha=0.5"
selection = "tournament:num_selections=10,tournament_size=3"
mutation = "gaussian:mutation_rate=0.1"

[pipelines.evosax_simple]
backend = "evosax"
evosax_strategy = "SimpleGA"
"""

        # Write to temporary file
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            temp_path = f.name

        try:
            result = Composer.from_toml(temp_path, shared_initial_population=True)

            # Check negate_map is built correctly
            assert result.negate_map["malthusjax_blend"] is False
            assert result.negate_map["evosax_simple"] is True

            # Check summary uses normalized values
            summary = result.summary_table()
            assert summary["malthusjax_blend"]["best_fitness"] < 0
            assert summary["evosax_simple"]["best_fitness"] < 0

        finally:
            os.unlink(temp_path)

    def test_manual_negate_on_top_of_auto(self):
        """Test that manual negate parameter works on top of automatic normalization."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        composer = Composer.create_default()

        pipelines = {
            "malthusjax_ga": {
                "backend": "malthusjax",
                "crossover": "blend:alpha=0.5",
                "selection": "tournament:num_selections=10,tournament_size=3",
                "mutation": "gaussian:mutation_rate=0.1",
            },
        }

        result = composer.compare(
            pipelines=pipelines,
            seeds=(42,),
            fitness="sphere:dim=5",
            pop_size=20,
            generations=5,
            genome_length=5,
            shared_initial_population=True,
        )

        fig, (ax1, ax2) = plt.subplots(1, 2)

        # Plot with automatic normalization only
        result.plot_convergence(seed_index=0, ax=ax1, title="Auto-normalized")

        # Plot with extra manual negation on top
        result.plot_convergence(
            seed_index=0, ax=ax2, title="Auto + Manual negate", negate={"malthusjax_ga": True}
        )

        # Get the y-data from both plots
        auto_y = ax1.get_lines()[0].get_ydata()
        manual_y = ax2.get_lines()[0].get_ydata()

        # The second plot should have opposite sign from first
        assert len(auto_y) == len(manual_y)
        for i in range(len(auto_y)):
            assert abs(auto_y[i] + manual_y[i]) < 1e-10, f"not negated at idx {i}"

        plt.close(fig)
