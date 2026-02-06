#!/usr/bin/env python3
"""
Quick reference examples for the parameter sweep scripts.

Run any of these examples as templates for your own sweeps.
"""

# ============================================================================
# EXAMPLE 1: Quick Test
# ============================================================================
# Good for: Testing the pipeline, debugging
# Runtime: ~5-10 minutes
# Command:
# python scripts/parameter_sweep_two_engines.py \
#     --sample 2 \
#     --seeds 1 \
#     --pop-size 20 \
#     --generations 20

# ============================================================================
# EXAMPLE 2: Standard Sweep
# ============================================================================
# Good for: Getting reliable results, typical use case
# Runtime: ~2-3 hours
# Command:
# python scripts/parameter_sweep_two_engines.py \
#     --pop-size 50 \
#     --generations 100 \
#     --seeds 1 2 3

# ============================================================================
# EXAMPLE 3: Comprehensive Sweep
# ============================================================================
# Good for: Publication-quality results, thorough comparison
# Runtime: ~8-10 hours
# Command:
# python scripts/parameter_sweep_two_engines.py \
#     --pop-size 100 \
#     --generations 200 \
#     --seeds 1 2 3 4 5

# ============================================================================
# EXAMPLE 4: High-Performance Sweep
# ============================================================================
# Good for: Large-scale experiments, parameter optimization studies
# Runtime: ~24+ hours
# Command:
# python scripts/parameter_sweep_two_engines.py \
#     --pop-size 200 \
#     --generations 500 \
#     --seeds 1 2 3 4 5 6 7 8 9 10 \
#     --output-dir results/large_sweep

# ============================================================================
# EXAMPLE 5: Analysis Only
# ============================================================================
# Good for: Analyzing existing results without re-running
# Runtime: < 1 minute
# Command:
# python scripts/analyze_sweep.py \
#     --results results/parameter_sweep/sweep_results.json \
#     --csv

# ============================================================================
# PYTHON API USAGE
# ============================================================================

if __name__ == "__main__":
    from scripts.parameter_sweep_two_engines import ParameterSweep
    from scripts.analyze_sweep import SweepAnalyzer

    # ========================================================================
    # API Example 1: Custom sweep with programmatic configuration
    # ========================================================================

    # Create sweep with custom parameters
    sweep = ParameterSweep(
        output_dir="results/my_custom_sweep",
        pop_size=75,
        generations=150,
        seeds=[42, 123, 456],  # Custom seeds
    )

    # Run sweep
    engine1_results, engine2_results = sweep.run_sweep()

    # Save results
    sweep.save_results(engine1_results, engine2_results)

    # ========================================================================
    # API Example 2: Analyzing saved results
    # ========================================================================

    # Load and analyze results
    analyzer = SweepAnalyzer(
        results_file="results/my_custom_sweep/sweep_results.json"
    )

    # Print detailed analysis
    analyzer.print_detailed_comparison()

    # Print sensitivity analysis
    analyzer.print_hyperparameter_sensitivity()

    # Export to CSV
    analyzer.save_csv_report(
        output_file="results/my_custom_sweep/analysis.csv"
    )

    # ========================================================================
    # API Example 3: Custom engine configurations
    # ========================================================================

    # Subclass to create custom configurations
    from scripts.parameter_sweep_two_engines import ParameterSweep, OperatorConfig

    class CustomSweep(ParameterSweep):
        """Custom sweep with different engine configurations."""

        def get_engine1_configs(self):
            """Override for binary GA configuration."""
            configs = []

            # Binary GA with different mutation rates
            for mutation_rate in [0.01, 0.03, 0.05, 0.1]:
                config = OperatorConfig(
                    selection=f"tournament:num_selections={self.pop_size},tournament_size=3",
                    crossover="single_point:num_offspring=2",
                    mutation=f"bitflip:mutation_rate={mutation_rate}",
                    fitness="knapsack",
                )
                configs.append(config)

            return configs

        def get_engine2_configs(self):
            """Override for different real GA configuration."""
            configs = []

            # Real GA with roulette selection
            for alpha in [0.2, 0.5, 0.8]:
                for eta in [10.0, 20.0, 30.0]:
                    config = OperatorConfig(
                        selection=f"roulette:num_selections={self.pop_size}",
                        crossover=f"blend:alpha={alpha}",
                        mutation=f"ball:mutation_rate=0.1,mutation_strength=0.5",
                        fitness="sphere:dim=10",
                    )
                    configs.append(config)

            return configs

    # Use custom sweep
    custom_sweep = CustomSweep(
        output_dir="results/custom_configuration",
        pop_size=50,
        generations=100,
        seeds=[1, 2, 3],
    )

    results1, results2 = custom_sweep.run_sweep()
    custom_sweep.save_results(results1, results2)

    print("✅ Custom sweep complete!")
