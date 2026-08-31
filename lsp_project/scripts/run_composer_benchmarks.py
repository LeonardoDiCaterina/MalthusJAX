import argparse
import os

import lsp.evaluator.differentiable_evaluator  # noqa
import lsp.evaluator.neural_evaluator  # noqa
import lsp.evaluator.neural_gymnax_evaluator  # noqa

# ---------------------------------------------------------------------------
# Import our new modules to trigger the decorators and register them!
# ---------------------------------------------------------------------------
import lsp.genome.neural  # noqa
import lsp.operators.crossover  # noqa
import lsp.operators.emitter  # noqa
import lsp.operators.neural_mutation  # noqa
import matplotlib.pyplot as plt
from lsp.genome.neural import NeuralGenomeConfig

from malthusjax.composer.composer import Composer


def main():
    parser = argparse.ArgumentParser(description="Run MalthusJAX Composer Benchmarks")
    parser.add_argument("--generations", type=int, default=50, help="Number of generations")
    parser.add_argument("--pop_size", type=int, default=128, help="Population size")
    parser.add_argument("--dry_run", action="store_true", help="Run a fast 2-generation test")
    parser.add_argument(
        "--env",
        type=str,
        default="Acrobot-v1",
        choices=["Acrobot-v1", "Pendulum-v1"],
        help="Gymnax env name",
    )
    args = parser.parse_args()

    gens = 2 if args.dry_run else args.generations
    pop = 32 if args.dry_run else args.pop_size

    # Env-specific dimensions
    if args.env == "Acrobot-v1":
        obs_dim = 6
        action_dim = 3
    else:  # Pendulum-v1
        obs_dim = 3
        action_dim = 1

    composer = Composer.create_default()

    # Define the pipelines as kwarg dictionaries for quick_run
    pipelines = {
        # 1. LSP Neural Pure Evolution on Gym Environment
        "LSP-Neural-PureEvo": dict(
            genome_type="lsp_neural",
            genome_config=NeuralGenomeConfig(
                num_inputs=obs_dim, output_dim=action_dim, length=16, num_ops=1, max_arity=5
            ),
            mutation="lsp_hybrid_mutation",
            crossover="homologous_prefix",
            fitness=f"lsp_neural_gymnax:env_name={args.env},obs_dim={obs_dim},action_dim={action_dim},refine_steps=0",
        ),
        # 2. LSP Hybrid (Gradient) on Gym Environment
        "LSP-Neural-Hybrid": dict(
            genome_type="lsp_neural",
            genome_config=NeuralGenomeConfig(
                num_inputs=obs_dim, output_dim=action_dim, length=16, num_ops=1, max_arity=5
            ),
            mutation="lsp_hybrid_mutation",
            crossover="homologous_prefix",
            fitness=f"lsp_neural_gymnax:env_name={args.env},obs_dim={obs_dim},action_dim={action_dim},refine_steps=5",
        ),
    }

    print(f"\n{'=' * 50}")
    print(f"Running RL Benchmarks ({args.env})")
    print(f"{'=' * 50}")

    # For Gym environments, fitness handles data generation internally
    result = composer.compare(
        pipelines=pipelines,
        fitness=f"gymnax:env_name={args.env}",  # Default fitness if not overridden
        engine_type="ga",
        selection="tournament",
        pop_size=pop,
        generations=gens,
        seeds=(42, 43, 44) if not args.dry_run else (42,),
        shared_initial_population=False,  # Genomes have different shapes, can't share init pop!
    )

    # Plotting
    plt.figure(figsize=(10, 6))
    result.plot_convergence(seed_index=0, ax=plt.gca())
    plt.title(f"RL Pipelines ({args.env}) - Convergence")
    plt.grid(True, alpha=0.3)

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "composer_benchmark_results.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved plot to {output_path}")

    # Summary table
    print("\nSummary Table:")
    for name, metrics in result.summary_table().items():
        print(f"  {name}: {metrics}")


if __name__ == "__main__":
    main()
