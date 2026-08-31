"""Fair Comparison Runner: LSP NeuralGenome vs. TensorNEAT on Gymnax.

Runs three configurations:
1. TensorNEAT (Baseline graph-based NAS)
2. LSP Pure Evo (Our linear NAS, no gradients)
3. LSP Hybrid (Our linear NAS, backprop weights)

Uses Composer.compare() to run across multiple seeds.
"""

import argparse

from lsp.evaluator.neural_gymnax_evaluator import NeuralGymnaxEvaluator, NeuralGymnaxEvaluatorConfig
from lsp.genome.neural import NeuralGenomeConfig
from lsp.operators.crossover import HomologousPrefixCrossover
from lsp.operators.neural_mutation import ArchitectureMutation, HybridMutation, WeightMutation
from lsp.operators.selection import LinearTournamentSelection

from malthusjax.composer import Composer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--pop", type=int, default=256)
    parser.add_argument("--gens", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--dry_run", action="store_true", help="Run 1 seed for 5 gens")
    return parser.parse_args()


def main():
    args = parse_args()

    seeds = list(range(42, 42 + args.seeds))
    gens = 5 if args.dry_run else args.gens
    pop = args.pop

    # Dimensions depend on the environment
    if args.env == "CartPole-v1":
        obs_dim, action_dim = 4, 2
    elif args.env == "Pendulum-v1":
        obs_dim, action_dim = 3, 1
    else:
        raise ValueError(f"Unknown env: {args.env}")

    composer = Composer.create_default()

    # Base configs for LSP
    base_genome_cfg = NeuralGenomeConfig(
        length=16,
        num_inputs=obs_dim,
        num_ops=1,
        max_arity=5,
        activation="tanh",
        output_dim=action_dim,
    )

    base_eval_cfg = NeuralGymnaxEvaluatorConfig(
        env_name=args.env,
        obs_dim=obs_dim,
        action_dim=action_dim,
        activation="tanh",
    )

    result = composer.compare(
        pipelines={
            # 1. TensorNEAT Baseline
            "TensorNEAT (NEAT)": dict(
                backend="tensorneat",
                tensorneat_problem=f"GymNaxEnv:env_name={args.env}",
                tensorneat_num_inputs=obs_dim,
                tensorneat_num_outputs=action_dim,
            ),
            # 2. LSP Pure Evolution (apples-to-apples with NEAT)
            "LSP NeuralGenome (Pure Evo)": dict(
                genome_config=base_genome_cfg,
                fitness=NeuralGymnaxEvaluator.create(base_eval_cfg),
                crossover=HomologousPrefixCrossover(),
                mutation=HybridMutation(
                    arch_mutation=ArchitectureMutation(p_input_start=0.1, p_input_end=0.9),
                    weight_mutation=WeightMutation(mutation_rate=0.1, sigma=0.1),
                ),
                selection=LinearTournamentSelection(num_selections=pop, tournament_size=3),
            ),
            # 3. LSP Hybrid (Architecture Evolution + Gradient Weights)
            "LSP NeuralGenome (Hybrid)": dict(
                genome_config=base_genome_cfg,
                fitness=NeuralGymnaxEvaluator.create(
                    base_eval_cfg.replace(refine_steps=5, refine_lr=1e-2)
                ),
                crossover=HomologousPrefixCrossover(),
                mutation=ArchitectureMutation(p_input_start=0.1, p_input_end=0.9),
                selection=LinearTournamentSelection(num_selections=pop, tournament_size=3),
            ),
        },
        seeds=seeds,
        shared_initial_population=False,  # TensorNEAT and LSP genomes are incompatible
        pop_size=pop,
        generations=gens,
    )

    print("\n--- Summary Table ---")
    summary = result.summary_table()
    import pandas as pd

    print(pd.DataFrame(summary).T)

    # Save convergence plot
    plot_path = f"results_comparison_{args.env}.png"
    ax = result.plot_convergence()
    ax.figure.savefig(plot_path)
    print(f"\nSaved convergence plot to {plot_path}")


if __name__ == "__main__":
    main()
