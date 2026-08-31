import argparse

import jax
import jax.numpy as jnp
from lsp.adapters.lsmf_adapter import build_lsmf_engine
from lsp.evaluator.neural_cartesian import NeuralCartesianEvaluator, NeuralCartesianEvaluatorConfig
from lsp.genome.neural_cartesian import NeuralCartesianGenomeConfig


def generate_regression_data(num_points: int = 100):
    """Generate simple regression data (Quartic Polynomial)."""
    # X in [-1, 1]
    X = jnp.linspace(-1, 1, num_points).reshape(-1, 1)

    # y = x^4 + x^3 + x^2 + x
    y = X**4 + X**3 + X**2 + X
    return X, y

def main():
    parser = argparse.ArgumentParser(description="Run dCGPANN LSMF Regression Experiment")
    parser.add_argument("--trials", type=int, default=5, help="Number of independent trials")
    parser.add_argument("--J", type=int, default=10, help="Number of evolutionary iterations")
    parser.add_argument("--K", type=int, default=10, help="Cycles per iteration before Forget step")
    parser.add_argument("--C", type=int, default=1, help="SGD Cooldown epochs per cycle")
    args = parser.parse_args()

    # Data
    X, y = generate_regression_data()

    # Network topology: Feed-Forward Template
    # 4 layers of 10 nodes (to mimic the paper's default setup roughly)
    from lsp.genome.neural_cartesian import ACTIVATIONS_LIST
    config = NeuralCartesianGenomeConfig(
        num_inputs=1,
        num_outputs=1,
        num_rows=10,
        num_cols=4,
        num_ops=len(ACTIVATIONS_LIST),
        max_arity=2, # fully connected feed forward usually implies high arity, but we stick to standard CGP for now
        levels_back=3,
    )

    eval_config = NeuralCartesianEvaluatorConfig(genome_config=config)
    evaluator = NeuralCartesianEvaluator(config=eval_config, data=(X, y))

    engine = build_lsmf_engine(
        config=config,
        base_evaluator=evaluator,
        pop_size=10, # small pop for demo speed
        elitism=1,
        mutation_rate=0.05,
        num_generations=args.K,
        learning_rate=0.01,
        cooldown_epochs=args.C,
    )

    print("=" * 60)
    print("dCGPANN LSMF Experiment (Quartic Polynomial)")
    print(f"Iterations (J): {args.J}")
    print(f"Cycles (K): {args.K}")
    print(f"Epochs (C): {args.C}")
    print("=" * 60)

    for trial in range(args.trials):
        print(f"\n--- Trial {trial+1}/{args.trials} ---")
        key = jax.random.PRNGKey(42 + trial)

        # Init engine state
        state = engine.init_state(key)

        for iteration in range(args.J):
            # Run K cycles
            for cycle in range(args.K):
                state, out = engine.step(state)

            best_fitness = state.best_fitness
            print(f"  Iteration {iteration+1:02d}/{args.J} | Best MSE: {best_fitness:.6e}")

            # The Forget Step
            # Re-initialize weights using random_init logic, keeping topology intact
            # We draw a new key to avoid deterministic repetitions
            k_forget, key = jax.random.split(key)

            # Create a dummy fresh genome to steal its weights
            dummy_population = config.init_population(k_forget, size=10)

            # Replace the continuous parameters of the current population
            # NOTE: We preserve state.population.genes.ops, args, out_nodes!
            new_genes = state.population.genes.replace(
                weights=dummy_population.genes.weights,
                biases=dummy_population.genes.biases
            )
            new_pop = state.population.replace(genes=new_genes)

            # Since fitness is out of date after randomizing weights, we evaluate
            # Note: We must dispatch evaluation manually using our evaluator

            k_eval, key = jax.random.split(key)

            # We use the LSMF evaluator directly to do a quick epoch, or the base evaluator
            # Since the forget step resets the loss, we should use base_evaluator
            new_pop = evaluator.evaluate_population(new_pop)

            # Update state with forgotten population
            # We reset best_fitness to the new minimum so it doesn't remember old weights' fitness
            state = state.replace(
                population=new_pop,
                best_fitness=jnp.min(new_pop.fitness)
            )

if __name__ == "__main__":
    main()
