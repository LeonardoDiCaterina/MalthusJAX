"""MEP Experiment 1: Success Rate vs Chromosome Length.

Reproduces the first experiment from Oltean & Dumitrescu (2002):
Analyzes the success rate of the standard MEP algorithm.
Target problem: Quartic polynomial (x^4 + x^3 + x^2 + x).

Parameters (from Table 1):
- Population size: 30
- Number of generations: 50
- Mutation: 2 genes / chromosome
- Crossover type: One-point-crossover
- Crossover probability: 0.7
- Selection: Binary tournament
- Elitism size: 1

We run 100 independent trials for each chromosome length and record
how many trials find a perfect solution (MSE < 1e-5).
"""

import argparse
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from lsp.genome import PrefixGenomeConfig
from lsp.operators.crossover import MEPOnePointCrossover
from lsp.operators.mutation import MEPMicroMutation

from malthusjax.core.fitness.linear_gp_evaluator import LinearGPEvaluator, LinearGPEvaluatorConfig
from malthusjax.core.genome.linear_genome import LinearGenome
from malthusjax.operators.selection.tournament import TournamentSelection


def get_quartic_data():
    """Generates training data for y = x^4 + x^3 + x^2 + x, with a constant feature."""
    # Standard GP practice: 20 points in [-1, 1]
    X_var = jnp.linspace(-1, 1, 20).reshape(-1, 1)
    Y = (X_var**4 + X_var**3 + X_var**2 + X_var).flatten()  # Fix: must be 1D
    # Inject a constant 1.0 as the second input feature
    X_const = jnp.ones_like(X_var)
    X = jnp.concatenate([X_var, X_const], axis=1)
    return X, Y


def run_experiment(args):
    print("=" * 60)
    print("MEP Experiment 1: Success Rate vs Chromosome Length")
    print("=" * 60)

    X, Y = get_quartic_data()

    # We will test a range of chromosome lengths
    lengths = jnp.arange(10, 61, 5) if not args.dry_run else jnp.array([10, 20])
    num_trials = 100 if not args.dry_run else 10

    success_rates = []
    symbol_counts = []

    for length in lengths.tolist():
        print(f"\nEvaluating length = {length} (running {num_trials} parallel trials)...")

        # In this case n = 2 (max_arity = 2)
        # Number_of_Symbols = 3 * Number_of_Genes - 2
        num_symbols = 3 * length - 2
        symbol_counts.append(num_symbols)

        # 1. Configuration
        config = PrefixGenomeConfig(
            length=length,
            num_inputs=2,  # Now 2: (x, 1.0)
            num_ops=4,  # +, -, *, /  (First 4 ops in TENSORGP_FUNCTIONS)
            max_arity=2,
        )

        # 2. Operators matching Table 1
        # Mutation: 2 genes / chromosome -> p_m = 2 / length
        mut_rate = 2.0 / length
        mutation = MEPMicroMutation(mutation_rate=mut_rate, num_offspring=1)

        # Crossover: One-point, p_c = 0.7
        crossover = MEPOnePointCrossover(num_offspring=1)

        # Selection: Binary tournament
        selection = TournamentSelection(num_selections=30, n_elites=1, tournament_size=2)

        # Evaluator
        evaluator_config = LinearGPEvaluatorConfig(num_inputs=2, length=length, maximize=False)
        evaluator = LinearGPEvaluator(config=evaluator_config, data=(X, Y))

        # --- JAX Evolution Loop for a single trial ---
        def run_single_trial(key):
            pop_size = 30
            k_init, k_loop = jax.random.split(key)

            # Initialize population
            k_inits = jax.random.split(k_init, pop_size)
            genomes = jax.vmap(LinearGenome.random_init, in_axes=(0, None))(k_inits, config)

            from malthusjax.core.base import BasePopulation

            pop = BasePopulation(genes=genomes, fitness=jnp.zeros(pop_size), info={})

            # Initial evaluation
            pop = evaluator.evaluate_population(pop)

            def step(pop, key):
                k_sel, k_cross, k_mut, k_cross_prob = jax.random.split(key, 4)

                # Selection & Elitism (Elitism size = 1)
                # TournamentSelection returns (parents, elites)
                parents_idx, elite_idx = selection(k_sel, pop)
                elite_genes = jax.tree_util.tree_map(lambda x: x[elite_idx], pop.genes)

                # We need pop_size - 1 offspring
                num_offspring = pop_size - 1
                parents_idx = parents_idx[:num_offspring]

                # Separate parents into pairs for crossover
                p1_idx = parents_idx
                p2_idx = jnp.roll(parents_idx, 1)  # simple pairing

                p1 = jax.tree_util.tree_map(lambda x: x[p1_idx], pop.genes)
                p2 = jax.tree_util.tree_map(lambda x: x[p2_idx], pop.genes)

                # Apply crossover with probability 0.7
                do_crossover = jax.random.bernoulli(k_cross_prob, p=0.7, shape=(num_offspring,))

                n_cross_keys = crossover.num_keys_per_atomic_operation
                k_cross_block = jax.random.split(k_cross, num_offspring * n_cross_keys)
                k_cross_block = k_cross_block.reshape((num_offspring, n_cross_keys, -1))
                if n_cross_keys == 1:
                    k_cross_block = k_cross_block.squeeze(1)

                def _cross(k, g1, g2, do_cross):
                    # We pass k directly since it now has shape (2,)
                    noise = crossover._generate_noise(jnp.expand_dims(k, 0), config)
                    # _generate_noise expects keys of shape (n_cross_keys, 2).
                    # If n_cross_keys is 1, it expects (1, 2)
                    crossed = crossover._recombine_one(g1, g2, noise, config)
                    ops = jnp.where(do_cross, crossed.ops, g1.ops)
                    args = jnp.where(do_cross, crossed.args, g1.args)
                    return g1.replace(ops=ops, args=args)

                # Fix: properly un-squeeze before vmap if needed
                k_cross_block = jax.random.split(k_cross, num_offspring)  # one key per pair

                def _cross2(k, g1, g2, do_cross):
                    k_reshaped = k.reshape((1, -1))  # shape (1, 2)
                    noise = crossover._generate_noise(k_reshaped, config)
                    crossed = crossover._recombine_one(g1, g2, noise, config)
                    ops = jnp.where(do_cross, crossed.ops, g1.ops)
                    args = jnp.where(do_cross, crossed.args, g1.args)
                    return g1.replace(ops=ops, args=args)

                offspring = jax.vmap(_cross2)(k_cross_block, p1, p2, do_crossover)

                # Apply mutation
                n_mut_keys = mutation.num_keys_per_atomic_operation
                k_mut_block = jax.random.split(k_mut, num_offspring * n_mut_keys)
                k_mut_block = k_mut_block.reshape((num_offspring, n_mut_keys, -1))

                def _mut(k_block, g):
                    noise = mutation._generate_noise(k_block, config)
                    return mutation._mutate_one(g, noise, config)

                offspring = jax.vmap(_mut)(k_mut_block, offspring)

                # Combine elites and offspring
                next_genes = jax.tree_util.tree_map(
                    lambda e, o: jnp.concatenate([e, o], axis=0), elite_genes, offspring
                )

                next_pop = pop.replace(genes=next_genes)
                next_pop = evaluator.evaluate_population(next_pop)

                return next_pop, jnp.min(next_pop.fitness)

            keys = jax.random.split(k_loop, 50)  # Restore to 50 generations (paper parameter)
            final_pop, best_fitnesses = jax.lax.scan(step, pop, keys)

            # The trial is a success if the best fitness ever found is < 1e-5
            global_best = jnp.minimum(jnp.min(pop.fitness), jnp.min(best_fitnesses))
            return global_best < 1e-5

        # --- Run trials in parallel ---
        master_key = jax.random.PRNGKey(length)
        trial_keys = jax.random.split(master_key, num_trials)

        # JIT compile the entire evolution loop!
        successes = jax.jit(jax.vmap(run_single_trial))(trial_keys)

        rate = jnp.mean(successes) * 100
        success_rates.append(rate.item())
        print(f"  -> Success Rate: {rate:.2f}% (Symbols: {num_symbols})")

    # --- Plotting ---
    plt.figure(figsize=(8, 5))
    plt.plot(
        symbol_counts,
        success_rates,
        marker="o",
        linestyle="-",
        linewidth=2,
        color="b",
        label="MEP (Our Impl)",
    )
    plt.xlabel("Number of Symbols in Chromosome")
    plt.ylabel("Success Rate (%)")
    plt.title("MEP Experiment 1: Success Rate vs Chromosome Length")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    out_path = os.path.join(os.path.dirname(__file__), "..", "mep_experiment_1.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true", help="Run a short test.")
    args = parser.parse_args()
    run_experiment(args)
