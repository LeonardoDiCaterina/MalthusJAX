"""Cartesian Genetic Programming (CGP) Experiment.

Reproduces symbolic regression using a 1-D Cartesian Genetic Programming
genome. The 1-D variant (num_rows=1) with full connectivity (levels_back=num_cols)
is highly effective and equivalent to a generic DAG.

Follows the 1 + λ ES strategy outlined in Miller & Thomson (2000).
"""

import argparse
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from lsp.adapters.cgp_adapter import build_cgp_engine
from lsp.evaluator.cartesian import CartesianGPEvaluator, CartesianGPEvaluatorConfig

from malthusjax.core.genome.cartesian_genome import CartesianGenomeConfig


def get_quartic_data():
    """Generates training data for y = x^4 + x^3 + x^2 + x, with a constant feature."""
    X_var = jnp.linspace(-1, 1, 20).reshape(-1, 1)
    Y = (X_var**4 + X_var**3 + X_var**2 + X_var).flatten()  # 1D targets
    X_const = jnp.ones_like(X_var)
    X = jnp.concatenate([X_var, X_const], axis=1)
    return X, Y


def run_experiment(args):
    print("=" * 60)
    print("Cartesian GP Experiment (1-D Variant)")
    print("=" * 60)

    X, Y = get_quartic_data()

    # CGP typically requires significantly more generations than MEP due to relying
    # entirely on point mutations. 500 is far too few for convergence; 5000 is a good baseline.
    num_generations = 5000 if not args.dry_run else 10

    # We will test a few different chromosome lengths.
    lengths = jnp.array([10, 30, 50, 100]) if not args.dry_run else jnp.array([10, 20])
    num_trials = 20 if not args.dry_run else 5

    success_rates = []

    for length in lengths.tolist():
        print(f"\nEvaluating num_cols = {length} (running {num_trials} trials)...")

        # 1. CGP Configuration (1D variant)
        config = CartesianGenomeConfig(
            num_rows=1,
            num_cols=length,
            num_inputs=2,  # (x, 1.0)
            num_outputs=1,  # single target
            num_ops=4,  # +, -, *, /
            max_arity=2,
            levels_back=-1,  # full connectivity
        )

        # 2. Evaluator
        evaluator_config = CartesianGPEvaluatorConfig(genome_config=config, maximize=False)
        evaluator = CartesianGPEvaluator(config=evaluator_config, data=(X, Y))

        from malthusjax.composer.engine_factory import GeneticEngineAdapter

        # 3. Build native MalthusJAX CGP engine (1+4 ES)
        # Miller typically mutates ~1-3% of the genome. We'll target ~2 genes mutated per chromosome.
        total_genes = config.num_nodes * (config.max_arity + 1) + config.num_outputs
        mut_rate = 3.0 / total_genes

        engine = build_cgp_engine(
            config=config,
            evaluator=evaluator,
            pop_size=5,  # 1 elite + 4 offspring (1+4 ES)
            elitism=1,
            mutation_rate=mut_rate,
            num_generations=num_generations,
        )

        adapter = GeneticEngineAdapter(engine, config, maximize=False)

        # --- Run trials in a loop ---
        master_key = jax.random.PRNGKey(length)
        trial_keys = jax.random.split(master_key, num_trials)

        successes = []
        for i, key in enumerate(trial_keys):
            res = adapter.run_once(key)
            best_fit = res["summary"]["best_fitness"]

            # CGP returns MSE, MEP returns best instruction MSE.
            # We consider it a success if MSE < 1e-5
            is_success = best_fit < 1e-5
            successes.append(is_success)

            if args.verbose:
                print(f"Trial {i}: MSE = {best_fit:.5f} ({'Success' if is_success else 'Fail'})")

        rate = jnp.mean(jnp.array(successes)) * 100
        success_rates.append(rate.item())
        print(f"  -> Success Rate: {rate:.2f}% (Generations: {num_generations})")

    # --- Plotting ---
    plt.figure(figsize=(8, 5))
    plt.plot(
        lengths,
        success_rates,
        marker="o",
        linestyle="-",
        linewidth=2,
        color="b",
        label="CGP (1+4 ES)",
    )
    plt.xlabel("Number of Columns (Nodes)")
    plt.ylabel("Success Rate (%)")
    plt.title(f"CGP Experiment: Success Rate vs Length ({num_generations} Gens)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    out_path = os.path.join(os.path.dirname(__file__), "..", "cgp_experiment.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true", help="Run a short test.")
    parser.add_argument("--verbose", action="store_true", help="Print trial details.")
    args = parser.parse_args()
    run_experiment(args)
