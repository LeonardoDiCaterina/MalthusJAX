"""MEP Experiment 2 using the native MalthusJAX engine.

Reproduces the second experiment from Oltean & Dumitrescu (2002), expanding
chromosome lengths up to 300 genes to show stability.
"""

import argparse
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from lsp.adapters.mep_adapter import build_mep_engine
from lsp.genome import PrefixGenomeConfig

from malthusjax.core.fitness.linear_gp_evaluator import LinearGPEvaluator, LinearGPEvaluatorConfig


def get_quartic_data():
    """Generates training data for y = x^4 + x^3 + x^2 + x, with a constant feature."""
    X_var = jnp.linspace(-1, 1, 20).reshape(-1, 1)
    Y = (X_var**4 + X_var**3 + X_var**2 + X_var).flatten()  # Fix: must be 1D
    X_const = jnp.ones_like(X_var)
    X = jnp.concatenate([X_var, X_const], axis=1)
    return X, Y


def run_experiment(args):
    print("=" * 60)
    print("MEP Experiment 2 (Native Engine): Success Rate up to L=300")
    print("=" * 60)

    X, Y = get_quartic_data()

    # Test lengths up to 300. We use a step of 40 to avoid 60+ recompilations
    # which would take a very long time during XLA warmup.
    lengths = jnp.arange(10, 310, 40) if not args.dry_run else jnp.array([10, 50])
    num_trials = 100 if not args.dry_run else 10

    success_rates = []
    symbol_counts = []

    for length in lengths.tolist():
        print(f"\nEvaluating length = {length} (running {num_trials} parallel trials)...")

        num_symbols = 3 * length - 2
        symbol_counts.append(num_symbols)

        # 1. Configuration
        config = PrefixGenomeConfig(
            length=length,
            num_inputs=2,  # (x, 1.0)
            num_ops=4,  # +, -, *, /
            max_arity=2,
        )

        # 2. Evaluator
        evaluator_config = LinearGPEvaluatorConfig(num_inputs=2, length=length, maximize=False)
        evaluator = LinearGPEvaluator(config=evaluator_config, data=(X, Y))

        from malthusjax.composer.engine_factory import GeneticEngineAdapter

        # 3. Build native MalthusJAX engine
        engine = build_mep_engine(
            config=config,
            evaluator=evaluator,
            pop_size=30,
            elitism=1,
            crossover_rate=0.7,
        )

        adapter = GeneticEngineAdapter(engine, config, maximize=False)

        # --- Run trials in a loop ---
        # The adapter JIT compiles the engine loop on the first call,
        # so subsequent calls are blazing fast.
        master_key = jax.random.PRNGKey(length)
        trial_keys = jax.random.split(master_key, num_trials)

        successes = []
        for i, key in enumerate(trial_keys):
            # run_once evaluates and executes the full pipeline
            res = adapter.run_once(key)
            best_fit = res["summary"]["best_fitness"]
            successes.append(best_fit < 1e-5)

        rate = jnp.mean(jnp.array(successes)) * 100
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
        color="g",
        label="MEP (Native Engine)",
    )
    plt.xlabel("Number of Symbols in Chromosome")
    plt.ylabel("Success Rate (%)")
    plt.title("MEP Experiment 2 (Native Engine): Success Rate vs Chromosome Length")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    out_path = os.path.join(os.path.dirname(__file__), "..", "mep_experiment_2_engine.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true", help="Run a short test.")
    args = parser.parse_args()
    run_experiment(args)
