import argparse

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from lsp.adapters.dmep_adapter import build_dmep_engine
from lsp.evaluator.neural_linear import NeuralPrefixEvaluator, NeuralPrefixEvaluatorConfig
from lsp.genome.neural_linear import NeuralPrefixGenomeConfig


def generate_nguyen7_data(num_points: int = 100):
    """Nguyen-7: y = ln(x+1) + ln(x^2+1), x in [0, 2]."""
    X = jnp.linspace(0.0, 2.0, num_points).reshape(-1, 1)
    y = jnp.log(X + 1.0) + jnp.log(X**2 + 1.0)
    return X, y

def main():
    parser = argparse.ArgumentParser(description="dMEP Benchmark")
    parser.add_argument("--generations", type=int, default=50, help="Evolutionary generations (J)")
    parser.add_argument("--epochs", type=int, default=10, help="Memetic SGD epochs per generation (C)")
    parser.add_argument("--lr", type=float, default=0.01, help="Adam learning rate")
    args = parser.parse_args()

    print("============================================================")
    print("Differentiable Multi Expression Programming (dMEP)")
    print(f"Generations (J): {args.generations}")
    print(f"SGD Epochs (C): {args.epochs}")
    print("Target: Nguyen-7 Polynomial")
    print("============================================================\n")

    X, y = generate_nguyen7_data()

    # 1. Configuration
    config = NeuralPrefixGenomeConfig(
        length=50,
        num_inputs=1,
        num_ops=6, # standard math ops mapping to ACTIVATIONS_LIST
        max_arity=2,
    )

    eval_config = NeuralPrefixEvaluatorConfig(num_inputs=config.num_inputs, length=config.length)
    base_evaluator = NeuralPrefixEvaluator(config=eval_config, data=(X, y))

    optimizer = optax.adam(learning_rate=args.lr)

    # 2. Build Memetic Engine
    engine = build_dmep_engine(
        config=config,
        base_evaluator=base_evaluator,
        optimizer=optimizer,
        epochs=args.epochs,
        pop_size=30,
        elitism=1,
    )

    # 3. Evolution Loop
    key = jax.random.PRNGKey(42)
    state = engine.init_state(key)

    history_loss = []

    for i in range(args.generations):
        state, metrics = engine.step(state)
        best_loss = float(state.best_fitness)
        history_loss.append(best_loss)

        if (i + 1) % 5 == 0 or i == 0:
            print(f"Generation {i + 1:02d}/{args.generations} | Best MSE: {best_loss:.6e}")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY: dMEP Symbolic Regression")
    print("=" * 60)
    print(f"{'Metric':<30} | {'Value':<20}")
    print("-" * 60)
    print(f"{'Final Best MSE':<30} | {history_loss[-1]:<20.6e}")
    print("=" * 60)

    # 4. Plot Results
    # Get the best genome predictions
    # Note: state.best_genome is a scalar pytree in MalthusJAX 0.x when elitism=1
    best_genome = state.best_genome

    # We need to find the specific instruction that had the best fitness
    # by re-evaluating predict_one and finding the argmin
    all_preds = jax.vmap(base_evaluator.predict_one, in_axes=(None, 0))(best_genome, X)
    Y_bcast = y[:, None]
    squared_errors = jnp.square(all_preds - Y_bcast)
    mse_per_tree = jnp.mean(squared_errors, axis=0)
    best_idx = jnp.argmin(mse_per_tree)

    y_pred = all_preds[:, best_idx]

    plt.figure(figsize=(10, 5))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(history_loss, label="dMEP MSE", color="red")
    plt.yscale("log")
    plt.title("dMEP Convergence")
    plt.xlabel("Generations")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)

    # Fit curve
    plt.subplot(1, 2, 2)
    plt.scatter(X, y, label="True (Nguyen-7)", color="black", alpha=0.5)
    plt.plot(X, y_pred, label=f"dMEP Best (Node {best_idx})", color="red", linewidth=2)
    plt.title("Function Fit")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("dmep_regression.png")
    print("Saved plot to dmep_regression.png")


if __name__ == "__main__":
    main()
