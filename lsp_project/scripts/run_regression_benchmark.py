import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from lsp.adapters.cgp_adapter import build_cgp_engine
from lsp.adapters.lsmf_adapter import build_lsmf_engine
from lsp.adapters.mep_adapter import build_mep_engine
from lsp.evaluator.cartesian import CartesianGPEvaluator, CartesianGPEvaluatorConfig
from lsp.evaluator.neural_cartesian import NeuralCartesianEvaluator, NeuralCartesianEvaluatorConfig

# MEP Imports
from lsp.genome.linear import PrefixGenomeConfig

# dCGPANN Imports
from lsp.genome.neural_cartesian import ACTIVATIONS_LIST, NeuralCartesianGenomeConfig

from malthusjax.core.fitness.linear_gp_evaluator import LinearGPEvaluator, LinearGPEvaluatorConfig

# CGP Imports
from malthusjax.core.genome.cartesian_genome import CartesianGenomeConfig


def generate_nguyen7_data(num_points: int = 100):
    """Nguyen-7: y = ln(x+1) + ln(x^2+1), x in [0, 2]."""
    X = jnp.linspace(0.0, 2.0, num_points).reshape(-1, 1)
    y = jnp.log(X + 1.0) + jnp.log(X**2 + 1.0)
    return X, y

def run_mep(X, y, num_generations, key):
    config = PrefixGenomeConfig(
        length=50,
        num_inputs=1,
        num_ops=6, # standard math ops
        max_arity=2,
    )
    eval_config = LinearGPEvaluatorConfig(num_inputs=config.num_inputs, length=config.length)
    evaluator = LinearGPEvaluator(config=eval_config, data=(X, y))

    engine = build_mep_engine(
        config=config,
        evaluator=evaluator,
        pop_size=50,
        elitism=1,
    )

    state = engine.init_state(key)
    history = []

    for i in range(num_generations):
        state, metrics = engine.step(state)
        history.append(float(state.best_fitness))

    return history

def run_cgp(X, y, num_generations, key):
    config = CartesianGenomeConfig(
        num_inputs=1,
        num_outputs=1,
        num_rows=1,
        num_cols=50,
        num_ops=6,
        max_arity=2,
        levels_back=50,
    )
    eval_config = CartesianGPEvaluatorConfig(genome_config=config)
    evaluator = CartesianGPEvaluator(config=eval_config, data=(X, y))

    engine = build_cgp_engine(
        config=config,
        evaluator=evaluator,
        pop_size=50,
        elitism=1,
        mutation_rate=0.05,
        num_generations=num_generations,
    )

    state = engine.init_state(key)
    history = []

    for i in range(num_generations):
        state, metrics = engine.step(state)
        history.append(float(state.best_fitness))

    return history

def run_dcgpann(X, y, num_generations, key):
    # dCGPANN runs inner SGD cycles, so it evaluates C times per generation.
    # To keep wall-clock roughly fair, we use a smaller graph and a small C.
    config = NeuralCartesianGenomeConfig(
        num_inputs=1,
        num_outputs=1,
        num_rows=1,
        num_cols=20,
        num_ops=len(ACTIVATIONS_LIST),
        max_arity=2,
        levels_back=20,
    )
    eval_config = NeuralCartesianEvaluatorConfig(genome_config=config)
    evaluator = NeuralCartesianEvaluator(config=eval_config, data=(X, y))

    engine = build_lsmf_engine(
        config=config,
        base_evaluator=evaluator,
        pop_size=10,
        elitism=1,
        mutation_rate=0.05,
        num_generations=num_generations,
        learning_rate=0.01,
        cooldown_epochs=1,
    )

    state = engine.init_state(key)
    history = []

    # Forget step configuration
    K_forget = 10

    for i in range(num_generations):
        state, metrics = engine.step(state)
        history.append(float(state.best_fitness))

        if (i + 1) % K_forget == 0:
            k_forget, key = jax.random.split(key)
            dummy_pop = config.init_population(k_forget, size=10)
            new_genes = state.population.genes.replace(
                weights=dummy_pop.genes.weights,
                biases=dummy_pop.genes.biases
            )
            new_pop = state.population.replace(genes=new_genes)
            new_pop = evaluator.evaluate_population(new_pop)
            state = state.replace(
                population=new_pop,
                best_fitness=jnp.min(new_pop.fitness)
            )

    return history

def main():
    X, y = generate_nguyen7_data(100)
    num_generations = 50
    trials = 3

    results = {"MEP": [], "CGP": [], "dCGPANN": []}

    print("Running Comparative Benchmark (Nguyen-7)...")
    for t in range(trials):
        print(f"Trial {t+1}/{trials}")
        k_mep, k_cgp, k_dcgp, base_k = jax.random.split(jax.random.PRNGKey(42 + t), 4)

        results["MEP"].append(run_mep(X, y, num_generations, k_mep))
        results["CGP"].append(run_cgp(X, y, num_generations, k_cgp))
        results["dCGPANN"].append(run_dcgpann(X, y, num_generations, k_dcgp))

    # Plotting
    plt.figure(figsize=(10, 6))
    for name, hist_list in results.items():
        arr = np.array(hist_list)
        mean_curve = np.mean(arr, axis=0)
        std_curve = np.std(arr, axis=0)

        plt.plot(mean_curve, label=name)
        plt.fill_between(range(num_generations), mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

    plt.yscale("log")
    plt.xlabel("Generation")
    plt.ylabel("MSE (Log Scale)")
    plt.title("Comparative Symbolic Regression: Nguyen-7")
    plt.legend()
    plt.grid(True)
    plt.savefig("regression_benchmark.png")
    print("Saved plot to regression_benchmark.png\n")

    # Print Quantitative Summary Table
    print("=" * 60)
    print(f"{'Algorithm':<15} | {'Mean Final MSE':<20} | {'Std Final MSE':<20}")
    print("-" * 60)
    for name, hist_list in results.items():
        arr = np.array(hist_list)
        final_mse = arr[:, -1]
        mean_final = np.mean(final_mse)
        std_final = np.std(final_mse)
        print(f"{name:<15} | {mean_final:<20.6e} | {std_final:<20.6e}")
    print("=" * 60)

if __name__ == "__main__":
    main()
