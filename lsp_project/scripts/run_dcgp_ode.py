import argparse

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import struct
from lsp.adapters.lsmf_adapter import build_lsmf_engine
from lsp.evaluator.neural_cartesian import NeuralCartesianEvaluatorConfig

# dCGPANN Imports
from lsp.genome.neural_cartesian import (
    ACTIVATIONS_LIST,
    NeuralCartesianGenome,
    NeuralCartesianGenomeConfig,
)

from malthusjax.core.fitness.base import BaseEvaluator


@struct.dataclass
class DifferentialEquationEvaluator(BaseEvaluator[NeuralCartesianGenome, NeuralCartesianEvaluatorConfig, chex.Array]):
    """Evaluates a NeuralCartesianGenome by its residual on the ODE dy/dt = -y, y(0)=1."""

    # We just need the domain T to evaluate on
    data: chex.Array = struct.field(pytree_node=False)

    def evaluate(self, genome: NeuralCartesianGenome, rng: chex.PRNGKey | None = None) -> chex.Array:
        gc = self.config.genome_config
        N = gc.num_inputs
        num_nodes = gc.num_nodes

        # Define the forward pass scalar function y(t)
        def predict_scalar(t: chex.Array) -> chex.Array:
            # Inputs: x is just [t]
            x = jnp.array([t])
            memory = jnp.zeros((N + num_nodes,))
            memory = memory.at[:N].set(x)

            def eval_node(mem: chex.Array, idx: int) -> tuple[chex.Array, None]:
                node_idx = idx - N
                node_args = genome.args[node_idx]
                gathered_inputs = mem[node_args]
                w = genome.weights[node_idx]
                b = genome.biases[node_idx]
                weighted_sum = jnp.sum(w * gathered_inputs) + b
                op_idx = genome.ops[node_idx]
                out_val = jax.lax.switch(op_idx, ACTIVATIONS_LIST, weighted_sum)
                mem = mem.at[idx].set(out_val)
                return mem, None

            final_mem, _ = jax.lax.scan(eval_node, memory, jnp.arange(N, N + num_nodes))
            return final_mem[genome.out_nodes][0]

        # Use JAX native AutoDiff to get dy/dt!
        grad_predict = jax.grad(predict_scalar)

        def compute_residual(t: chex.Array) -> chex.Array:
            y_pred = predict_scalar(t)
            dy_dt = grad_predict(t)
            # ODE: dy/dt + y = 0
            res = (dy_dt + y_pred)**2
            return res

        # Evaluate mean residual over the domain T
        residuals = jax.vmap(compute_residual)(self.data)
        mean_res = jnp.mean(residuals)

        # Boundary condition penalty: y(0) = 1
        y_0 = predict_scalar(0.0)
        bc_penalty = (y_0 - 1.0)**2

        # Total loss
        return mean_res + bc_penalty * 10.0


def main():
    parser = argparse.ArgumentParser(description="Run dCGP ODE Solver Experiment")
    parser.add_argument("--J", type=int, default=10, help="Number of evolutionary iterations")
    parser.add_argument("--K", type=int, default=5, help="Cycles per iteration before Forget step")
    parser.add_argument("--C", type=int, default=5, help="SGD Cooldown epochs per cycle")
    args = parser.parse_args()

    # Time domain t in [0, 2]
    T = jnp.linspace(0.0, 2.0, 100)

    config = NeuralCartesianGenomeConfig(
        num_inputs=1,  # t
        num_outputs=1, # y(t)
        num_rows=1,
        num_cols=15,
        num_ops=len(ACTIVATIONS_LIST),
        max_arity=2,
        levels_back=15,
    )

    eval_config = NeuralCartesianEvaluatorConfig(genome_config=config)
    evaluator = DifferentialEquationEvaluator(config=eval_config, data=T)

    engine = build_lsmf_engine(
        config=config,
        base_evaluator=evaluator,
        pop_size=10,
        elitism=1,
        mutation_rate=0.1,
        num_generations=args.K,
        learning_rate=0.01,
        cooldown_epochs=args.C,
    )

    print("=" * 60)
    print("Physics-Informed dCGP: ODE dy/dt = -y, y(0)=1")
    print(f"Iterations (J): {args.J}")
    print(f"Cycles (K): {args.K}")
    print(f"Epochs (C): {args.C}")
    print("=" * 60)

    key = jax.random.PRNGKey(123)
    state = engine.init_state(key)

    for iteration in range(args.J):
        for cycle in range(args.K):
            state, _ = engine.step(state)

        best_loss = state.best_fitness
        print(f"Iteration {iteration+1:02d}/{args.J} | ODE Loss: {best_loss:.6e}")

        # Forget Step
        k_forget, key = jax.random.split(key)
        dummy_pop = config.init_population(k_forget, size=10)
        new_genes = state.population.genes.replace(
            weights=dummy_pop.genes.weights,
            biases=dummy_pop.genes.biases
        )
        new_pop = state.population.replace(genes=new_genes)
        new_pop = evaluator.evaluate_population(new_pop)
        state = state.replace(population=new_pop, best_fitness=jnp.min(new_pop.fitness))

    # Evaluate best genome vs analytical solution y = e^-t
    best_genome = state.best_genome

    gc = config
    N = gc.num_inputs
    num_nodes = gc.num_nodes
    def predict_single(x: chex.Array) -> chex.Array:
        memory = jnp.zeros((N + num_nodes,))
        memory = memory.at[:N].set(x)
        def eval_node(mem: chex.Array, idx: int) -> tuple[chex.Array, None]:
            node_idx = idx - N
            node_args = best_genome.args[node_idx]
            w = best_genome.weights[node_idx]
            b = best_genome.biases[node_idx]
            gathered_inputs = mem[node_args]
            weighted_sum = jnp.sum(w * gathered_inputs) + b
            op_idx = best_genome.ops[node_idx]
            out_val = jax.lax.switch(op_idx, ACTIVATIONS_LIST, weighted_sum)
            mem = mem.at[idx].set(out_val)
            return mem, None
        final_mem, _ = jax.lax.scan(eval_node, memory, jnp.arange(N, N + num_nodes))
        return final_mem[best_genome.out_nodes][0]

    t_plot = np.linspace(0.0, 2.0, 100)
    y_pred = jax.vmap(predict_single)(jnp.expand_dims(jnp.array(t_plot), axis=-1))
    y_true = np.exp(-t_plot)

    plt.figure(figsize=(8, 5))
    plt.plot(t_plot, y_true, 'k--', label="Analytical (y = e^-t)", linewidth=2)
    plt.plot(t_plot, y_pred, 'r-', label="dCGPANN Evolved Solution", linewidth=2)
    plt.title("dCGPANN ODE Solver")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.grid(True)
    plt.savefig("ode_solution.png")
    print("Saved analytical comparison to ode_solution.png\n")

    # Calculate MSE against analytical solution
    analytical_mse = np.mean((y_pred - y_true)**2)

    # Print Quantitative Summary Table
    print("=" * 60)
    print("FINAL SUMMARY: Physics-Informed dCGP (ODE)")
    print("=" * 60)
    print(f"{'Metric':<30} | {'Value':<20}")
    print("-" * 60)
    print(f"{'Final ODE Residual Loss':<30} | {best_loss:<20.6e}")
    print(f"{'MSE vs Analytical Solution':<30} | {analytical_mse:<20.6e}")
    print("=" * 60)

if __name__ == "__main__":
    main()
