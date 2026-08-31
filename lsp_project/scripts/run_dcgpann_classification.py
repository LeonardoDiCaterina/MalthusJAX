import argparse

import chex
import jax
import jax.numpy as jnp
from flax import struct
from lsp.adapters.lsmf_adapter import build_lsmf_engine
from lsp.evaluator.neural_cartesian import NeuralCartesianEvaluatorConfig

# dCGPANN Imports
from lsp.genome.neural_cartesian import (
    ACTIVATIONS_LIST,
    NeuralCartesianGenome,
    NeuralCartesianGenomeConfig,
)

# Use sklearn for dataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from malthusjax.core.fitness.base import BaseEvaluator


@struct.dataclass
class ClassificationNeuralCartesianEvaluator(BaseEvaluator[NeuralCartesianGenome, NeuralCartesianEvaluatorConfig, tuple[chex.Array, chex.Array]]):
    """Evaluates a NeuralCartesianGenome on a classification dataset using Binary Cross Entropy."""

    def evaluate(
        self, genome: NeuralCartesianGenome, rng: chex.PRNGKey | None = None
    ) -> chex.Array:
        gc = self.config.genome_config
        N = gc.num_inputs
        num_nodes = gc.num_nodes

        def predict_single(x: chex.Array) -> chex.Array:
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

            final_mem, _ = jax.lax.scan(
                eval_node,
                memory,
                jnp.arange(N, N + num_nodes),
            )

            # Raw output
            raw = final_mem[genome.out_nodes][0]
            # Sigmoid activation for binary classification
            prob = jax.nn.sigmoid(raw)
            return prob

        probs = jax.vmap(predict_single)(self.data[0])
        targets = self.data[1]

        # Binary Cross Entropy Loss
        # add epsilon to avoid log(0)
        eps = 1e-7
        probs = jnp.clip(probs, eps, 1.0 - eps)
        bce = -jnp.mean(targets * jnp.log(probs) + (1 - targets) * jnp.log(1 - probs))

        return bce


def main():
    parser = argparse.ArgumentParser(description="Run dCGPANN Classification Experiment")
    parser.add_argument("--J", type=int, default=10, help="Number of evolutionary iterations")
    parser.add_argument("--K", type=int, default=10, help="Cycles per iteration before Forget step")
    parser.add_argument("--C", type=int, default=5, help="SGD Cooldown epochs per cycle")
    args = parser.parse_args()

    print("Loading Breast Cancer Dataset...")
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Cast to jnp arrays
    X_train, y_train = jnp.array(X_train), jnp.array(y_train)
    X_test, y_test = jnp.array(X_test), jnp.array(y_test)

    num_features = X_train.shape[1]

    config = NeuralCartesianGenomeConfig(
        num_inputs=num_features,
        num_outputs=1,
        num_rows=1,
        num_cols=20,
        num_ops=len(ACTIVATIONS_LIST),
        max_arity=4, # Give it access to 4 previous nodes
        levels_back=20,
    )

    eval_config = NeuralCartesianEvaluatorConfig(genome_config=config)
    evaluator = ClassificationNeuralCartesianEvaluator(config=eval_config, data=(X_train, y_train))

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
    print("dCGPANN LSMF Classification (Breast Cancer)")
    print(f"Features: {num_features}, Training samples: {len(X_train)}")
    print(f"Iterations (J): {args.J}")
    print(f"Cycles (K): {args.K}")
    print(f"Epochs (C): {args.C}")
    print("=" * 60)

    key = jax.random.PRNGKey(42)
    state = engine.init_state(key)

    history_train_loss = []

    # Evaluate a genome to get accuracy
    def get_accuracy(genome, X_data, y_data):
        test_evaluator = ClassificationNeuralCartesianEvaluator(config=eval_config, data=(X_data, y_data))
        loss = test_evaluator.evaluate(genome)
        # Re-run forward pass for predictions
        gc = config
        N = gc.num_inputs
        num_nodes = gc.num_nodes

        def predict_single(x: chex.Array) -> chex.Array:
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
            raw = final_mem[genome.out_nodes][0]
            return jax.nn.sigmoid(raw)

        probs = jax.vmap(predict_single)(X_data)
        preds = (probs > 0.5).astype(jnp.int32)
        acc = jnp.mean(preds == y_data)
        return acc, loss

    for iteration in range(args.J):
        for cycle in range(args.K):
            state, _ = engine.step(state)

        best_loss = state.best_fitness
        history_train_loss.append(float(best_loss))

        best_genome = state.best_genome

        train_acc, _ = get_accuracy(best_genome, X_train, y_train)
        test_acc, test_loss = get_accuracy(best_genome, X_test, y_test)

        print(f"Iteration {iteration+1:02d}/{args.J} | Train BCE: {best_loss:.4f} (Acc: {train_acc:.2%}) | Test BCE: {test_loss:.4f} (Acc: {test_acc:.2%})")

        # Forget Step
        k_forget, key = jax.random.split(key)
        dummy_pop = config.init_population(k_forget, size=10)
        new_genes = state.population.genes.replace(
            weights=dummy_pop.genes.weights,
            biases=dummy_pop.genes.biases
        )
        new_pop = state.population.replace(genes=new_genes)
        # Using base_evaluator for standard BCE (not LSMF wrapped)
        # In our script we actually just call the evaluator manually
        new_pop = evaluator.evaluate_population(new_pop)
        state = state.replace(population=new_pop, best_fitness=jnp.min(new_pop.fitness))

    # Print Quantitative Summary Table
    print("\n" + "=" * 60)
    print("FINAL SUMMARY: dCGPANN Classification (Breast Cancer)")
    print("=" * 60)
    print(f"{'Metric':<25} | {'Value':<20}")
    print("-" * 60)
    print(f"{'Best Train BCE Loss':<25} | {best_loss:<20.6f}")
    print(f"{'Best Train Accuracy':<25} | {train_acc:<20.2%}")
    print(f"{'Best Test BCE Loss':<25} | {test_loss:<20.6f}")
    print(f"{'Best Test Accuracy':<25} | {test_acc:<20.2%}")
    print("=" * 60)

if __name__ == "__main__":
    main()
