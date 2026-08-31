import argparse
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import struct
from lsp.evaluator.lsmf_evaluator import LSMFEvaluator
from lsp.evaluator.neural_linear import NeuralPrefixEvaluator, NeuralPrefixEvaluatorConfig

# dMEP Imports
from lsp.genome.neural_linear import NeuralPrefixGenomeConfig
from lsp.operators.neural_mutation import ArchitectureMutation
from lsp.operators.selection import TournamentSelection

# Use sklearn for dataset
from sklearn.datasets import fetch_openml, load_digits
from sklearn.model_selection import train_test_split

from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.operators.base import BaseCrossover


@struct.dataclass
class DummyCrossover(BaseCrossover[Any, Any]):
    @property
    def num_keys_per_atomic_operation(self) -> int: return 1
    def _generate_noise(self, keys, config, generation=0): return keys[0]
    def _recombine_one(self, p1, p2, noise_data, config, **kwargs): return p1

def load_dataset(name: str):
    print(f"Loading {name} dataset...")
    if name.lower() == "digits":
        data = load_digits()
        X = data.data
        y = data.target
    elif name.lower() == "mnist":
        print("Fetching MNIST (this may take a minute)...")
        data = fetch_openml('mnist_784', version=1, parser='auto')
        X = data.data.values
        y = data.target.values.astype(int)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Standardize and normalize to [0, 1]
    X = X / 255.0 if name.lower() == "mnist" else X / 16.0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Cast to jnp arrays
    X_train, y_train = jnp.array(X_train), jnp.array(y_train)
    X_test, y_test = jnp.array(X_test), jnp.array(y_test)

    return X_train, X_test, y_train, y_test

def main():
    parser = argparse.ArgumentParser(description="Run dMEP Image Classification Benchmark")
    parser.add_argument("--dataset", type=str, default="digits", choices=["digits", "mnist"], help="Dataset to use")
    parser.add_argument("--generations", type=int, default=50, help="Number of evolutionary generations")
    parser.add_argument("--pop-size", type=int, default=16, help="Population size")
    parser.add_argument("--epochs", type=int, default=10, help="Lamarckian SGD epochs per generation")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for SGD and evaluation")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for Adam optimizer")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_dataset(args.dataset)
    num_features = X_train.shape[1]
    num_classes = 10

    print(f"Dataset shape: {X_train.shape} features, {num_classes} classes.")

    # 1. Genome Config
    gc = NeuralPrefixGenomeConfig(
        length=50,
        num_inputs=num_features,
        num_ops=5,
        max_arity=2,
        mep_output_strategy="linear_readout",
        num_outputs=num_classes
    )

    # 2. Base Evaluator
    ec = NeuralPrefixEvaluatorConfig(
        num_inputs=num_features,
        length=gc.length,
        batch_size=args.batch_size,
        loss_function="cce"
    )
    base_eval = NeuralPrefixEvaluator(config=ec, data=(X_train, y_train))

    # 3. Lamarckian Wrapper
    optimizer = optax.adam(learning_rate=args.lr)
    lsmf_evaluator = LSMFEvaluator(
        config=ec,
        data=(X_train, y_train),
        base_evaluator=base_eval,
        optimizer=optimizer,
        epochs=args.epochs
    )

    # Test Evaluator (full batch, no SGD)
    test_ec = ec.replace(batch_size=min(1024, len(X_test)))
    test_eval = NeuralPrefixEvaluator(config=test_ec, data=(X_test, y_test))

    # 4. Engine setup
    engine_params = GeneticEngineParams(pop_size=args.pop_size, elitism=1, num_generations=args.generations)
    selection = TournamentSelection(num_selections=args.pop_size - 1, tournament_size=2)
    engine = GeneticEngine(
        engine_params=engine_params,
        genome_config=gc,
        evaluator=lsmf_evaluator,
        mutation=ArchitectureMutation(op_rate=0.1, arg_rate=0.2),
        selection=selection,
        crossover=DummyCrossover()
    )

    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)

    print("Initializing population...")
    state = engine.init_state(subkey)

    print("Starting Evolution...")
    jitted_step = jax.jit(engine.step)
    for gen in range(args.generations):
        state, _ = jitted_step(state)

        # JIT compilation will happen on first step, so we print progress
        if gen % 5 == 0 or gen == args.generations - 1:
            best_idx = jnp.argmin(state.population.fitness)
            best_loss = state.population.fitness[best_idx]
            print(f"Gen {gen:03d} | Best Train CCE: {best_loss:.4f}")

    # Final Test Evaluation
    print("Evaluating best individual on test set...")
    best_idx = jnp.argmin(state.population.fitness)
    # Extract the best genome (we need to un-batch it)
    best_genome = jax.tree_util.tree_map(lambda x: x[best_idx], state.population.genes)

    # Evaluate test loss
    key, test_key = jax.random.split(key)
    test_loss = test_eval.evaluate(best_genome, test_key)
    print(f"Test CCE: {test_loss:.4f}")

    # To compute accuracy, we need to extract logits
    def predict(x):
        memory = jnp.zeros((num_features + gc.length,))
        memory = memory.at[:num_features].set(x)

        def eval_node(mem, idx):
            node_idx = idx - num_features
            node_args = best_genome.args[node_idx]
            gathered = mem[node_args]

            w = best_genome.weights[node_idx]
            b = best_genome.biases[node_idx]
            weighted_sum = jnp.sum(w * gathered) + b

            from lsp.evaluator.neural_linear import ACTIVATIONS_LIST
            op_idx = best_genome.ops[node_idx]
            out_val = jax.lax.switch(op_idx, ACTIVATIONS_LIST, weighted_sum)

            mem = mem.at[idx].set(out_val)
            return mem, out_val

        final_mem, all_preds = jax.lax.scan(eval_node, memory, jnp.arange(num_features, num_features + gc.length))
        logits = jnp.dot(all_preds, best_genome.readout_weights) + best_genome.readout_biases
        return logits

    logits = jax.vmap(predict)(X_test)
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(preds == y_test)

    print(f"Test Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()
