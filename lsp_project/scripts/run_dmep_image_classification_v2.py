import argparse
import jax
import jax.numpy as jnp
import optax
import numpy as np

# Use sklearn for dataset
from sklearn.datasets import load_digits, fetch_openml
from sklearn.model_selection import train_test_split

# dMEP Imports
from lsp.genome.neural_linear import NeuralPrefixGenomeConfig, NeuralPrefixPopulation
from lsp.evaluator.neural_linear import NeuralPrefixEvaluatorConfig, NeuralPrefixEvaluator
from lsp.evaluator.lsmf_evaluator import LSMFEvaluator
from lsp.operators.neural_mutation import ArchitectureMutation
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.engine.schedules import TrackBest, TrackMetrics
from lsp.operators.selection import TournamentSelection
from malthusjax.operators.base import BaseCrossover
from flax import struct
from typing import Any

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
    parser = argparse.ArgumentParser(description="Run dMEP Image Classification Benchmark V2")
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

    # 4. Engine setup
    engine_params = GeneticEngineParams(
        pop_size=args.pop_size, 
        elitism=1, 
        num_generations=args.generations,
        track_best=TrackBest.FULL,
        track_metrics=TrackMetrics.ALL
    )
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
    # Using our new strongly-typed tailored population directly via init_state!
    state = engine.init_state(subkey)
    
    print(f"Starting XLA Compiled Evolution ({args.generations} generations)...")
    # Using engine.run() to push the ENTIRE nested loop into XLA
    final_state, history, _ = jax.jit(
        lambda s: engine.run(s)
    )(state)
    
    # The history object contains metrics for every generation
    print("\nTraining Metrics (Sampled every 10 gens):")
    for gen in range(0, args.generations, max(1, args.generations // 5)):
        print(f"Gen {gen:03d} | Best Train CCE: {history.best_fitness[gen]:.4f} | Mean Train CCE: {history.mean_fitness[gen]:.4f}")
            
    # Final Test Evaluation
    print("\nEvaluating best individual on test set...")
    # engine.run with TrackBest.FULL automatically retains the best global genome 
    # across ALL generations in the final state!
    best_genome = final_state.best_genome
    
    # Evaluate test loss
    test_ec = ec.replace(batch_size=None) # Full batch evaluation
    test_eval = NeuralPrefixEvaluator(config=test_ec, data=(X_test, y_test))
    
    key, test_key = jax.random.split(key)
    test_loss = test_eval.evaluate(best_genome, test_key)
    print(f"Test CCE: {test_loss:.4f}")
    
    # Use the new inference helper to easily get logits
    logits = jax.jit(test_eval.get_logits)(best_genome, X_test)
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(preds == y_test)
    
    print(f"Test Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()
