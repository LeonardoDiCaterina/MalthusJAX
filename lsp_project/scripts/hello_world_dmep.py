import jax
import jax.numpy as jnp
import optax
import time

from lsp.genome.neural_linear import NeuralPrefixGenomeConfig
from lsp.evaluator.neural_linear import NeuralPrefixEvaluatorConfig, NeuralPrefixEvaluator
from lsp.adapters.dmep_adapter import build_dmep_engine

def get_toy_dataset(key, n_samples=100):
    """Generates y = x_0^2 + sin(x_1) + noise"""
    X = jax.random.uniform(key, (n_samples, 2), minval=-2.0, maxval=2.0)
    y = X[:, 0]**2 + jnp.sin(X[:, 1])
    
    # Add a tiny bit of noise
    noise_key, _ = jax.random.split(key)
    y += jax.random.normal(noise_key, shape=y.shape) * 0.05
    
    return X, y

def run_hello_world():
    print("--- Starting dMEP Hello World ---")
    key = jax.random.PRNGKey(42)
    key, data_key = jax.random.split(key)
    
    # 1. Generate Dataset
    X, y = get_toy_dataset(data_key, n_samples=100)
    print(f"Dataset generated: X shape {X.shape}, y shape {y.shape}")
    
    # 2. Configure Genome and Evaluator (Using Global Softmax!)
    genome_config = NeuralPrefixGenomeConfig(
        num_inputs=2, 
        num_outputs=1, 
        length=20,          # 20 instructions
        num_ops=6,          # standard math + trig
        max_arity=2, 
    )
    
    eval_config = NeuralPrefixEvaluatorConfig(
        num_inputs=2, 
        length=20, 
        loss_function="mse",
        mep_output_strategy="global_softmax", 
        attention_temperature=0.1 # low temp = sharper attention
    )
    
    evaluator = NeuralPrefixEvaluator(config=eval_config, data=(X, y))
    
    # 3. Build Engine
    optimizer = optax.adam(learning_rate=0.01)
    engine = build_dmep_engine(genome_config, evaluator, optimizer, pop_size=50)
    
    # 4. Initialize State
    print("Initializing evolutionary state...")
    state = engine.init_state(key)
    
    # 5. Training Loop
    num_generations = 50
    print(f"Starting evolutionary loop for {num_generations} generations...")
    
    @jax.jit
    def step_fn(state):
        return engine.step(state)
    
    start_time = time.time()
    for gen in range(num_generations):
        state, metrics = step_fn(state)
        if gen % 10 == 0:
            best_fit = -jnp.max(state.population.fitness)
            print(f"Generation {gen} | Best MSE: {best_fit:.4f}")
            
    end_time = time.time()
    
    print(f"Training completed in {end_time - start_time:.2f} seconds!")
    
    best_fitness = jnp.max(state.population.fitness)
    print(f"Best fitness found: {best_fitness:.4f} (MSE: {-best_fitness:.4f})")

if __name__ == "__main__":
    run_hello_world()
