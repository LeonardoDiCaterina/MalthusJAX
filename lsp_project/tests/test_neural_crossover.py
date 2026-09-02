import jax
import jax.numpy as jnp
from lsp.genome.neural_cartesian import NeuralCartesianGenome
from lsp.operators.neural_crossover import ContinuousBlendCrossover, HybridNeuralCrossover
from lsp.operators.cartesian_crossover import SubgraphCrossover
from lsp.genome.neural_cartesian import NeuralCartesianGenomeConfig

def test_hybrid_crossover():
    config = NeuralCartesianGenomeConfig(
        num_rows=1, num_cols=10, num_inputs=2, num_outputs=1, num_ops=3, max_arity=2
    )
    
    key = jax.random.PRNGKey(0)
    pop = config.init_population(key, 2)
    p1 = jax.tree_util.tree_map(lambda x: x[0], pop.genes)
    p2 = jax.tree_util.tree_map(lambda x: x[1], pop.genes)
    
    top_crossover = SubgraphCrossover()
    weight_crossover = ContinuousBlendCrossover()
    hybrid = HybridNeuralCrossover(topology_crossover=top_crossover, weight_crossover=weight_crossover)
    
    keys = jax.random.split(key, hybrid.num_keys_per_atomic_operation)
    noise = hybrid._generate_noise(keys, config)
    offspring = hybrid._recombine_one(p1, p2, noise, config)
    
    # Assert offspring has correct shapes
    assert offspring.ops.shape == p1.ops.shape
    assert offspring.weights.shape == p1.weights.shape
    assert offspring.biases.shape == p1.biases.shape
    
    # Since alpha could be anything, let's just ensure it's not strictly equal to p1 or p2 for weights
    # and strictly discrete for ops
    print("Hybrid crossover test passed! Offspring ops shape:", offspring.ops.shape, "Weights shape:", offspring.weights.shape)

if __name__ == "__main__":
    test_hybrid_crossover()
