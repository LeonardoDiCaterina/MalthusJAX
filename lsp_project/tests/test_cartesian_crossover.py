import jax
import jax.numpy as jnp
import chex
from malthusjax.core.genome.cartesian_genome import CartesianGenome, CartesianGenomeConfig
from lsp.operators.cartesian_crossover import SubgraphCrossover

def get_test_genomes(key, config):
    k1, k2 = jax.random.split(key)
    p1 = CartesianGenome.random_init(k1, config)
    p2 = CartesianGenome.random_init(k2, config)
    return p1, p2

def test_subgraph_crossover_scan():
    config = CartesianGenomeConfig(
        num_inputs=2,
        num_outputs=1,
        num_rows=2,
        num_cols=5,
        num_ops=4,
        max_arity=2,
        levels_back=3,
    )
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    p1, p2 = get_test_genomes(k1, config)
    
    crossover_op = SubgraphCrossover(crossover_rate=1.0, mask_algorithm="scan")
    
    # Apply crossover (normally vmapped, but we test single)
    o1 = crossover_op.cross_single_pair(k2, p1, p2, config)
    o1 = jax.tree_util.tree_map(lambda x: x[0], o1)
    
    # Assert offspring have correct shapes
    chex.assert_equal_shape([p1.ops, o1.ops])
    chex.assert_equal_shape([p1.args, o1.args])
    chex.assert_equal_shape([p1.out_nodes, o1.out_nodes])
    
    # Output nodes should be properly constrained within (0, total_nodes)
    total_nodes = config.num_inputs + config.num_rows * config.num_cols
    assert jnp.all(o1.out_nodes >= 0) and jnp.all(o1.out_nodes < total_nodes)

def test_subgraph_crossover_matrix():
    config = CartesianGenomeConfig(
        num_inputs=2,
        num_outputs=1,
        num_rows=2,
        num_cols=5,
        num_ops=4,
        max_arity=2,
        levels_back=3,
    )
    key = jax.random.PRNGKey(43)
    k1, k2 = jax.random.split(key)
    p1, p2 = get_test_genomes(k1, config)
    
    crossover_op = SubgraphCrossover(crossover_rate=1.0, mask_algorithm="matrix")
    
    # Apply crossover (normally vmapped, but we test single)
    o1 = crossover_op.cross_single_pair(k2, p1, p2, config)
    o1 = jax.tree_util.tree_map(lambda x: x[0], o1)
    
    # Assert offspring have correct shapes
    chex.assert_equal_shape([p1.ops, o1.ops])
    chex.assert_equal_shape([p1.args, o1.args])
    chex.assert_equal_shape([p1.out_nodes, o1.out_nodes])

if __name__ == "__main__":
    test_subgraph_crossover_scan()
    test_subgraph_crossover_matrix()
    print("SubgraphCrossover tests passed!")
