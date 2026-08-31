import jax
import jax.numpy as jnp
from lsp.genome.linear import BasePrefixAwareGenome, PrefixGenomeConfig
from lsp.genome.neural_cartesian import NeuralCartesianGenome, NeuralCartesianGenomeConfig
from lsp.genome.neural_linear import NeuralPrefixGenome, NeuralPrefixGenomeConfig

from malthusjax.core.genome.cartesian_genome import CartesianGenome, CartesianGenomeConfig


def test_neural_prefix_genome_init():
    key = jax.random.PRNGKey(42)
    config = NeuralPrefixGenomeConfig(
        length=20,
        num_inputs=5,
        num_ops=5,
        max_arity=2,
        num_outputs=3,
        mep_output_strategy="linear_readout"
    )

    genome = NeuralPrefixGenome.random_init(key, config)

    # Check discrete topologies
    assert genome.ops.shape == (20,)
    assert genome.ops.dtype == jnp.int32
    assert genome.args.shape == (20, 2)
    assert genome.args.dtype == jnp.int32

    # Check continuous topology
    assert genome.weights.shape == (20, 2)
    assert genome.weights.dtype == jnp.float32
    assert genome.biases.shape == (20,)

    # Check multi-output parameters (Strategy: linear_readout)
    assert genome.out_nodes is None
    assert genome.readout_weights.shape == (20, 3)
    assert genome.readout_biases.shape == (3,)

def test_neural_prefix_genome_init_out_nodes():
    key = jax.random.PRNGKey(99)
    config = NeuralPrefixGenomeConfig(
        length=10,
        num_inputs=2,
        num_ops=5,
        max_arity=2,
        num_outputs=5,
        mep_output_strategy="out_nodes"
    )

    genome = NeuralPrefixGenome.random_init(key, config)
    assert genome.out_nodes.shape == (5,)
    assert jnp.all(genome.out_nodes >= 0)
    assert jnp.all(genome.out_nodes < 10)
    assert genome.readout_weights is None

def test_neural_cartesian_genome_init():
    key = jax.random.PRNGKey(77)
    config = NeuralCartesianGenomeConfig(
        num_inputs=3,
        num_outputs=2,
        num_rows=5,
        num_cols=5,
        num_ops=5,
        levels_back=3,
        max_arity=2,
    )

    genome = NeuralCartesianGenome.random_init(key, config)

    num_nodes = 5 * 5
    # Check discrete topologies
    assert genome.ops.shape == (num_nodes,)
    assert genome.ops.dtype == jnp.int32
    assert genome.args.shape == (num_nodes, 2)
    assert genome.args.dtype == jnp.int32

    # Check continuous
    assert genome.weights.shape == (num_nodes, 2)
    assert genome.weights.dtype == jnp.float32
    assert genome.biases.shape == (num_nodes,)
    assert genome.biases.dtype == jnp.float32

    # Check outputs
    assert genome.out_nodes.shape == (2,)

def test_discrete_bounds():
    """Verify that argument pointers strictly obey causal constraints."""
    key = jax.random.PRNGKey(123)

    # 1. Prefix Constraints: node i can only point to [0, N + i - 1]
    p_config = PrefixGenomeConfig(length=50, num_inputs=5, num_ops=5, max_arity=2)
    p_genome = BasePrefixAwareGenome.random_init(key, p_config)

    for i in range(50):
        # The node index is N + i
        valid_max = 5 + i
        args = p_genome.args[i]
        assert jnp.all(args >= 0)
        assert jnp.all(args < valid_max)

    # 2. Cartesian Constraints: node (c, r) can only point to nodes in [max(0, c-levels_back), c-1]
    c_config = CartesianGenomeConfig(
        num_inputs=2,
        num_outputs=1,
        num_rows=10,
        num_cols=10,
        num_ops=5,
        levels_back=3,
        max_arity=2
    )
    c_genome = CartesianGenome.random_init(key, c_config)

    for c in range(10):
        for r in range(10):
            node_idx = c * 10 + r
            args = c_genome.args[node_idx]

            # The maximum allowed index is the last node of the previous column, plus inputs
            max_allowed = 2 + (c * 10)

            assert jnp.all(args >= 0)
            assert jnp.all(args < max_allowed)
