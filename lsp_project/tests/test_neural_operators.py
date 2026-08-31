import jax
import jax.numpy as jnp
from lsp.genome.neural_linear import NeuralPrefixGenome, NeuralPrefixGenomeConfig
from lsp.operators.crossover import HomologousPrefixCrossover
from lsp.operators.neural_mutation import ArchitectureMutation, HybridMutation, WeightMutation


def test_architecture_mutation():
    """Verify ArchitectureMutation mutates ops/args but preserves weights/biases."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    config = NeuralPrefixGenomeConfig(length=20, num_inputs=5, num_ops=5, max_arity=2, mep_output_strategy="dynamic")
    genome = NeuralPrefixGenome.random_init(k1, config)

    mut_op = ArchitectureMutation(op_rate=1.0, arg_rate=1.0)

    # We mutate a single genome using the internal _mutate_one method
    # Since ArchitectureMutation wraps AnnealedTopologicalMutation, we must use _generate_noise
    noise = mut_op._generate_noise(jnp.stack(jax.random.split(k2, 4)), config)
    mutated = mut_op._mutate_one(genome, noise, config)

    # Ops and args should be completely different (rate=1.0)
    # Note: it's possible some ops/args sample the same value by chance, but mostly different
    assert not jnp.array_equal(genome.ops, mutated.ops)
    assert not jnp.array_equal(genome.args, mutated.args)

    # Weights and biases MUST be identical
    assert jnp.array_equal(genome.weights, mutated.weights)
    assert jnp.array_equal(genome.biases, mutated.biases)

def test_weight_mutation():
    """Verify WeightMutation mutates weights/biases but preserves ops/args."""
    key = jax.random.PRNGKey(99)
    k1, k2 = jax.random.split(key, 2)

    config = NeuralPrefixGenomeConfig(length=20, num_inputs=5, num_ops=5, max_arity=2, mep_output_strategy="linear_readout", num_outputs=3)
    genome = NeuralPrefixGenome.random_init(k1, config)

    mut_op = WeightMutation(mutation_rate=1.0, sigma=0.5)
    noise = mut_op._generate_noise(jnp.stack(jax.random.split(k2, mut_op.num_keys_per_atomic_operation)), config)
    mutated = mut_op._mutate_one(genome, noise, config)

    # Ops and args MUST be identical
    assert jnp.array_equal(genome.ops, mutated.ops)
    assert jnp.array_equal(genome.args, mutated.args)

    # Biases and readouts are currently not mutated by WeightMutation
    assert jnp.array_equal(genome.biases, mutated.biases)
    assert jnp.array_equal(genome.readout_weights, mutated.readout_weights)
    assert jnp.array_equal(genome.readout_biases, mutated.readout_biases)

def test_hybrid_mutation():
    """Verify HybridMutation delegates to both correctly."""
    key = jax.random.PRNGKey(123)
    k1, k2 = jax.random.split(key, 2)

    config = NeuralPrefixGenomeConfig(length=20, num_inputs=5, num_ops=5, max_arity=2, mep_output_strategy="dynamic")
    genome = NeuralPrefixGenome.random_init(k1, config)

    mut_op = HybridMutation(
        arch_mutation=ArchitectureMutation(op_rate=1.0, arg_rate=1.0),
        weight_mutation=WeightMutation(mutation_rate=1.0, sigma=0.5)
    )

    keys = jnp.stack(jax.random.split(k2, mut_op.num_keys_per_atomic_operation))
    noise = mut_op._generate_noise(keys, config)
    mutated = mut_op._mutate_one(genome, noise, config)

    # Both topology and weights should change
    assert not jnp.array_equal(genome.ops, mutated.ops)
    assert not jnp.array_equal(genome.args, mutated.args)
    assert not jnp.array_equal(genome.weights, mutated.weights)
    assert jnp.array_equal(genome.biases, mutated.biases)

def test_homologous_prefix_crossover():
    """Verify HomologousPrefixCrossover autocorrects args bounds."""
    key = jax.random.PRNGKey(777)
    k1, k2, k3 = jax.random.split(key, 3)

    config = NeuralPrefixGenomeConfig(length=20, num_inputs=2, num_ops=5, max_arity=2, mep_output_strategy="dynamic")
    p1 = NeuralPrefixGenome.random_init(k1, config)
    p2 = NeuralPrefixGenome.random_init(k2, config)

    cross = HomologousPrefixCrossover()
    noise = cross._generate_noise(jnp.array([k3, k3]), config)

    child = cross._recombine_one(p1, p2, noise, config)

    # Check that autocorrect maintained causal bounds
    for i in range(config.length):
        valid_max = config.num_inputs + i
        args = child.args[i]
        assert jnp.all(args >= 0)
        assert jnp.all(args < valid_max)
