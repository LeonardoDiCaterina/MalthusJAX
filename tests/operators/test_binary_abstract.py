import jax
import jax.numpy as jnp

from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig
from malthusjax.operators.crossover.binary import SinglePointCrossover, UniformCrossover
from malthusjax.operators.mutation.binary import BitFlipMutation, ScrambleMutation, SwapMutation


def test_binary_crossover_coverage():
    config = BinaryGenomeConfig(shape=(4,))
    p1 = BinaryGenome(values=jnp.array([True, True, True, True]))
    p2 = BinaryGenome(values=jnp.array([False, False, False, False]))

    crossovers = [(UniformCrossover(), 1), (SinglePointCrossover(), 1)]

    master_key = jax.random.PRNGKey(42)
    for op, expected_keys in crossovers:
        assert op.num_keys_per_atomic_operation == expected_keys

        keys = jax.random.split(master_key, op.num_keys_per_atomic_operation)
        noise = op._generate_noise(keys, config)
        out = op._recombine_one(p1, p2, noise, config)

        assert isinstance(out, BinaryGenome)
        assert out.values.shape == config.shape


def test_binary_mutation_coverage():
    config = BinaryGenomeConfig(shape=(4,))

    # Test mutations with both bool and float to hit branching in BitFlipMutation
    g_bool = BinaryGenome(values=jnp.array([True, False, True, False]))
    g_float = BinaryGenome(values=jnp.array([1.0, 0.0, 1.0, 0.0]))

    mutations = [(BitFlipMutation(), 1), (ScrambleMutation(), 2), (SwapMutation(), 3)]

    master_key = jax.random.PRNGKey(42)
    for op, expected_keys in mutations:
        assert op.num_keys_per_atomic_operation == expected_keys

        keys = jax.random.split(master_key, op.num_keys_per_atomic_operation)
        noise = op._generate_noise(keys, config)

        out_bool = op._mutate_one(g_bool, noise, config)
        assert isinstance(out_bool, BinaryGenome)
        assert out_bool.values.shape == config.shape

        out_float = op._mutate_one(g_float, noise, config)
        assert isinstance(out_float, BinaryGenome)
        assert out_float.values.shape == config.shape
