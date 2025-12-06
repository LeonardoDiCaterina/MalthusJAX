import jax.random as jar
import jax.numpy as jnp
import numpy as np

from malthusjax.operators.mutation.binary import BitFlipMutation
from malthusjax.operators.crossover.binary import SinglePointCrossover
from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig


def test_bitflip_kernel_parity_with_legacy():
    """Parity: legacy `_mutate_one` vs `apply_kernel` for BitFlipMutation."""

    key = jar.PRNGKey(123)

    # Create a random binary genome
    cfg = BinaryGenomeConfig(length=16)
    _, init_key = jar.split(key)
    bits = jar.bernoulli(init_key, p=0.5, shape=(cfg.length,)).astype(jnp.int8)
    genome = BinaryGenome(bits=bits)

    op = BitFlipMutation(mutation_rate=0.3)

    K = jar.PRNGKey(42)

    # Legacy single-sample call uses the provided key directly in _mutate_one
    legacy_out = op._mutate_one(K, genome, cfg)

    # Kernel should accept the same key and produce identical result
    kernel_out = op.apply_kernel(K, genome, cfg)

    np.testing.assert_array_equal(np.array(legacy_out.bits), np.array(kernel_out.bits))


def test_onepoint_kernel_parity_with_legacy():
    """Parity: legacy `_cross_one` vs `apply_kernel` for SinglePointCrossover."""

    key = jar.PRNGKey(2025)

    cfg = BinaryGenomeConfig(length=20)
    k1, k2, k3 = jar.split(key, 3)
    p1_bits = jar.bernoulli(k1, p=0.5, shape=(cfg.length,)).astype(jnp.int8)
    p2_bits = jar.bernoulli(k2, p=0.5, shape=(cfg.length,)).astype(jnp.int8)

    p1 = BinaryGenome(bits=p1_bits)
    p2 = BinaryGenome(bits=p2_bits)

    op = SinglePointCrossover()

    K = jar.PRNGKey(99)

    legacy_child = op._cross_one(K, p1, p2, cfg)
    kernel_child = op.apply_kernel(K, p1, p2, cfg)

    np.testing.assert_array_equal(np.array(legacy_child.bits), np.array(kernel_child.bits))
