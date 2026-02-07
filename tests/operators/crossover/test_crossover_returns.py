import jax
import jax.random as jr
import jax.numpy as jnp

from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig
from malthusjax.operators.crossover.binary import UniformCrossover as BinaryUniformCrossover
from malthusjax.operators.crossover.real import UniformCrossover as RealUniformCrossover


def test_binary_single_pair_returns_batched_genome():
    key = jr.PRNGKey(0)
    cfg = BinaryGenomeConfig(shape=(10,), dtype=jnp.bool_)

    p1 = BinaryGenome.random_init(jr.PRNGKey(1), cfg)
    p2 = BinaryGenome.random_init(jr.PRNGKey(2), cfg)

    op = BinaryUniformCrossover(crossover_rate=0.5, num_offspring=3)

    # _cross_fused expects keys already split into num_keys_per_atomic_operation pieces
    key = jr.PRNGKey(3)
    keys = jr.split(key, op.num_keys_per_atomic_operation)
    out = op._cross_fused(keys, p1, p2, cfg)

    # _cross_fused returns a single offspring per pair (not batched)
    assert isinstance(out, BinaryGenome)
    assert out.values.shape == cfg.shape 

def test_real_single_pair_returns_batched_genome_and_vmap():
    key = jr.PRNGKey(10)
    cfg = RealGenomeConfig(shape=(5,), bounds=(-1.0, 1.0))

    p1 = RealGenome.random_init(jr.PRNGKey(11), cfg)
    p2 = RealGenome.random_init(jr.PRNGKey(12), cfg)

    op = RealUniformCrossover(crossover_rate=0.5)  # num_offspring defaults to 1

    # _cross_fused expects keys already split into num_keys_per_atomic_operation pieces
    key = jr.PRNGKey(13)
    keys = jr.split(key, op.num_keys_per_atomic_operation)
    out = op._cross_fused(keys, p1, p2, cfg)

    assert isinstance(out, RealGenome)
    # Single offspring from _cross_fused has shape (5,)
    assert out.values.shape == (5,)

    # Test vmap of _cross_fused to simulate batched processing
    # Each call generates num_keys_per_atomic_operation keys
    def single_cross(k):
        ks = jr.split(k, op.num_keys_per_atomic_operation)
        return op._cross_fused(ks, p1, p2, cfg)

    keys = jr.split(jr.PRNGKey(20), 4)
    results = jax.vmap(single_cross)(keys)

    # results should be a RealGenome with values shape (4, 5) from vmap
    assert isinstance(results, RealGenome)
    assert results.values.shape == (4, 5)